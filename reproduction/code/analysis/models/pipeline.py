"""Portable execution of author scientific kernels; no completed results are bundled."""

from pathlib import Path
import argparse, json, sys, subprocess, os, hashlib, shutil, re, datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_config import load_config
from price_validation import verify_prices

sys.path.insert(0, str(Path(__file__).parent / "kernels/scripts"))
from source_bindings import bind_assignment

STAGES = {
    "B2",
    "B2D",
    "B2_threads4",
    "B2D_threads4",
    "E1",
    "Risk",
    "C2",
    "C1",
    "C1S",
    "N1",
    "R2",
    "R4",
    "S6bounds",
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def writej(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--stages", default="B2")
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare source/configuration without model fits or resampling",
    )
    a = p.parse_args()
    cfg = load_config(a.config)
    stages = set(a.stages.split(","))
    if not stages or not stages <= STAGES:
        raise ValueError(f"Unknown stages: {stages-STAGES}")
    options = (
        json.loads(cfg.config_file.read_text(encoding="utf-8-sig"))
        if cfg.config_file
        else {}
    )
    pit_mode = options.get("pit_mode", options.get("input_mode", "raw-rebuilt"))
    if pit_mode not in ("frozen-reference", "raw-rebuilt"):
        raise ValueError("pit_mode must be frozen-reference or raw-rebuilt")
    pit = (
        cfg.output("pit/stock_pit_covariates_daily.universe.parquet")
        if pit_mode == "raw-rebuilt"
        else cfg.path("input_recovery/stock_pit_covariates_daily.universe.parquet")
    )
    attention_mode = options.get(
        "attention_mode", options.get("input_mode", "raw-rebuilt")
    )
    if attention_mode not in ("frozen-reference", "raw-rebuilt"):
        raise ValueError("attention_mode must be frozen-reference or raw-rebuilt")
    attention = (
        cfg.output("attention-inputs/attention_daily.parquet")
        if attention_mode == "raw-rebuilt"
        else cfg.path("attention/inputs/attention_daily.parquet")
    )
    input_mode = options.get("input_mode", "raw-rebuilt")
    if input_mode not in ("frozen-reference", "raw-rebuilt"):
        raise ValueError("input_mode must be frozen-reference or raw-rebuilt")
    root = cfg.output("models")
    root.mkdir(parents=True, exist_ok=True)
    repl = {
        "__MODELS__": root.as_posix(),
        "__SNAPSHOT__": cfg.snapshot_root.as_posix(),
        "__CORE_CACHE__": cfg.output("core/cache").as_posix(),
        "__N1__": (root / "N1").as_posix(),
        "__RESAMPLING__": (root / "resampling").as_posix(),
        "__BOUNDS__": (root / "S6bounds").as_posix(),
    }
    template = Path(__file__).parent / "kernels"
    for src in template.rglob("*"):
        if not src.is_file():
            continue
        text = src.read_text(encoding="utf8")
        for k, v in repl.items():
            text = text.replace(k, v)
        dst = root / src.relative_to(template)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf8")
        if dst.suffix == ".py":
            compile(text, str(dst), "exec")
    for d in ["logs", "tmp"]:
        (root / d).mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CQ_C2_RUN_NAME"] = "C2"
    records = []

    def call(script, args=()):
        cmd = [sys.executable, "-B", "-X", "utf8", str(script), *map(str, args)]
        rec = {
            "command": cmd,
            "script_sha256": sha(script),
            "prepared_only": a.prepare_only,
        }
        records.append(rec)
        if not a.prepare_only:
            rec["started_utc"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            subprocess.run(cmd, cwd=root, env=env, check=True)
            rec["completed_utc"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()

    def b2(threads, suffix):
        receipt = root / f"b2_public_dependency_receipt{suffix}.json"
        inputs = {
            str(cfg.output("core/cache") / n): sha(cfg.output("core/cache") / n)
            for n in [
                "calendar.parquet",
                "prices_latest.parquet",
                "frozen_common_scores.parquet",
            ]
        }
        inputs.update(
            {
                str(root / "runtime_source/feature_increment" / n): sha(
                    root / "runtime_source/feature_increment" / n
                )
                for n in ["b2_model.py", "b2_block_sensitivity.py"]
            }
        )
        inputs[str(root / "scripts/rerun_b2.py")] = sha(root / "scripts/rerun_b2.py")
        if (root / f"b2_completion{suffix}.json").exists():
            if not receipt.exists():
                raise ValueError(
                    "Completed branch lacks a public input/output hash receipt; use a fresh output_root."
                )
            old = json.loads(receipt.read_text(encoding="utf8"))
            if old["inputs"] != inputs:
                raise ValueError(
                    "Completed B2 dependency inputs or code changed; use a fresh output_root."
                )
            for path, digest in old["outputs"].items():
                if sha(path) != digest:
                    raise ValueError("Completed B2 output changed: " + path)
            records.append(
                {
                    "reused_dependency": str(root / ("B2" + suffix)),
                    "input_output_hashes_checked": True,
                    "new_fits": 0,
                }
            )
            return
        c = {
            "threads": threads,
            "suffix": suffix,
            "input_cache": str(cfg.output("core/cache")),
            "overlay": str(cfg.path("input_recovery/patched_raw")),
        }
        for n in [
            "calendar.parquet",
            "prices_latest.parquet",
            "frozen_common_scores.parquet",
        ]:
            if not (Path(c["input_cache"]) / n).is_file():
                raise FileNotFoundError(
                    "Run core first: " + str(Path(c["input_cache"]) / n)
                )
        if input_mode == "frozen-reference":
            verify_prices(
                cfg.path("input_recovery/prices_latest.parquet"),
                Path(c["input_cache"]) / "prices_latest.parquet",
                root / "price_authority_validation.json",
            )
        else:
            writej(
                root / "price_authority_validation.json",
                {
                    "status": "SKIPPED_ARCHIVED_AUTHORITY_IN_RAW_MODE",
                    "fresh_prices": str(
                        Path(c["input_cache"]) / "prices_latest.parquet"
                    ),
                    "sha256": sha(Path(c["input_cache"]) / "prices_latest.parquet"),
                    "scope": "Uses freshly constructed core prices; no private archived price file is required.",
                },
            )
        if suffix and (root / "B2/audit/feature_freeze.json").exists():
            c.update(
                feature_cache=str(root / "B2/cache"),
                feature_audit=str(root / "B2/audit"),
            )
        cf = root / f"scripts/config_b2{suffix}.json"
        writej(cf, c)
        call(root / "scripts/rerun_b2.py", ["--config", cf])
        if not a.prepare_only:
            outputs = {
                str(f): sha(f)
                for folder in ["B2" + suffix, "B2D" + suffix]
                for f in (root / folder / "data").glob("*")
                if f.suffix in (".csv", ".parquet")
            }
            writej(
                receipt,
                {
                    "inputs": inputs,
                    "outputs": outputs,
                    "threads": threads,
                    "new_fits": 252,
                    "source_scope": "Raw source inventories and feature hashes are retained by B2/audit. Reusing this dependency is not another raw reconstruction or refit.",
                },
            )

    if stages & {"B2", "B2D"}:
        b2(2, "")
    if stages & {"B2_threads4", "B2D_threads4", "R2", "R4"}:
        b2(4, "_threads4")
    if stages & {"C2", "C1", "C1S"}:
        c = {
            "output_name": "C2",
            "input_cache": str(cfg.output("core/cache")),
            "pit_path": str(pit),
            "attention_path": str(attention),
            "attention_mode": attention_mode,
            "completion_file": "C2_completion.json",
            "pit_mode": pit_mode,
            "compare_archived_reference": options.get(
                "compare_archived_reference", input_mode == "frozen-reference"
            ),
        }
        cf = root / "scripts/config_C2.json"
        writej(cf, c)
        if not (root / "C2/audit/input_driver_complete.json").exists():
            call(root / "scripts/rerun_c2.py", ["--config", cf, "--inputs-only"])
        if stages & {"C1", "C1S"}:
            call(root / "scripts/parallel_c2_models_two_jobs.py")
    if stages & {"E1", "Risk"}:
        bindings = {
            "ranked_features": root / "B2/cache/ranked_feature_panel.parquet",
            "prices": cfg.output("core/cache/prices_latest.parquet"),
            "calendar": cfg.output("core/cache/calendar.parquet"),
            "csi300": cfg.path("raw_snapshot/stock-main-index-data/sh000300.csv"),
        }
        for name in ["E1", "market_risk"]:
            file = root / name / "protocol_frozen.json"
            protocol = json.loads(file.read_text(encoding="utf8"))
            for z in protocol["inputs"]:
                q = bindings[z["role"]]
                z.update(runtime_path=str(q), bytes=q.stat().st_size, sha256=sha(q))
            writej(file, protocol)
            code = (
                root
                / name
                / ("code/run_e1.py" if name == "E1" else "run_common_risk.py")
            )
            text = code.read_text(encoding="utf8")
            text = bind_assignment(
                text,
                "EXPECTED_PROTOCOL_SHA" if name == "E1" else "EXPECTED",
                repr(sha(file)),
            )
            code.write_text(text, encoding="utf8")
        if "E1" in stages:
            call(root / "E1/code/run_e1.py", ["--execute"])
        if "Risk" in stages:
            call(root / "market_risk/run_common_risk.py", ["--execute"])
    if "N1" in stages:
        nr = root / "N1"
        inp = nr / "run/inputs"
        inp.mkdir(parents=True, exist_ok=True)
        for src, name in [
            (
                root / "C2/cache/mechanism/mechanism_analysis_panel.parquet",
                "panel.parquet",
            ),
            (
                root / "C2/data/mechanism/mechanism_expanding_time_folds.csv",
                "folds.csv",
            ),
            (pit, "daily_covariates.parquet"),
        ]:
            dst = inp / name
            if dst.exists():
                if sha(src) != sha(dst):
                    raise ValueError("N1 frozen input differs: " + str(dst))
            else:
                shutil.copy2(src, dst)
        if not (nr / "run/input_manifest.json").exists():
            call(nr / "strict_controls.py", ["--prepare"])
        for name in [
            "strict_controls.py",
            "verify_control_values.py",
            "verify_diagnostics.py",
        ]:
            call(nr / name)
    for module in ["R2", "R4"]:
        if module not in stages:
            continue
        b = root / "B2_threads4"
        inp = root / f"resampling/work/root_{module}_inputs"
        inp.mkdir(parents=True, exist_ok=True)
        if a.prepare_only:
            continue
        if not (root / "b2_completion_threads4.json").exists():
            raise FileNotFoundError("R2/R4 require completed B2_threads4")
        names = (
            [
                f"{kind}_H5_{year}.parquet"
                for year in range(2020, 2027)
                for kind in ["daily", "selections"]
            ]
            if module == "R2"
            else [
                f"{kind}_H{h}_{year}.parquet"
                for h in [1, 3, 5, 10, 20, 60, 120, 250]
                for year in range(2020, 2027)
                for kind in ["daily", "ic", "industry_ic"]
            ]
        )
        files = []
        for name in names + ["increment_inference.csv"]:
            src = b / "data" / name
            dst = inp / name
            shutil.copy2(src, dst)
            files.append(
                {
                    "source": str(src),
                    "local": str(dst.relative_to(root / "resampling")),
                    "sha256": sha(dst),
                }
            )
        plan = next((root / "resampling/protocols").glob(module + "_*.md"))
        writej(
            inp / "manifest.json",
            {
                "plan_sha256": sha(plan),
                "files": files,
                "scope": "Fresh four-thread B2 output",
            },
        )
        call(
            root
            / "resampling/code"
            / ("root_r2_width.py" if module == "R2" else "root_r4_precision.py"),
            ["--root", root / "resampling", "--output", root / "resampling" / module],
        )
    if "S6bounds" in stages:
        import polars as pl

        d4 = cfg.output("core/replay/D4")
        br = root / "S6bounds"
        (br / "inputs").mkdir(exist_ok=True)
        pl.read_parquet(d4 / "frozen_top20_selections.parquet").select(
            "date", "code"
        ).with_columns(pl.lit("FW").alias("method")).write_parquet(
            br / "inputs/all_top20_frozen.parquet"
        )
        shutil.copy2(
            d4 / "daily_matched_controls.csv",
            br / "inputs/daily_matched_independent.csv",
        )
        for name in ["missing_outcome_bounds.py", "verify_s6_independent.py"]:
            call(br / name)
    writej(
        root / "public_execution_manifest.json",
        {
            "stages": sorted(stages),
            "prepare_only": a.prepare_only,
            "input_mode": input_mode,
            "pit_mode": pit_mode,
            "attention_mode": attention_mode,
            "commands": records,
            "scope": "Only subprocess commands with completed_utc performed new calculations. Existing completed B2 outputs may supply dependencies and are not counted again as fits.",
        },
    )


if __name__ == "__main__":
    main()
