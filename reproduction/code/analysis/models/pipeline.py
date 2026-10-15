"""Prepare model kernels, bind local inputs, and record staged execution."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_config import RuntimeConfig, load_config
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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--stages", default="B2")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare source/configuration without model fits or resampling",
    )
    return parser.parse_args()


def prepare_sources(config: RuntimeConfig, root: Path) -> None:
    """Bind configured paths in copied kernels and compile generated Python."""
    replacements = {
        "__MODELS__": root.as_posix(),
        "__SNAPSHOT__": config.snapshot_root.as_posix(),
        "__CORE_CACHE__": config.output("core/cache").as_posix(),
        "__N1__": (root / "N1").as_posix(),
        "__RESAMPLING__": (root / "resampling").as_posix(),
        "__BOUNDS__": (root / "S6bounds").as_posix(),
    }
    template = Path(__file__).parent / "kernels"
    for src in template.rglob("*"):
        if not src.is_file():
            continue
        text = src.read_text(encoding="utf8")
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        dst = root / src.relative_to(template)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf8")
        if dst.suffix == ".py":
            compile(text, str(dst), "exec")
    for directory in ["logs", "tmp"]:
        (root / directory).mkdir(exist_ok=True)


def bind_risk_protocols(config: RuntimeConfig, root: Path) -> None:
    """Seal E1 and market-risk protocols against the configured input files."""
    bindings = {
        "ranked_features": root / "B2/cache/ranked_feature_panel.parquet",
        "prices": config.output("core/cache/prices_latest.parquet"),
        "calendar": config.output("core/cache/calendar.parquet"),
        "csi300": config.path("raw_snapshot/stock-main-index-data/sh000300.csv"),
    }
    for name in ["E1", "market_risk"]:
        protocol_path = root / name / "protocol_frozen.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf8"))
        for entry in protocol["inputs"]:
            input_path = bindings[entry["role"]]
            entry.update(
                runtime_path=str(input_path),
                bytes=input_path.stat().st_size,
                sha256=file_sha256(input_path),
            )
        write_json(protocol_path, protocol)
        code = (
            root / name / ("code/run_e1.py" if name == "E1" else "run_common_risk.py")
        )
        text = code.read_text(encoding="utf8")
        text = bind_assignment(
            text,
            "EXPECTED_PROTOCOL_SHA" if name == "E1" else "EXPECTED",
            repr(file_sha256(protocol_path)),
        )
        code.write_text(text, encoding="utf8")


def prepare_control_inputs(root: Path, pit: Path) -> None:
    """Copy N1 inputs once; reject differing copies on subsequent runs."""
    control_root = root / "N1"
    inputs_dir = control_root / "run/inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
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
        dst = inputs_dir / name
        if dst.exists():
            if file_sha256(src) != file_sha256(dst):
                raise ValueError("N1 frozen input differs: " + str(dst))
        else:
            shutil.copy2(src, dst)


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    stages = set(args.stages.split(","))
    if not stages or not stages <= STAGES:
        raise ValueError(f"Unknown stages: {stages-STAGES}")
    options = (
        json.loads(config.config_file.read_text(encoding="utf-8-sig"))
        if config.config_file
        else {}
    )
    pit_mode = options.get("pit_mode", options.get("input_mode", "raw-rebuilt"))
    if pit_mode not in ("frozen-reference", "raw-rebuilt"):
        raise ValueError("pit_mode must be frozen-reference or raw-rebuilt")
    pit = (
        config.output("pit/stock_pit_covariates_daily.universe.parquet")
        if pit_mode == "raw-rebuilt"
        else config.path("input_recovery/stock_pit_covariates_daily.universe.parquet")
    )
    attention_mode = options.get(
        "attention_mode", options.get("input_mode", "raw-rebuilt")
    )
    if attention_mode not in ("frozen-reference", "raw-rebuilt"):
        raise ValueError("attention_mode must be frozen-reference or raw-rebuilt")
    attention = (
        config.output("attention-inputs/attention_daily.parquet")
        if attention_mode == "raw-rebuilt"
        else config.path("attention/inputs/attention_daily.parquet")
    )
    input_mode = options.get("input_mode", "raw-rebuilt")
    if input_mode not in ("frozen-reference", "raw-rebuilt"):
        raise ValueError("input_mode must be frozen-reference or raw-rebuilt")
    root = config.output("models")
    root.mkdir(parents=True, exist_ok=True)
    prepare_sources(config, root)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CQ_C2_RUN_NAME"] = "C2"
    records = []

    def run_script(script: Path, script_args=()) -> None:
        command = [
            sys.executable,
            "-B",
            "-X",
            "utf8",
            str(script),
            *map(str, script_args),
        ]
        record = {
            "command": command,
            "script_sha256": file_sha256(script),
            "prepared_only": args.prepare_only,
        }
        records.append(record)
        if not args.prepare_only:
            record["started_utc"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            subprocess.run(command, cwd=root, env=env, check=True)
            record["completed_utc"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()

    def run_b2_dependency(threads: int, suffix: str) -> None:
        receipt = root / f"b2_public_dependency_receipt{suffix}.json"
        inputs = {
            str(config.output("core/cache") / filename): file_sha256(
                config.output("core/cache") / filename
            )
            for filename in [
                "calendar.parquet",
                "prices_latest.parquet",
                "frozen_common_scores.parquet",
            ]
        }
        inputs.update(
            {
                str(root / "runtime_source/feature_increment" / filename): file_sha256(
                    root / "runtime_source/feature_increment" / filename
                )
                for filename in ["b2_model.py", "b2_block_sensitivity.py"]
            }
        )
        inputs[str(root / "scripts/rerun_b2.py")] = file_sha256(
            root / "scripts/rerun_b2.py"
        )
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
                if file_sha256(path) != digest:
                    raise ValueError("Completed B2 output changed: " + path)
            records.append(
                {
                    "reused_dependency": str(root / ("B2" + suffix)),
                    "input_output_hashes_checked": True,
                    "new_fits": 0,
                }
            )
            return
        settings = {
            "threads": threads,
            "suffix": suffix,
            "input_cache": str(config.output("core/cache")),
            "overlay": str(config.path("input_recovery/patched_raw")),
        }
        for filename in [
            "calendar.parquet",
            "prices_latest.parquet",
            "frozen_common_scores.parquet",
        ]:
            if not (Path(settings["input_cache"]) / filename).is_file():
                raise FileNotFoundError(
                    "Run core first: " + str(Path(settings["input_cache"]) / filename)
                )
        if input_mode == "frozen-reference":
            verify_prices(
                config.path("input_recovery/prices_latest.parquet"),
                Path(settings["input_cache"]) / "prices_latest.parquet",
                root / "price_authority_validation.json",
            )
        else:
            write_json(
                root / "price_authority_validation.json",
                {
                    "status": "SKIPPED_ARCHIVED_AUTHORITY_IN_RAW_MODE",
                    "fresh_prices": str(
                        Path(settings["input_cache"]) / "prices_latest.parquet"
                    ),
                    "sha256": file_sha256(
                        Path(settings["input_cache"]) / "prices_latest.parquet"
                    ),
                    "scope": "Uses freshly constructed core prices; no private archived price file is required.",
                },
            )
        if suffix and (root / "B2/audit/feature_freeze.json").exists():
            settings.update(
                feature_cache=str(root / "B2/cache"),
                feature_audit=str(root / "B2/audit"),
            )
        config_path = root / f"scripts/config_b2{suffix}.json"
        write_json(config_path, settings)
        run_script(root / "scripts/rerun_b2.py", ["--config", config_path])
        if not args.prepare_only:
            outputs = {
                str(f): file_sha256(f)
                for folder in ["B2" + suffix, "B2D" + suffix]
                for f in (root / folder / "data").glob("*")
                if f.suffix in (".csv", ".parquet")
            }
            write_json(
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
        run_b2_dependency(2, "")
    if stages & {"B2_threads4", "B2D_threads4", "R2", "R4"}:
        run_b2_dependency(4, "_threads4")
    if stages & {"C2", "C1", "C1S"}:
        settings = {
            "output_name": "C2",
            "input_cache": str(config.output("core/cache")),
            "pit_path": str(pit),
            "attention_path": str(attention),
            "attention_mode": attention_mode,
            "completion_file": "C2_completion.json",
            "pit_mode": pit_mode,
            "compare_archived_reference": options.get(
                "compare_archived_reference", input_mode == "frozen-reference"
            ),
        }
        config_path = root / "scripts/config_C2.json"
        write_json(config_path, settings)
        if not (root / "C2/audit/input_driver_complete.json").exists():
            run_script(
                root / "scripts/rerun_c2.py", ["--config", config_path, "--inputs-only"]
            )
        if stages & {"C1", "C1S"}:
            run_script(root / "scripts/parallel_c2_models_two_jobs.py")
    if stages & {"E1", "Risk"}:
        bind_risk_protocols(config, root)
        if "E1" in stages:
            run_script(root / "E1/code/run_e1.py", ["--execute"])
        if "Risk" in stages:
            run_script(root / "market_risk/run_common_risk.py", ["--execute"])
    if "N1" in stages:
        control_root = root / "N1"
        prepare_control_inputs(root, pit)
        if not (control_root / "run/input_manifest.json").exists():
            run_script(control_root / "strict_controls.py", ["--prepare"])
        for name in [
            "strict_controls.py",
            "verify_control_values.py",
            "verify_diagnostics.py",
        ]:
            run_script(control_root / name)
    for module in ["R2", "R4"]:
        if module not in stages:
            continue
        b2_root = root / "B2_threads4"
        inputs_dir = root / f"resampling/work/root_{module}_inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        if args.prepare_only:
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
            src = b2_root / "data" / name
            dst = inputs_dir / name
            shutil.copy2(src, dst)
            files.append(
                {
                    "source": str(src),
                    "local": str(dst.relative_to(root / "resampling")),
                    "sha256": file_sha256(dst),
                }
            )
        plan = next((root / "resampling/protocols").glob(module + "_*.md"))
        write_json(
            inputs_dir / "manifest.json",
            {
                "plan_sha256": file_sha256(plan),
                "files": files,
                "scope": "Fresh four-thread B2 output",
            },
        )
        run_script(
            root
            / "resampling/code"
            / ("root_r2_width.py" if module == "R2" else "root_r4_precision.py"),
            ["--root", root / "resampling", "--output", root / "resampling" / module],
        )
    if "S6bounds" in stages:
        import polars as pl

        replay_root = config.output("core/replay/D4")
        bounds_root = root / "S6bounds"
        (bounds_root / "inputs").mkdir(exist_ok=True)
        pl.read_parquet(replay_root / "frozen_top20_selections.parquet").select(
            "date", "code"
        ).with_columns(pl.lit("FW").alias("method")).write_parquet(
            bounds_root / "inputs/all_top20_frozen.parquet"
        )
        shutil.copy2(
            replay_root / "daily_matched_controls.csv",
            bounds_root / "inputs/daily_matched_independent.csv",
        )
        for name in ["missing_outcome_bounds.py", "verify_s6_independent.py"]:
            run_script(bounds_root / name)
    write_json(
        root / "public_execution_manifest.json",
        {
            "stages": sorted(stages),
            "prepare_only": args.prepare_only,
            "input_mode": input_mode,
            "pit_mode": pit_mode,
            "attention_mode": attention_mode,
            "commands": records,
            "scope": "Only subprocess commands with completed_utc performed new calculations. Existing completed B2 outputs may supply dependencies and are not counted again as fits.",
        },
    )


if __name__ == "__main__":
    main()
