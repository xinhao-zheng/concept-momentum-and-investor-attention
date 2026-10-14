"""Execute the retained core design from newly constructed input caches."""

from __future__ import annotations
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE.parent))
from runtime_config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["scores", "S2", "AGG", "D1", "B1", "B3", "D4", "V", "T2", "S9", "T3"],
        choices=["scores", "S2", "AGG", "D1", "B1", "B3", "D4", "V", "T2", "S9", "T3"],
    )
    parser.add_argument(
        "--stage",
        choices=["scores", "S2", "AGG", "D1", "B1", "B3", "D4", "V", "T2", "S9", "T3"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--check-inputs", action="store_true")
    return parser.parse_args()


def js(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def design_note(dest, stage):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "design.md").write_text(
        "# Retained design: " + stage + "\n\n"
        "The published calculation settings are encoded in the accompanying kernel. "
        "This run records current input hashes and code fingerprints before outcome linkage. "
        "It does not establish preregistration or a new holdout. The observation cutoff is 2026-08-31.\n",
        encoding="utf-8",
    )
    if stage in ["V", "T2"]:
        shutil.copy2(dest / "design.md", dest / "protocol.md")
    if stage in ["D1", "D4"]:
        js(
            dest / "frozen_protocol.json",
            {
                "stage": stage,
                "cutoff": "2026-08-31",
                "execution_scope": "Retained published calculation; new execution receipt",
                "kernel_sha256": sha(
                    CODE
                    / "kernels"
                    / stage
                    / (
                        "run_parameter_bridge.py"
                        if stage == "D1"
                        else "run_matched_controls.py"
                    )
                ),
            },
        )


def setup(config):
    import polars as pl
    import replay_frozen_estimands as RP

    cache = config.output("core/cache")
    dest = RP.S2
    for folder in ["cache", "audit", "data"]:
        (dest / folder).mkdir(parents=True, exist_ok=True)
    for source in cache.glob("*.parquet"):
        target = dest / "cache" / source.name
        if target.exists():
            if not os.path.samefile(target, source) and sha(target) != sha(source):
                raise RuntimeError(
                    "Stale cache in the output directory: " + source.name
                )
        else:
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
    keys = pl.read_parquet(
        cache / "frozen_common_scores.parquet", columns=["date", "code"]
    )
    prices = pl.scan_parquet(cache / "prices_latest.parquet")
    duplicates = (
        prices.select(pl.len() - pl.struct("date", "code").n_unique()).collect().item()
    )
    if duplicates:
        raise RuntimeError("Duplicate stock-date observations")
    spec = {
        "cutoff": config.cutoff,
        "F": 5,
        "W": 10,
        "clock": ["calendar", "observations"],
        "designs": ["beta", "gamma"],
        "top_n": [5, 20, 100],
        "H": [1, 3, 5, 10, 20, 60, 120, 250],
        "bootstrap_B": 100000,
        "seed": 20260922,
        "block": "max(20,H)",
        "confirmation_gate": "floor(n_signal_dates/block)>=20",
        "execution_scope": "Current replay of retained design; not preregistration",
    }
    for name, value in [
        (
            "source_duplicate_dates.json",
            {
                "source_duplicate_dates": duplicates,
                "scope": "Rebuilt input cache stock-date keys",
            },
        ),
        (
            "execution_precision_amendment.json",
            {
                "bootstrap_B_execution": 100000,
                "approved_before_real_outcome_evaluation": True,
                "meaning": "Retained historical calculation setting written before this replay evaluates endpoints; not a new prospective approval",
            },
        ),
        ("frozen_specification.json", spec),
        (
            "selection_freeze.json",
            {
                "rows": len(keys),
                "dates": keys["date"].n_unique(),
                "common_scores_sha256": sha(cache / "frozen_common_scores.parquet"),
                "ranks_sha256": sha(cache / "frozen_top100_ranks.parquet"),
                "cutoff": config.cutoff,
                "execution_scope": "Fresh scoring; exact integer-ratio ranks; saved before evaluation",
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        ),
    ]:
        p = dest / "audit" / name
        if p.exists() and name == "selection_freeze.json":
            prior = json.loads(p.read_text(encoding="utf-8"))
            for key in [
                "rows",
                "dates",
                "common_scores_sha256",
                "ranks_sha256",
                "cutoff",
            ]:
                if prior[key] != value[key]:
                    raise RuntimeError(
                        "Frozen selection changed; use a new output_root"
                    )
        else:
            js(p, value)
    (dest / "audit/pre_evaluation_specification.md").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(CODE / "kernels/S2/s2_prepare.py", dest / "s2_prepare.py")
    RP.CFG.NEW_S2 = dest
    RP.CFG.NEW_S2_CACHE = dest / "cache"
    RP.CFG.NEW_S2_AUDIT = dest / "audit"
    RP.CFG.NEW_S2_DATA = dest / "data"
    return dest


def execute(stage, config):
    import replay_frozen_estimands as RP
    import rebuild_secondary_scores as RS
    import score_engine as IS
    from receipt_paths import bind_receipt_paths

    cache = config.output("core/cache")
    replay = config.output("core/replay")
    if stage == "scores":
        state_path = config.output("core/scoring_input_contract.json")
        contract = {
            "inputs": {
                n: sha(cache / n)
                for n in [
                    "calendar.parquet",
                    "prices_latest.parquet",
                    "membership_latest.parquet",
                ]
            },
            "code": {
                n: sha(CODE / n) for n in ["score_engine.py", "exact_rank_util.py"]
            },
            "cutoff": config.cutoff,
        }
        if state_path.exists():
            if json.loads(state_path.read_text(encoding="utf8")) != contract:
                raise ValueError(
                    "Scoring inputs or code changed; use a new output_root"
                )
        elif config.output("core/score_partitions").is_dir() and any(
            config.output("core/score_partitions").glob("*.parquet")
        ):
            raise ValueError(
                "Existing score partitions lack their input contract; use a new output_root"
            )
        else:
            js(state_path, contract)
        for design in ["beta", "gamma"]:
            IS.pairs(design)
            for clock in ["calendar", "observations"]:
                IS.score(design, clock)
        IS.ranks()
        return
    if stage == "S9":
        m = RP.load(
            "S9",
            "rebuild_s9_membership_lag.py",
            {"PACKAGE": config.snapshot_root, "CACHE": cache},
        )
        bind_receipt_paths(m, config.snapshot_root, config.output_root)
        m.main()
        return
    dest = setup(config)
    if stage == "S2":
        RS.s2(dest)
    elif stage == "AGG":
        design_note(replay / stage, stage)
        m = RP.load(stage, "run_aggregation_extension.py")
        m.freeze()
        m.select_only()
        d = m.evaluate()
        m.statistics(d)
        m.identity_audit()
        RP.compare(stage)
    elif stage == "D1":
        design_note(replay / stage, stage)
        RS.d1()
    elif stage == "B1":
        RS.b1()
    elif stage == "B3":
        RS.b3()
    elif stage == "D4":
        design_note(replay / stage, stage)
        RP.load(
            stage,
            "run_matched_controls.py",
            {"D1": replay / "D1/cache", "B2": config.output("models/B2/cache")},
        )
        RP.compare(stage)
    elif stage == "V":
        design_note(replay / stage, stage)
        m = RP.load(
            stage, "validate_core.py", {"D1": replay / "D1", "D4": replay / "D4"}
        )
        m.main()
        RP.compare(stage)
    elif stage == "T2":
        design_note(replay / stage, stage)
        m = RP.load(stage, "revalidate_timing_2x2.py")
        m.main()
        RP.compare(stage)
    elif stage == "T3":
        m = RP.load(
            stage,
            "run_abg80.py",
            {
                "EXP": replay,
                "PACKAGE": config.snapshot_root,
                "S2": cache,
                "C2": config.output("models/C2/cache/mechanism"),
            },
        )
        bind_receipt_paths(m, config.snapshot_root, config.output_root)
        m.evaluate()
        RP.compare(stage)
    else:
        raise ValueError(stage)


def main():
    args = parse_args()
    config = load_config(args.config)
    required = [
        config.output("core/cache") / n
        for n in [
            "calendar.parquet",
            "prices_latest.parquet",
            "membership_latest.parquet",
        ]
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run core_inputs.py first; missing: " + ", ".join(missing)
        )
    if args.check_inputs:
        print(
            json.dumps(
                {
                    "required_inputs_present": True,
                    "cutoff": config.cutoff,
                    "output_root": str(config.output_root),
                },
                indent=2,
            )
        )
        return
    if args.stage:
        execute(args.stage, config)
        return
    root = config.output("core")
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    receipt = root / "portable_execution.json"
    state = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else []
    if not isinstance(state, list):
        raise ValueError("Execution receipt must contain a list of stage events")
    for stage in args.stages:
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            "-u",
            str(CODE / "run_core.py"),
            "--stage",
            stage,
        ]
        if args.config:
            cmd += ["--config", str(args.config.resolve())]
        logfile = logs / (stage + ".log")
        row = {
            "stage": stage,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "entry_sha256": sha(CODE / "run_core.py"),
            "log": str(logfile.relative_to(config.output_root)),
        }
        state.append(row)
        js(receipt, state)
        print("START", stage, flush=True)
        with logfile.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                cmd,
                cwd=CODE,
                env=os.environ.copy(),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        row.update(
            exit_code=result.returncode,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        js(receipt, state)
        print("DONE", stage, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
