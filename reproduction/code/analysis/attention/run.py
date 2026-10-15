"""Run the retained attention estimators from explicit local inputs."""

from pathlib import Path
import argparse
import sys
import os
import subprocess
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_config import load_config
from input_checks import validate_measurement_calendar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--stages", default="M2,CF1,CF2,CF3,D2")
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    stages = set(a.stages.upper().split(","))
    allowed = {"M2", "CF1", "CF2", "CF3", "D2"}
    if not stages <= allowed:
        raise ValueError(f"Unknown stages: {stages-allowed}")
    groups = set()
    if stages & {"CF1", "CF3", "D2"}:
        groups.add("cf1_cf3_d2")
    if "CF2" in stages:
        groups.add("cf2")
    if "M2" in stages:
        groups.add("m2")
    if len(groups) > 1:
        branch = "all"
    else:
        branch = next(iter(groups))
    if branch in {"all", "m2"}:
        validate_measurement_calendar(cfg.snapshot_root, cfg.cutoff)
    env = os.environ.copy()
    env["CQ_SNAPSHOT_ROOT"] = str(cfg.snapshot_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    options = (
        json.loads(cfg.config_file.read_text(encoding="utf-8-sig"))
        if cfg.config_file
        else {}
    )
    mode = options.get("pit_mode", options.get("input_mode", "raw-rebuilt"))
    if mode not in ("frozen-reference", "raw-rebuilt"):
        raise ValueError("pit_mode must be frozen-reference or raw-rebuilt")
    attention_mode = options.get(
        "attention_mode", options.get("input_mode", "raw-rebuilt")
    )
    if attention_mode not in ("frozen-reference", "raw-rebuilt"):
        raise ValueError("attention_mode must be frozen-reference or raw-rebuilt")
    env["CQ_ATTENTION_MODE"] = attention_mode
    env["CQ_ATTENTION_PANEL"] = str(
        cfg.output("attention-inputs/attention_daily.parquet")
        if attention_mode == "raw-rebuilt"
        else cfg.path("attention/inputs/attention_daily.parquet")
    )
    input_mode = options.get("input_mode", "raw-rebuilt")
    if input_mode not in ("frozen-reference", "raw-rebuilt"):
        raise ValueError("input_mode must be frozen-reference or raw-rebuilt")
    env["CQ_INPUT_MODE"] = input_mode
    if input_mode == "raw-rebuilt":
        env["CQ_C2_ROOT"] = str(cfg.output("models/C2"))
    env["CQ_PIT_MODE"] = mode
    if mode == "raw-rebuilt":
        env["CQ_PIT_PANEL"] = str(
            cfg.output("pit/stock_pit_covariates_daily.universe.parquet")
        )
    cmd = [
        sys.executable,
        "-B",
        "-X",
        "utf8",
        str(Path(__file__).with_name("portable_replay.py")),
        "--destination",
        str(cfg.output("attention")),
        "--branch",
        branch,
    ]
    if a.prepare_only:
        cmd.append("--prepare-only")
    print(
        json.dumps(
            {
                "attention_mode": attention_mode,
                "pit_mode": mode,
                "requested_stages": sorted(stages),
                "executed_pipeline": branch,
                "dependency_note": "CF1, CF3 and D2 share one ordered pipeline. Multiple pipeline groups run the complete attention chain. Selected daily attention/PIT inputs and C2 keys/cards are explicit dependencies.",
            },
            indent=2,
        ),
        flush=True,
    )
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
