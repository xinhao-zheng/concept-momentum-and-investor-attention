"""Construct all research inputs from raw files and published sample indices."""

from __future__ import annotations

from pathlib import Path
import argparse
import datetime
import json
import subprocess
import sys

from runtime_config import REPRODUCTION_ROOT, load_config


def preparation_plan(config):
    values = (
        json.loads(config.config_file.read_text(encoding="utf-8-sig"))
        if config.config_file
        else {}
    )
    if values.get("input_mode", "raw-rebuilt") != "raw-rebuilt":
        raise ValueError("Input preparation requires input_mode=raw-rebuilt")
    overrides = values.get("raw_inputs", {})

    def source(key, relative):
        value = Path(overrides.get(key, relative)).expanduser()
        return (
            value.resolve()
            if value.is_absolute()
            else (config.snapshot_root / value).resolve()
        )

    calendar = source("calendar", "raw_snapshot/stock-main-index-data/sh000300.csv")
    prices = source("prices", "raw_snapshot/stock-trading-data")
    pit_prices = source("pit_prices", "raw_snapshot/stock-trading-data-pro")
    attention = source("attention", "raw_snapshot/stock-popular-concept-detail")
    membership = Path(
        overrides.get(
            "membership_archive",
            REPRODUCTION_ROOT.parent / "concept-data/concept-data.zip",
        )
    )
    if not membership.is_absolute():
        membership = config.snapshot_root / membership
    membership = membership.resolve()
    overlays = {}
    for key, suffix in [
        ("price_overlay", "stock-trading-data"),
        ("pit_overlay", "stock-trading-data-pro"),
    ]:
        value = overrides.get(key)
        if key in overrides and value is None:
            overlays[key] = None
        else:
            path = source(key, "input_recovery/patched_raw/" + suffix)
            overlays[key] = path if key in overrides or path.is_dir() else None
    workers = min(config.threads, 4)
    if workers == 3:
        workers = 2
    index = REPRODUCTION_ROOT / "data/pit-observation-index.parquet"
    attention_index = REPRODUCTION_ROOT / "data/attention-observation-index.parquet"
    core = config.output("core/cache")
    pit = config.output("pit/raw-build")
    selected = config.output("pit/stock_pit_covariates_daily.universe.parquet")
    daily = config.output("attention-inputs/attention_daily.parquet")
    commands = [
        (
            "core-inputs",
            "core_inputs.py",
            [
                "--raw-prices",
                str(prices),
                "--membership-archive",
                str(membership),
                "--calendar-source",
                str(calendar),
                "--output",
                str(core),
                "--workers",
                str(workers),
            ],
            [prices, membership, calendar],
            [core],
        ),
        (
            "pit-values",
            "input_build.py",
            [
                "--daily-prices",
                str(pit_prices),
                "--market-index",
                str(calendar),
                "--output",
                str(pit),
                "--cutoff",
                config.cutoff,
                "--output-start",
                "2016-11-25",
                "--row-policy",
                "calendar",
                "--preprocessing-contract",
                "reconstructed-legacy",
                "--workers",
                str(workers),
            ],
            [pit_prices, calendar],
            [pit],
        ),
        (
            "pit-sample",
            "apply_observation_index.py",
            [
                "--features",
                str(pit),
                "--observation-index",
                str(index),
                "--output",
                str(selected),
            ],
            [index],
            [selected],
        ),
        (
            "attention-inputs",
            "attention_inputs.py",
            [
                "--raw-attention",
                str(attention),
                "--calendar",
                str(calendar),
                "--observation-index",
                str(attention_index),
                "--output",
                str(daily),
                "--workers",
                str(workers),
            ],
            [attention, calendar, attention_index],
            [
                daily,
                daily.with_suffix(".raw.parquet"),
                daily.with_suffix(".staging.parquet"),
            ],
        ),
    ]
    jobs = []
    for name, filename, params, inputs, outputs in commands:
        overlay_key = {"core-inputs": "price_overlay", "pit-values": "pit_overlay"}.get(
            name
        )
        if overlay_key and overlays[overlay_key]:
            overlay = overlays[overlay_key]
            params += [
                "--price-overlay" if name == "core-inputs" else "--overlays",
                str(overlay),
            ]
            inputs.append(overlay)
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            str(REPRODUCTION_ROOT / "code/analysis" / filename),
            *params,
        ]
        if filename == "attention_inputs.py" and config.config_file:
            cmd += ["--config", str(config.config_file)]
        jobs.append(
            {
                "module": name,
                "command": cmd,
                "required_inputs": [str(x) for x in inputs],
                "fresh_outputs": [str(x) for x in outputs],
                "entry_present": (
                    REPRODUCTION_ROOT / "code/analysis" / filename
                ).is_file(),
            }
        )
    return jobs


def check_plan(jobs):
    missing = [
        p for job in jobs for p in job["required_inputs"] if not Path(p).exists()
    ]
    occupied = [
        p
        for job in jobs
        for p in job["fresh_outputs"]
        if Path(p).exists() and (not Path(p).is_dir() or any(Path(p).iterdir()))
    ]
    if missing:
        raise FileNotFoundError(
            "Required raw input or published sample index is absent: "
            + "; ".join(missing)
        )
    if occupied:
        raise FileExistsError(
            "Input preparation requires fresh output locations: " + "; ".join(occupied)
        )
    if not all(job["entry_present"] for job in jobs):
        raise FileNotFoundError("A declared input constructor is absent")


def execute_preparation(config, jobs):
    check_plan(jobs)
    destination = config.output("input_preparation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "scope": "Raw research-input construction; no model fitting or account-engine execution",
        "input_mode": "raw-rebuilt",
        "cutoff": config.cutoff,
        "steps": [],
    }
    for job in jobs:
        step = {
            "module": job["module"],
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command": job["command"],
        }
        receipt["steps"].append(step)
        destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        result = subprocess.run(job["command"], cwd=REPRODUCTION_ROOT)
        step.update(
            exit_code=result.returncode,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            raise SystemExit(result.returncode)
    receipt["status"] = "INPUTS_CONSTRUCTED"
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    jobs = preparation_plan(config)
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return
    execute_preparation(config, jobs)


if __name__ == "__main__":
    main()
