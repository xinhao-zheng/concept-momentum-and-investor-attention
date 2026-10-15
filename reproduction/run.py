"""Locate, render, analyse or verify a manuscript figure, table or section."""

from __future__ import annotations
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code/analysis"))
from runtime_config import load_config
from prepare_inputs import preparation_plan, execute_preparation
from attention.input_checks import validate_measurement_calendar

ALIASES = {
    "Core": ["scores"],
    "schematic": ["Definitions"],
    "M3_CF1": ["CF1"],
    "M3_CF3": ["CF3"],
    "V_annual": ["V"],
    "M2_reference": ["M2"],
    "M2_decomposition": ["M2"],
    "C2/C1/C1S": ["C2", "C1", "C1S"],
    "R2/R4": ["R2", "R4"],
}
CORE = ["scores", "S2", "AGG", "D1", "B1", "B3", "D4", "V", "T2", "S9", "T3"]
MODELS = ["B2", "B2D", "E1", "Risk", "C2", "C1", "C1S", "N1", "R2", "R4", "S6bounds"]
ATTENTION = ["M2", "CF1", "CF2", "CF3", "D2"]
EXPLANATORY = {"Definitions", "Protocol"}
PREREQUISITES = {
    "scores": "Run code/analysis/core_inputs.py into <output_root>/core/cache.",
    "S2": "Core scores and ranks.",
    "AGG": "Core S2 evaluated daily results.",
    "D1": "Core scores and ranks.",
    "B1": "Core scores and ranks.",
    "B3": "Core scores and ranks.",
    "D4": "Core D1 and models B2 feature/industry caches.",
    "V": "Core D1 and D4 outputs.",
    "T2": "Core S2 outputs.",
    "S9": "Three rebuilt core input caches.",
    "T3": "models C2 full scores and core input caches.",
    "Sample": "Core input/score caches, B1 results and B2 model results.",
    "AccountMetrics": "Released daily account paths in section S17.",
    "M2": "Attention/PIT panels, C2 keys/cards and raw_snapshot/stock-main-index-data/sh000001.csv.",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def inventory():
    entries = {}
    for record in read_json(ROOT / "figures/index.json")["figures"]:
        entries[record["id"]] = {
            "kind": "figure",
            "id": record["id"],
            "directory": record["directory"],
            "title": record["display_label"],
            "modules": record["experiment_modules"],
            "record": record,
        }
    for record in read_json(ROOT / "tables/index.json"):
        entries[record["id"]] = {
            "kind": "table",
            "id": record["id"],
            "directory": record["path"],
            "title": record["title"],
            "modules": record["modules"],
            "record": record,
        }
    for record in read_json(ROOT / "sections/index.json"):
        identity = "section-s" + record["id"][1:].zfill(2)
        entries[identity] = {
            "kind": "section",
            "id": identity,
            "directory": record["path"],
            "title": record["title"],
            "modules": record["modules"],
            "record": record,
        }
    return entries


def modules(entries):
    result = []
    for entry in entries:
        for value in entry["modules"]:
            for module in ALIASES.get(value, [value]):
                if module not in result:
                    result.append(module)
    return result


def command(script, args, config=None):
    cmd = [sys.executable, "-X", "utf8", "-B", str(ROOT / script), *args]
    if config:
        cmd += ["--config", str(config.resolve())]
    return cmd


def analysis_plan(entries, args):
    selected = modules(entries)
    jobs = []
    # Order shared calculations before the analyses that consume them.
    ordering = [
        "Definitions",
        "Protocol",
        "scores",
        "S2",
        "AGG",
        "D1",
        "B1",
        "B3",
        "S9",
        *[x for x in MODELS if x != "S6bounds"],
        "D4",
        "S6bounds",
        "V",
        "T2",
        "T3",
        *ATTENTION,
        "Sample",
        "AccountMetrics",
    ]
    unknown = [x for x in selected if x not in ordering]
    if unknown:
        raise ValueError("Unknown calculation module: " + ", ".join(unknown))
    for module in [x for x in ordering if x in selected]:
        if module in EXPLANATORY:
            jobs.append(
                {
                    "modules": [module],
                    "kind": "explanation",
                    "reason": "Definitions/protocol metadata; no empirical model is fitted.",
                }
            )
            continue
        if module in CORE:
            script = "code/analysis/core/run_core.py"
            params = ["--stages", module]
        elif module in MODELS:
            script = "code/analysis/models/run.py"
            params = ["--stages", module]
        elif module in ATTENTION:
            script = "code/analysis/attention/run.py"
            params = ["--stages", module]
        elif module == "Sample":
            script = "code/analysis/sample.py"
            params = []
        elif module == "AccountMetrics":
            script = "code/tables/account_metrics.py"
            params = []
        else:
            raise ValueError("Unknown calculation module: " + module)
        jobs.append(
            {
                "modules": [module],
                "kind": "calculation",
                "script": script,
                "entry_present": (ROOT / script).is_file(),
                "prerequisite": PREREQUISITES.get(
                    module, "See the model or attention runner input contract."
                ),
                "command": command(script, params, args.config),
            }
        )
    grouped = []
    for job in jobs:
        if (
            grouped
            and job["kind"] == "calculation"
            and grouped[-1]["kind"] == "calculation"
            and job["script"] == grouped[-1]["script"]
            and job["script"]
            in [
                "code/analysis/core/run_core.py",
                "code/analysis/models/run.py",
                "code/analysis/attention/run.py",
            ]
        ):
            previous = grouped[-1]
            previous["modules"] += job["modules"]
            previous["prerequisite"] += " " + job["prerequisite"]
            params = (
                ["--stages", *previous["modules"]]
                if "/core/" in job["script"]
                else ["--stages", ",".join(previous["modules"])]
            )
            previous["command"] = command(job["script"], params, args.config)
        else:
            grouped.append(job)
    return grouped


def verify(entry):
    folder = ROOT / entry["directory"]
    checks = []
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    if entry["kind"] == "figure":
        r = entry["record"]
        for group in ["data", "figures", "reference_plot_data"]:
            for item in r.get(group, []):
                p = folder / item["file"]
                checks.append(
                    {
                        "file": str(p.relative_to(ROOT)).replace("\\", "/"),
                        "sha256_equal": p.is_file()
                        and file_sha256(p) == item["sha256"],
                    }
                )
    elif entry["kind"] == "table":
        spec = entry["record"]
        for item in spec.get("numerical_sources", []):
            p = folder / item["file"]
            checks.append(
                {
                    "file": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "sha256_equal": p.is_file() and file_sha256(p) == item["sha256"],
                }
            )
        for p in sorted((folder / "data").glob("published*.csv")):
            with p.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
            rectangular = bool(rows) and all(len(row) == len(rows[0]) for row in rows)
            checks.append(
                {
                    "file": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "rectangular_csv": rectangular,
                    "rows": max(0, len(rows) - 1),
                }
            )
            escaped = [[x.replace("|", r"\|") for x in row] for row in rows]
            expected = (
                "\n".join(
                    [
                        "| " + " | ".join(escaped[0]) + " |",
                        "| " + " | ".join(["---"] * len(escaped[0])) + " |",
                        *["| " + " | ".join(row) + " |" for row in escaped[1:]],
                    ]
                )
                + "\n"
            )
            table = folder / "tables" / (p.stem + ".md")
            checks.append(
                {
                    "file": str(table.relative_to(ROOT)).replace("\\", "/"),
                    "published_layout_matches": table.is_file()
                    and table.read_text(encoding="utf-8") == expected,
                }
            )
    else:
        checks.append(
            {"check": "navigation_present", "equal": (folder / "README.md").is_file()}
        )
    status = all(
        value
        for check in checks
        for key, value in check.items()
        if isinstance(value, bool)
    )
    return {
        "artifact": entry["id"],
        "status": "PASS" if status else "FAIL",
        "scope": "File integrity, published table layout and execution-index consistency. This mode does not rerun empirical analyses.",
        "checks": checks,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifact",
        default="all",
        help="fig-01, fig-s01, table-s01, section-s01, a comma-separated list, or all",
    )
    p.add_argument(
        "--mode", choices=["render", "analyse", "verify", "prepare"], default="verify"
    )
    p.add_argument("--config", type=Path)
    p.add_argument(
        "--plan",
        action="store_true",
        help="Print actual entry points and prerequisites without executing them",
    )
    p.add_argument("--list", action="store_true")
    p.add_argument(
        "--prepare-inputs",
        action="store_true",
        help="Construct raw research inputs before analyse; requires fresh output locations",
    )
    args = p.parse_args()
    all_entries = inventory()
    if args.prepare_inputs and args.mode != "analyse":
        p.error(
            "--prepare-inputs is used with --mode analyse; use --mode prepare for input construction alone."
        )
    if args.list:
        for x in all_entries.values():
            print(x["id"] + "\t" + x["title"] + "\t" + ",".join(x["modules"]))
        return
    names = (
        list(all_entries)
        if args.artifact == "all"
        else [x.strip().lower() for x in args.artifact.split(",")]
    )
    unknown = [x for x in names if x not in all_entries]
    if unknown:
        p.error("Unknown artifact: " + ", ".join(unknown))
    entries = [all_entries[x] for x in names]
    if args.mode == "prepare":
        config = load_config(args.config)
        jobs = preparation_plan(config)
        if args.plan:
            print(json.dumps(jobs, ensure_ascii=False, indent=2))
            return
        execute_preparation(config, jobs)
        return
    if args.mode == "verify":
        reports = [verify(x) for x in entries]
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        if any(x["status"] != "PASS" for x in reports):
            raise SystemExit(1)
        return
    if args.mode == "analyse":
        jobs = analysis_plan(entries, args)
        config = load_config(args.config)
        preparation = preparation_plan(config) if args.prepare_inputs else []
        if args.plan:
            plan = {
                "scope": "Research calculations and summaries of released account paths.",
                "input_preparation": preparation,
                "analysis_jobs": jobs,
            }
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        missing = [
            x["script"]
            for x in jobs
            if x["kind"] == "calculation" and not x["entry_present"]
        ]
        if missing:
            p.error(
                "Required analysis entry is absent: " + ", ".join(sorted(set(missing)))
            )
        if "M2" in modules(entries):
            validate_measurement_calendar(config.snapshot_root, config.cutoff)
        if preparation:
            execute_preparation(config, preparation)
        receipt_path = config.output("execution-index.json")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt = {
            "scope": "Research calculations and summaries of released account paths.",
            "artifacts": names,
            "steps": [],
        }
        for job in jobs:
            if job["kind"] == "explanation":
                print(job["reason"])
                continue
            print(
                "Executing "
                + ",".join(job["modules"])
                + ". Prerequisite: "
                + job["prerequisite"],
                flush=True,
            )
            step = {
                "modules": job["modules"],
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "command": job["command"],
            }
            receipt["steps"].append(step)
            receipt_path.write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            result = subprocess.run(job["command"], cwd=ROOT)
            step.update(
                exit_code=result.returncode,
                completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            receipt_path.write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            if result.returncode:
                raise SystemExit(result.returncode)
        receipt["status"] = "REQUESTED_MODULES_COMPLETED"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return
    for entry in entries:
        if entry["kind"] == "figure":
            cmd = command("code/plotting/render_all.py", ["--figure", entry["id"]])
        elif entry["kind"] == "table":
            cmd = command("code/tables/render_tables.py", ["--table", entry["id"]])
        else:
            print(
                entry["id"]
                + ": section navigation; no standalone figure renderer. "
                + str(ROOT / entry["directory"] / "README.md")
            )
            continue
        if args.plan:
            print(json.dumps({"artifact": entry["id"], "command": cmd}))
            continue
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
