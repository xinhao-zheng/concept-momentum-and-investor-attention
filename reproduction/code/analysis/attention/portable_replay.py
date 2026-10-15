"""Prepare and run attention analyses in a separate output directory.

The complete chain uses two concurrent pipelines and two numerical threads
per process. Input hashes, source bindings and package versions are recorded."""

from pathlib import Path
import argparse
import os
import sys
import subprocess
import json
import re
import shutil
import hashlib
import datetime
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models/kernels/scripts"))
from source_bindings import bind_paths
from input_checks import validate_measurement_calendar

PACKAGE = Path(__file__).resolve().parent / "kernels"
SCI = Path(os.environ["CQ_SNAPSHOT_ROOT"])
parser = argparse.ArgumentParser()
parser.add_argument(
    "--destination", required=True, help="New, nonexistent output directory"
)
parser.add_argument(
    "--branch", choices=["all", "cf1_cf3_d2", "cf2", "m2"], default="all"
)
parser.add_argument(
    "--prepare-only",
    action="store_true",
    help="Copy and compile scripts, validate input hashes; do not estimate",
)
args = parser.parse_args()
DEST = Path(args.destination).resolve()
if DEST.exists():
    prior = DEST / "portable_replay_manifest.json"
    if (
        not prior.exists()
        or not json.loads(prior.read_text(encoding="utf8")).get("prepare_only")
        or list((DEST / "logs").glob("*_complete.json"))
    ):
        raise FileExistsError(
            "Use a new directory; completed outputs are protected: " + str(DEST)
        )
for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ[key] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["MPLBACKEND"] = "Agg"
vendor = (
    PACKAGE / "vendor"
)  # Optional import directory; install pinned dependencies in the environment.
import numpy
import pandas
import scipy
import polars
import pyarrow
import sklearn
import statsmodels
import matplotlib

versions = {
    "python": sys.version,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scipy": scipy.__version__,
    "polars": polars.__version__,
    "pyarrow": pyarrow.__version__,
    "sklearn": sklearn.__version__,
    "statsmodels": statsmodels.__version__,
    "matplotlib": matplotlib.__version__,
}
expected = {
    "numpy": "2.3.5",
    "pandas": "2.2.3",
    "scipy": "1.16.3",
    "polars": "1.41.2",
    "pyarrow": "21.0.0",
    "sklearn": "1.8.0",
}
for key, value in expected.items():
    if versions[key] != value:
        raise RuntimeError(
            f"Version mismatch for {key}: {versions[key]} vs frozen {value}; do not silently change the scientific runtime."
        )


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


inputs = SCI / "attention/inputs"
PIT = Path(
    os.environ.get(
        "CQ_PIT_PANEL",
        str(SCI / "input_recovery/stock_pit_covariates_daily.universe.parquet"),
    )
)
RAW = SCI / "raw_snapshot"
C2 = Path(os.environ.get("CQ_C2_ROOT", str(SCI / "models/C2")))
measurement_calendar = (
    validate_measurement_calendar(SCI, "2026-08-31")
    if args.branch in {"all", "m2"}
    else None
)
targets = {
    "attention_daily.parquet": Path(
        os.environ.get("CQ_ATTENTION_PANEL", str(inputs / "attention_daily.parquet"))
    ),
    "stock_pit_covariates_daily.universe.parquet": PIT,
    "sh000300.csv": RAW / "stock-main-index-data/sh000300.csv",
    "沪深_1h.csv": inputs / "沪深_1h.csv",
    "沪深_24h.csv": inputs / "沪深_24h.csv",
    "mechanism_sample_keys_before_outcomes.parquet": C2
    / "cache/mechanism/mechanism_sample_keys_before_outcomes.parquet",
    "protocol_membership_cards_beta_sample_dates.parquet": C2
    / "cache/mechanism/protocol_membership_cards_beta_sample_dates.parquet",
}
orig = json.loads((PACKAGE / "input_checksums.json").read_text(encoding="utf-8"))
checked = []
for rec in orig["inputs"]:
    p = targets[Path(rec["path"]).name]
    assert p.exists(), p
    actual = sha(p)
    if os.environ.get("CQ_INPUT_MODE") != "raw-rebuilt" and not (
        (p == PIT and os.environ.get("CQ_PIT_MODE") == "raw-rebuilt")
        or (
            p == targets["attention_daily.parquet"]
            and os.environ.get("CQ_ATTENTION_MODE") == "raw-rebuilt"
        )
    ):
        assert actual == rec["sha256"], f"Input changed: {p}"
    checked.append({"path": str(p), "sha256": actual, "bytes": p.stat().st_size})
if measurement_calendar is not None:
    checked.append(
        {
            "path": str(measurement_calendar),
            "sha256": sha(measurement_calendar),
            "bytes": measurement_calendar.stat().st_size,
        }
    )
DEST.mkdir(parents=True, exist_ok=True)
for rel in [
    "logs",
    "tmp",
    "cache/matplotlib",
    "dynamics/shared",
    "config",
]:
    (DEST / rel).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(DEST / "tmp")
os.environ["MPLCONFIGDIR"] = str(DEST / "cache/matplotlib")
bindings = {
    "ATTENTION": targets["attention_daily.parquet"],
    "ATTENTION_PATH": targets["attention_daily.parquet"],
    "PIT": PIT,
    "PIT_PATH": PIT,
    "PIT_PANEL": PIT,
    "MARKET": targets["sh000300.csv"],
    "RAW_XQ": inputs,
    "C2": C2,
}
copied = []
for folder in ["dynamics", "measurement", "diagnostics"]:
    for src in (PACKAGE / folder).rglob("*"):
        if not src.is_file() or src.suffix not in (".py", ".md"):
            continue
        # Generated reports do not supply input to a fresh estimator run.
        if any(
            x
            in (
                "reports",
                "figures",
                "cache",
                "audit",
                "results",
                "figures",
                "results",
                "__pycache__",
            )
            for x in src.relative_to(PACKAGE / folder).parts
        ):
            continue
        old = src.read_text(encoding="utf-8-sig")
        new = old.replace("__SNAPSHOT__", SCI.as_posix()).replace(
            "__ATTENTION__", DEST.as_posix()
        )
        if src.suffix == ".py":
            active_bindings = dict(bindings)
            if folder == "measurement":
                active_bindings["RAW"] = RAW
            new = bind_paths(new, active_bindings)
        new = new.replace("max_workers=8", "max_workers=2").replace(
            "max_workers=6", "max_workers=2"
        )
        if src.suffix == ".py":
            compile(new, str(src), "exec")
        dest = DEST / src.relative_to(PACKAGE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(new, encoding="utf-8")
        copied.append(
            {
                "source": str(src),
                "source_sha256": sha(src),
                "destination": str(dest),
                "destination_sha256": sha(dest),
                "change": "local input-path rebinding only; scientific algorithms unchanged",
            }
        )
(DEST / "config/cutoff.py").write_text(
    f"from pathlib import Path\nOBSERVATION_CUTOFF='2026-08-31'\nEXP=Path({str(SCI)!r})\n",
    encoding="utf-8",
)
legacy_rel = Path(
    "measurement/diagnostics/float32_reference/results/daily_correlations.csv"
)
legacy = SCI / "attention" / legacy_rel
if os.environ.get("CQ_INPUT_MODE") != "raw-rebuilt" and legacy.exists():
    legacy_dest = DEST / legacy_rel
    legacy_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy, legacy_dest)
    assert sha(legacy_dest) == sha(legacy)
    checked.append(
        {
            "path": str(legacy_dest),
            "sha256": sha(legacy_dest),
            "bytes": legacy_dest.stat().st_size,
            "scope": "Historical Float32 diagnostic reference, optional and not a new result",
        }
    )
else:
    checked.append(
        {
            "role": "historical_float32_comparison",
            "status": "SKIPPED_ARCHIVED_REFERENCE_UNAVAILABLE_OR_DISABLED",
        }
    )
diagnostic = PACKAGE / "cf2_partial_fold_reaggregation.py"
t = diagnostic.read_text(encoding="utf-8")
t = re.sub(
    r"^sys\.path\.insert\(0,.*$",
    lambda _: f"sys.path.insert(0,{str(vendor)!r})",
    t,
    flags=re.M,
)
(DEST / diagnostic.name).write_text(t, encoding="utf-8")
manifest = {
    "created_at": datetime.datetime.now().isoformat(),
    "package": str(PACKAGE),
    "destination": str(DEST),
    "prepare_only": args.prepare_only,
    "versions": versions,
    "input_checks": checked,
    "scripts": copied,
    "input_mode": os.environ.get("CQ_INPUT_MODE", "frozen-reference"),
    "attention_mode": os.environ.get("CQ_ATTENTION_MODE", "frozen-reference"),
    "pit_mode": os.environ.get("CQ_PIT_MODE", "frozen-reference"),
    "boundary": "Daily attention and PIT modes explicitly select frozen references or separately raw-rebuilt indexed inputs.",
}
(DEST / "portable_replay_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
if args.prepare_only:
    print(
        json.dumps(
            {
                "status": "PREPARED_AND_COMPILED_NOT_ESTIMATED",
                "scripts": len(copied),
                "inputs": len(checked),
                "destination": str(DEST),
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0)
M3 = DEST / "dynamics"
pipelines = {
    "cf1_cf3_d2": [
        (M3 / "scripts/extensions/run_construct_level_irf.py", ["--all", "--force"]),
        (
            M3 / "extensions/feedback/code/run_cf3_bidirectional_feedback.py",
            ["--force-base", "--force-h"],
        ),
        (
            M3 / "return_bridge/run_d2_bridge.py",
            ["--split-reference", "--freeze", "--run"],
        ),
    ],
    "cf2": [
        (
            M3 / "scripts/crossmarket/run_tsk_measurement_validation.py",
            ["--stage", "build"],
        ),
        (
            M3 / "scripts/extensions/run_construct_crossplatform_freshness.py",
            ["--stage", "all", "--force"],
        ),
        (DEST / "cf2_partial_fold_reaggregation.py", []),
    ],
    "m2": [
        (DEST / "measurement/run_m2_measurement.py", []),
        (DEST / "measurement/diagnose_m2_measurement.py", []),
        (DEST / "measurement/measurement_extension/paired_rerank_audit.py", []),
        (DEST / "measurement/measurement_composition/run_composition.py", []),
        (DEST / "diagnostics/audit_m2_nonoverlap.py", []),
        (DEST / "diagnostics/recheck_m2_tail.py", []),
    ],
}


def run(name):
    stages = []
    for script, argv in pipelines[name]:
        start = datetime.datetime.now().isoformat()
        log = DEST / "logs" / f"{name}_{script.stem}.log"
        print(start, name, script.name, flush=True)
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [sys.executable, "-B", "-X", "utf8", str(script), *argv],
                cwd=DEST,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        stage = {
            "script": str(script),
            "sha256": sha(script),
            "args": argv,
            "start": start,
            "end": datetime.datetime.now().isoformat(),
            "returncode": result.returncode,
            "log": str(log),
        }
        stages.append(stage)
        (DEST / "logs" / f"{name}_progress.json").write_text(
            json.dumps(stages, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError(f"{name}/{script.name} failed: inspect {log}")
    (DEST / "logs" / f"{name}_complete.json").write_text(
        json.dumps(
            {"stages": stages, "environment": versions}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )


if args.branch == "all":
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, name) for name in ["cf1_cf3_d2", "cf2", "m2"]]
        for f in futures:
            f.result()
else:
    run(args.branch)
if args.branch in ("all", "cf1_cf3_d2"):
    d = M3 / "return_bridge/primary_split_reference/data"
    rows = pandas.read_parquet(d / "bridge_stock_date_outcomes.parquet")
    cols = [
        "future_real_corr",
        "past_real_corr",
        "future_pseudo_corr",
        "past_pseudo_corr",
    ]
    complete = rows.dropna(subset=cols)
    daily = complete.groupby("signal_date")[cols].mean()
    for p in ["future", "past"]:
        daily[p + "_real_minus_pseudo"] = (
            daily[p + "_real_corr"] - daily[p + "_pseudo_corr"]
        )
    daily["n_common_stock_dates"] = complete.groupby("signal_date").size()
    summary = pandas.DataFrame(
        [
            dict(
                metric=c,
                equal_date_mean=float(daily[c].mean()),
                n_dates=len(daily),
                n_common_stock_dates=len(complete),
                inference="descriptive_context_only",
            )
            for c in cols + ["future_real_minus_pseudo", "past_real_minus_pseudo"]
        ]
    )
    daily.reset_index().to_csv(
        d / "context_correlations_daily.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        d / "context_correlations_summary.csv", index=False, encoding="utf-8-sig"
    )
print("COMPLETE", DEST, flush=True)
