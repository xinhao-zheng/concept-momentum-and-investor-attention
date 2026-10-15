from pathlib import Path
import os
import sys
import importlib.util
import json
import hashlib

OUT = Path(__file__).resolve().parent
W = OUT.parents[2]
for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ[key] = "2"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(OUT / "cache/matplotlib")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import pandas as pd

M3 = OUT / "dynamics"
script = M3 / "scripts/extensions/run_construct_crossplatform_freshness.py"
spec = importlib.util.spec_from_file_location("cf2_partial_reaggregate", script)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
src = M3 / "data/extensions/CF2_crossplatform_freshness_daily_metrics_all.csv"
daily = pd.read_csv(src, parse_dates=["signal_date"])
partial = "2026H2_partial"
assert partial in set(daily["fold"])
dest = OUT / "text_only_diagnostics"
dest.mkdir(exist_ok=True)
summary = m.paired_summary(daily.loc[daily["fold"] != partial].copy())
summary.to_csv(
    dest / "CF2_without_partial_fold_summary.csv", index=False, encoding="utf-8-sig"
)
primary = summary.loc[
    summary["comparison"].eq("current_increment_X2_minus_X1")
    & summary.apply(lambda r: m.PRIMARY_METRIC.get(r["task"]) == r["metric"], axis=1)
].copy()
rotation = summary.loc[
    summary["comparison"].eq("real_current_X2_minus_rotation_X3")
    & summary.apply(lambda r: m.PRIMARY_METRIC.get(r["task"]) == r["metric"], axis=1),
    ["horizon", "task", "improvement"],
].rename(columns={"improvement": "X2_minus_X3"})
gates = primary.merge(rotation, on=["horizon", "task"], validate="one_to_one")
gates["current_increment_confirmed"] = (
    (gates["holm_p_8_headlines"] < 0.05)
    & (gates["positive_fold_count"] >= 5)
    & (gates["X2_minus_X3"] > 0)
)
gates.to_csv(
    dest / "CF2_without_partial_fold_headlines.csv", index=False, encoding="utf-8-sig"
)
counts = (
    daily[daily["sample"] == "main_common_support"]
    .groupby(["horizon", "fold"])["signal_date"]
    .nunique()
    .reset_index(name="evaluation_dates")
)
counts.to_csv(dest / "CF2_dates_by_fold.csv", index=False, encoding="utf-8-sig")
last = (
    daily[daily["sample"] == "main_common_support"]
    .groupby("horizon")["signal_date"]
    .agg(["min", "max", "nunique"])
    .reset_index()
)
last.to_csv(dest / "CF2_evaluation_date_support.csv", index=False, encoding="utf-8-sig")
row = gates[(gates.horizon == "1h") & (gates.task == "new_entry")].iloc[0]
checks = {
    "partial_dates_1h_15": int(
        counts[
            (counts.horizon == "1h") & (counts.fold == partial)
        ].evaluation_dates.iloc[0]
    )
    == 15,
    "partial_dates_24h_16": int(
        counts[
            (counts.horizon == "24h") & (counts.fold == partial)
        ].evaluation_dates.iloc[0]
    )
    == 16,
    "retention_two_only_confirmed": set(
        zip(
            gates.loc[gates.current_increment_confirmed, "horizon"],
            gates.loc[gates.current_increment_confirmed, "task"],
        )
    )
    == {("1h", "retention"), ("24h", "retention")},
    "new_entry_mean_6dp": round(float(row.improvement), 6) == 0.004424,
    "new_entry_ci_low_6dp": round(float(row.ci_low), 6) == 0.000540,
    "new_entry_ci_high_6dp": round(float(row.ci_high), 6) == 0.008337,
    "new_entry_Holm_6dp": round(float(row.holm_p_8_headlines), 6) == 0.191808,
    "new_entry_fold_signs": int(row.positive_fold_count) == 5
    and int(row.available_fold_count) == 7,
}
receipt = {
    "scope": "Reaggregation of existing partial-fold sensitivity from newly refitted OOF daily metrics; main estimates unchanged.",
    "source": str(src),
    "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    "module_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    "dropped_fold": partial,
    "B": m.B,
    "block_days": m.BLOCK_DAYS,
    "seed": m.SEED,
    "checks_against_v3_printed_claims": checks,
    "all_printed_claims_match": all(checks.values()),
}
(dest / "CF2_partial_fold_receipt.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
