"""Reconstruct missing-outcome weights on the declared 2349/2346-date supports."""

from pathlib import Path
import os
import json
import hashlib
import datetime
import time
import gc

for k in [
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
]:
    os.environ[k] = "2"
import polars as pl
import numpy as np
import pandas as pd

O = Path(__file__).resolve().parent
C = Path("__CORE_CACHE__")
N = O / "inputs"
F = Path("__MODELS__/B2/cache")
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
inputs = {
    k: p
    for k, p in [
        ("scores", C / "frozen_common_scores.parquet"),
        ("prices", C / "prices_latest.parquet"),
        ("calendar", C / "calendar.parquet"),
        ("features", F / "feature_panel.parquet"),
        ("industry", F / "industry_labels.parquet"),
        ("selections", N / "all_top20_frozen.parquet"),
        ("matched_daily", N / "daily_matched_independent.csv"),
    ]
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for z in iter(lambda: f.read(8388608), b""):
            h.update(z)
    return h.hexdigest()


print("BUILD shared stock-date support from copied inputs", flush=True)
cal = pl.read_parquet(inputs["calendar"]).filter(
    pl.col("date") <= datetime.datetime(2026, 8, 31)
)
imax = cal["idx"].max()
pool = (
    pl.read_parquet(inputs["scores"], columns=["date", "code"])
    .join(cal, on="date", validate="m:1")
    .filter(pl.col("idx") + 4 <= imax)
)
prices = pl.read_parquet(inputs["prices"], columns=["date", "code", "open_adj"]).join(
    cal, on="date", validate="m:1"
)
for offset, name in [(1, "entry"), (4, "exit")]:
    pool = pool.join(
        prices.select(
            "code",
            (pl.col("idx") - offset).alias("idx"),
            pl.col("open_adj").alias(name),
        ),
        on=["code", "idx"],
        how="left",
        validate="1:1",
    )
del prices
gc.collect()
pool = pool.with_columns(
    pl.when(
        pl.col("entry").is_finite()
        & (pl.col("entry") > 0)
        & pl.col("exit").is_finite()
        & (pl.col("exit") > 0)
    )
    .then(pl.col("exit") / pl.col("entry") - 1)
    .otherwise(None)
    .alias("outcome")
).drop("entry", "exit")
assert (
    len(pool) == 8814334
    and pool["date"].n_unique() == 2349
    and pool["date"].max() == datetime.datetime(2026, 8, 25)
)
features = pl.read_parquet(inputs["features"], columns=["date", "code", "float_cap"])
industry = pl.read_parquet(inputs["industry"]).with_columns(
    pl.col("date").cast(pl.Datetime("ns"))
)
pool = pool.join(features, on=["date", "code"], how="left", validate="1:1").join(
    industry, on=["date", "code"], how="left", validate="1:1"
)
del features, industry
gc.collect()
eligible = pool.filter(
    pl.col("float_cap").is_finite()
    & (pl.col("float_cap") > 0)
    & pl.col("industry").is_not_null()
    & pl.col("industry").str.strip_chars().ne("")
)
assert len(eligible) == len(
    pool
), "Eligibility changes must not silently change pool scope"
pool = (
    eligible.sort("date", "float_cap", "code")
    .with_columns(
        (
            (
                (pl.col("code").cum_count().over("date") - 1)
                * 10
                // pl.len().over("date")
            )
            + 1
        ).alias("size_decile")
    )
    .drop("float_cap")
)
del eligible
gc.collect()
sel = (
    pl.read_parquet(inputs["selections"])
    .filter(pl.col("method") == "FW")
    .select("date", "code")
    .with_columns(pl.lit(1).alias("selected"))
)
pool = pool.join(sel, on=["date", "code"], how="left", validate="1:1").with_columns(
    pl.col("selected").fill_null(0)
)
assert pool.group_by("date").agg(pl.col("selected").sum())["selected"].eq(20).all()
mat = pl.read_csv(inputs["matched_daily"], try_parse_dates=True).filter(
    (pl.col("N") == 20) & (pl.col("benchmark") == "industry_x_size")
)
paired = mat.filter(pl.col("delta").is_finite()).select(
    pl.col("date").cast(pl.Datetime("ns"))
)
assert len(paired) == 2346
removed = mat.filter(~pl.col("delta").is_finite().fill_null(False)).select("date")
removed.write_csv(O / "joint_excluded_complete_dates.csv")
results = []
dailyall = []
stats = []
for bench, keys in [
    ("pool", ["date"]),
    ("industry_x_size", ["date", "industry", "size_decile"]),
]:
    cnt = pool.group_by(keys).agg(
        pl.len().alias("members"), pl.col("selected").sum().alias("selected_members")
    )
    w = pool.join(cnt, on=keys, validate="m:1").with_columns(
        (
            pl.col("selected") / 20
            - pl.col("selected_members") / (20 * pl.col("members"))
        ).alias("weight")
    )
    max_sum = float(
        w.group_by("date").agg(pl.col("weight").sum())["weight"].abs().max()
    )
    assert max_sum < 1e-12
    # One unknown per stock-date, shared by selected and benchmark portfolios.
    day = (
        w.group_by("date")
        .agg(
            (pl.col("weight") * pl.col("outcome")).sum().alias("known_sum"),
            pl.when(pl.col("outcome").is_null() & (pl.col("weight") > 0))
            .then(pl.col("weight"))
            .otherwise(0)
            .sum()
            .alias("missing_positive_weight"),
            pl.when(pl.col("outcome").is_null() & (pl.col("weight") < 0))
            .then(pl.col("weight"))
            .otherwise(0)
            .sum()
            .alias("missing_negative_weight"),
            pl.col("outcome").null_count().alias("missing_stock_returns"),
        )
        .sort("date")
    )
    day = day.with_columns(pl.lit(bench).alias("benchmark"))
    dailyall.append(day)
    stats.append(
        {
            "benchmark": bench,
            "rows": len(w),
            "max_abs_daily_weight_sum": max_sum,
            "missing_rows": w["outcome"].null_count(),
            "same_unknown_on_both_sides": True,
        }
    )
    for scope, dd in [("all_mature_2349", day)] + (
        [
            (
                "paired_complete_2346",
                day.join(paired, on="date", how="inner", validate="1:1"),
            )
        ]
        if bench == "industry_x_size"
        else []
    ):
        for upper in [0.0, 0.2, 1.0]:
            lo = (
                dd["known_sum"]
                - dd["missing_positive_weight"]
                + upper * dd["missing_negative_weight"]
            )
            hi = (
                dd["known_sum"]
                + upper * dd["missing_positive_weight"]
                - dd["missing_negative_weight"]
            )
            results.append(
                {
                    "benchmark": bench,
                    "date_scope": scope,
                    "dates": len(dd),
                    "lower_return_support": -1.0,
                    "upper_return_support": upper,
                    "lower_spread_pp": 100 * float(lo.mean()),
                    "upper_spread_pp": 100 * float(hi.mean()),
                }
            )
    print("BOUNDS", bench, flush=True)
    del w
    gc.collect()
pd.DataFrame(results).to_csv(O / "missing_bounds_independent.csv", index=False)
pl.concat(dailyall).write_csv(O / "shared_weight_daily_components.csv")
receipt = {
    "started": started,
    "completed": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "input_hashes": {k: {"path": str(p), "sha256": sha(p)} for k, p in inputs.items()},
    "code_sha256": sha(__file__),
    "rows": len(pool),
    "dates": 2349,
    "last_formation": str(pool["date"].max()),
    "last_calendar": str(cal["date"].max()),
    "linear_weights": stats,
    "old_output_used_in_estimation": False,
    "date_scope_difference_retained": True,
    "output_sha256": sha(O / "missing_bounds_independent.csv"),
}
(O / "missing_bounds_calculation_sealed.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
)
# Original outputs are opened only after independently computed outputs are sealed.
oldpath = Path("__SNAPSHOT__/references/missing_outcome_bounds.csv")
old = pd.read_csv(oldpath)
new = pd.DataFrame(results)
j = new[new.date_scope.eq("all_mature_2349")].merge(
    old,
    on=["benchmark", "dates", "lower_return_support", "upper_return_support"],
    suffixes=("_new", "_old"),
    validate="one_to_one",
)
diff = float(
    np.max(
        abs(
            j[["lower_spread_pp_new", "upper_spread_pp_new"]].to_numpy()
            - j[["lower_spread_pp_old", "upper_spread_pp_old"]].to_numpy()
        )
    )
)
(O / "missing_bounds_original_comparison.json").write_text(
    json.dumps(
        {
            "original_path": str(oldpath),
            "original_sha256": sha(oldpath),
            "compared_same_scope_rows": len(j),
            "max_abs_difference_pp": diff,
            "original_joint_dates": old.loc[
                old.benchmark.eq("industry_x_size"), "dates"
            ].tolist(),
            "manuscript_S6_claims_joint_2346": True,
            "actual_original_bound_scope_joint_2349": True,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(pd.DataFrame(results).to_string(index=False))
print("ORIGINAL_SCOPE_MAX_ABS_PP", diff, flush=True)
