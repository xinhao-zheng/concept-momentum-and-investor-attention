"""Post-hoc WY 6-1-6 concept mapping with read-only inputs and local outputs."""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")
import numpy as np
import polars as pl

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))
import cutoff as CFG  # noqa: E402

SRC = CFG.NEW_S2_CACHE
INPUTS = [
    "gamma_past_pairs.parquet",
    "calendar.parquet",
    "prices_latest.parquet",
    "frozen_common_scores.parquet",
]
NS = [20, 100]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name, obj):
    (HERE / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def mid():
    return (
        pl.col("date").dt.year() * 12 + pl.col("date").dt.month().cast(pl.Int32) - 1
    ).alias("month")


def month_label(m):
    return f"{m//12:04d}-{m%12+1:02d}"


def freeze():
    if (HERE / "frozen_spec.json").exists():
        raise FileExistsError("Already frozen; do not overwrite.")
    dump(
        "frozen_spec.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "post_hoc": True,
            "stage": "before monthly selections and outcomes",
            "rerun_branch": "2026-08-31 unified observation cutoff; design unchanged",
            "observation_cutoff": CFG.OBSERVATION_CUTOFF,
            "design": "gamma",
            "formation_months": 6,
            "beta_months": 3,
            "beta_min_pairs": 20,
            "variance_ss_min": 1e-15,
            "own_return_rule": "All market-calendar days in six full months must have finite return > -1",
            "skip_whole_months": 1,
            "holding_whole_months": 6,
            "pool_min": 300,
            "every_3x10_cell_min": 10,
            "N": NS,
            "sorting": "Independent full-pool beta terciles and individual momentum deciles; deterministic code tie-breaks",
            "outcome": "open_adj[first day m+8] / open_adj[first day m+2] - 1",
            "missing": ["complete_observed", "fixed_denominator_missing_zero"],
            "inference": "Descriptive only; no p-values, CI, annualization or Sharpe",
            "potential_inference_gate": "At least 20 full nonoverlapping 6-month blocks; would require a new frozen plan",
            "protocol_sha256": sha(HERE / "design.md"),
            "script_sha256": sha(__file__),
            "inputs": [{"path": str(SRC / n), "sha256": sha(SRC / n)} for n in INPUTS],
        },
    )
    print("FROZEN", flush=True)


def assert_design():
    frozen = json.loads((HERE / "frozen_spec.json").read_text(encoding="utf-8"))
    assert sha(__file__) == frozen["script_sha256"], "Code changed after design freeze"
    assert sha(HERE / "design.md") == frozen["protocol_sha256"]


def build_calendar():
    cal = pl.read_parquet(SRC / "calendar.parquet").with_columns(mid())
    months = (
        cal.group_by("month")
        .agg(
            pl.col("date").min().alias("month_first"),
            pl.col("date").max().alias("formation_date"),
            pl.len().alias("calendar_n"),
        )
        .sort("month")
    )
    # Both boundary months are deliberately excluded from feature windows.
    lo, hi = months["month"].min(), months["month"].max()
    months = months.with_columns(
        ((pl.col("month") > lo) & (pl.col("month") < hi)).alias("full_month")
    )
    return cal, months


def select():
    assert_design()
    if (HERE / "selection_freeze.json").exists():
        raise FileExistsError("Selection already frozen")
    cal, months = build_calendar()
    full = months.filter(pl.col("full_month"))
    cal6 = (
        full.select("month", "calendar_n")
        .sort("month")
        .with_columns(
            pl.col("calendar_n")
            .rolling_sum_by("month", window_size="6i", min_samples=6)
            .alias("required_n6")
        )
    )
    # Read only the historical return column. No adjusted opens are projected before selection freeze.
    prices = (
        pl.scan_parquet(SRC / "prices_latest.parquet")
        .select("date", "code", "ret")
        .with_columns(mid())
    )
    pmon = (
        prices.join(full.lazy().select("month"), on="month", how="inner")
        .filter(pl.col("ret").is_finite() & (pl.col("ret") > -1))
        .group_by(["code", "month"])
        .agg(pl.col("ret").log1p().sum().alias("logsum"), pl.len().alias("n"))
        .collect()
        .sort(["code", "month"])
    )
    mom = (
        pmon.with_columns(
            pl.col("logsum")
            .rolling_sum_by("month", window_size="6i", min_samples=6)
            .over("code")
            .alias("sum6"),
            pl.col("n")
            .rolling_sum_by("month", window_size="6i", min_samples=6)
            .over("code")
            .alias("n6"),
        )
        .join(cal6.select("month", "required_n6"), on="month")
        .filter(
            pl.col("required_n6").is_not_null()
            & (pl.col("n6") == pl.col("required_n6"))
        )
        .with_columns(pl.col("sum6").exp().sub(1).alias("mom6"))
        .select("month", "code", "mom6", "n6")
    )
    print("MOM6", mom.height, flush=True)

    # Aggregate daily pairs to exact calendar months, then sum three contiguous calendar months.
    q = (
        pl.scan_parquet(SRC / "gamma_past_pairs.parquet")
        .with_columns(mid())
        .join(full.lazy().select("month", "formation_date"), on="month", how="inner")
    )
    valid = q.filter(pl.col("ret").is_finite() & pl.col("loo").is_finite())
    active = (
        valid.filter(pl.col("date") == pl.col("formation_date"))
        .select("month", "code", "concept")
        .unique()
        .collect()
    )
    bm = (
        valid.group_by(["code", "concept", "month"])
        .agg(
            pl.len().alias("n"),
            pl.col("ret").sum().alias("sx"),
            pl.col("loo").sum().alias("sy"),
            (pl.col("ret") ** 2).sum().alias("sxx"),
            (pl.col("loo") ** 2).sum().alias("syy"),
            (pl.col("ret") * pl.col("loo")).sum().alias("sxy"),
        )
        .collect()
        .sort(["code", "concept", "month"])
    )
    bm = bm.with_columns(
        [
            pl.col(c)
            .rolling_sum_by("month", window_size="3i", min_samples=1)
            .over(["code", "concept"])
            .alias(c + "3")
            for c in ["n", "sx", "sy", "sxx", "syy", "sxy"]
        ]
    )
    vx = pl.col("sxx3") - pl.col("sx3") ** 2 / pl.col("n3")
    vy = pl.col("syy3") - pl.col("sy3") ** 2 / pl.col("n3")
    cov = pl.col("sxy3") - pl.col("sx3") * pl.col("sy3") / pl.col("n3")
    bc = (
        bm.join(active, on=["month", "code", "concept"])
        .filter((pl.col("n3") >= 20) & (vx > 1e-15) & (vy > 1e-15))
        .with_columns((cov / vy).alias("beta"))
    )
    beta = bc.group_by(["month", "code"]).agg(
        pl.col("beta").mean().alias("beta3"), pl.len().alias("n_cards")
    )
    print("BETA3", beta.height, flush=True)
    fw = (
        pl.read_parquet(
            SRC / "frozen_common_scores.parquet",
            columns=["date", "code", "gamma_calendar_FW"],
        )
        .rename({"date": "formation_date", "gamma_calendar_FW": "FW"})
        .join(full.select("month", "formation_date"), on="formation_date")
    )
    pool = (
        fw.join(mom, on=["month", "code"])
        .join(beta, on=["month", "code"])
        .filter(pl.all_horizontal(pl.col(["FW", "mom6", "beta3"]).is_finite()))
    )
    # Independent ranks: retain beta rank before the separate momentum and FW sorts.
    pool = pool.sort(
        ["month", "beta3", "code"], descending=[False, True, False]
    ).with_columns(
        pl.len().over("month").alias("pool_n"),
        pl.col("code").cum_count().over("month").alias("rank_beta"),
    )
    pool = pool.with_columns(
        (
            3 - ((pl.col("rank_beta") - 1) * 3 / pl.col("pool_n")).floor().cast(pl.Int8)
        ).alias("C")
    )
    pool = pool.sort(
        ["month", "mom6", "code"], descending=[False, True, False]
    ).with_columns(pl.col("code").cum_count().over("month").alias("rank_mom"))
    pool = pool.with_columns(
        (
            10
            - ((pl.col("rank_mom") - 1) * 10 / pl.col("pool_n")).floor().cast(pl.Int8)
        ).alias("D")
    )
    counts = pool.group_by(["month", "C", "D"]).len()
    support = counts.group_by("month").agg(
        pl.col("len").min().alias("min_cell"), pl.len().alias("n_cells")
    )
    n = pool.group_by("month").agg(pl.len().alias("pool_n"))
    support = (
        full.select("month", "formation_date")
        .join(n, on="month", how="left")
        .join(support, on="month", how="left")
        .with_columns(
            (
                (pl.col("pool_n") >= 300)
                & (pl.col("min_cell") >= 10)
                & (pl.col("n_cells") == 30)
            )
            .fill_null(False)
            .alias("eligible")
        )
    )
    support = support.with_columns(
        pl.col("month")
        .map_elements(month_label, return_dtype=pl.String)
        .alias("formation_month")
    )
    support.write_csv(HERE / "past_support_all_months.csv")
    pool = pool.join(support.filter(pl.col("eligible")).select("month"), on="month")
    pool = pool.sort(
        ["month", "FW", "code"], descending=[False, True, False]
    ).with_columns(pl.col("code").cum_count().over("month").alias("rank_FW"))
    pool = pool.sort(
        ["month", "C", "mom6", "code"], descending=[False, True, True, False]
    ).with_columns(pl.col("code").cum_count().over("month").alias("rank_BC"))
    pool = pool.with_columns(
        (pl.col("month") + 2).alias("entry_month"),
        (pl.col("month") + 8).alias("exit_month"),
    )
    pool.sort(["month", "code"]).write_parquet(
        HERE / "frozen_monthly_pool.parquet", compression="zstd"
    )
    counts.join(
        support.filter(pl.col("eligible")).select("month"), on="month"
    ).write_csv(HERE / "frozen_cell_counts.csv")
    dump(
        "selection_freeze.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": sha(HERE / "frozen_monthly_pool.parquet"),
            "rows": pool.height,
            "months": pool["month"].n_unique(),
            "first_month": month_label(pool["month"].min()),
            "last_month": month_label(pool["month"].max()),
            "outcome_open_adj_not_read": True,
            "min_pool": pool["pool_n"].min(),
            "min_cell": counts.join(
                support.filter(pl.col("eligible")).select("month"), on="month"
            )["len"].min(),
        },
    )
    print("SELECTION FROZEN", pool.height, pool["month"].n_unique(), flush=True)


def agg(frame, keys):
    return (
        frame.group_by(keys)
        .agg(
            pl.len().alias("intended"),
            pl.col("r").count().alias("observed"),
            pl.col("r").mean().alias("complete"),
            pl.col("r").fill_null(0).mean().alias("zero"),
        )
        .with_columns((pl.col("observed") / pl.col("intended")).alias("coverage"))
    )


def evaluate():
    assert_design()
    f = json.loads((HERE / "selection_freeze.json").read_text(encoding="utf-8"))
    assert sha(HERE / "frozen_monthly_pool.parquet") == f["sha256"]
    _, months = build_calendar()
    pool = pl.read_parquet(HERE / "frozen_monthly_pool.parquet")
    ep = months.select(
        pl.col("month").alias("entry_month"), pl.col("month_first").alias("entry_date")
    )
    xp = months.select(
        pl.col("month").alias("exit_month"), pl.col("month_first").alias("exit_date")
    )
    pool = pool.join(ep, on="entry_month", how="left").join(
        xp, on="exit_month", how="left"
    )
    maturity = pool.group_by("month").agg(
        pl.col("entry_date").first(),
        pl.col("exit_date").first(),
        pl.len().alias("pool_n"),
    )
    maturity = maturity.with_columns(
        (pl.col("entry_date").is_not_null() & pl.col("exit_date").is_not_null()).alias(
            "mature"
        )
    )
    maturity.write_csv(HERE / "maturity.csv")
    z = pool.filter(
        pl.col("exit_date").is_not_null() & pl.col("entry_date").is_not_null()
    )
    prices = pl.read_parquet(
        SRC / "prices_latest.parquet", columns=["date", "code", "open_adj"]
    )
    z = z.join(
        prices.rename({"date": "entry_date", "open_adj": "entry_open"}),
        on=["entry_date", "code"],
        how="left",
    ).join(
        prices.rename({"date": "exit_date", "open_adj": "exit_open"}),
        on=["exit_date", "code"],
        how="left",
    )
    z = z.with_columns(
        pl.when(
            pl.col("entry_open").is_finite()
            & (pl.col("entry_open") > 0)
            & pl.col("exit_open").is_finite()
            & (pl.col("exit_open") > 0)
        )
        .then(pl.col("exit_open") / pl.col("entry_open") - 1)
        .alias("r")
    )
    assert z.select((pl.col("formation_date") < pl.col("entry_date")).all()).item()
    assert z.select((pl.col("entry_date") < pl.col("exit_date")).all()).item()
    z.write_parquet(HERE / "monthly_endpoints.parquet", compression="zstd")
    cells = agg(z, ["month", "C", "D"]).sort(["month", "C", "D"])
    cells.write_csv(HERE / "monthly_cells.csv")
    base = agg(z, ["month"]).sort("month")
    base.write_csv(HERE / "monthly_pool_returns.csv")
    tops = []
    for method in ["FW", "BC"]:
        for n in NS:
            x = agg(z.filter(pl.col("rank_" + method) <= n), ["month"]).with_columns(
                pl.lit(method).alias("method"), pl.lit(n).alias("N")
            )
            assert x.select((pl.col("intended") == n).all()).item()
            tops.append(x)
    tops = pl.concat(tops).sort(["N", "method", "month"])
    tops.write_csv(HERE / "monthly_top_returns.csv")
    rows = []
    for m in sorted(z["month"].unique().to_list()):
        cell = {
            (int(a["C"]), int(a["D"])): a
            for a in cells.filter(pl.col("month") == m).to_dicts()
        }
        ts = {
            (a["method"], int(a["N"])): a
            for a in tops.filter(pl.col("month") == m).to_dicts()
        }
        b = base.filter(pl.col("month") == m).to_dicts()[0]
        for miss in ["complete", "zero"]:
            for c in [1, 2, 3]:
                val = cell[(c, 10)][miss] - cell[(c, 1)][miss]
                rows.append(
                    {
                        "month": m,
                        "label": f"C{c}_winner_minus_loser",
                        "N": 0,
                        "missing": miss,
                        "value": val,
                    }
                )
            win = cell[(3, 10)][miss] - cell[(1, 10)][miss]
            dd = win - (cell[(3, 1)][miss] - cell[(1, 1)][miss])
            rows.extend(
                [
                    {
                        "month": m,
                        "label": "C3_minus_C1_winners",
                        "N": 0,
                        "missing": miss,
                        "value": win,
                    },
                    {
                        "month": m,
                        "label": "double_difference",
                        "N": 0,
                        "missing": miss,
                        "value": dd,
                    },
                ]
            )
            for n in NS:
                fw, bc = ts[("FW", n)][miss], ts[("BC", n)][miss]
                for label, value in [
                    ("FW_minus_BC", fw - bc),
                    ("FW_minus_pool", fw - b[miss]),
                    ("BC_minus_pool", bc - b[miss]),
                ]:
                    rows.append(
                        {
                            "month": m,
                            "label": label,
                            "N": n,
                            "missing": miss,
                            "value": value,
                        }
                    )
    contrasts = pl.DataFrame(rows).sort(["label", "N", "missing", "month"])
    contrasts.write_csv(HERE / "monthly_contrasts.csv")
    summ = (
        contrasts.group_by(["label", "N", "missing"])
        .agg(
            pl.col("value").count().alias("months"),
            pl.col("value").mean().alias("mean"),
            pl.col("value").median().alias("median"),
            pl.col("value").std().alias("std"),
            pl.col("value").quantile(0.1).alias("p10"),
            pl.col("value").quantile(0.9).alias("p90"),
            (pl.col("value") > 0).mean().alias("positive_fraction"),
        )
        .sort(["label", "N", "missing"])
    )
    summ.with_columns((pl.col("mean") * 100).alias("mean_pp")).write_csv(
        HERE / "descriptive_contrasts.csv"
    )
    cells.group_by(["C", "D"]).agg(
        pl.col("complete").count().alias("months"),
        pl.col("complete").mean().alias("mean_complete"),
        pl.col("zero").mean().alias("mean_zero"),
        pl.col("coverage").mean().alias("mean_coverage"),
        pl.col("coverage").min().alias("min_coverage"),
    ).sort(["C", "D"]).write_csv(HERE / "descriptive_cells.csv")
    nm = z["month"].n_unique()
    dump(
        "run_summary.json",
        {
            "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "mature_months": nm,
            "first_mature_month": month_label(z["month"].min()),
            "last_mature_month": month_label(z["month"].max()),
            "full_6month_blocks_upper_bound": nm // 6,
            "inference_gate_met": nm // 6 >= 20,
            "inference_performed": False,
            "outcome_unit": "six-calendar-month cohort gross holding return",
            "overall_endpoint_coverage": z["r"].count() / z.height,
            "cell_min_coverage": cells["coverage"].min(),
            "selection_hash_verified": True,
            "no_future_reselection": True,
            "not_exact_replication": True,
        },
    )
    print(
        "EVALUATED", nm, "months;", nm // 6, "six-month blocks upper bound", flush=True
    )


def test():
    # Boundary and independent-sort definitions, no production outcomes involved.
    jan = 2020 * 12
    assert month_label(jan + 2) == "2020-03" and month_label(jan + 8) == "2020-09"
    x = np.array([0.01, 0.02, -0.01, 0.03, 0.005])
    y = 0.04 + 2 * x
    b = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    assert abs(b - 2) < 1e-12
    q = pl.DataFrame({"m": [0, 1, 3], "x": [1.0, 2.0, 4.0]}).with_columns(
        pl.col("x").rolling_sum_by("m", window_size="3i", min_samples=1).alias("r")
    )
    assert q["r"].to_list() == [
        1.0,
        3.0,
        6.0,
    ]  # The window ending at m=3 includes m=1,2,3 and excludes m=0.
    dump(
        "unit_tests.json",
        {
            "month_skip_and_exit": "PASS",
            "OLS_with_intercept": "PASS",
            "integer_calendar_window": "PASS",
        },
    )
    print("TESTS PASS", flush=True)
