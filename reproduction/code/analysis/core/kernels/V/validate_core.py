"""Reconstruction, annual stability and temporal-sensitivity checks for retained core results."""

from pathlib import Path
import os
import sys
import json
import hashlib
import datetime
import math
import platform
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))
import cutoff as CFG

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
PROTOCOL = HERE / "protocol.md"
os.environ.setdefault("POLARS_MAX_THREADS", "8")
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

S2 = CFG.NEW_S2
D4 = CFG.EXP / "D4"
D1 = CFG.EXP / "D1"
CUT = datetime.datetime.fromisoformat(CFG.OBSERVATION_CUTOFF)
NS = [5, 10, 20]
BASELINES = {
    "pool": [],
    "industry": ["industry"],
    "size": ["size_decile"],
    "industry_x_size": ["industry", "size_decile"],
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name, value):
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def say(*args):
    print(*args, flush=True)


def save(name, df):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def manifest(paths):
    out = []
    for p in paths:
        x = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
        if p.suffix == ".parquet":
            pf = pq.ParquetFile(p)
            x.update(rows=pf.metadata.num_rows, columns=pf.schema_arrow.names)
        out.append(x)
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps, kind="stable")
    adjusted = np.maximum.accumulate((len(ps) - np.arange(len(ps))) * ps[order])
    result = np.ones(len(ps))
    result[order] = np.minimum(1, adjusted)
    return result


def bootstrap(x, L, B=20000, seed=20260925):
    x = np.asarray(x, float)
    assert np.isfinite(x).all()
    n = len(x)
    rng = np.random.default_rng(seed)
    m = math.ceil(n / L)
    tail = n - L * (m - 1)
    ext = np.r_[x, x[:L]]
    cs = np.r_[0, np.cumsum(ext)]
    sums = cs[np.arange(n) + L] - cs[np.arange(n)]
    tails = cs[np.arange(n) + tail] - cs[np.arange(n)]
    starts = rng.integers(0, n, size=(B, m))
    means = (sums[starts[:, :-1]].sum(1) + tails[starts[:, -1]]) / n
    mean = x.mean()
    ci = np.quantile(means, [0.025, 0.975])
    p = (1 + np.sum(np.abs(means - mean) >= abs(mean))) / (B + 1)
    return mean, ci[0], ci[1], p


def reconstruct():
    say("Reconstructing frozen pool, rankings and exact endpoints")
    pool = pl.read_parquet(D4 / "frozen_eligible_pool.parquet")
    duplicate = pool.height - pool.unique(["date", "code"]).height
    assert duplicate == 0
    chk = pool.sort(["date", "float_cap", "code"]).with_columns(
        (
            (
                (pl.col("code").cum_count().over("date") - 1)
                * 10
                // pl.len().over("date")
            )
            + 1
        )
        .cast(pl.Int8)
        .alias("size_rebuilt")
    )
    bad_size = chk.filter(pl.col("size_decile") != pl.col("size_rebuilt")).height
    assert bad_size == 0
    del chk
    from exact_rank_util import rank_table

    ranked = (
        rank_table(pool, "FW", "FW", {})
        .filter(pl.col("rank") <= 20)
        .select("date", "code", "rank")
        .join(pool, on=["date", "code"])
        .sort(["date", "rank"])
    )
    saved_rank = pl.read_parquet(D4 / "frozen_top20_selections.parquet")
    keys = ["date", "code", "rank"]
    missing_rank = (
        ranked.select(keys).join(saved_rank.select(keys), on=keys, how="anti").height
    )
    extra_rank = (
        saved_rank.select(keys).join(ranked.select(keys), on=keys, how="anti").height
    )
    assert missing_rank == extra_rank == 0
    ranked.write_parquet(OUT / "reconstructed_frozen_top20.parquet")
    cal = pl.read_parquet(S2 / "cache/calendar.parquet")
    assert cal["date"].n_unique() == cal.height and cal["idx"].n_unique() == cal.height
    maxidx = cal.filter(pl.col("date") <= CUT)["idx"].max()
    prices = pl.read_parquet(
        S2 / "cache/prices_latest.parquet", columns=["date", "code", "open_adj"]
    ).join(cal, on="date", how="inner")
    entry = prices.select(
        "code",
        (pl.col("idx") - 1).alias("idx"),
        pl.col("open_adj").alias("entry"),
        pl.col("date").alias("entry_date"),
    )
    exitp = prices.select(
        "code",
        (pl.col("idx") - 4).alias("idx"),
        pl.col("open_adj").alias("exit"),
        pl.col("date").alias("exit_date"),
    )
    ev = (
        pool.filter(pl.col("idx") <= maxidx - 4)
        .join(entry, on=["idx", "code"], how="left")
        .join(exitp, on=["idx", "code"], how="left")
    )
    valid = (
        pl.col("entry").is_finite()
        & (pl.col("entry") > 0)
        & pl.col("exit").is_finite()
        & (pl.col("exit") > 0)
    )
    ev = ev.with_columns(
        pl.when(valid)
        .then(pl.col("exit") / pl.col("entry") - 1)
        .otherwise(None)
        .alias("r3")
    )
    ev = ev.with_columns(pl.col("r3").fill_null(0).alias("r3_zero"))
    selected = ranked.select("date", "code", "rank").join(
        ev, on=["date", "code"], how="inner"
    )
    selected.write_parquet(OUT / "independent_selected_endpoints.parquet")
    selected.filter(pl.col("r3").is_null()).write_csv(
        OUT / "missing_selected_endpoints.csv"
    )
    del pool, prices, entry, exitp
    daily = []
    for baseline, groups in BASELINES.items():
        gkeys = ["date"] + groups
        groupmeans = ev.group_by(gkeys).agg(
            pl.col("r3").mean().alias("group_mean"),
            pl.col("r3_zero").mean().alias("group_zero"),
        )
        sel = selected.join(groupmeans, on=gkeys, how="left")
        for n in NS:
            d = (
                sel.filter(pl.col("rank") <= n)
                .group_by(["date", "idx"])
                .agg(
                    pl.len().alias("intended"),
                    pl.col("r3").count().alias("observed"),
                    pl.col("r3").mean().alias("actual"),
                    pl.col("r3_zero").mean().alias("actual_zero"),
                    pl.col("group_mean").count().alias("valid_group_members"),
                    pl.col("group_mean").mean().alias("group_mean"),
                    pl.col("group_zero").mean().alias("group_mean_zero"),
                )
            )
            d = d.with_columns(
                pl.when(pl.col("valid_group_members") == n)
                .then(pl.col("group_mean"))
                .otherwise(None)
                .alias("group_mean")
            )
            d = d.with_columns(
                (pl.col("actual") - pl.col("group_mean")).alias("delta"),
                (pl.col("actual_zero") - pl.col("group_mean_zero")).alias("delta_zero"),
                pl.lit(n).alias("N"),
                pl.lit(baseline).alias("benchmark"),
            )
            d = d.with_columns(
                (
                    pl.col("actual").is_not_null() & pl.col("group_mean").is_not_null()
                ).alias("paired_day")
            )
            daily.append(d)
    d = pl.concat(daily).sort(["date", "N", "benchmark"])
    d.write_csv(OUT / "daily_reconstructed.csv")
    old = pl.read_csv(
        D4 / "daily_matched_controls.csv", try_parse_dates=True
    ).with_columns(pl.col("date").cast(pl.Datetime("ns")))
    paired = d.join(
        old.select(
            [
                "date",
                "N",
                "benchmark",
                "actual",
                "actual_zero",
                "group_mean",
                "group_mean_zero",
                "delta",
                "delta_zero",
                "paired_day",
            ]
        ),
        on=["date", "N", "benchmark"],
        suffix="_saved",
    )
    errors = {
        c: paired.select((pl.col(c) - pl.col(c + "_saved")).abs().max()).item()
        for c in [
            "actual",
            "actual_zero",
            "group_mean",
            "group_mean_zero",
            "delta",
            "delta_zero",
        ]
    }
    null_mismatch = {
        c: paired.filter(pl.col(c).is_null() != pl.col(c + "_saved").is_null()).height
        for c in ["actual", "group_mean", "delta"]
    }
    assert paired.height == d.height == old.height
    assert max(errors.values()) < 1e-12 and max(null_mismatch.values()) == 0
    stats = {
        "pool_stock_dates": ev.height,
        "mature_dates": ev["date"].n_unique(),
        "pool_missing_returns": ev["r3"].null_count(),
        "first_signal": str(ev["date"].min()),
        "last_signal": str(ev["date"].max()),
        "max_exit": str(ev["exit_date"].max()),
        "frozen_rank_rows": ranked.height,
        "rank_key_disagreements": missing_rank + extra_rank,
        "size_decile_disagreements": bad_size,
        "duplicate_stock_dates": duplicate,
        "max_abs_daily_errors": errors,
        "null_pattern_disagreements": null_mismatch,
        "selected_missing_top20": selected["r3"].null_count(),
        "selected_intended_top20": selected.height,
        "source_cutoff": CFG.OBSERVATION_CUTOFF,
        "not_certified": "Historical first-publication availability, source accuracy and survivorship",
    }
    pdaily = d.to_pandas()
    pdaily["date"] = pd.to_datetime(pdaily.date)
    summary = (
        pdaily.groupby(["N", "benchmark"], sort=False)
        .agg(
            dates=("delta", "count"),
            mean_pp=("delta", lambda x: x.mean() * 100),
            zero_mean_pp=("delta_zero", lambda x: x.mean() * 100),
            actual_mean_pp=("actual", lambda x: x.mean() * 100),
        )
        .reset_index()
    )
    save("main_reconstruction_summary.csv", summary)
    dump("core_qa.json", stats)
    say("Core complete", stats)
    return pdaily


def temporal(daily):
    say("Prespecified calendar and block-length sensitivities")
    d = daily[
        (daily.N == 20) & daily.benchmark.isin(["pool", "industry_x_size"])
    ].copy()
    d["year"] = d.date.dt.year
    annual = (
        d.groupby(["benchmark", "year"])
        .agg(
            n_dates=("delta", "count"),
            mean_pp=("delta", lambda x: x.mean() * 100),
            zero_mean_pp=("delta_zero", lambda x: x.mean() * 100),
            positive_day_share=("delta", lambda x: (x.dropna() > 0).mean()),
            first=("date", "min"),
            last=("date", "max"),
        )
        .reset_index()
    )
    annual["partial_year"] = annual.year.isin([2016, 2026])
    save("annual_primary_spreads.csv", annual)
    leave = []
    for baseline, df in d.groupby("benchmark"):
        for year in sorted(df.year.unique()):
            z = df[df.year != year]
            leave.append(
                {
                    "benchmark": baseline,
                    "omitted_year": year,
                    "n_dates": z.delta.count(),
                    "mean_pp": z.delta.mean() * 100,
                    "zero_mean_pp": z.delta_zero.mean() * 100,
                }
            )
    save("leave_one_year_out.csv", pd.DataFrame(leave))
    offsets = (
        d.groupby(["benchmark", d.idx % 3])
        .agg(
            n_dates=("delta", "count"),
            mean_pp=("delta", lambda x: x.mean() * 100),
            zero_mean_pp=("delta_zero", lambda x: x.mean() * 100),
        )
        .reset_index()
        .rename(columns={"idx": "calendar_offset_mod3"})
    )
    save("nonoverlap_h3_offsets.csv", offsets)
    rows = []
    for L in [20, 40, 60]:
        family = []
        for baseline, df in d.groupby("benchmark"):
            for label, y0, y1 in [
                ("2017-2021", 2017, 2021),
                ("2022-2026_partial", 2022, 2026),
            ]:
                z = df[df.year.between(y0, y1)].sort_values("date")
                z = z[z.delta.notna()]
                x = z.delta.to_numpy() * 100
                m, lo, hi, p = bootstrap(x, L, seed=20260925 + L)
                family.append(
                    {
                        "benchmark": baseline,
                        "period": label,
                        "n_dates": len(z),
                        "block_length": L,
                        "replicates": 20000,
                        "mean_pp": m,
                        "lo_pp": lo,
                        "hi_pp": hi,
                        "p_raw": p,
                        "nominal_blocks": len(z) // L,
                        "calendar_gaps": int((z.idx.diff().dropna() > 1).sum()),
                        "classification": "retrospective sensitivity, not holdout",
                    }
                )
        ps = holm([x["p_raw"] if x["nominal_blocks"] >= 20 else 1.0 for x in family])
        for x, p in zip(family, ps):
            x["p_Holm4"] = p
            x["formally_supported"] = x["nominal_blocks"] >= 20
        rows += family
    save("fixed_period_bootstrap.csv", pd.DataFrame(rows))
    dump(
        "temporal_summary.json",
        {
            "complete_years": list(range(2017, 2026)),
            "positive_complete_year_counts": {
                b: int((g.loc[g.year.between(2017, 2025), "mean_pp"] > 0).sum())
                for b, g in annual.groupby("benchmark")
            },
            "interpretation": "Shared historical sample; annual differences are descriptive, not yearly discoveries.",
        },
    )


def parameters():
    say("Six-parameter annual sensitivity")
    q = (
        pl.scan_parquet(D1 / "daily_returns.parquet")
        .filter((pl.col("H") == 3) & (pl.col("N") == 20))
        .collect()
        .to_pandas()
    )
    p = (
        pl.scan_parquet(D1 / "daily_pool_returns.parquet")
        .filter(pl.col("H") == 3)
        .collect()
        .to_pandas()
    )
    q["date"] = pd.to_datetime(q.date)
    p["date"] = pd.to_datetime(p.date)
    qwide = q.pivot(
        index=["F", "W", "date"], columns="method", values="complete"
    ).reset_index()
    qwide = qwide.merge(
        p[["F", "W", "date", "complete"]].rename(columns={"complete": "pool"}),
        on=["F", "W", "date"],
        validate="one_to_one",
    )
    out = []
    for b in ["pool", "F_only", "W_only", "A", "P"]:
        z = qwide[["F", "W", "date"]].copy()
        z["baseline"] = b
        z["delta_pp"] = (qwide.FW - qwide[b]) * 100
        out.append(z)
    d = pd.concat(out)
    d["year"] = d.date.dt.year
    annual = (
        d.groupby(["F", "W", "baseline", "year"])
        .agg(n_dates=("delta_pp", "count"), mean_pp=("delta_pp", "mean"))
        .reset_index()
    )
    save("parameter_annual.csv", annual)
    rows = []
    for label, y0, y1 in [
        ("full", 2016, 2026),
        ("2017-2021", 2017, 2021),
        ("2022-2026_partial", 2022, 2026),
    ]:
        z = (
            d[d.year.between(y0, y1)]
            .groupby(["F", "W", "baseline"])
            .agg(n_dates=("delta_pp", "count"), mean_pp=("delta_pp", "mean"))
            .reset_index()
        )
        z["period"] = label
        rows.append(z)
    save("parameter_fixed_periods.csv", pd.concat(rows))
    df = annual[annual.baseline == "pool"]
    a = (
        df.groupby("year")
        .agg(
            positive_pairs=("mean_pp", lambda x: int((x > 0).sum())),
            pairs=("mean_pp", "size"),
            min_pp=("mean_pp", "min"),
            max_pp=("mean_pp", "max"),
        )
        .reset_index()
    )
    save("parameter_year_signs.csv", a)
    # Duplicate primary data is an arithmetic bridge, not extra evidence.
    daily = pd.read_csv(OUT / "daily_reconstructed.csv", parse_dates=["date"])
    v = d[(d.F == 5) & (d.W == 10) & (d.baseline == "pool")].merge(
        daily[(daily.N == 20) & (daily.benchmark == "pool")][["date", "delta"]],
        on="date",
    )
    dump(
        "parameter_qa.json",
        {
            "default_pair_daily_max_abs_error_pp": float(
                np.max(np.abs(v.delta_pp - v.delta * 100))
            ),
            "dates": len(v),
            "n_parameter_cells": len(annual),
            "scope": "all six existing settings, H3 Top20 only; no parameter optimization",
        },
    )


def simulation():
    p = EX / "account_reference/daily_equity_sanitized.csv"
    d = pd.read_csv(p, parse_dates=["date"])
    nav = d.net_nav
    dd = nav / nav.cummax() - 1
    trough = int(dd.idxmin())
    peak = int(nav.iloc[: trough + 1].idxmax())
    days = (d.date.iloc[-1] - d.date.iloc[0]).days
    dump(
        "simulation_arithmetic.json",
        {
            "n_dates": len(d),
            "first": str(d.date.iloc[0]),
            "last": str(d.date.iloc[-1]),
            "terminal_nav": float(nav.iloc[-1]),
            "cagr36525": float((nav.iloc[-1] / nav.iloc[0]) ** (365.25 / days) - 1),
            "max_drawdown": float(dd.min()),
            "peak": str(d.date.iloc[peak]),
            "trough": str(d.date.iloc[trough]),
            "mean_exposure": float(d.market_exposure.mean()),
            "fees_CNY": float(d.total_fees_CNY.sum()),
            "scope": "Saved historical simulation arithmetic only. Not verification of live fills, economic tradability or causal attribution.",
        },
    )


def main():
    start = time.time()
    assert PROTOCOL.exists()
    inputs = [
        D4 / "frozen_eligible_pool.parquet",
        D4 / "frozen_top20_selections.parquet",
        D4 / "daily_matched_controls.csv",
        S2 / "cache/calendar.parquet",
        S2 / "cache/prices_latest.parquet",
        D1 / "daily_returns.parquet",
        D1 / "daily_pool_returns.parquet",
    ]
    before = manifest(inputs)
    dump("core_input_manifest_before.json", before)
    dump(
        "execution_environment.json",
        {
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "polars": pl.__version__,
            "script_sha256": sha(__file__),
            "protocol_sha256": sha(PROTOCOL),
            "observation_cutoff": CFG.OBSERVATION_CUTOFF,
        },
    )
    daily = reconstruct()
    temporal(daily)
    parameters()  # Trading-account simulation is outside this research-statistics check.
    after = manifest(inputs)
    dump("core_input_manifest_after.json", after)
    assert before == after
    dump(
        "core_completion.json",
        {
            "seconds": time.time() - start,
            "source_hashes_unchanged": True,
            "script_sha256": sha(__file__),
            "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    say("Completed", time.time() - start)
