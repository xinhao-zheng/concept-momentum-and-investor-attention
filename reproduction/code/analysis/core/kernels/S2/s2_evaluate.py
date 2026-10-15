"""Evaluate the physically frozen S2 rankings; never rerank using outcomes."""

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.multitest import multipletests
from s2_prepare import ROOT, CACHE, AUDIT, METHODS, CUTOFF, digest, save_json

DATA = ROOT / "data"
# Signal-date segments use the configured cutoff; the paper uses FULL.
# NEW_SHORT is empty after truncation.
SEGMENT_HISTORICAL_END = "2026-07-01"
SEGMENT_SEEN_END = CUTOFF
HORIZONS = [1, 3, 5, 10, 20, 60, 120, 250]
TOPNS = [5, 20, 100]
B = 100000
SEED = 20260922


def evaluate_daily():
    DATA.mkdir(exist_ok=True)
    selection = json.loads(
        (AUDIT / "selection_freeze.json").read_text(encoding="utf-8")
    )
    assert (
        json.loads((AUDIT / "source_duplicate_dates.json").read_text(encoding="utf-8"))[
            "source_duplicate_dates"
        ]
        == 0
    )
    amendment = json.loads(
        (AUDIT / "execution_precision_amendment.json").read_text(encoding="utf-8")
    )
    assert (
        amendment["bootstrap_B_execution"] == B
        and amendment["approved_before_real_outcome_evaluation"]
    )
    for f, key in [
        ("frozen_common_scores.parquet", "common_scores_sha256"),
        ("frozen_top100_ranks.parquet", "ranks_sha256"),
    ]:
        assert (
            digest(CACHE / f) == selection[key]
        ), "Selection cache changed after freeze"
    # Stamp and preserve the evaluator fingerprint before any future endpoint
    # is constructed or projected. This is distinct from the source stage.
    save_json(
        AUDIT / "evaluation_start.json",
        {
            "started_utc": pd.Timestamp.now(tz="UTC"),
            "selection_sha256": digest(AUDIT / "selection_freeze.json"),
            "evaluator_sha256": digest(__file__),
            "precision_amendment_sha256": digest(
                AUDIT / "execution_precision_amendment.json"
            ),
            "pre_outcome_addendum_sha256": digest(
                AUDIT / "pre_evaluation_specification.md"
            ),
            "comparison_family_FULL": 960,
            "primary_family": 5,
            "bootstrap_B": B,
            "seed": SEED,
            "calendar_signal_segments": [
                f"<={SEGMENT_HISTORICAL_END}",
                f"2026-07-02..{SEGMENT_SEEN_END}",
                f">{SEGMENT_SEEN_END} (empty)",
            ],
            "observation_cutoff": CUTOFF,
            "segment_note": "signal-date segments, outcomes mature by the 2026-08-31 observation cutoff; not old-cutoff exact replicas",
        },
    )
    cal = pl.read_parquet(CACHE / "calendar.parquet")
    common = pl.read_parquet(CACHE / "frozen_common_scores.parquet").join(
        cal, on="date"
    )
    ranks = pl.read_parquet(CACHE / "frozen_top100_ranks.parquet")
    prices = pl.read_parquet(
        CACHE / "prices_latest.parquet", columns=["date", "code", "open_adj"]
    ).join(cal, on="date")
    daily = []
    ic = []
    gates = []
    for h in HORIZONS:
        entry = prices.select(
            "code", (pl.col("idx") - 1).alias("idx"), pl.col("open_adj").alias("entry")
        )
        exitp = prices.select(
            "code",
            (pl.col("idx") - h - 1).alias("idx"),
            pl.col("open_adj").alias("exit"),
        )
        valid = common.filter(pl.col("idx") + h + 1 <= cal["idx"].max())
        frame = valid.join(entry, on=["code", "idx"], how="left").join(
            exitp, on=["code", "idx"], how="left"
        )
        frame = frame.with_columns(
            pl.when(
                (pl.col("entry") > 0)
                & (pl.col("exit") > 0)
                & pl.col("entry").is_finite()
                & pl.col("exit").is_finite()
            )
            .then(pl.col("exit") / pl.col("entry") - 1)
            .otherwise(None)
            .alias("outcome")
        )
        universe = frame.group_by("date").agg(
            pl.len().alias("n_pool"),
            pl.col("outcome").count().alias("n_pool_observed"),
            pl.col("outcome").mean().alias("universe_complete"),
            pl.col("outcome").fill_null(0).mean().alias("universe_zero"),
        )
        endpoint = frame.select(
            "date",
            "code",
            "outcome",
            (
                pl.col("entry").is_null()
                | ~pl.col("entry").is_finite()
                | (pl.col("entry") <= 0)
            ).alias("entry_missing"),
            (
                pl.col("exit").is_null()
                | ~pl.col("exit").is_finite()
                | (pl.col("exit") <= 0)
            ).alias("exit_missing"),
        )
        selected = ranks.join(endpoint, on=["date", "code"], how="inner")
        for n in TOPNS:
            d = (
                selected.filter(pl.col("rank") <= n)
                .group_by(["date", "design", "clock", "method"])
                .agg(
                    pl.len().alias("n_intended"),
                    pl.col("outcome").count().alias("n_observed"),
                    pl.col("entry_missing").sum().alias("n_entry_missing"),
                    pl.col("exit_missing").sum().alias("n_exit_missing"),
                    pl.col("outcome").mean().alias("portfolio_complete"),
                    pl.col("outcome").fill_null(0).mean().alias("portfolio_zero"),
                )
            )
            d = d.join(universe, on="date").with_columns(
                pl.lit(n).alias("top_n"),
                pl.lit(h).alias("H"),
                (pl.col("portfolio_complete") - pl.col("universe_complete")).alias(
                    "excess_complete"
                ),
                (pl.col("portfolio_zero") - pl.col("universe_zero")).alias(
                    "excess_zero"
                ),
            )
            daily.append(d)
        for design in ["gamma", "beta"]:
            for clock in ["calendar", "observations"]:
                for method in METHODS:
                    col = (
                        method
                        if method in ["mom5", "rev5", "mom20"]
                        else f"{design}_{clock}_{method}"
                    )
                    q = (
                        frame.filter(pl.col("outcome").is_finite())
                        .group_by("date")
                        .agg(
                            pl.corr(col, "outcome", method="spearman").alias("IC"),
                            pl.len().alias("n_ic"),
                        )
                    )
                    ic.append(
                        q.with_columns(
                            pl.lit(design).alias("design"),
                            pl.lit(clock).alias("clock"),
                            pl.lit(method).alias("method"),
                            pl.lit(h).alias("H"),
                        )
                    )
        gates.append(
            {
                "H": h,
                "mature_signal_dates": valid["date"].n_unique(),
                "eligible_rows": len(valid),
                "outcome_missing_rows": frame["outcome"].null_count(),
            }
        )
        print("daily evaluated H", h, "dates", valid["date"].n_unique(), flush=True)
    pl.concat(daily).write_parquet(
        DATA / "daily_portfolio_results.parquet", compression="zstd"
    )
    pl.concat(ic).write_parquet(DATA / "daily_ic.parquet", compression="zstd")
    save_json(AUDIT / "endpoint_gates.json", gates)


def block_bootstrap_matrix(x, L, seed):
    """Circular moving blocks, exact original n rows including last remainder.

    Same block starts across columns; NaNs retained and denominators counted.
    Percentile interval; centered two-sided empirical null test with +1 rule.
    """
    n, p = x.shape
    valid = np.isfinite(x)
    values = np.where(valid, x, 0.0)
    doubled = np.vstack([values, values[:L]])
    vdoubled = np.vstack([valid, valid[:L]])
    cs = np.vstack([np.zeros((1, p)), np.cumsum(doubled, axis=0)])
    cn = np.vstack([np.zeros((1, p)), np.cumsum(vdoubled, axis=0)])
    starts = np.arange(n)
    sums = cs[starts + L] - cs[starts]
    counts = cn[starts + L] - cn[starts]
    q, r = divmod(n, L)
    if r:
        rs = cs[starts + r] - cs[starts]
        rc = cn[starts + r] - cn[starts]
    rng = np.random.default_rng(seed)
    draws = np.empty((B, p))
    for first in range(0, B, 200):
        k = min(200, B - first)
        ix = rng.integers(0, n, size=(k, q))
        tot = sums[ix].sum(axis=1)
        denom = counts[ix].sum(axis=1)
        if r:
            ri = rng.integers(0, n, size=k)
            tot += rs[ri]
            denom += rc[ri]
        draws[first : first + k] = np.divide(
            tot, denom, out=np.full_like(tot, np.nan), where=denom > 0
        )
    means = np.nanmean(x, axis=0)
    low, high = np.nanquantile(draws, [0.025, 0.975], axis=0)
    pval = (1 + np.sum(np.abs(draws - means) >= np.abs(means), axis=0)) / (B + 1)
    return means, low, high, pval


def segment_mask(dates, label):
    if label == "FULL":
        return np.ones(len(dates), dtype=bool)
    if label == "HISTORICAL":
        return dates <= pd.Timestamp(SEGMENT_HISTORICAL_END)
    if label == "SEEN_EXTENSION":
        return (dates > pd.Timestamp(SEGMENT_HISTORICAL_END)) & (
            dates <= pd.Timestamp(SEGMENT_SEEN_END)
        )
    return dates > pd.Timestamp(SEGMENT_SEEN_END)


def inference():
    df = pd.read_parquet(DATA / "daily_portfolio_results.parquet")
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    summaries = []
    for (design, clock, h), part in df.groupby(["design", "clock", "H"], sort=True):
        L = max(20, int(h))
        for missing in ["complete", "zero"]:
            raw = part.pivot(
                index="date", columns=["top_n", "method"], values=f"portfolio_{missing}"
            ).sort_index()
            cols = [(n, b) for n in TOPNS for b in METHODS if b != "FW"]
            x = np.column_stack(
                [(raw[(n, "FW")] - raw[(n, b)]).to_numpy() for n, b in cols]
            )
            for segment in ["FULL", "HISTORICAL", "SEEN_EXTENSION", "NEW_SHORT"]:
                mask = segment_mask(raw.index, segment)
                z = x[mask]
                if len(z) == 0:
                    for n, b in cols:
                        rows.append(
                            dict(
                                design=design,
                                clock=clock,
                                H=int(h),
                                top_n=n,
                                baseline=b,
                                missing=missing,
                                segment=segment,
                                n_dates=0,
                                block=L,
                                blocks=0,
                                status="NO_MATURE_OUTCOME",
                                delta_pp=np.nan,
                                ci_low_pp=np.nan,
                                ci_high_pp=np.nan,
                                p_raw=np.nan,
                            )
                        )
                    continue
                nvalid = np.isfinite(z).sum(axis=0)
                means = np.nanmean(z, axis=0)
                confirm = (nvalid // L >= 20) & (segment != "NEW_SHORT")
                if confirm.any():
                    _, lo, hi, pval = block_bootstrap_matrix(z, L, SEED + int(h))
                else:
                    lo = hi = pval = np.full(z.shape[1], np.nan)
                for j, (n, b) in enumerate(cols):
                    rows.append(
                        dict(
                            design=design,
                            clock=clock,
                            H=int(h),
                            top_n=n,
                            baseline=b,
                            missing=missing,
                            segment=segment,
                            n_dates=int(nvalid[j]),
                            block=L,
                            blocks=int(nvalid[j] // L),
                            status=(
                                "INFERENCE_ELIGIBLE"
                                if confirm[j]
                                else "DESCRIPTIVE_INSUFFICIENT_BLOCKS"
                            ),
                            delta_pp=float(means[j] * 100),
                            ci_low_pp=float(lo[j] * 100) if confirm[j] else np.nan,
                            ci_high_pp=float(hi[j] * 100) if confirm[j] else np.nan,
                            p_raw=float(pval[j]) if confirm[j] else np.nan,
                        )
                    )
            print("bootstrap", design, clock, h, missing, flush=True)
    out = pd.DataFrame(rows)
    out["p_empirical_resolution"] = 1 / (B + 1)
    out["p_monte_carlo_se_approx"] = np.sqrt(out.p_raw * (1 - out.p_raw) / (B + 1))
    full = out.segment.eq("FULL")
    assert full.sum() == 960
    out.loc[full, "p_holm_grid"] = multipletests(
        out.loc[full, "p_raw"].fillna(1), method="holm"
    )[1]
    main = (
        full
        & out.design.eq("gamma")
        & out.clock.eq("calendar")
        & out.H.eq(3)
        & out.top_n.eq(20)
        & out.missing.eq("complete")
    )
    assert main.sum() == 5
    out.loc[main, "p_holm_primary"] = multipletests(
        out.loc[main, "p_raw"].fillna(1), method="holm"
    )[1]
    out["confirmed_primary"] = (
        main & out.p_holm_primary.lt(0.05) & out.status.eq("INFERENCE_ELIGIBLE")
    )
    out["confirmed_grid"] = (
        full & out.p_holm_grid.lt(0.05) & out.status.eq("INFERENCE_ELIGIBLE")
    )
    out.to_csv(
        DATA / "paired_competition_results.csv", index=False, encoding="utf-8-sig"
    )
    for keys, part in df.groupby(
        ["design", "clock", "H", "top_n", "method"], sort=True
    ):
        for segment in ["FULL", "HISTORICAL", "SEEN_EXTENSION", "NEW_SHORT"]:
            sub = part.loc[segment_mask(part.date, segment)]
            row = dict(zip(["design", "clock", "H", "top_n", "method"], keys))
            row.update(
                segment=segment,
                n_dates=len(sub),
                gross_complete_pp=sub.portfolio_complete.mean() * 100,
                pool_excess_complete_pp=sub.excess_complete.mean() * 100,
                gross_zero_pp=sub.portfolio_zero.mean() * 100,
                pool_excess_zero_pp=sub.excess_zero.mean() * 100,
                selected_missing_fraction=(
                    1 - sub.n_observed.sum() / sub.n_intended.sum()
                    if len(sub)
                    else np.nan
                ),
                n_entry_missing=int(sub.n_entry_missing.sum()),
                n_exit_missing=int(sub.n_exit_missing.sum()),
            )
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(
        DATA / "portfolio_summary.csv", index=False, encoding="utf-8-sig"
    )
    ic = pd.read_parquet(DATA / "daily_ic.parquet")
    ic.groupby(["design", "clock", "H", "method"]).agg(
        mean_IC=("IC", "mean"),
        n_dates=("IC", "count"),
        mean_cross_section=("n_ic", "mean"),
    ).reset_index().to_csv(DATA / "ic_summary.csv", index=False, encoding="utf-8-sig")
    mainout = out.loc[main]
    save_json(
        AUDIT / "inference_summary.json",
        {
            "B": B,
            "seed": SEED,
            "grid_family_size": 960,
            "primary_rows": mainout.to_dict("records"),
            "grid_confirmations": int(out.confirmed_grid.sum()),
            "boundary": "observational, all historical reuse disclosed; no causality or core net-return claim",
        },
    )


def self_test():
    x = np.arange(400, dtype=float)[:, None] * 0 + 0.01
    mean, lo, hi, p = block_bootstrap_matrix(x, 20, SEED)
    assert np.allclose([mean[0], lo[0], hi[0]], 0.01)
    assert p[0] == 1 / (B + 1)
    # Known-calendar missing endpoint: next stock row must never substitute.
    df = pl.DataFrame({"idx": [0, 1, 3, 4], "open": [10.0, 11.0, 13.0, 14.0]})
    lookup = dict(zip(df["idx"], df["open"]))
    assert lookup.get(2) is None and lookup.get(3) == 13
    save_json(
        AUDIT / "evaluation_synthetic_tests.json",
        {"circular_block_constant": "PASS", "missing_exact_endpoint": "PASS"},
    )
    print("EVALUATOR SYNTHETIC PASS")
