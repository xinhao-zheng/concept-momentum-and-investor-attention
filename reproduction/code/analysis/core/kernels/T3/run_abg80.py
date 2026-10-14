"""Eight-horizon observation-row timing sensitivity, through 2026-08-31.

Scores are supplied by the original-kernel C2 reconstruction. Freeze eligible
stock identities and ranked lists before opening outcome prices. No trading
simulation is run here. Run using Python -B, after C2 score acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from runtime_config import load_config

CFG_RUN = load_config()
ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
PACKAGE = EXP.parent
C2 = CFG_RUN.output("models/C2/cache/mechanism")
S2 = EXP / "S2/cache"
OUT = ROOT / "results"
AUDIT = ROOT / "audit"
HORIZONS = [1, 3, 5, 10, 20, 60, 120, 250]
SIZES = [20, 30]
CELLS = {
    "alpha": ("beta", "close"),
    "beta": ("beta", "open"),
    "gamma": ("gamma", "open"),
    "delta_internal": ("gamma", "close"),
}
CONTRASTS = {
    "membership_at_open_beta_minus_gamma": np.array([0, 1, -1, 0.0]),
    "membership_at_close_alpha_minus_delta": np.array([1, 0, 0, -1.0]),
    "return_clock_sync_alpha_minus_beta": np.array([1, -1, 0, 0.0]),
    "return_clock_lag_delta_minus_gamma": np.array([0, 0, -1, 1.0]),
    "difference_in_differences_timing_interaction": np.array([1, -1, 1, -1.0]),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsave(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def frozen_protocol():
    return {
        "version": "T3-observation-row-ABG80-20260831-v1",
        "execution_role": "research replication",
        "cutoff_inclusive": "2026-08-31",
        "F": 5,
        "W": 10,
        "holding_h": HORIZONS,
        "top_n": SIZES,
        "score_source": "C2 full_stock_scores_beta/gamma.parquet, original float32-return, observation-row fully-LOO kernel reconstruction. Read the C2 prefix reconciliation audit for numerical provenance.",
        "gamma": "Member-only lag gamma_M: latest membership snapshot dated no later than the previous market day; price information still through signal t close. Not S9 gamma_F full-information lag.",
        "formation_window": "Valid stock-concept observation rows, not fixed full-market calendar windows. The current main paired experiment uses calendar windows and is a separate sensitivity.",
        "pool": "Stock-date intersection of finite strictly positive beta and gamma F5W10 scores. No additional calendar-score eligibility or future-outcome requirement.",
        "selection": "Within each day and design, descending full-precision score then ascending stock code. Save common identities and Top30 ranks before opening outcome price values. Top20 is the first20; no replacement when outcomes are missing.",
        "cells": {
            "alpha": "beta list, close(t) to close(t+H)",
            "beta": "beta list, open(t+1) to open(t+H+1)",
            "gamma": "gamma_M list, open(t+1) to open(t+H+1)",
            "delta_internal": "gamma_M list, close(t) to close(t+H)",
        },
        "outcomes": "Exact global market calendar and same stock adjusted-price endpoints from frozen S2 prices. Both finite and >0. All complete exits no later than 2026-08-31. No shortened interval or next available stock date.",
        "missing_conventions": ["complete_observed", "missing_as_cash"],
        "complete_observed": "Mean observed selected endpoints minus mean observed pool endpoints.",
        "missing_as_cash": "Missing outcomes count as zero at original intended list/pool denominator; accounting sensitivity, not realized execution.",
        "pairing": "Within every H and N and missing convention retain dates with all four finite cell spreads. All five contrasts use exactly those same dates.",
        "inference": "Original date-HAC Newey-West lag max(H-1,5), Bartlett weights, denominator n; normal two-sided P and pointwise95%CI. Holm and BH across 5 contrasts*8H*2N=80 in each missing convention.",
        "scope": "Only the earlier eight-horizon F5W10 timing sensitivity. No other six-pair grids, no return-linked stock eligibility, no trading backtest.",
        "historical_review": "Original results were already known. Saving selections before outcome linkage documents execution order, not preregistration or an independent holdout.",
    }


def freeze():
    for folder in [ROOT, OUT, AUDIT]:
        folder.mkdir(exist_ok=True)
    f = AUDIT / "selection_freeze.json"
    if f.exists():
        saved = json.loads(f.read_text(encoding="utf-8"))
        for field in ["common_keys", "ranked_lists", "common_scores"]:
            assert sha(ROOT / saved[field]["path"]) == saved[field]["sha256"]
        return saved
    protocol = frozen_protocol()
    jsave(ROOT / "protocol_frozen.json", protocol)
    (ROOT / "protocol_frozen.sha256").write_text(
        sha(ROOT / "protocol_frozen.json") + "  protocol_frozen.json\n",
        encoding="utf-8",
    )
    frames = {}
    sources = []
    for design in ["beta", "gamma"]:
        path = C2 / f"full_stock_scores_{design}.parquet"
        if not path.exists():
            raise FileNotFoundError("C2 accepted score is required: " + str(path))
        sources.append(
            dict(
                design=design,
                **_receipt_path(path),
                sha256=sha(path),
                bytes=path.stat().st_size,
            )
        )
        frame = pl.read_parquet(path).select(
            pl.col("signal_date").cast(pl.Datetime("ns")).alias("date"),
            pl.col("stock_code").cast(pl.String).alias("code"),
            pl.col("FW").cast(pl.Float64).alias(design + "_score"),
        )
        assert frame.select(pl.struct("date", "code").n_unique()).item() == frame.height
        assert frame["date"].max() <= datetime(2026, 8, 31)
        frames[design] = frame
    common = (
        frames["beta"]
        .join(frames["gamma"], on=["date", "code"], how="inner")
        .filter(
            pl.col("beta_score").is_finite()
            & (pl.col("beta_score") > 0)
            & pl.col("gamma_score").is_finite()
            & (pl.col("gamma_score") > 0)
        )
        .sort(["date", "code"])
    )
    common.write_parquet(OUT / "frozen_common_scores.parquet", compression="zstd")
    common.select("date", "code").write_parquet(
        OUT / "frozen_common_keys.parquet", compression="zstd"
    )
    ranks = []
    for design in ["beta", "gamma"]:
        ranked = (
            common.select("date", "code", pl.col(design + "_score").alias("score"))
            .sort(["date", "score", "code"], descending=[False, True, False])
            .with_columns(pl.int_range(1, pl.len() + 1).over("date").alias("rank"))
            .filter(pl.col("rank") <= 30)
            .with_columns(pl.lit(design).alias("design"))
        )
        ranks.append(ranked)
    ranks = pl.concat(ranks).sort(["date", "design", "rank"])
    ranks.write_parquet(OUT / "frozen_top30.parquet", compression="zstd")
    info = dict(
        frozen_at=datetime.now().astimezone().isoformat(),
        sources=sources,
        pool_rows=common.height,
        pool_dates=common["date"].n_unique(),
        first_signal=str(common["date"].min()),
        last_signal=str(common["date"].max()),
        outcome_price_values_opened_before_freeze=False,
        protocol_sha256=sha(ROOT / "protocol_frozen.json"),
    )
    for field, name in [
        ("common_keys", "frozen_common_keys.parquet"),
        ("ranked_lists", "frozen_top30.parquet"),
        ("common_scores", "frozen_common_scores.parquet"),
    ]:
        info[field] = dict(path="results/" + name, sha256=sha(OUT / name))
    jsave(f, info)
    print(
        "Selection freeze complete:",
        info["pool_rows"],
        "pool rows;",
        info["pool_dates"],
        "days",
        flush=True,
    )
    return info


def nw_stats(values, h):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    mu = float(x.mean())
    residual = x - mu
    lag = min(max(h - 1, 5), n - 1)
    lrv = float(residual @ residual / n)
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (residual[k:] @ residual[:-k] / n)
    se = math.sqrt(max(lrv, 0) / n)
    t = mu / se if se > 0 else np.nan
    return dict(
        mean=mu,
        se=se,
        t=t,
        ci95_low=mu - 1.959963984540054 * se,
        ci95_high=mu + 1.959963984540054 * se,
        ci90_low=mu - 1.6448536269514722 * se,
        ci90_high=mu + 1.6448536269514722 * se,
        n=n,
        hac_lag=lag,
        p_two_sided_hac=math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else np.nan,
    )


def evaluate():
    frozen = freeze()
    if (OUT / "completion.json").exists():
        raise FileExistsError("Completed result cannot be overwritten")
    prices_path = S2 / "prices_latest.parquet"
    prices_hash = sha(prices_path)
    calendar = pl.read_parquet(S2 / "calendar.parquet").sort("idx")
    dates = calendar["date"].to_numpy()
    nd = len(dates)
    assert str(dates[-1])[:10] == "2026-08-31"
    pool = pl.read_parquet(OUT / "frozen_common_scores.parquet").join(
        calendar, on="date"
    )
    ranks = pl.read_parquet(OUT / "frozen_top30.parquet").join(calendar, on="date")
    codes = pool.select("code").unique().sort("code").with_row_index("sid")
    pool = pool.join(codes, on="code").sort(["idx", "sid"])
    ranks = ranks.join(codes, on="code").sort(["idx", "design", "rank"])
    p = (
        pl.read_parquet(prices_path, columns=["date", "code", "open_adj", "close_adj"])
        .join(codes, on="code")
        .join(calendar, on="date")
    )
    assert p["date"].max() <= datetime(2026, 8, 31)
    mats = {}
    for col in ["open_adj", "close_adj"]:
        matrix = np.full((nd, len(codes)), np.nan)
        matrix[p["idx"].to_numpy(), p["sid"].to_numpy()] = p[col].to_numpy()
        mats[col] = matrix
    pool_by_day = {
        int(k[0]): g["sid"].to_numpy()
        for k, g in pool.partition_by("idx", as_dict=True).items()
    }
    ranked_by_day = {
        (int(k[0]), k[1]): g.sort("rank")["sid"].to_numpy()
        for k, g in ranks.partition_by(["idx", "design"], as_dict=True).items()
    }
    daily = []
    for h in HORIZONS:
        for t, ids in pool_by_day.items():
            for clock in ["close", "open"]:
                entry = t + (clock == "open")
                exitidx = t + h + (clock == "open")
                if exitidx >= nd:
                    continue
                px = mats[clock + "_adj"]
                p0, p1 = px[entry], px[exitidx]
                finite = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0) & (p1 > 0)
                returns = (
                    np.divide(p1, p0, out=np.full(len(codes), np.nan), where=finite) - 1
                )
                universe = returns[ids]
                n_pool = len(universe)
                observed = int(np.isfinite(universe).sum())
                pool_complete = float(np.nanmean(universe)) if observed else np.nan
                pool_cash = float(np.nansum(universe) / n_pool)
                for lens, (design, c) in CELLS.items():
                    if c != clock:
                        continue
                    intended = ranked_by_day[t, design]
                    for n in SIZES:
                        selected = intended[:n]
                        vals = returns[selected]
                        n_observed = int(np.isfinite(vals).sum())
                        n_intended = len(vals)
                        complete = float(np.nanmean(vals)) if n_observed else np.nan
                        cash = float(np.nansum(vals) / n_intended)
                        daily.append(
                            dict(
                                lens=lens,
                                display_lens="gamma_M" if lens == "gamma" else lens,
                                date=dates[t],
                                entry_date=dates[entry],
                                exit_date=dates[exitidx],
                                score_name="F5_W10_DEFAULT",
                                f=5,
                                w=10,
                                holding_h=h,
                                top_n=n,
                                portfolio_return_complete_observed=complete,
                                universe_return_complete_observed=pool_complete,
                                excess_complete_observed=complete - pool_complete,
                                portfolio_return_missing_as_cash=cash,
                                universe_return_missing_as_cash=pool_cash,
                                excess_missing_as_cash=cash - pool_cash,
                                n_intended=n_intended,
                                n_observed=n_observed,
                                n_common_universe=n_pool,
                                n_universe_observed=observed,
                                selected_missing_rate=1 - n_observed / n_intended,
                                universe_missing_rate=1 - observed / n_pool,
                                selection_frozen_before_outcome=True,
                            )
                        )
        print("Evaluated H", h, flush=True)
    daily = pd.DataFrame(daily).sort_values(["date", "holding_h", "top_n", "lens"])
    daily.to_parquet(OUT / "abg_daily_portfolios.parquet", index=False)
    rows = []
    contrasts_daily = []
    cell_rows = []
    support = []
    for convention in ["complete_observed", "missing_as_cash"]:
        key = "excess_" + convention
        pivot = daily.pivot(
            index=["date", "holding_h", "top_n"], columns="lens", values=key
        )
        for (h, n), sub in pivot.groupby(level=["holding_h", "top_n"], sort=True):
            matched = sub.dropna(subset=list(CELLS))
            arr = matched[list(CELLS)].to_numpy()
            sig = matched.index.get_level_values("date")
            support.append(
                dict(
                    missing_convention=convention,
                    holding_h=h,
                    top_n=n,
                    n_dates=len(matched),
                    signal_start=str(sig.min().date()),
                    signal_end=str(sig.max().date()),
                    max_close_exit=str(
                        dates[np.searchsorted(dates, np.datetime64(sig.max())) + h]
                    )[:10],
                    max_open_exit=str(
                        dates[np.searchsorted(dates, np.datetime64(sig.max())) + h + 1]
                    )[:10],
                )
            )
            for li, lens in enumerate(CELLS):
                cell_rows.append(
                    dict(
                        lens=lens,
                        missing_convention=convention,
                        holding_h=h,
                        top_n=n,
                        **nw_stats(arr[:, li], h),
                    )
                )
            for name, weights in CONTRASTS.items():
                vals = arr @ weights
                rows.append(
                    dict(
                        contrast=name,
                        missing_convention=convention,
                        holding_h=h,
                        top_n=n,
                        signal_start=str(sig.min().date()),
                        signal_end=str(sig.max().date()),
                        **nw_stats(vals, h),
                    )
                )
                contrasts_daily.append(
                    pd.DataFrame(
                        dict(
                            contrast=name,
                            missing_convention=convention,
                            holding_h=h,
                            top_n=n,
                            date=sig,
                            value=vals,
                        )
                    )
                )
    result = pd.DataFrame(rows)
    for convention, idx in result.groupby("missing_convention").groups.items():
        indices = np.array(list(idx))
        p = result.loc[indices, "p_two_sided_hac"].to_numpy()
        assert len(p) == 80 and np.isfinite(p).all()
        order = np.argsort(p, kind="stable")
        ordered = p[order]
        holm = np.minimum(1, np.maximum.accumulate(ordered * (80 - np.arange(80))))
        bh = np.minimum(
            1, np.minimum.accumulate((ordered * 80 / (np.arange(80) + 1))[::-1])[::-1]
        )
        result.loc[indices[order], "p_holm_family"] = holm
        result.loc[indices[order], "q_bh_family"] = bh
        result.loc[indices, "family_n"] = 80
        result.loc[indices, "multiplicity_family"] = (
            "timing_2x2_all_five_contrasts_HN|" + convention + "|n=80"
        )
    result.to_csv(
        OUT / "abg_2x2_timing_decomposition.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(cell_rows).to_csv(
        OUT / "paired_cell_levels.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(support).to_csv(OUT / "paired_sample_support_32.csv", index=False)
    pd.concat(contrasts_daily, ignore_index=True).to_parquet(
        OUT / "paired_daily_contrasts.parquet", index=False
    )
    result[result.holding_h.eq(3) & result.top_n.eq(20)].to_csv(
        OUT / "H3_Top20_five_contrasts.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(cell_rows).query("holding_h==3 and top_n==20").to_csv(
        OUT / "H3_Top20_four_cells.csv", index=False, encoding="utf-8-sig"
    )
    assert sha(prices_path) == prices_hash
    assert sha(ROOT / "protocol_frozen.json") == frozen["protocol_sha256"]
    for item in frozen["sources"]:
        assert sha(_resolve_receipt_path(item)) == item["sha256"]
    checks = dict(
        completed=datetime.now().astimezone().isoformat(),
        cutoff="2026-08-31",
        daily_portfolio_rows=len(daily),
        contrasts=len(result),
        families=2,
        tests_per_family=80,
        full_contrast_support=True,
        last_outcome=str(daily.exit_date.max().date()),
        all_outcomes_within_cutoff=bool(daily.exit_date.le("2026-08-31").all()),
        missing_outcome_replacements=0,
        selection_hashes_unchanged=True,
        price_input_sha256=prices_hash,
        protocol_id="observation_row_member_lag_abg80",
        score_source="C2 full_stock_scores",
        holm_significant_by_missing=result.groupby("missing_convention")
        .p_holm_family.apply(lambda x: int(x.lt(0.05).sum()))
        .to_dict(),
    )
    jsave(OUT / "qa_results.json", checks)
    jsave(OUT / "completion.json", checks)
    print(json.dumps(checks, ensure_ascii=False), flush=True)
