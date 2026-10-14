#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Attach exact price and attention endpoints after feature/sample freeze.

Price outcomes are copied only from the audited ABG exact-calendar outcome
cache.  Future attention changes are reconstructed here from conservative
attention *levels* at the exact market-calendar t+H date; historical
``future_*`` columns are never projected.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CQ = ROOT.parents[2]
CACHE = ROOT / "cache" / "mechanism"
DATA = ROOT / "data" / "mechanism"
AUDIT = ROOT / "audit" / "mechanism"
SAMPLE = CACHE / "mechanism_sample_keys_before_outcomes.parquet"
OUTCOMES = ROOT / "cache" / "abg" / "abg_outcomes_cutoff.parquet"
CALENDAR = ROOT / "data" / "abg" / "abg_horizon_calendar.csv"
COVARIATES = ROOT.parent / "dynamics/shared/stock_pit_covariates_daily.universe.parquet"
ATTENTION = ROOT.parent / "dynamics/shared/attention_daily.parquet"
FUTURE_ATTENTION = CACHE / "mechanism_future_attention_changes.parquet"
OUTPUT = CACHE / "mechanism_analysis_panel.parquet"
SUPPORT_COUNTS = DATA / "mechanism_outcome_support.csv"
AUDIT_PATH = AUDIT / "mechanism_outcome_attachment.json"

CUTOFF = pd.Timestamp("2026-09-01")
HORIZONS = (1, 3, 5, 10, 20, 60, 120, 250)
CORE_COVARIATES = (
    "feature_asof_date",
    "source_last_trade_date",
    "source_trade_observed_t",
    "log_amount_t",
    "amount_to_float_mv_t",
    "log_amount_20_mean",
    "amount_to_float_mv_20_mean",
    "amihud_20",
    "suspension_days_20",
    "zero_amount_days_20",
    "observed_trade_days_20",
    "limit_up_proxy_days_20",
    "limit_down_proxy_days_20",
    "one_price_limit_proxy_days_20",
    "market_return_t",
    "ret_1",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "drawdown_from_60d_high",
    "beta_hs300_60",
    "idio_vol_hs300_60",
    "log_float_mv_t",
    "log_total_mv_t",
    "member_hs300_t",
    "member_sz50_t",
    "member_zz500_t",
    "member_cyb_t",
    "member_zz1000_t",
    "member_zz2000_t",
    "sw1_industry",
    "sw2_industry",
    "sw3_industry",
    "institution_net_inflow_to_amount_t",
    "institution_net_inflow_to_amount_5",
    "institution_net_inflow_to_amount_20",
    "large_net_inflow_to_amount_t",
    "large_net_inflow_to_amount_5",
    "large_net_inflow_to_amount_20",
    "mid_net_inflow_to_amount_t",
    "mid_net_inflow_to_amount_5",
    "mid_net_inflow_to_amount_20",
    "retail_net_inflow_to_amount_t",
    "retail_net_inflow_to_amount_5",
    "retail_net_inflow_to_amount_20",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parquet_identity(path: Path) -> dict[str, object]:
    pf = pq.ParquetFile(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "columns": pf.schema_arrow.names,
    }


def calendar_wide() -> pd.DataFrame:
    frame = pd.read_csv(CALENDAR)
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "signal_date"})
    for column in (
        "signal_date",
        "close_exit_date",
        "open_entry_date",
        "open_exit_date",
    ):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    rows: list[pd.DataFrame] = []
    dates = sorted(
        pd.Timestamp(value) for value in frame["signal_date"].dropna().unique()
    )
    index_lookup = {date: index for index, date in enumerate(dates)}
    base = pd.DataFrame(
        {"signal_date": dates, "market_calendar_index": range(len(dates))}
    )
    for horizon in HORIZONS:
        part = frame[frame["holding_h"].eq(horizon)][
            ["signal_date", "close_exit_date", "open_entry_date", "open_exit_date"]
        ].copy()
        part = part.rename(
            columns={
                "close_exit_date": f"attention_target_date_{horizon}",
                "open_entry_date": f"open_entry_date_{horizon}",
                "open_exit_date": f"open_exit_date_{horizon}",
            }
        )
        part[f"target_market_calendar_index_{horizon}"] = part[
            f"attention_target_date_{horizon}"
        ].map(index_lookup)
        base = base.merge(part, on="signal_date", how="left", validate="one_to_one")
    return base


def build_future_attention(calendar: pd.DataFrame) -> None:
    sample = pl.scan_parquet(SAMPLE).select("signal_date", "stock_code")
    current = (
        pl.scan_parquet(ATTENTION)
        .select("signal_date", "stock_code", "log_T", "log_S", "log_K")
        .filter(pl.col("signal_date") < pl.lit(CUTOFF.to_pydatetime()))
        .unique(subset=["signal_date", "stock_code"], keep="last")
    )
    result = sample.join(current, on=["signal_date", "stock_code"], how="left")
    calendar_pl = pl.from_pandas(calendar, include_index=False).lazy()
    result = result.join(calendar_pl, on="signal_date", how="left")
    for horizon in HORIZONS:
        target = current.select(
            pl.col("signal_date").alias(f"attention_target_date_{horizon}"),
            "stock_code",
            pl.col("log_T").alias(f"future_log_T_{horizon}"),
            pl.col("log_S").alias(f"future_log_S_{horizon}"),
            pl.col("log_K").alias(f"future_log_K_{horizon}"),
        )
        result = (
            result.join(
                target,
                on=[f"attention_target_date_{horizon}", "stock_code"],
                how="left",
            )
            .with_columns(
                (pl.col(f"future_log_T_{horizon}") - pl.col("log_T"))
                .cast(pl.Float32)
                .alias(f"future_attention_change_T_{horizon}"),
                (pl.col(f"future_log_S_{horizon}") - pl.col("log_S"))
                .cast(pl.Float32)
                .alias(f"future_attention_change_S_{horizon}"),
                (pl.col(f"future_log_K_{horizon}") - pl.col("log_K"))
                .cast(pl.Float32)
                .alias(f"future_attention_change_K_{horizon}"),
            )
            .drop(
                f"future_log_T_{horizon}",
                f"future_log_S_{horizon}",
                f"future_log_K_{horizon}",
            )
        )
    result = result.select(
        "signal_date",
        "stock_code",
        "market_calendar_index",
        *[f"attention_target_date_{h}" for h in HORIZONS],
        *[f"target_market_calendar_index_{h}" for h in HORIZONS],
        *[f"open_entry_date_{h}" for h in HORIZONS],
        *[f"open_exit_date_{h}" for h in HORIZONS],
        *[
            f"future_attention_change_{feature}_{h}"
            for feature in ("T", "S", "K")
            for h in HORIZONS
        ],
    ).sort(["signal_date", "stock_code"])
    temporary = FUTURE_ATTENTION.with_name(f".{FUTURE_ATTENTION.name}.partial")
    temporary.unlink(missing_ok=True)
    result.sink_parquet(
        temporary,
        compression="zstd",
        compression_level=6,
        statistics=True,
        engine="streaming",
        mkdir=True,
    )
    temporary.replace(FUTURE_ATTENTION)


def main() -> None:
    args = parse_args()
    started = time.time()
    for directory in (CACHE, DATA, AUDIT):
        directory.mkdir(parents=True, exist_ok=True)
    beta_features = CACHE / "mechanism_features_beta_primary_preoutcome.parquet"
    gamma_features = CACHE / "mechanism_features_gamma_primary_preoutcome.parquet"
    for path in (
        SAMPLE,
        OUTCOMES,
        CALENDAR,
        COVARIATES,
        ATTENTION,
        beta_features,
        gamma_features,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT.exists() and not args.force:
        raise FileExistsError(f"analysis panel exists; pass --force: {OUTPUT}")
    attention_columns = pq.ParquetFile(ATTENTION).schema_arrow.names
    if not {"signal_date", "stock_code", "log_T", "log_S", "log_K"}.issubset(
        attention_columns
    ):
        raise ValueError("attention source lacks level columns")
    calendar = calendar_wide()
    build_future_attention(calendar)

    outcome_names = pq.ParquetFile(OUTCOMES).schema_arrow.names
    outcome_date, outcome_code = outcome_names[:2]
    outcomes = pl.scan_parquet(OUTCOMES).select(
        pl.col(outcome_date).alias("signal_date"),
        pl.col(outcome_code).alias("stock_code"),
        *[pl.col(f"future_open_ret_{h}") for h in HORIZONS],
        *[pl.col(f"future_close_ret_{h}") for h in HORIZONS],
    )
    covariate_names = set(pq.ParquetFile(COVARIATES).schema_arrow.names)
    required_covariates = {"signal_date", "stock_code", *CORE_COVARIATES}
    missing_covariates = required_covariates - covariate_names
    if missing_covariates:
        raise ValueError(f"covariate panel missing {sorted(missing_covariates)}")
    covariates = pl.scan_parquet(COVARIATES).select(
        "signal_date", "stock_code", *CORE_COVARIATES
    )
    future_attention = pl.scan_parquet(FUTURE_ATTENTION)
    sample_scores = pl.scan_parquet(SAMPLE).select(
        "signal_date",
        "stock_code",
        "beta_score_F5_W10_DEFAULT",
        "gamma_score_F5_W10_DEFAULT",
        "beta_concept_state_signal_date",
        "beta_concept_asof_date",
        "gamma_concept_state_signal_date",
        "gamma_concept_asof_date",
    )

    frames: list[pl.LazyFrame] = []
    for lens, feature_path, score_design, outcome_clock in (
        ("alpha", beta_features, "beta", "close"),
        ("beta", beta_features, "beta", "open"),
        ("gamma", gamma_features, "gamma", "open"),
    ):
        frame = (
            pl.scan_parquet(feature_path)
            .join(sample_scores, on=["signal_date", "stock_code"], how="left")
            .join(covariates, on=["signal_date", "stock_code"], how="left")
            .join(future_attention, on=["signal_date", "stock_code"], how="left")
            .join(outcomes, on=["signal_date", "stock_code"], how="left")
            .with_columns(
                pl.lit(lens).alias("lens"),
                pl.lit(outcome_clock).alias("price_clock"),
                pl.lit(score_design).alias("feature_membership_design"),
                pl.col(f"{score_design}_score_F5_W10_DEFAULT")
                .cast(pl.Float32)
                .alias("score_F5_W10"),
                pl.col(f"{score_design}_concept_state_signal_date").alias(
                    "concept_state_signal_date"
                ),
                pl.col(f"{score_design}_concept_asof_date").alias("concept_asof_date"),
            )
            .with_columns(
                *[
                    pl.col(f"future_{outcome_clock}_ret_{h}")
                    .cast(pl.Float32)
                    .alias(f"price_return_{h}")
                    for h in HORIZONS
                ]
            )
            .drop(
                *[f"future_open_ret_{h}" for h in HORIZONS],
                *[f"future_close_ret_{h}" for h in HORIZONS],
                "beta_score_F5_W10_DEFAULT",
                "gamma_score_F5_W10_DEFAULT",
                "beta_concept_state_signal_date",
                "beta_concept_asof_date",
                "gamma_concept_state_signal_date",
                "gamma_concept_asof_date",
            )
        )
        frames.append(frame)
    panel = pl.concat(frames, how="vertical_relaxed")
    panel = panel.with_columns(
        *[
            (
                pl.col(f"price_return_{h}")
                - pl.col(f"price_return_{h}").mean().over(["lens", "signal_date"])
            )
            .cast(pl.Float32)
            .alias(f"price_excess_{h}")
            for h in HORIZONS
        ],
        *[
            (
                pl.col(f"future_attention_change_{feature}_{h}")
                - pl.col(f"future_attention_change_{feature}_{h}")
                .mean()
                .over(["lens", "signal_date"])
            )
            .cast(pl.Float32)
            .alias(f"future_attention_change_{feature}_{h}_date_demeaned")
            for feature in ("T", "S", "K")
            for h in HORIZONS
        ],
    ).sort(["lens", "signal_date", "stock_code"])
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.partial")
    temporary.unlink(missing_ok=True)
    panel.sink_parquet(
        temporary,
        compression="zstd",
        compression_level=6,
        statistics=True,
        engine="streaming",
        mkdir=True,
    )
    temporary.replace(OUTPUT)

    support = (
        pl.scan_parquet(OUTPUT)
        .group_by("lens")
        .agg(
            pl.len().alias("rows"),
            pl.col("signal_date").n_unique().alias("dates"),
            pl.col("stock_code").n_unique().alias("stocks"),
            *[
                pl.col(f"price_return_{h}")
                .is_not_null()
                .sum()
                .alias(f"price_observed_{h}")
                for h in HORIZONS
            ],
            *[
                pl.col(f"future_attention_change_{feature}_{h}")
                .is_not_null()
                .sum()
                .alias(f"attention_{feature}_observed_{h}")
                for feature in ("T", "S", "K")
                for h in HORIZONS
            ],
        )
        .sort("lens")
        .collect(engine="streaming")
    )
    support.write_csv(SUPPORT_COUNTS, include_header=True)
    checks = (
        pl.scan_parquet(OUTPUT)
        .select(
            pl.len().alias("rows"),
            pl.struct(["lens", "signal_date", "stock_code"])
            .n_unique()
            .alias("unique_keys"),
            pl.col("lens").n_unique().alias("lenses"),
            pl.col("signal_date").max().alias("signal_max"),
            (pl.col("feature_asof_date") > pl.col("signal_date"))
            .fill_null(False)
            .sum()
            .alias("feature_asof_violations"),
            (pl.col("source_last_trade_date") > pl.col("signal_date"))
            .fill_null(False)
            .sum()
            .alias("trade_source_after_signal"),
            (pl.col("observation_date") > pl.col("attention_asof_cutoff"))
            .fill_null(False)
            .sum()
            .alias("attention_asof_violations"),
            *[
                (pl.col(f"open_exit_date_{h}") >= pl.lit(CUTOFF.to_pydatetime()))
                .fill_null(False)
                .sum()
                .alias(f"open_exit_cutoff_violations_{h}")
                for h in HORIZONS
            ],
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    sample_rows = pq.ParquetFile(SAMPLE).metadata.num_rows
    hard_gates = {
        "three_lenses_one_row_per_sample_key": int(checks["rows"])
        == 3 * sample_rows
        == int(checks["unique_keys"]),
        "alpha_beta_gamma_present": int(checks["lenses"]) == 3,
        "signals_before_cutoff": pd.Timestamp(checks["signal_max"]) < CUTOFF,
        "covariates_point_in_time": int(checks["feature_asof_violations"]) == 0
        and int(checks["trade_source_after_signal"]) == 0,
        "attention_features_conservative": int(checks["attention_asof_violations"])
        == 0,
        "all_open_exits_before_cutoff": all(
            int(checks[f"open_exit_cutoff_violations_{h}"]) == 0 for h in HORIZONS
        ),
        "old_future_attention_columns_projected": False,
        "features_or_sample_reselected_after_outcome": False,
    }
    descriptive_false = {
        "old_future_attention_columns_projected",
        "features_or_sample_reselected_after_outcome",
    }
    status = (
        "PASS"
        if all(
            value for key, value in hard_gates.items() if key not in descriptive_false
        )
        else "FAIL"
    )
    payload = {
        "schema": "current-mechanism-outcome-attachment-1.0",
        "status": status,
        "hard_gates": hard_gates,
        "checks": checks,
        "definitions": {
            "alpha": "beta/synchronous mechanism features with exact close[t] to close[t+H] reference return",
            "beta": "beta/synchronous mechanism features with exact open[t+1] to open[t+H+1] return",
            "gamma": "gamma/lag-one-state mechanism features with exact open[t+1] to open[t+H+1] return",
            "price_excess": "stock return minus frozen-sample equal-weight return within lens and signal date",
            "future_attention": "level at conservative panel row t+H minus level at t; target t+H is exact global market-calendar date",
            "sample_frequency": "every fifth eligible market date; market_calendar_index retains original daily spacing",
        },
        "inputs": {
            "outcomes": parquet_identity(OUTCOMES),
            "covariates": parquet_identity(COVARIATES),
            "attention_level_source": parquet_identity(ATTENTION),
        },
        "outputs": {
            "future_attention": parquet_identity(FUTURE_ATTENTION),
            "analysis_panel": parquet_identity(OUTPUT),
            "support_counts": str(SUPPORT_COUNTS),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    AUDIT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": status, "checks": checks, "output": str(OUTPUT)},
            ensure_ascii=False,
            default=str,
        )
    )
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
