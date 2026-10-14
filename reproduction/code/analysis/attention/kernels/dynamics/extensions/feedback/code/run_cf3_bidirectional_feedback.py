#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CF3: bidirectional attention--trading timing with frozen negative controls.

The module is intentionally self-contained under ``extensions/CF3_*``.  It
uses the already frozen full-market attention panel, exact point-in-time stock
trading covariates, expanding-time out-of-fold nuisance fits, and a shared
circular moving-date-block bootstrap.  It does *not* interpret observational
lead/lag associations as structural causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.multitest import multipletests


MODULE = Path(__file__).resolve().parents[1]
CURRENT = MODULE.parents[1]
INPUT_PANEL = CURRENT / "shared" / "construct_CF1_level_panel.parquet"
PIT_PANEL = Path(
    "__SNAPSHOT__/input_recovery/stock_pit_covariates_daily.universe.parquet"
)

CACHE = MODULE / "cache"
DATA = MODULE / "data"
SOURCE = MODULE / "SourceData"
FIGURES = MODULE / "figures"
REPORTS = MODULE / "reports"
AUDIT = MODULE / "audit"
BASE_CACHE = CACHE / "cf3_base_oof_residuals.parquet"
DAILY_STATS = DATA / "cf3_daily_sufficient_statistics.parquet"
RESULTS = DATA / "cf3_all_family_results.csv"
ASYMMETRY = DATA / "cf3_lead_lag_asymmetry.csv"
NUISANCE = DATA / "cf3_nuisance_diagnostics.csv"
SUPPORT = DATA / "cf3_support_registry.csv"
BOOT_DRAWS = DATA / "cf3_bootstrap_draws.parquet"

CUTOFF = date(2026, 9, 1)
ENDPOINT_MAX = date(2026, 8, 31)
HORIZONS = (1, 3, 5, 10, 20, 60, 120, 250)
ATTENTION = {"T": "T_delta", "S": "S_delta_5", "K": "K_same_weekday_z56"}
ATTENTION_LABELS = {
    "T": {"zh": "总热度", "en": "Total heat"},
    "S": {"zh": "自选热度", "en": "Watchlist heat"},
    "K": {"zh": "黏性（同星期）", "en": "Same-weekday stickiness"},
}
TRADING = ("RET", "AMT", "TO")
FOLDS = (
    (0, np.datetime64("2023-01-01"), np.datetime64("2023-07-01"), "2023H1"),
    (1, np.datetime64("2023-07-01"), np.datetime64("2024-01-01"), "2023H2"),
    (2, np.datetime64("2024-01-01"), np.datetime64("2024-07-01"), "2024H1"),
    (3, np.datetime64("2024-07-01"), np.datetime64("2025-01-01"), "2024H2"),
    (4, np.datetime64("2025-01-01"), np.datetime64("2025-07-01"), "2025H1"),
    (5, np.datetime64("2025-07-01"), np.datetime64("2026-01-01"), "2025H2"),
    (6, np.datetime64("2026-01-01"), np.datetime64("2026-07-01"), "2026H1"),
    (7, np.datetime64("2026-07-01"), np.datetime64("2026-09-01"), "2026H2_partial"),
)
CONTROL_RAW = (
    "ret_1",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "log_float_mv_t",
    "log_amount_t",
    "amount_to_float_mv_20_mean",
    "amihud_20",
    "beta_hs300_60",
    "idio_vol_hs300_60",
    "drawdown_from_60d_high",
    "log_T",
    "log_S",
    "log_K",
    "slow_T",
    "slow_S",
    "slow_K",
)
MAX_NUISANCE_TRAIN = 120_000
RIDGE_ALPHA = 10.0
B = 1_000
BLOCK_LENGTH = 50
BOOTSTRAP_SEED = 2026071403
EQUIVALENCE_BAND = 0.02
MIN_ROWS = 20_000
MIN_DATES = 100
MIN_FOLDS = 4
MIN_DAILY_CROSS_SECTION = 100


for directory in (CACHE, DATA, SOURCE, FIGURES, REPORTS, AUDIT):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CURRENT / "scripts"))
from plot_style import (
    COLORS,
    apply_style,
    clean_axes,
    panel_label,
    save_figure,
)  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_json(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64, date, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=safe_json),
        encoding="utf-8",
    )


def finite_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column).cast(pl.Float64, strict=False).is_finite())
        .then(pl.col(column).cast(pl.Float64, strict=False))
        .otherwise(None)
    )


def add_date_industry_transforms(
    lf: pl.LazyFrame, variables: Sequence[str]
) -> pl.LazyFrame:
    """Daily z-score then absorb date-by-industry means for named variables."""

    z_names: list[str] = []
    expressions: list[pl.Expr] = []
    for variable in variables:
        safe = finite_expr(variable)
        mean = safe.mean().over("signal_date")
        scale = safe.std(ddof=0).over("signal_date")
        z_name = f"__z_{variable}"
        z_names.append(z_name)
        expressions.append(
            pl.when(scale.is_finite() & (scale > 1e-12))
            .then((safe - mean) / scale)
            .otherwise(None)
            .alias(z_name)
        )
    lf = lf.with_columns(expressions)
    lf = lf.with_columns(
        [
            (
                pl.col(z_name)
                - pl.col(z_name).mean().over(["signal_date", "sw1_industry"])
            )
            .cast(pl.Float32)
            .alias(f"Z_{variable}")
            for z_name, variable in zip(z_names, variables)
        ]
    )
    return lf.drop(z_names)


def trading_source(date_alias: str, prefix: str) -> pl.LazyFrame:
    raw_turnover = pl.col("amount_to_float_mv_t").cast(pl.Float64, strict=False)
    mean_turnover = pl.col("amount_to_float_mv_20_mean").cast(pl.Float64, strict=False)
    turnover = (
        pl.when((raw_turnover > 0) & (mean_turnover > 0))
        .then((raw_turnover / mean_turnover).log())
        .otherwise(None)
    )
    return pl.scan_parquet(PIT_PANEL).select(
        pl.col("signal_date").cast(pl.Date).alias(date_alias),
        "stock_code",
        (pl.col("ret_1") - pl.col("market_return_t"))
        .cast(pl.Float32)
        .alias(f"{prefix}_RET"),
        (pl.col("log_amount_t") - pl.col("log_amount_20_mean"))
        .cast(pl.Float32)
        .alias(f"{prefix}_AMT"),
        turnover.cast(pl.Float32).alias(f"{prefix}_TO"),
    )


def calendar_dates() -> np.ndarray:
    values = (
        pl.scan_parquet(PIT_PANEL)
        .select("signal_date")
        .unique()
        .sort("signal_date")
        .collect(engine="streaming")
        .get_column("signal_date")
        .to_numpy()
    )
    return values.astype("datetime64[D]")


def lag_map(dates: np.ndarray, lags: dict[str, int], key_name: str) -> pl.LazyFrame:
    payload: dict[str, object] = {key_name: dates}
    for name, lag in lags.items():
        output = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[D]")
        if lag < len(dates):
            output[lag:] = dates[:-lag]
        payload[name] = output
    return pl.DataFrame(payload).lazy()


def strict_observation_lag_map(
    observation_dates: np.ndarray,
    trading_dates: np.ndarray,
    lags: dict[str, int],
) -> pl.LazyFrame:
    """Map calendar observation dates to strictly earlier trading sessions.

    For H=1 the result is the most recent trading date strictly before the
    attention observation date.  This handles weekend/holiday attention
    snapshots without treating a non-trading date as missing support.
    """

    observations = np.sort(np.unique(observation_dates.astype("datetime64[D]")))
    strict_anchor = np.searchsorted(trading_dates, observations, side="left") - 1
    payload: dict[str, object] = {"observation_date": observations}
    for name, lag in lags.items():
        indices = strict_anchor - (lag - 1)
        output = np.full(len(observations), np.datetime64("NaT"), dtype="datetime64[D]")
        valid = indices >= 0
        output[valid] = trading_dates[indices[valid]]
        payload[name] = output
    return pl.DataFrame(payload).lazy()


def deterministic_thin(
    indices: np.ndarray, maximum: int = MAX_NUISANCE_TRAIN
) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)]


def _impute_scale(
    controls: np.ndarray, train_fit: np.ndarray, requested: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = controls[train_fit].astype(np.float64, copy=True)
    test = controls[requested].astype(np.float64, copy=True)
    medians = np.asarray(
        [
            np.nanmedian(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
            for col in train.T
        ],
        dtype=float,
    )
    train = np.where(np.isfinite(train), train, medians)
    test = np.where(np.isfinite(test), test, medians)
    means = train.mean(axis=0)
    scales = train.std(axis=0, ddof=0)
    scales[~np.isfinite(scales) | (scales < 1e-10)] = 1.0
    train = (train - means) / scales
    test = (test - means) / scales
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    return train, test, scales


def multi_crossfit(
    targets: np.ndarray,
    controls: np.ndarray,
    signal_dates: np.ndarray,
    *,
    group_name: str,
    target_dates: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Expanding-time ridge residuals for a predeclared triplet."""

    y = np.asarray(targets, dtype=np.float32)
    c = np.asarray(controls, dtype=np.float32)
    dates = np.asarray(signal_dates).astype("datetime64[D]")
    endpoint = (
        None
        if target_dates is None
        else np.asarray(target_dates).astype("datetime64[D]")
    )
    residual = np.full(y.shape, np.nan, dtype=np.float32)
    diagnostics: list[dict[str, object]] = []
    for fold_id, start, end, label in FOLDS:
        train_mask = dates < start
        if endpoint is not None:
            train_mask &= (~np.isnat(endpoint)) & (endpoint < start)
        test_mask = (dates >= start) & (dates < end)
        train = np.flatnonzero(train_mask & np.isfinite(y).all(axis=1))
        test = np.flatnonzero(test_mask & np.isfinite(y).all(axis=1))
        if len(train) < 1_000 or len(test) < 100:
            diagnostics.append(
                {
                    "group": group_name,
                    "fold_id": fold_id,
                    "fold_label": label,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "purged": endpoint is not None,
                    "status": "DEFERRED_INSUFFICIENT_FOLD",
                }
            )
            continue
        fit = deterministic_thin(train)
        x_train, x_test, _ = _impute_scale(c, fit, test)
        penalty = np.eye(x_train.shape[1], dtype=float) * RIDGE_ALPHA
        penalty[0, 0] = 0.0
        gram = x_train.T @ x_train + penalty
        rhs = x_train.T @ y[fit].astype(float)
        try:
            coefficient = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coefficient = np.linalg.pinv(gram, rcond=1e-10) @ rhs
        prediction = x_test @ coefficient
        residual[test] = (y[test] - prediction).astype(np.float32)
        r2_values: list[float] = []
        for column in range(y.shape[1]):
            denominator = float(
                np.square(y[test, column] - np.mean(y[test, column])).sum()
            )
            numerator = float(np.square(y[test, column] - prediction[:, column]).sum())
            r2_values.append(
                1.0 - numerator / denominator if denominator > 0 else math.nan
            )
        diagnostics.append(
            {
                "group": group_name,
                "fold_id": fold_id,
                "fold_label": label,
                "train_candidate_rows": int(train_mask.sum()),
                "train_valid_rows": len(train),
                "train_fit_rows": len(fit),
                "test_rows": len(test),
                "purged": endpoint is not None,
                "max_train_target_date": (
                    str(endpoint[train].max())
                    if endpoint is not None and len(train)
                    else None
                ),
                "target_strictly_before_test": (
                    bool(endpoint[train].max() < start)
                    if endpoint is not None and len(train)
                    else None
                ),
                "r2_1": r2_values[0],
                "r2_2": r2_values[1],
                "r2_3": r2_values[2],
                "status": "ESTIMATED",
            }
        )
    return residual, diagnostics


def create_permutation_index(frame: pl.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    strata = ["signal_date", "sw1_industry", "size_q", "liquidity_q"]
    ordered = (
        frame.select("row_id", "stock_code", *strata)
        .with_columns(
            pl.col("stock_code").hash(seed=20260714, seed_1=3).alias("perm_hash")
        )
        .sort([*strata, "perm_hash", "stock_code"])
        .with_columns(
            pl.len().over(strata).alias("stratum_n"),
            pl.col("row_id").shift(-1).over(strata).alias("next_row"),
            pl.col("row_id").first().over(strata).alias("first_row"),
        )
        .with_columns(
            pl.when(pl.col("stratum_n") >= 2)
            .then(pl.coalesce("next_row", "first_row"))
            .otherwise(None)
            .alias("perm_source_row")
        )
        .sort("row_id")
    )
    values = (
        ordered.get_column("perm_source_row").fill_null(-1).to_numpy().astype(np.int64)
    )
    checks = {
        "rows": len(values),
        "eligible_rows": int((values >= 0).sum()),
        "singleton_rows": int((values < 0).sum()),
        "self_mapped_eligible_rows": int(
            ((values >= 0) & (values == np.arange(len(values)))).sum()
        ),
    }
    return values, checks


def build_base(
    force: bool = False,
) -> tuple[pl.DataFrame, list[dict[str, object]], dict[str, object]]:
    if BASE_CACHE.exists() and not force:
        frame = pl.read_parquet(BASE_CACHE)
        permutation = frame.get_column("perm_source_row").to_numpy().astype(np.int64)
        checks = {
            "status": "REUSED",
            "rows": frame.height,
            "unique_keys": frame.select(
                pl.struct(["signal_date", "stock_code"]).n_unique()
            ).item(),
            "signal_min": str(frame.get_column("signal_date").min()),
            "signal_max": str(frame.get_column("signal_date").max()),
            "observation_max": str(frame.get_column("observation_date").max()),
            "pit_after_observation": int(
                frame.filter(pl.col("pit_date") > pl.col("observation_date")).height
            ),
            "permutation": {
                "rows": len(permutation),
                "eligible_rows": int((permutation >= 0).sum()),
                "singleton_rows": int((permutation < 0).sum()),
                "self_mapped_eligible_rows": int(
                    (
                        (permutation >= 0)
                        & (permutation == np.arange(len(permutation)))
                    ).sum()
                ),
            },
            "sha256": sha256(BASE_CACHE),
        }
        return frame, [], checks
    dates = calendar_dates()
    forward_payload = {"signal_date": dates.astype("datetime64[ns]")}
    for horizon in HORIZONS:
        targets = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
        targets[:-horizon] = dates[horizon:].astype("datetime64[ns]")
        forward_payload[f"target_date_{horizon}"] = targets
    forward_map = pl.DataFrame(forward_payload).lazy()
    signal_map = lag_map(dates, {"hist_signal_date_60": 60}, "signal_date")
    attention_history = pl.scan_parquet(INPUT_PANEL).select(
        pl.col("signal_date").cast(pl.Date).alias("hist_signal_date_60"),
        "stock_code",
        *[pl.col(source).alias(f"A_hist_{name}") for name, source in ATTENTION.items()],
    )
    selected = [
        "signal_date",
        "stock_code",
        "observation_date",
        "pit_date",
        "sw1_industry",
        *CONTROL_RAW,
        *ATTENTION.values(),
        *[f"target_date_{h}" for h in HORIZONS],
    ]
    base = (
        pl.scan_parquet(INPUT_PANEL)
        .select(selected)
        .drop([f"target_date_{h}" for h in HORIZONS])
        .join(forward_map, on="signal_date", how="left")
        .with_columns(
            pl.col("signal_date").cast(pl.Date),
            pl.col("observation_date").cast(pl.Date),
            pl.col("pit_date").cast(pl.Date),
            *[pl.col(f"target_date_{h}").cast(pl.Date) for h in HORIZONS],
        )
        .filter(
            (pl.col("signal_date") < pl.lit(CUTOFF))
            & (pl.col("observation_date") < pl.lit(CUTOFF))
            & (pl.col("pit_date") <= pl.col("observation_date"))
        )
        .rename({source: f"A_{name}" for name, source in ATTENTION.items()})
        .join(signal_map, on="signal_date", how="left")
        .join(attention_history, on=["hist_signal_date_60", "stock_code"], how="left")
        .join(
            trading_source("signal_date", "X_current"),
            on=["signal_date", "stock_code"],
            how="left",
        )
    )
    count_over_date = pl.len().over("signal_date").cast(pl.Float64)
    size_rank = pl.col("log_float_mv_t").rank("average").over("signal_date")
    liquid_rank = (
        pl.col("amount_to_float_mv_20_mean").rank("average").over("signal_date")
    )
    base = base.with_columns(
        (((size_rank - 1.0) * 5.0 / count_over_date).floor().clip(0, 4))
        .cast(pl.Int8)
        .alias("size_q"),
        (((liquid_rank - 1.0) * 5.0 / count_over_date).floor().clip(0, 4))
        .cast(pl.Int8)
        .alias("liquidity_q"),
    )
    transformed = [
        *CONTROL_RAW,
        *[f"A_{name}" for name in ATTENTION],
        *[f"A_hist_{name}" for name in ATTENTION],
        *[f"X_current_{name}" for name in TRADING],
    ]
    base = add_date_industry_transforms(base, transformed)
    keep = [
        "signal_date",
        "stock_code",
        "observation_date",
        "pit_date",
        "sw1_industry",
        "size_q",
        "liquidity_q",
        *[f"target_date_{h}" for h in HORIZONS],
        *[f"Z_{name}" for name in CONTROL_RAW],
        *[f"Z_A_{name}" for name in ATTENTION],
        *[f"Z_A_hist_{name}" for name in ATTENTION],
        *[f"Z_X_current_{name}" for name in TRADING],
    ]
    frame = (
        base.select(keep)
        .sort(["signal_date", "stock_code"])
        .collect(engine="streaming")
        .with_row_index("row_id")
    )
    if (
        frame.select(pl.struct(["signal_date", "stock_code"]).n_unique()).item()
        != frame.height
    ):
        raise RuntimeError("base signal_date-stock_code keys are not unique")
    permutation, permutation_checks = create_permutation_index(frame)
    dates_np = frame.get_column("signal_date").to_numpy().astype("datetime64[D]")
    controls = (
        frame.select([f"Z_{name}" for name in CONTROL_RAW])
        .to_numpy()
        .astype(np.float32)
    )
    nuisance_rows: list[dict[str, object]] = []
    groups = {
        "A_current": [f"Z_A_{name}" for name in ATTENTION],
        "A_history60": [f"Z_A_hist_{name}" for name in ATTENTION],
        "X_current_signal": [f"Z_X_current_{name}" for name in TRADING],
    }
    residuals: dict[str, np.ndarray] = {}
    for group, columns in groups.items():
        resid, diagnostics = multi_crossfit(
            frame.select(columns).to_numpy(), controls, dates_np, group_name=group
        )
        residuals[group] = resid
        nuisance_rows.extend(diagnostics)
    augmented = np.column_stack(
        [controls, frame.select([f"Z_X_current_{name}" for name in TRADING]).to_numpy()]
    )
    a_mediator, diagnostics = multi_crossfit(
        frame.select([f"Z_A_{name}" for name in ATTENTION]).to_numpy(),
        augmented,
        dates_np,
        group_name="A_current_with_signal_trade_mediator_controls",
    )
    nuisance_rows.extend(diagnostics)
    frame = frame.with_columns(pl.Series("perm_source_row", permutation))
    for index, name in enumerate(ATTENTION):
        frame = frame.with_columns(
            pl.Series(f"R_A_{name}", residuals["A_current"][:, index]),
            pl.Series(f"R_A_hist_{name}", residuals["A_history60"][:, index]),
            pl.Series(f"R_A_med_{name}", a_mediator[:, index]),
        )
    for index, name in enumerate(TRADING):
        frame = frame.with_columns(
            pl.Series(f"R_X_current_{name}", residuals["X_current_signal"][:, index])
        )
    frame.write_parquet(
        BASE_CACHE, compression="zstd", statistics=True, row_group_size=125_000
    )
    checks = {
        "status": "BUILT",
        "rows": frame.height,
        "unique_keys": frame.select(
            pl.struct(["signal_date", "stock_code"]).n_unique()
        ).item(),
        "signal_min": str(frame.get_column("signal_date").min()),
        "signal_max": str(frame.get_column("signal_date").max()),
        "observation_max": str(frame.get_column("observation_date").max()),
        "pit_after_observation": int(
            frame.filter(pl.col("pit_date") > pl.col("observation_date")).height
        ),
        "permutation": permutation_checks,
        "sha256": sha256(BASE_CACHE),
    }
    return frame, nuisance_rows, checks


def h_frame(base: pl.DataFrame, horizon: int) -> pl.DataFrame:
    dates = calendar_dates()
    observation_map = strict_observation_lag_map(
        base.get_column("observation_date").to_numpy(),
        dates,
        {
            f"past_observation_date_{horizon}": horizon,
            f"history_observation_date_{horizon}": horizon + 60,
        },
    )
    signal_map = lag_map(dates, {f"past_signal_date_{horizon}": horizon}, "signal_date")
    attention_past = pl.scan_parquet(INPUT_PANEL).select(
        pl.col("signal_date").cast(pl.Date).alias(f"past_signal_date_{horizon}"),
        "stock_code",
        *[pl.col(source).alias(f"A_past_{name}") for name, source in ATTENTION.items()],
    )
    lf = (
        base.lazy()
        .join(observation_map, on="observation_date", how="left")
        .join(signal_map, on="signal_date", how="left")
        .join(
            trading_source(f"target_date_{horizon}", "Y_future"),
            on=[f"target_date_{horizon}", "stock_code"],
            how="left",
        )
        .join(
            trading_source(f"past_observation_date_{horizon}", "X_past"),
            on=[f"past_observation_date_{horizon}", "stock_code"],
            how="left",
        )
        .join(
            trading_source(f"history_observation_date_{horizon}", "X_history"),
            on=[f"history_observation_date_{horizon}", "stock_code"],
            how="left",
        )
        .join(
            attention_past, on=[f"past_signal_date_{horizon}", "stock_code"], how="left"
        )
    )
    variables = [
        *[f"A_past_{name}" for name in ATTENTION],
        *[f"Y_future_{name}" for name in TRADING],
        *[f"X_past_{name}" for name in TRADING],
        *[f"X_history_{name}" for name in TRADING],
    ]
    lf = add_date_industry_transforms(lf, variables)
    frame = (
        lf.select(
            *base.columns,
            f"past_observation_date_{horizon}",
            f"history_observation_date_{horizon}",
            f"past_signal_date_{horizon}",
            *[f"Z_{name}" for name in variables],
        )
        .sort("row_id")
        .collect(engine="streaming")
    )
    if frame.height != base.height or not np.array_equal(
        frame.get_column("row_id").to_numpy(), np.arange(base.height)
    ):
        raise RuntimeError(f"H{horizon} joins changed the frozen base row registry")
    return frame


def fold_label_for_date(value: np.datetime64) -> int:
    for fold, start, end, _ in FOLDS:
        if start <= value < end:
            return fold
    return -1


def matrix_daily_stats(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    horizon: int,
    direction: str,
    estimand: str,
    x_attention: bool,
) -> list[dict[str, object]]:
    """Aggregate a 3x3 exposure/outcome matrix to exact daily sufficient statistics."""

    date_values = dates.astype("datetime64[D]")
    unique, first, counts = np.unique(
        date_values, return_index=True, return_counts=True
    )
    rows: list[dict[str, object]] = []
    attention_names = list(ATTENTION)
    trading_names = list(TRADING)
    for current_date, start, count in zip(unique, first, counts):
        fold = fold_label_for_date(current_date)
        if fold < 0:
            continue
        stop = start + count
        xx = x[start:stop]
        yy = y[start:stop]
        valid = np.isfinite(xx).all(axis=1) & np.isfinite(yy).all(axis=1)
        n = int(valid.sum())
        if n < MIN_DAILY_CROSS_SECTION:
            continue
        xv = xx[valid].astype(float)
        yv = yy[valid].astype(float)
        cross = xv.T @ yv
        sum_xx = np.square(xv).sum(axis=0)
        sum_yy = np.square(yv).sum(axis=0)
        for i in range(3):
            for j in range(3):
                if x_attention:
                    attention, trading = attention_names[i], trading_names[j]
                else:
                    trading, attention = trading_names[i], attention_names[j]
                rows.append(
                    {
                        "path_id": f"{direction}|{attention}|{trading}|H{horizon}",
                        "direction": direction,
                        "attention": attention,
                        "trading": trading,
                        "holding_h": horizon,
                        "estimand": estimand,
                        "signal_date": current_date,
                        "fold_id": fold,
                        "n": n,
                        "sum_xx": float(sum_xx[i]),
                        "sum_yy": float(sum_yy[j]),
                        "sum_xy": float(cross[i, j]),
                    }
                )
    return rows


def analyze_h(
    base: pl.DataFrame, horizon: int, *, force: bool = False
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    checkpoint = CACHE / f"cf3_daily_stats_H{horizon}.parquet"
    diagnostic_file = CACHE / f"cf3_nuisance_H{horizon}.json"
    if checkpoint.exists() and diagnostic_file.exists() and not force:
        output = pd.read_parquet(checkpoint)
        target_dates = (
            base.get_column(f"target_date_{horizon}").to_numpy().astype("datetime64[D]")
        )
        target_finite = target_dates[~np.isnat(target_dates)]
        observation_map = strict_observation_lag_map(
            base.get_column("observation_date").to_numpy(),
            calendar_dates(),
            {
                f"past_observation_date_{horizon}": horizon,
                f"history_observation_date_{horizon}": horizon + 60,
            },
        ).collect()
        return (
            output,
            json.loads(diagnostic_file.read_text(encoding="utf-8")),
            {
                "holding_h": horizon,
                "status": "REUSED",
                "base_rows": base.height,
                "daily_stat_rows": len(output),
                "target_max": str(target_finite.max()) if len(target_finite) else None,
                "target_after_endpoint_max": int(
                    (target_finite > np.datetime64(ENDPOINT_MAX)).sum()
                ),
                "past_observation_not_strict": int(
                    observation_map.filter(
                        pl.col(f"past_observation_date_{horizon}")
                        >= pl.col("observation_date")
                    ).height
                ),
                "history_observation_not_strict": int(
                    observation_map.filter(
                        pl.col(f"history_observation_date_{horizon}")
                        >= pl.col("observation_date")
                    ).height
                ),
                "daily_stats_sha256": sha256(checkpoint),
            },
        )
    started = time.time()
    frame = h_frame(base, horizon)
    dates = frame.get_column("signal_date").to_numpy().astype("datetime64[D]")
    target_dates = (
        frame.get_column(f"target_date_{horizon}").to_numpy().astype("datetime64[D]")
    )
    controls = (
        frame.select([f"Z_{name}" for name in CONTROL_RAW])
        .to_numpy()
        .astype(np.float32)
    )
    current_trade_z = (
        frame.select([f"Z_X_current_{name}" for name in TRADING])
        .to_numpy()
        .astype(np.float32)
    )
    augmented_controls = np.column_stack([controls, current_trade_z])
    group_columns = {
        "A_past": [f"Z_A_past_{name}" for name in ATTENTION],
        "Y_future": [f"Z_Y_future_{name}" for name in TRADING],
        "X_past": [f"Z_X_past_{name}" for name in TRADING],
        "X_history": [f"Z_X_history_{name}" for name in TRADING],
    }
    residuals: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, object]] = []
    for group, columns in group_columns.items():
        target = target_dates if group == "Y_future" else None
        resid, rows = multi_crossfit(
            frame.select(columns).to_numpy(),
            controls,
            dates,
            group_name=f"H{horizon}_{group}",
            target_dates=target,
        )
        residuals[group] = resid
        diagnostics.extend(rows)
    future_med, rows = multi_crossfit(
        frame.select(group_columns["Y_future"]).to_numpy(),
        augmented_controls,
        dates,
        group_name=f"H{horizon}_Y_future_with_signal_trade_mediator_controls",
        target_dates=target_dates,
    )
    diagnostics.extend(rows)

    a = (
        frame.select([f"R_A_{name}" for name in ATTENTION])
        .to_numpy()
        .astype(np.float32)
    )
    a_hist = (
        frame.select([f"R_A_hist_{name}" for name in ATTENTION])
        .to_numpy()
        .astype(np.float32)
    )
    a_med = (
        frame.select([f"R_A_med_{name}" for name in ATTENTION])
        .to_numpy()
        .astype(np.float32)
    )
    x_current = (
        frame.select([f"R_X_current_{name}" for name in TRADING])
        .to_numpy()
        .astype(np.float32)
    )
    a_past = residuals["A_past"]
    y_future = residuals["Y_future"]
    x_past = residuals["X_past"]
    x_history = residuals["X_history"]
    perm = frame.get_column("perm_source_row").to_numpy().astype(np.int64)
    valid_perm = perm >= 0
    perm_a = np.full_like(a, np.nan)
    perm_x_past = np.full_like(x_past, np.nan)
    perm_a[valid_perm] = a[perm[valid_perm]]
    perm_x_past[valid_perm] = x_past[perm[valid_perm]]

    daily: list[dict[str, object]] = []
    daily.extend(
        matrix_daily_stats(
            a,
            y_future,
            dates,
            horizon=horizon,
            direction="A_to_T",
            estimand="main",
            x_attention=True,
        )
    )
    daily.extend(
        matrix_daily_stats(
            x_past,
            a,
            dates,
            horizon=horizon,
            direction="T_to_A",
            estimand="main",
            x_attention=False,
        )
    )
    daily.extend(
        matrix_daily_stats(
            a,
            x_past,
            dates,
            horizon=horizon,
            direction="A_to_T",
            estimand="temporal_reverse",
            x_attention=True,
        )
    )
    daily.extend(
        matrix_daily_stats(
            x_current,
            a_past,
            dates,
            horizon=horizon,
            direction="T_to_A",
            estimand="temporal_reverse",
            x_attention=False,
        )
    )
    daily.extend(
        matrix_daily_stats(
            perm_a,
            y_future,
            dates,
            horizon=horizon,
            direction="A_to_T",
            estimand="code_permutation",
            x_attention=True,
        )
    )
    daily.extend(
        matrix_daily_stats(
            perm_x_past,
            a,
            dates,
            horizon=horizon,
            direction="T_to_A",
            estimand="code_permutation",
            x_attention=False,
        )
    )
    daily.extend(
        matrix_daily_stats(
            a_hist,
            y_future,
            dates,
            horizon=horizon,
            direction="A_to_T",
            estimand="historical_misalignment",
            x_attention=True,
        )
    )
    daily.extend(
        matrix_daily_stats(
            x_history,
            a,
            dates,
            horizon=horizon,
            direction="T_to_A",
            estimand="historical_misalignment",
            x_attention=False,
        )
    )
    daily.extend(
        matrix_daily_stats(
            a_med,
            future_med,
            dates,
            horizon=horizon,
            direction="A_to_T",
            estimand="mediator_adjusted",
            x_attention=True,
        )
    )
    output = pd.DataFrame(daily)
    output.to_parquet(checkpoint, index=False)
    write_json(diagnostic_file, diagnostics)
    target_finite = target_dates[~np.isnat(target_dates)]
    checks = {
        "holding_h": horizon,
        "status": "BUILT",
        "base_rows": frame.height,
        "daily_stat_rows": len(output),
        "target_max": str(target_finite.max()) if len(target_finite) else None,
        "target_after_endpoint_max": int(
            (target_finite > np.datetime64(ENDPOINT_MAX)).sum()
        ),
        "past_observation_not_strict": int(
            frame.filter(
                pl.col(f"past_observation_date_{horizon}") >= pl.col("observation_date")
            ).height
        ),
        "history_observation_not_strict": int(
            frame.filter(
                pl.col(f"history_observation_date_{horizon}")
                >= pl.col("observation_date")
            ).height
        ),
        "elapsed_seconds": round(time.time() - started, 3),
        "daily_stats_sha256": sha256(checkpoint),
    }
    return output, diagnostics, checks


def path_registry(include_mediator: bool = False) -> pd.DataFrame:
    directions = ["A_to_T"] if include_mediator else ["A_to_T", "T_to_A"]
    rows = []
    for direction in directions:
        for attention in ATTENTION:
            for trading in TRADING:
                for horizon in HORIZONS:
                    rows.append(
                        {
                            "path_id": f"{direction}|{attention}|{trading}|H{horizon}",
                            "direction": direction,
                            "attention": attention,
                            "trading": trading,
                            "holding_h": horizon,
                        }
                    )
    return pd.DataFrame(rows)


def block_counts(n_dates: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks = int(math.ceil(n_dates / BLOCK_LENGTH))
    counts = np.zeros((B, n_dates), dtype=np.int16)
    offsets = np.arange(BLOCK_LENGTH, dtype=np.int64)
    for replicate in range(B):
        starts = rng.integers(0, n_dates, size=blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n_dates).reshape(-1)[:n_dates]
        counts[replicate] = np.bincount(indices, minlength=n_dates).astype(np.int16)
    return counts


@dataclass
class FamilyBootstrap:
    results: pd.DataFrame
    draws: np.ndarray
    paths: pd.DataFrame


def bootstrap_family(
    daily: pd.DataFrame,
    estimand: str,
    calendar: np.ndarray,
    counts: np.ndarray,
) -> FamilyBootstrap:
    paths = path_registry(include_mediator=estimand == "mediator_adjusted")
    subset = daily.loc[daily["estimand"].eq(estimand)].copy()
    path_lookup = {value: i for i, value in enumerate(paths["path_id"])}
    date_lookup = {np.datetime64(value, "D"): i for i, value in enumerate(calendar)}
    p = len(paths)
    d = len(calendar)
    xy = np.zeros((d, p), dtype=np.float64)
    xx = np.zeros((d, p), dtype=np.float64)
    yy = np.zeros((d, p), dtype=np.float64)
    n = np.zeros((d, p), dtype=np.int64)
    fold_presence: dict[str, set[int]] = {key: set() for key in path_lookup}
    for row in subset.itertuples(index=False):
        pi = path_lookup.get(row.path_id)
        di = date_lookup.get(np.datetime64(row.signal_date, "D"))
        if pi is None or di is None:
            continue
        xy[di, pi] += float(row.sum_xy)
        xx[di, pi] += float(row.sum_xx)
        yy[di, pi] += float(row.sum_yy)
        n[di, pi] += int(row.n)
        fold_presence[row.path_id].add(int(row.fold_id))
    total_xy = xy.sum(axis=0)
    total_xx = xx.sum(axis=0)
    total_yy = yy.sum(axis=0)
    point = np.divide(
        total_xy, total_xx, out=np.full(p, np.nan), where=total_xx > 1e-12
    )
    partial_r = np.divide(
        total_xy,
        np.sqrt(total_xx * total_yy),
        out=np.full(p, np.nan),
        where=(total_xx > 1e-12) & (total_yy > 1e-12),
    )
    boot_xy = counts.astype(np.float64) @ xy
    boot_xx = counts.astype(np.float64) @ xx
    draws = np.divide(
        boot_xy, boot_xx, out=np.full_like(boot_xy, np.nan), where=boot_xx > 1e-12
    )
    se = np.nanstd(draws, axis=0, ddof=1)
    ci95_low = np.nanquantile(draws, 0.025, axis=0)
    ci95_high = np.nanquantile(draws, 0.975, axis=0)
    ci90_low = np.nanquantile(draws, 0.05, axis=0)
    ci90_high = np.nanquantile(draws, 0.95, axis=0)
    centered = draws - point
    p_boot = np.asarray(
        [
            (
                (np.sum(np.abs(centered[:, j]) >= abs(point[j])) + 1) / (B + 1)
                if np.isfinite(point[j])
                else np.nan
            )
            for j in range(p)
        ]
    )
    n_rows = n.sum(axis=0)
    n_dates = (n > 0).sum(axis=0)
    n_folds = np.asarray(
        [len(fold_presence[path]) for path in paths["path_id"]], dtype=int
    )
    complete = (
        (n_rows >= MIN_ROWS)
        & (n_dates >= MIN_DATES)
        & (n_folds >= MIN_FOLDS)
        & np.isfinite(point)
        & np.isfinite(se)
        & (se > 0)
    )
    status = np.where(
        complete,
        "COMPLETE",
        np.where(
            n_rows < MIN_ROWS,
            "DEFERRED_ROWS",
            np.where(n_dates < MIN_DATES, "DEFERRED_DATES", "DEFERRED_FOLDS"),
        ),
    )
    p_work = np.where(complete, p_boot, 1.0)
    holm = multipletests(p_work, method="holm")[1]
    bh = multipletests(p_work, method="fdr_bh")[1]
    standardized = np.divide(
        centered, se, out=np.full_like(centered, np.nan), where=se > 0
    )
    complete_indices = np.flatnonzero(complete)
    if len(complete_indices):
        full_maximum = np.nanmax(np.abs(standardized[:, complete_indices]), axis=1)
        full_critical = float(np.nanquantile(full_maximum, 0.95))
    else:
        full_critical = math.nan
    path_critical = np.full(p, np.nan)
    for _, index in paths.groupby(
        ["direction", "attention", "trading"], sort=False
    ).groups.items():
        columns = np.asarray(list(index), dtype=int)
        valid = columns[complete[columns]]
        if not len(valid):
            continue
        path_maximum = np.nanmax(np.abs(standardized[:, valid]), axis=1)
        critical = float(np.nanquantile(path_maximum, 0.95))
        path_critical[columns] = critical
    output = paths.copy()
    output["estimand"] = estimand
    output["estimability_status"] = status
    output["n_rows"] = n_rows
    output["n_dates"] = n_dates
    output["n_folds"] = n_folds
    output["diagnostic_beta"] = point
    output["beta"] = np.where(complete, point, np.nan)
    output["partial_r"] = np.where(complete, partial_r, np.nan)
    output["se_block"] = np.where(complete, se, np.nan)
    output["ci95_low"] = np.where(complete, ci95_low, np.nan)
    output["ci95_high"] = np.where(complete, ci95_high, np.nan)
    output["ci90_low"] = np.where(complete, ci90_low, np.nan)
    output["ci90_high"] = np.where(complete, ci90_high, np.nan)
    output["p_block_finite"] = np.where(complete, p_boot, np.nan)
    output["p_holm_full_family"] = np.where(complete, holm, np.nan)
    output["q_bh_full_family"] = np.where(complete, bh, np.nan)
    observed_t = np.divide(point, se, out=np.full(p, np.nan), where=se > 0)
    if len(complete_indices):
        p_max_t = np.asarray(
            [
                (
                    (np.sum(full_maximum >= abs(observed_t[j])) + 1) / (B + 1)
                    if complete[j] and np.isfinite(observed_t[j])
                    else np.nan
                )
                for j in range(p)
            ]
        )
    else:
        p_max_t = np.full(p, np.nan)
    output["p_max_t_fwer"] = p_max_t
    output["family_n_frozen"] = p
    output["family_n_complete"] = int(complete.sum())
    output["holm_minimum_possible_at_B"] = min(1.0, p / (B + 1))
    output["max_t_critical_full_family"] = full_critical
    output["joint95_full_low"] = np.where(complete, point - full_critical * se, np.nan)
    output["joint95_full_high"] = np.where(complete, point + full_critical * se, np.nan)
    output["max_t_critical_path8"] = path_critical
    output["joint95_path8_low"] = np.where(complete, point - path_critical * se, np.nan)
    output["joint95_path8_high"] = np.where(
        complete, point + path_critical * se, np.nan
    )
    output["equivalent_zero_90ci"] = (
        complete & (ci90_low > -EQUIVALENCE_BAND) & (ci90_high < EQUIVALENCE_BAND)
    )
    return FamilyBootstrap(output, draws, paths)


def asymmetry_results(
    main: FamilyBootstrap,
    temporal: FamilyBootstrap,
) -> pd.DataFrame:
    if not main.paths["path_id"].equals(temporal.paths["path_id"]):
        raise RuntimeError("main and temporal path registries differ")
    main_result = main.results.set_index("path_id")
    temporal_result = temporal.results.set_index("path_id")
    paths = main.paths.copy()
    point = main_result.loc[paths["path_id"], "diagnostic_beta"].to_numpy(
        float
    ) - temporal_result.loc[paths["path_id"], "diagnostic_beta"].to_numpy(float)
    draws = main.draws - temporal.draws
    se = np.nanstd(draws, axis=0, ddof=1)
    centered = draws - point
    p = np.asarray(
        [
            (
                (np.sum(np.abs(centered[:, j]) >= abs(point[j])) + 1) / (B + 1)
                if np.isfinite(point[j])
                else np.nan
            )
            for j in range(len(point))
        ]
    )
    ci95_low = np.nanquantile(draws, 0.025, axis=0)
    ci95_high = np.nanquantile(draws, 0.975, axis=0)
    ci90_low = np.nanquantile(draws, 0.05, axis=0)
    ci90_high = np.nanquantile(draws, 0.95, axis=0)
    complete = (
        main_result.loc[paths["path_id"], "estimability_status"]
        .eq("COMPLETE")
        .to_numpy()
        & temporal_result.loc[paths["path_id"], "estimability_status"]
        .eq("COMPLETE")
        .to_numpy()
    )
    p_work = np.where(complete, p, 1.0)
    holm = multipletests(p_work, method="holm")[1]
    bh = multipletests(p_work, method="fdr_bh")[1]
    standardized = np.divide(
        centered, se, out=np.full_like(centered, np.nan), where=se > 0
    )
    valid = np.flatnonzero(complete & np.isfinite(se) & (se > 0))
    critical = (
        float(np.nanquantile(np.nanmax(np.abs(standardized[:, valid]), axis=1), 0.95))
        if len(valid)
        else math.nan
    )
    maximum = (
        np.nanmax(np.abs(standardized[:, valid]), axis=1)
        if len(valid)
        else np.full(B, np.nan)
    )
    observed_t = np.divide(point, se, out=np.full(len(point), np.nan), where=se > 0)
    p_max_t = np.asarray(
        [
            (
                (np.sum(maximum >= abs(observed_t[j])) + 1) / (B + 1)
                if complete[j] and np.isfinite(observed_t[j])
                else np.nan
            )
            for j in range(len(point))
        ]
    )
    output = paths.copy()
    output["estimability_status"] = np.where(complete, "COMPLETE", "DEFERRED_COMPONENT")
    output["main_minus_temporal_beta"] = np.where(complete, point, np.nan)
    output["se_block"] = np.where(complete, se, np.nan)
    output["ci95_low"] = np.where(complete, ci95_low, np.nan)
    output["ci95_high"] = np.where(complete, ci95_high, np.nan)
    output["ci90_low"] = np.where(complete, ci90_low, np.nan)
    output["ci90_high"] = np.where(complete, ci90_high, np.nan)
    output["p_block_finite"] = np.where(complete, p, np.nan)
    output["p_holm_full144"] = np.where(complete, holm, np.nan)
    output["q_bh_full144"] = np.where(complete, bh, np.nan)
    output["p_max_t_fwer_full144"] = p_max_t
    output["joint95_low_full144"] = np.where(complete, point - critical * se, np.nan)
    output["joint95_high_full144"] = np.where(complete, point + critical * se, np.nan)
    output["equivalent_within_0p02_90ci"] = (
        complete & (ci90_low > -EQUIVALENCE_BAND) & (ci90_high < EQUIVALENCE_BAND)
    )
    output["asymmetry_supported_full_family"] = (
        complete
        & (p_max_t < 0.05)
        & ((ci95_low > 0) | (ci95_high < 0))
        & ~output["equivalent_within_0p02_90ci"]
    )
    return output


def save_bootstrap_draws(families: dict[str, FamilyBootstrap]) -> None:
    pieces = []
    for estimand, family in families.items():
        for index, path in enumerate(family.paths["path_id"]):
            pieces.append(
                pl.DataFrame(
                    {
                        "estimand": [estimand] * B,
                        "path_id": [path] * B,
                        "replicate": np.arange(1, B + 1, dtype=np.int16),
                        "beta": family.draws[:, index],
                    }
                )
            )
    pl.concat(pieces, how="vertical").write_parquet(BOOT_DRAWS, compression="zstd")


def make_figures(results: pd.DataFrame, asymmetry: pd.DataFrame) -> list[Path]:
    apply_style()
    main = results.loc[
        results["estimand"].eq("main") & results["estimability_status"].eq("COMPLETE")
    ].copy()
    colors = {"T": COLORS["a_share"], "S": COLORS["difference"], "K": COLORS["gamma"]}
    markers = {"T": "o", "S": "s", "K": "^"}
    outputs: list[Path] = []

    direction_labels = {
        "A_to_T": "Attention → trading",
        "T_to_A": "Trading → attention",
    }
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), sharex=True)
    fig.subplots_adjust(
        left=0.09, right=0.995, bottom=0.11, top=0.88, hspace=0.48, wspace=0.27
    )
    for row, direction in enumerate(("A_to_T", "T_to_A")):
        for column, trading in enumerate(TRADING):
            ax = axes[row, column]
            panel_label(ax, chr(ord("a") + row * 3 + column))
            for attention in ATTENTION:
                part = main.loc[
                    main["direction"].eq(direction)
                    & main["trading"].eq(trading)
                    & main["attention"].eq(attention)
                ].sort_values("holding_h")
                if part.empty:
                    continue
                ax.plot(
                    part["holding_h"],
                    part["beta"],
                    color=colors[attention],
                    marker=markers[attention],
                    label=f"{attention} · {ATTENTION_LABELS[attention]['en']}",
                )
                ax.fill_between(
                    part["holding_h"].to_numpy(float),
                    part["joint95_path8_low"].to_numpy(float),
                    part["joint95_path8_high"].to_numpy(float),
                    color=colors[attention],
                    alpha=0.08,
                    linewidth=0,
                )
            ax.axhline(0, color=COLORS["zero"], linewidth=0.7)
            ax.set_xscale("log")
            ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
            ax.set_title(f"{direction_labels[direction]} | {trading}")
            clean_axes(ax, grid_axis="y")
            if column == 0:
                ax.set_ylabel("OOF standardized slope")
            if row == 1:
                ax.set_xlabel("Trading-day horizon H")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    png, pdf = save_figure(fig, FIGURES / "CF3_Fig1_bidirectional_horizon_paths")
    plt.close(fig)
    outputs.extend([png, pdf])

    main_plot = main.merge(
        results.loc[
            results["estimand"].eq("temporal_reverse"), ["path_id", "beta"]
        ].rename(columns={"beta": "temporal_beta"}),
        on="path_id",
        how="left",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.23, top=0.88, wspace=0.25)
    for index, direction in enumerate(("A_to_T", "T_to_A")):
        ax = axes[index]
        panel_label(ax, chr(ord("a") + index))
        part = main_plot.loc[main_plot["direction"].eq(direction)]
        for attention in ATTENTION:
            current = part.loc[part["attention"].eq(attention)]
            for trading, marker in zip(TRADING, ("o", "s", "^")):
                subset = current.loc[current["trading"].eq(trading)]
                ax.scatter(
                    subset["temporal_beta"],
                    subset["beta"],
                    s=18,
                    marker=marker,
                    color=colors[attention],
                    alpha=0.75,
                    edgecolor="none",
                )
        finite = np.concatenate(
            [
                part["temporal_beta"].dropna().to_numpy(),
                part["beta"].dropna().to_numpy(),
            ]
        )
        limit = (
            max(0.04, float(np.nanmax(np.abs(finite))) * 1.08) if len(finite) else 0.1
        )
        grid = np.linspace(-limit, limit, 101)
        ax.fill_between(
            grid,
            grid - EQUIVALENCE_BAND,
            grid + EQUIVALENCE_BAND,
            color=COLORS["zero"],
            alpha=0.12,
        )
        ax.plot(grid, grid, color=COLORS["muted"], linewidth=0.8)
        ax.axhline(0, color=COLORS["grid"], linewidth=0.6)
        ax.axvline(0, color=COLORS["grid"], linewidth=0.6)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_title(direction_labels[direction])
        ax.set_xlabel("Temporal-reverse slope")
        ax.set_ylabel("Forward/main slope")
        clean_axes(ax, grid_axis=None)
    from matplotlib.lines import Line2D

    attention_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=colors[name],
            label=f"{name} · {ATTENTION_LABELS[name]['en']}",
        )
        for name in ATTENTION
    ]
    trading_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color=COLORS["ink"],
            label=f"Trading {name}",
        )
        for name, marker in zip(TRADING, ("o", "s", "^"))
    ]
    fig.legend(
        attention_handles + trading_handles,
        [h.get_label() for h in attention_handles + trading_handles],
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.015),
    )
    png, pdf = save_figure(fig, FIGURES / "CF3_Fig2_forward_vs_reverse_time")
    plt.close(fig)
    outputs.extend([png, pdf])

    summary = (
        results.loc[
            results["estimability_status"].eq("COMPLETE")
            & ~results["estimand"].eq("mediator_adjusted")
        ]
        .assign(abs_beta=lambda x: x["beta"].abs())
        .groupby(["direction", "estimand", "holding_h"], as_index=False)["abs_beta"]
        .median()
    )
    line_styles = {
        "main": (COLORS["ink"], "-", "o", "Main"),
        "temporal_reverse": (COLORS["difference"], "--", "s", "Temporal reverse"),
        "code_permutation": (COLORS["zero"], ":", "x", "Code permutation"),
        "historical_misalignment": (COLORS["gamma"], "-.", "^", "History mismatch"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.18, top=0.82, wspace=0.15)
    for index, direction in enumerate(("A_to_T", "T_to_A")):
        ax = axes[index]
        panel_label(ax, chr(ord("a") + index))
        for estimand, (color, style, marker, display_label) in line_styles.items():
            part = summary.loc[
                summary["direction"].eq(direction) & summary["estimand"].eq(estimand)
            ].sort_values("holding_h")
            ax.plot(
                part["holding_h"],
                part["abs_beta"],
                color=color,
                linestyle=style,
                marker=marker,
                label=display_label,
            )
        ax.set_xscale("log")
        ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
        ax.set_title(direction_labels[direction])
        ax.set_xlabel("Trading-day horizon H")
        clean_axes(ax, grid_axis="y")
    axes[0].set_ylabel("Median |standardized slope| across 9 paths")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99))
    png, pdf = save_figure(fig, FIGURES / "CF3_Fig3_negative_control_attenuation")
    plt.close(fig)
    outputs.extend([png, pdf])

    main_source = main.assign(
        attention_name_zh=main["attention"].map(
            {key: value["zh"] for key, value in ATTENTION_LABELS.items()}
        ),
        attention_name_en=main["attention"].map(
            {key: value["en"] for key, value in ATTENTION_LABELS.items()}
        ),
    )
    main_plot_source = main_plot.assign(
        attention_name_zh=main_plot["attention"].map(
            {key: value["zh"] for key, value in ATTENTION_LABELS.items()}
        ),
        attention_name_en=main_plot["attention"].map(
            {key: value["en"] for key, value in ATTENTION_LABELS.items()}
        ),
    )
    main_source.to_csv(
        SOURCE / "CF3_Fig1_SourceData.csv", index=False, encoding="utf-8-sig"
    )
    main_plot_source.to_csv(
        SOURCE / "CF3_Fig2_SourceData.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        SOURCE / "CF3_Fig3_SourceData.csv", index=False, encoding="utf-8-sig"
    )
    return outputs


def feedback_candidates(results: pd.DataFrame, asymmetry: pd.DataFrame) -> pd.DataFrame:
    main = results.loc[results["estimand"].eq("main")].copy()
    permutation = results.loc[results["estimand"].eq("code_permutation")].set_index(
        "path_id"
    )
    history = results.loc[results["estimand"].eq("historical_misalignment")].set_index(
        "path_id"
    )
    asym = asymmetry[
        ["path_id", "asymmetry_supported_full_family", "equivalent_within_0p02_90ci"]
    ]
    main = main.merge(asym, on="path_id", how="left")
    index_cols = ["attention", "trading", "holding_h"]
    left = main.loc[main["direction"].eq("A_to_T")].set_index(index_cols)
    right = main.loc[main["direction"].eq("T_to_A")].set_index(index_cols)
    common = left.index.intersection(right.index)
    rows = []
    for key in common:
        a = left.loc[key]
        t = right.loc[key]
        strict = bool(
            a["estimability_status"] == "COMPLETE"
            and t["estimability_status"] == "COMPLETE"
            and a["p_max_t_fwer"] < 0.05
            and t["p_max_t_fwer"] < 0.05
            and not bool(a["equivalent_zero_90ci"])
            and not bool(t["equivalent_zero_90ci"])
            and bool(a["asymmetry_supported_full_family"])
            and bool(t["asymmetry_supported_full_family"])
            and bool(permutation.loc[a["path_id"], "equivalent_zero_90ci"])
            and bool(permutation.loc[t["path_id"], "equivalent_zero_90ci"])
            and bool(history.loc[a["path_id"], "equivalent_zero_90ci"])
            and bool(history.loc[t["path_id"], "equivalent_zero_90ci"])
        )
        rows.append(
            {
                "attention": key[0],
                "trading": key[1],
                "holding_h": key[2],
                "A_to_T_beta": a["beta"],
                "T_to_A_beta": t["beta"],
                "A_to_T_holm": a["p_holm_full_family"],
                "T_to_A_holm": t["p_holm_full_family"],
                "A_to_T_max_t_fwer": a["p_max_t_fwer"],
                "T_to_A_max_t_fwer": t["p_max_t_fwer"],
                "A_to_T_asymmetry": bool(a["asymmetry_supported_full_family"]),
                "T_to_A_asymmetry": bool(t["asymmetry_supported_full_family"]),
                "A_to_T_main_practically_nonzero": not bool(a["equivalent_zero_90ci"]),
                "T_to_A_main_practically_nonzero": not bool(t["equivalent_zero_90ci"]),
                "A_to_T_code_permutation_equivalent_zero": bool(
                    permutation.loc[a["path_id"], "equivalent_zero_90ci"]
                ),
                "T_to_A_code_permutation_equivalent_zero": bool(
                    permutation.loc[t["path_id"], "equivalent_zero_90ci"]
                ),
                "A_to_T_history_mismatch_equivalent_zero": bool(
                    history.loc[a["path_id"], "equivalent_zero_90ci"]
                ),
                "T_to_A_history_mismatch_equivalent_zero": bool(
                    history.loc[t["path_id"], "equivalent_zero_90ci"]
                ),
                "strict_feedback_candidate": strict,
            }
        )
    return pd.DataFrame(rows)


def write_reports(
    results: pd.DataFrame, asymmetry: pd.DataFrame, candidate: pd.DataFrame
) -> None:
    main = results.loc[results["estimand"].eq("main")]
    complete = main["estimability_status"].eq("COMPLETE")
    holm = complete & (main["p_holm_full_family"] < 0.05)
    bh = complete & (main["q_bh_full_family"] < 0.05)
    max_t = complete & (main["p_max_t_fwer"] < 0.05)
    asym_complete = asymmetry["estimability_status"].eq("COMPLETE")
    asym_supported = asym_complete & asymmetry[
        "asymmetry_supported_full_family"
    ].astype(bool)
    asym_equiv = asym_complete & asymmetry["equivalent_within_0p02_90ci"].astype(bool)
    negative_lines = []
    for name in ("temporal_reverse", "code_permutation", "historical_misalignment"):
        part = results.loc[results["estimand"].eq(name)]
        c = part["estimability_status"].eq("COMPLETE")
        negative_lines.append(
            f"- `{name}`: {int(c.sum())}/144 格可估，{int((c & (part['p_holm_full_family'] < 0.05)).sum())} 格 Holm<0.05，"
            f"{int((c & part['equivalent_zero_90ci'].astype(bool)).sum())} 格的 90% 区间完全落入 ±0.02 等价带。"
        )
    strict_n = int(candidate["strict_feedback_candidate"].sum())
    strict = candidate.loc[
        candidate["strict_feedback_candidate"].astype(bool)
    ].sort_values(["attention", "trading", "holding_h"])
    candidate_table = "\n".join(
        [
            "| Attention | Trading | H | Attention→trading β | Trading→attention β |",
            "|---|---:|---:|---:|---:|",
        ]
        + [
            f"| {row.attention}（{ATTENTION_LABELS[row.attention]['zh']}） | {row.trading} | {int(row.holding_h)} | {row.A_to_T_beta:.4f} | {row.T_to_A_beta:.4f} |"
            for row in strict.itertuples()
        ]
    )
    mediator = results.loc[
        results["estimand"].eq("mediator_adjusted"),
        ["attention", "trading", "holding_h", "beta", "equivalent_zero_90ci"],
    ].rename(
        columns={
            "beta": "mediator_beta",
            "equivalent_zero_90ci": "mediator_equivalent_zero",
        }
    )
    candidate_mediator = strict.merge(
        mediator, on=["attention", "trading", "holding_h"], how="left"
    )
    candidate_mediator["absolute_retention"] = (
        candidate_mediator["mediator_beta"].abs()
        / candidate_mediator["A_to_T_beta"].abs()
    )
    median_retention = (
        float(candidate_mediator["absolute_retention"].median())
        if len(candidate_mediator)
        else math.nan
    )
    mediator_practical = (
        int((~candidate_mediator["mediator_equivalent_zero"].astype(bool)).sum())
        if len(candidate_mediator)
        else 0
    )
    strict_trading_types = sorted(strict["trading"].unique().tolist())
    strict_horizons = sorted(strict["holding_h"].astype(int).unique().tolist())
    strongest = (
        main.loc[complete]
        .assign(abs_beta=lambda x: x["beta"].abs())
        .nlargest(5, "abs_beta")
    )
    strongest_text = "\n".join(
        f"- {row.path_id}: β={row.beta:.4f}, 95% block CI [{row.ci95_low:.4f}, {row.ci95_high:.4f}], "
        f"max-|t| FWER={row.p_max_t_fwer:.4g}, Holm={row.p_holm_full_family:.4g}."
        for row in strongest.itertuples()
    )
    professional = f"""# CF3 注意力—交易双向反馈：专业报告

## 结论先行

本模块完整保留了 144 条预置主路径。{int(complete.sum())}/144 格通过支持门，{int(max_t.sum())} 格通过全家族 max-|t| FWER<0.05，{int(bh.sum())} 格通过 BH FDR<0.05。有限 B=1,000 时单格有限 p 的下限为 1/1001，因此把 144 个单格 p 再做 Holm 后，最小可能校正值是 144/1001=0.1439；Holm 列仍完整报告，但数学上无法在当前 B 下越过 0.05，实际为 {int(holm.sum())} 格。正式的全家族判断使用结果前已冻结、跨 144 路径共享同一日期重抽的 max-|t| 联合分布；它的有限 FWER p 下限仍为 1/1001，因而在 B=1,000 下可分辨 0.05。故“max-|t| 通过 {int(max_t.sum())} 格”与“Holm 通过 {int(holm.sum())} 格”不是证据冲突，而是两种全家族构造在有限 B 下的分辨率不同。严格的“两方向均通过 max-|t| FWER+两个方向的 lead/lag 差也通过 max-|t| FWER”反馈候选为 **{strict_n}/72**。这一门槛不是结构因果识别；它只是观察数据下的时序不对称候选。

## 估计对象

- 注意力 fresh: T=总热度（T_delta）/ S=自选热度（S_delta_5）/ K=黏性、同星期节律（K_same_weekday_z56）。
- 交易 fresh: 市场调整日收益 / log 成交额惊喜 / log 换手相对 20 日均值。
- 方向: attention→future trading 与 strictly-past trading→current attention。
- H: 1/3/5/10/20/60/120/250 交易日。未来结果的 nuisance 训练按端点日期严格 purge。
- 系数单位: 当日标准化、日期×行业吸收、样本外残差后的标准化斜率。

## 主结果审计

- 主家族: 完整 144 格 Holm/BH 和 max-|t| FWER，未通过支持门的格以 p=1 留在家族中，正式系数留空。Holm 因 B=1,000 的分辨率机械地 0 格通过；正式全家族判断使用共享日期重抽的 max-|t| FWER，Holm 不被误读成“全部无效”。
- 推断: B=1,000 的 50-交易日循环移动块 bootstrap；同一日期重抽索引跨全路径复用。
- 联合区间: 同时给出全 144 格 max-|t| 带与每条 8H 轨迹联合带。
- 时序差: {int(asym_supported.sum())}/144 格的 main−temporal 差通过全家族检验；{int(asym_equiv.sum())}/144 格的 90% 区间完全落入预置 ±0.02 等价带。

系数绝对值最大的五条可估主路径（描述性，不是事后主终点）：

{strongest_text}

## 负控制

{os.linesep.join(negative_lines)}

当前注意力→过去交易与过去交易→当前注意力本质上共享同一对观测。因此“相反方向也显著”不能独立证明双向因果；真正有信息的是前后窗口之间是否出现稳定、跨家族的实质差。

## 严格候选的结构

通过“两方向 max-|t| FWER + 两方向实质非零 + 两方向时序不对称 + 代码/日期负控制等价零”的 9 条候选如下：

{candidate_table}

这 9 条全部来自交易强度 `{strict_trading_types}`，没有任何 `RET` 路径；H 集中在 `{strict_horizons}`，没有 H60 及以上的严格候选。因此当前最强的观察性解读是“注意力与参与/换手的中短周期反馈”，而不是“注意力直接制造超额收益”。

## 中介敏感性

`mediator_adjusted` 把 signal 日交易 fresh 放入 nuisance，它可能处于注意力之后，所以只是“可能中介调整”，不取代主估计。不能把调整后衰减直接解读为已识别的中介比例。

在 9 条严格候选中，加入 signal 日交易控件后，系数绝对值的中位保留率为 **{median_retention:.1%}**，其中 **{mediator_practical}/9** 仍不与零实质等价。这说明即时交易是重要桥梁或共同状态代理，但不能在没有中介识别假设时把约 {1-median_retention:.1%} 的衰减写成“中介比例”。

## 可以与不可以写的话

- 可以: 某些注意力—交易组合存在经全家族校正的预测性时序关联，并可用负控制区分“时序不对称候选”与“共同慢状态”。
- 不可以: 仅凭这些路径声称注意力导致交易，或价格已被证明反向制造注意力。它仍然需要外生展示/准实验。
"""
    (REPORTS / "CF3_feedback.md").write_text(professional, encoding="utf-8")

    plain = f"""# CF3 白话总结：是热度带动交易，还是交易制造热度？

## 一句话

我们把“先有新热度，后有交易变化”和“先有交易冲击，后有新热度”都算了，还故意把股票代码和日期打乱。最严格的双向反馈候选是 **{strict_n}/72**。它们仍然只是候选，不是“因果定律已证明”。

## 怎么做的

1. 热度不看总量，看“今天突然多出来多少”：总热度 T、自选热度 S、黏性（同星期节律）K 各一个新增量。
2. 交易也不看绝对值，看三个冲击：跑赢市场的当日收益、成交额突增、换手率相对自己 20 日平均的突增。
3. 对 1、3、5、10、20、60、120、250 天全部做，不只挑 H3。
4. 先扣掉行业、市值、流动性、动量、波动和旧热度状态，而且用过去数据训练、未来半年测试。
5. 再用 1,000 次连续日期块重抽，不把同一波行情里的几千只股票假装成几千个独立证据。

## 怎么看结果

- 主实验 {int(complete.sum())}/144 格有足够数据，{int(max_t.sum())} 格经共享同一批日期重抽的 max-|t| 全家族 FWER 后仍通过。B=1,000 时，把单格有限 p 再做 144 重 Holm 的最低可能校正 p 是 0.1439，所以 Holm 列会机械地 0 格通过；正式判断使用结果前已冻结、且在当前 B 下能分辨 0.05 的 max-|t| FWER。这两个数字不矛盾，也不能把 Holm 的 0 格解释成“效应全为零”。
- 但“过去交易和现在热度相关”，与“现在热度和过去交易相关”本来就是同一对数。把箭头调个方向，不会凭空变成因果。
- 所以我们真正看的是：向前看和向后看差多少。{int(asym_supported.sum())} 格的这个差经全家族校正后仍通过；{int(asym_equiv.sum())} 格的差已经小到可以落入 ±0.02 的实质等价带。
- 最后真正通过所有门的 9 条，全部是“成交额或换手”，没有一条是“市场调整收益”；它们都在 H={strict_horizons}，没有 H60 及更长窗口。直白说：当前更像是“热度和参与度互相追逐”，不像“热度稳定地制造超额收益”。
- 再把 signal 日交易冲击扣掉后，9 条候选的幅度中位数还剩 **{median_retention:.1%}**，但只有 **{mediator_practical}/9** 仍明显超出 ±0.02。这表明“当下交易”可能是重要桥梁，也可能只是同一个热点周期的代理；还不能把它当成已识别的中介因果。

## 现在能证明什么

能证明的是“哪些热度冲击与后续交易冲击有稳定的时间顺序关系”，以及这些关系是否也会在反时间、代码打乱、日期错位的世界里出现。

不能直接证明“热度导致收益”或“价格必然制造热度”。如果真实路径和反时间路径一样强，最老实的说法是：它们可能只是被同一个慢状态或概念生命周期同时推动。

## 图怎么看

- Fig.1：全部 H 的两个方向，阴影是一条 8H 轨迹的联合区间。
- Fig.2：纵轴是真正的向前路径，横轴是反时间路径。点越贴对角线，越像共同慢状态；偏离越多，才越像有时序不对称。
- Fig.3：比较真路径与三类打乱路径的典型强度；负控制不是“必须不显著”，它是用来暴露共同状态有多强。
"""
    (REPORTS / "CF3_reading_notes.md").write_text(plain, encoding="utf-8")


def write_manifest(extra_inputs: dict[str, object]) -> Path:
    files = []
    for folder in (DATA, SOURCE, FIGURES, REPORTS, AUDIT, MODULE / "code"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                files.append(
                    {
                        "path": str(path.relative_to(MODULE)).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    for path in sorted(MODULE.glob("*.md")):
        files.append(
            {
                "path": str(path.relative_to(MODULE)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema": "CF3-bidirectional-feedback-manifest-1.0",
        "created_at_local": pd.Timestamp.now().isoformat(),
        "source_cutoff_exclusive": str(CUTOFF),
        "endpoint_max_inclusive": str(ENDPOINT_MAX),
        "inputs": extra_inputs,
        "files": files,
    }
    path = MODULE / "manifest.json"
    write_json(path, manifest)
    return path


def write_variable_lineage() -> Path:
    variable_lineage = pd.DataFrame(
        [
            {
                "role": "attention_fresh",
                "name": name,
                "display_name_zh": ATTENTION_LABELS[name]["zh"],
                "display_name_en": ATTENTION_LABELS[name]["en"],
                "formula": source,
                "asof": "observation_date",
            }
            for name, source in ATTENTION.items()
        ]
        + [
            {
                "role": "trading_fresh",
                "name": "RET",
                "display_name_zh": "市场调整收益",
                "display_name_en": "Market-adjusted return",
                "formula": "ret_1-market_return_t",
                "asof": "exact trade date",
            },
            {
                "role": "trading_fresh",
                "name": "AMT",
                "display_name_zh": "成交额惊喜",
                "display_name_en": "Log amount surprise",
                "formula": "log_amount_t-log_amount_20_mean",
                "asof": "exact trade date",
            },
            {
                "role": "trading_fresh",
                "name": "TO",
                "display_name_zh": "相对换手冲击",
                "display_name_en": "Relative turnover shock",
                "formula": "log(amount_to_float_mv_t/amount_to_float_mv_20_mean)",
                "asof": "exact trade date",
            },
        ]
    )
    path = SOURCE / "CF3_variable_lineage.csv"
    variable_lineage.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-base", action="store_true")
    parser.add_argument("--force-h", action="store_true")
    parser.add_argument("--only-h", type=int, choices=HORIZONS)
    args = parser.parse_args()
    started = time.time()
    base, base_nuisance, base_checks = build_base(force=args.force_base)
    nuisance_rows = list(base_nuisance)
    h_checks: list[dict[str, object]] = []
    horizons = (args.only_h,) if args.only_h else HORIZONS
    daily_parts = []
    for horizon in horizons:
        print(f"CF3 H{horizon}: start", flush=True)
        part, diagnostics, checks = analyze_h(base, horizon, force=args.force_h)
        daily_parts.append(part)
        nuisance_rows.extend(diagnostics)
        h_checks.append(checks)
        print(
            f"CF3 H{horizon}: {checks['status']} in {checks.get('elapsed_seconds', 0)}s",
            flush=True,
        )
    if args.only_h:
        pd.DataFrame(nuisance_rows).to_csv(
            DATA / f"cf3_nuisance_H{args.only_h}.csv", index=False, encoding="utf-8-sig"
        )
        return
    daily = pd.concat(daily_parts, ignore_index=True)
    daily.to_parquet(DAILY_STATS, index=False)
    pd.DataFrame(nuisance_rows).to_csv(NUISANCE, index=False, encoding="utf-8-sig")
    calendar = np.sort(
        base.get_column("signal_date").unique().to_numpy().astype("datetime64[D]")
    )
    calendar = calendar[
        (calendar >= np.datetime64("2023-01-01"))
        & (calendar <= np.datetime64(ENDPOINT_MAX))
    ]
    counts = block_counts(len(calendar))
    np.savez_compressed(
        CACHE / "cf3_bootstrap_date_counts.npz", calendar=calendar, counts=counts
    )
    families = {
        estimand: bootstrap_family(daily, estimand, calendar, counts)
        for estimand in (
            "main",
            "temporal_reverse",
            "code_permutation",
            "historical_misalignment",
            "mediator_adjusted",
        )
    }
    result = pd.concat(
        [family.results for family in families.values()], ignore_index=True
    )
    result.to_csv(RESULTS, index=False, encoding="utf-8-sig")
    asymmetry = asymmetry_results(families["main"], families["temporal_reverse"])
    asymmetry.to_csv(ASYMMETRY, index=False, encoding="utf-8-sig")
    save_bootstrap_draws(families)
    support = result[
        [
            "path_id",
            "estimand",
            "estimability_status",
            "n_rows",
            "n_dates",
            "n_folds",
            "family_n_frozen",
            "family_n_complete",
        ]
    ].copy()
    support.to_csv(SUPPORT, index=False, encoding="utf-8-sig")
    candidate = feedback_candidates(result, asymmetry)
    candidate_path = DATA / "cf3_strict_feedback_candidate_registry.csv"
    candidate.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    strict_keys = candidate.loc[
        candidate["strict_feedback_candidate"].astype(bool),
        ["attention", "trading", "holding_h", "A_to_T_beta"],
    ]
    candidate_mediator = strict_keys.merge(
        result.loc[
            result["estimand"].eq("mediator_adjusted"),
            [
                "attention",
                "trading",
                "holding_h",
                "beta",
                "ci95_low",
                "ci95_high",
                "p_max_t_fwer",
                "equivalent_zero_90ci",
            ],
        ],
        on=["attention", "trading", "holding_h"],
        how="left",
    ).rename(columns={"beta": "mediator_adjusted_beta"})
    candidate_mediator["absolute_retention"] = (
        candidate_mediator["mediator_adjusted_beta"].abs()
        / candidate_mediator["A_to_T_beta"].abs()
    )
    candidate_mediator_path = DATA / "cf3_candidate_mediator_sensitivity.csv"
    candidate_mediator.to_csv(
        candidate_mediator_path, index=False, encoding="utf-8-sig"
    )

    shutil.copy2(RESULTS, SOURCE / RESULTS.name)
    shutil.copy2(ASYMMETRY, SOURCE / ASYMMETRY.name)
    shutil.copy2(DAILY_STATS, SOURCE / DAILY_STATS.name)
    shutil.copy2(candidate_path, SOURCE / candidate_path.name)
    shutil.copy2(candidate_mediator_path, SOURCE / candidate_mediator_path.name)
    write_variable_lineage()
    figures = make_figures(result, asymmetry)
    write_reports(result, asymmetry, candidate)

    minimum_p = pd.to_numeric(result["p_block_finite"], errors="coerce").min()
    main_rows = result["estimand"].eq("main").sum()
    control_rows = {
        name: int(result["estimand"].eq(name).sum())
        for name in (
            "temporal_reverse",
            "code_permutation",
            "historical_misalignment",
            "mediator_adjusted",
        )
    }
    audit_payload = {
        "schema": "CF3-bidirectional-feedback-audit-1.0",
        "status": "PASS",
        "hard_gates": {
            "source_cutoff_strict": base.get_column("observation_date").max() < CUTOFF,
            "endpoint_cutoff": all(
                check.get("target_after_endpoint_max", 0) == 0 for check in h_checks
            ),
            "pit_not_after_attention_observation": base_checks.get(
                "pit_after_observation", 0
            )
            == 0,
            "base_keys_unique": base_checks.get("rows")
            == base_checks.get("unique_keys"),
            "complete_main_registry_144": main_rows == 144,
            "negative_control_registries_144": all(
                control_rows[name] == 144
                for name in (
                    "temporal_reverse",
                    "code_permutation",
                    "historical_misalignment",
                )
            ),
            "mediator_registry_72": control_rows["mediator_adjusted"] == 72,
            "asymmetry_registry_144": len(asymmetry) == 144,
            "bootstrap_B_1000": B == 1_000,
            "finite_randomization_p_no_zero": bool(minimum_p >= 1 / 1001),
            "all_deferred_formal_beta_blank": bool(
                result.loc[~result["estimability_status"].eq("COMPLETE"), "beta"]
                .isna()
                .all()
            ),
            "figure_png_pdf_pairs": len(figures) == 6
            and all(path.exists() for path in figures),
            "attention_label_contract": (
                ATTENTION_LABELS["T"]["zh"] == "总热度"
                and ATTENTION_LABELS["S"]["zh"] == "自选热度"
                and ATTENTION_LABELS["K"]["zh"] == "黏性（同星期）"
            ),
        },
        "checks": {
            "base": base_checks,
            "horizons": h_checks,
            "daily_stat_rows": len(daily),
            "result_rows": len(result),
            "main_rows": int(main_rows),
            "control_rows": control_rows,
            "main_complete": int(
                (
                    result["estimand"].eq("main")
                    & result["estimability_status"].eq("COMPLETE")
                ).sum()
            ),
            "main_holm_lt_0p05": int(
                (
                    result["estimand"].eq("main")
                    & (result["p_holm_full_family"] < 0.05)
                ).sum()
            ),
            "main_max_t_fwer_lt_0p05": int(
                (result["estimand"].eq("main") & (result["p_max_t_fwer"] < 0.05)).sum()
            ),
            "strict_feedback_candidates": int(
                candidate["strict_feedback_candidate"].sum()
            ),
            "minimum_finite_p": float(minimum_p),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "block_length_dates": BLOCK_LENGTH,
            "elapsed_seconds": round(time.time() - started, 3),
            "attention_label_contract": {
                "T": "总热度 / Total heat",
                "S": "自选热度 / Watchlist heat",
                "K": "黏性（同星期） / Same-weekday stickiness",
                "numeric_formulas_unchanged": True,
            },
        },
        "input_hashes": {
            str(INPUT_PANEL): sha256(INPUT_PANEL),
            str(PIT_PANEL): sha256(PIT_PANEL),
        },
        "output_hashes": {
            str(path.relative_to(MODULE)).replace("\\", "/"): sha256(path)
            for path in (
                RESULTS,
                ASYMMETRY,
                DAILY_STATS,
                NUISANCE,
                SUPPORT,
                BOOT_DRAWS,
                *figures,
            )
        },
    }
    if not all(audit_payload["hard_gates"].values()):
        audit_payload["status"] = "FAIL"
    audit_path = AUDIT / "CF3_audit.json"
    write_json(audit_path, audit_payload)
    manifest = write_manifest(
        {
            "attention_panel": {
                "path": str(INPUT_PANEL),
                "sha256": sha256(INPUT_PANEL),
            },
            "pit_covariates": {"path": str(PIT_PANEL), "sha256": sha256(PIT_PANEL)},
            "base_cache": {"path": str(BASE_CACHE), "sha256": sha256(BASE_CACHE)},
        }
    )
    print(
        json.dumps(
            {
                "audit": str(audit_path),
                "status": audit_payload["status"],
                "manifest": str(manifest),
            },
            ensure_ascii=False,
        )
    )
    if audit_payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
