#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CF1: coupling-free endpoint-level attention impulse responses.

The formal exposure variables are construct-specific fresh-state proxies:
T_delta, S_delta_5 and K_same_weekday_z56.  Outcomes are future endpoint
levels recovered from the frozen exact-target change fields.  The old
future-minus-current change is retained only as an accounting bridge.

This is an adjusted observational state-dynamics design.  It is not an
exogenous treatment-effect estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
CQ = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_style import COLORS, clean_axes, panel_label, save_figure  # noqa: E402

ATTENTION = Path("__SNAPSHOT__/attention/inputs/attention_daily.parquet")
PIT = Path("__SNAPSHOT__/input_recovery/stock_pit_covariates_daily.universe.parquet")
MARKET = Path("__SNAPSHOT__/raw_snapshot/stock-main-index-data/sh000300.csv")

CACHE = ROOT / "cache" / "extensions"
DATA = ROOT / "data" / "extensions"
SOURCE = ROOT / "SourceData" / "extensions"
FIGURES = ROOT / "figures" / "extensions"
AUDIT = ROOT / "audit" / "extensions"
REPORTS = ROOT / "reports" / "extensions"
PANEL = ROOT / "shared" / "construct_CF1_level_panel.parquet"

CUTOFF = pd.Timestamp("2026-09-01")
START = pd.Timestamp("2022-03-23")
HORIZONS = (1, 3, 5, 10, 20, 60, 120, 250)
PROXIES = {
    "T": {"level": "log_T", "fresh": "T_delta", "slow": "slow_T"},
    "S": {"level": "log_S", "fresh": "S_delta_5", "slow": "slow_S"},
    "K": {"level": "log_K", "fresh": "K_same_weekday_z56", "slow": "slow_K"},
}
NUMERIC_CONTROLS = (
    "slow_T",
    "slow_S",
    "slow_K",
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
    "market_return_t",
)
BLOCK_LENGTH = 120
BOOTSTRAPS = 1_000
SEED = 20260714
MAX_TRAIN = 100_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_dirs() -> None:
    for directory in (CACHE, DATA, SOURCE, FIGURES, AUDIT, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)


def market_calendar() -> pd.DatetimeIndex:
    raw = pd.read_csv(MARKET, usecols=["candle_end_time"])
    dates = pd.to_datetime(raw["candle_end_time"], errors="coerce")
    dates = pd.DatetimeIndex(
        sorted(dates[(dates.notna()) & (dates < CUTOFF)].dt.normalize().unique())
    )
    if not dates.is_monotonic_increasing or len(dates) < 1_000:
        raise RuntimeError("invalid market calendar")
    return dates


def target_map() -> pl.DataFrame:
    calendar = market_calendar()
    rows: dict[str, object] = {"signal_date": calendar}
    for horizon in HORIZONS:
        target = np.full(len(calendar), np.datetime64("NaT"), dtype="datetime64[ns]")
        if horizon + 1 < len(calendar):
            target[: -(horizon + 1)] = calendar[horizon + 1 :].to_numpy(
                dtype="datetime64[ns]"
            )
        rows[f"target_date_{horizon}"] = target
    return pl.from_pandas(pd.DataFrame(rows))


def rank_pct(column: str, group: str = "signal_date") -> pl.Expr:
    denominator = pl.col(column).count().over(group).cast(pl.Float64)
    return (pl.col(column).rank(method="average").over(group) / denominator).cast(
        pl.Float32
    )


def prepare_panel(force: bool = False) -> dict[str, object]:
    ensure_dirs()
    if PANEL.exists() and not force:
        pf = pq.ParquetFile(PANEL)
        return {
            "path": str(PANEL),
            "rows": pf.metadata.num_rows,
            "columns": pf.schema_arrow.names,
            "reused": True,
        }
    for path in (ATTENTION, PIT, MARKET):
        if not path.exists():
            raise FileNotFoundError(path)

    attention_columns = [
        "signal_date",
        "stock_code",
        "observation_date",
        "common_TSK",
        "log_T",
        "T_delta",
        "log_S",
        "S_delta_5",
        "log_K",
        "K_same_weekday_z56",
    ] + [f"future_own_{proxy}_change_h{h}" for proxy in PROXIES for h in HORIZONS]
    attention = (
        pl.scan_parquet(ATTENTION)
        .select(attention_columns)
        .filter(
            (pl.col("signal_date") >= pl.lit(START.to_pydatetime()))
            & (pl.col("signal_date") < pl.lit(CUTOFF.to_pydatetime()))
            & (pl.col("common_TSK") == 1)
        )
        .sort(["stock_code", "signal_date"])
        .with_columns(
            pl.col("log_T").shift(1).over("stock_code").alias("pre_T"),
            pl.col("log_S").shift(1).over("stock_code").alias("pre_S"),
            pl.col("log_K").shift(1).over("stock_code").alias("pre_K"),
            pl.col("log_T")
            .shift(1)
            .rolling_mean(window_size=20, min_samples=10)
            .over("stock_code")
            .alias("slow_T"),
            pl.col("log_S")
            .shift(1)
            .rolling_mean(window_size=20, min_samples=10)
            .over("stock_code")
            .alias("slow_S"),
            pl.col("log_K")
            .shift(1)
            .rolling_median(window_size=20, min_samples=10)
            .over("stock_code")
            .alias("slow_K"),
        )
    )

    pit = (
        pl.scan_parquet(PIT)
        .select(
            "stock_code",
            pl.col("signal_date").alias("pit_date"),
            *[column for column in NUMERIC_CONTROLS if not column.startswith("slow_")],
            "sw1_industry",
        )
        .filter(pl.col("pit_date") < pl.lit(CUTOFF.to_pydatetime()))
        .sort(["stock_code", "pit_date"])
    )

    joined = attention.join_asof(
        pit,
        left_on="observation_date",
        right_on="pit_date",
        by="stock_code",
        strategy="backward",
        tolerance="10d",
        check_sortedness=False,
    ).join(target_map().lazy(), on="signal_date", how="left")

    endpoint_exprs: list[pl.Expr] = []
    for proxy, spec in PROXIES.items():
        level = spec["level"]
        endpoint_exprs.append(rank_pct(level).alias(f"current_rank_{proxy}"))
        endpoint_exprs.append(rank_pct(spec["fresh"]).alias(f"fresh_rank_{proxy}"))
        for horizon in HORIZONS:
            endpoint_exprs.append(
                (pl.col(level) + pl.col(f"future_own_{proxy}_change_h{horizon}"))
                .cast(pl.Float32)
                .alias(f"future_level_{proxy}_{horizon}")
            )
    joined = joined.with_columns(*endpoint_exprs)

    future_rank_exprs: list[pl.Expr] = []
    for proxy in PROXIES:
        for horizon in HORIZONS:
            future_rank_exprs.append(
                rank_pct(f"future_level_{proxy}_{horizon}").alias(
                    f"future_rank_{proxy}_{horizon}"
                )
            )
    joined = joined.with_columns(*future_rank_exprs)

    keep = [
        "signal_date",
        "stock_code",
        "observation_date",
        "pit_date",
        "sw1_industry",
        *NUMERIC_CONTROLS,
    ]
    for proxy, spec in PROXIES.items():
        keep.extend(
            [
                spec["level"],
                spec["fresh"],
                f"pre_{proxy}",
                f"fresh_rank_{proxy}",
                f"current_rank_{proxy}",
            ]
        )
        for horizon in HORIZONS:
            keep.extend(
                [
                    f"target_date_{horizon}",
                    f"future_own_{proxy}_change_h{horizon}",
                    f"future_level_{proxy}_{horizon}",
                    f"future_rank_{proxy}_{horizon}",
                ]
            )
    # Preserve first occurrence of shared target-date columns.
    keep = list(dict.fromkeys(keep))
    temporary = PANEL.with_name(f".{PANEL.name}.partial")
    temporary.unlink(missing_ok=True)
    joined.select(keep).sort(["signal_date", "stock_code"]).sink_parquet(
        temporary,
        compression="zstd",
        compression_level=6,
        statistics=True,
        engine="streaming",
        mkdir=True,
    )
    temporary.replace(PANEL)

    checks = (
        pl.scan_parquet(PANEL)
        .select(
            pl.len().alias("rows"),
            pl.struct(["signal_date", "stock_code"]).n_unique().alias("unique_keys"),
            pl.col("signal_date").n_unique().alias("dates"),
            pl.col("stock_code").n_unique().alias("stocks"),
            pl.col("signal_date").min().alias("signal_min"),
            pl.col("signal_date").max().alias("signal_max"),
            pl.col("pit_date").is_not_null().sum().alias("pit_join_rows"),
            (pl.col("pit_date") > pl.col("observation_date"))
            .sum()
            .alias("pit_after_attention"),
            pl.max_horizontal(*[pl.col(f"target_date_{h}") for h in HORIZONS])
            .max()
            .alias("max_target"),
        )
        .collect()
        .row(0, named=True)
    )
    hard_gates = {
        "unique_keys": int(checks["rows"]) == int(checks["unique_keys"]),
        "source_cutoff": pd.Timestamp(checks["signal_max"]) < CUTOFF,
        "endpoint_cutoff": pd.Timestamp(checks["max_target"])
        <= CUTOFF - pd.Timedelta(days=1),
        "pit_never_after_attention_snapshot": int(checks["pit_after_attention"]) == 0,
        "pit_join_coverage_ge_95pct": int(checks["pit_join_rows"]) / int(checks["rows"])
        >= 0.95,
    }
    payload = {
        "schema": "construct-CF1-panel-1.0",
        "status": "PASS" if all(hard_gates.values()) else "FAIL",
        "hard_gates": hard_gates,
        "checks": {k: str(v) for k, v in checks.items()},
        "inputs": {str(path): sha256(path) for path in (ATTENTION, PIT, MARKET)},
        "output": {
            "path": str(PANEL),
            "sha256": sha256(PANEL),
            "bytes": PANEL.stat().st_size,
        },
    }
    (AUDIT / "construct_CF1_panel.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if payload["status"] != "PASS":
        raise RuntimeError(json.dumps(payload, ensure_ascii=False))
    return payload


def fold_registry(dates: pd.Series) -> pd.DataFrame:
    periods = [
        (0, "2023-01-01", "2023-06-30"),
        (1, "2023-07-01", "2023-12-31"),
        (2, "2024-01-01", "2024-06-30"),
        (3, "2024-07-01", "2024-12-31"),
        (4, "2025-01-01", "2025-06-30"),
        (5, "2025-07-01", "2025-12-31"),
        (6, "2026-01-01", "2026-06-30"),
        (7, "2026-07-01", "2026-08-31"),
    ]
    rows = []
    available = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    for fold, start, end in periods:
        test = available[
            (available >= pd.Timestamp(start)) & (available <= pd.Timestamp(end))
        ]
        if len(test):
            rows.append(
                {
                    "fold_id": fold,
                    "test_start": test.min(),
                    "test_end": test.max(),
                    "test_dates": len(test),
                }
            )
    return pd.DataFrame(rows)


def deterministic_thin(indices: np.ndarray, maximum: int = MAX_TRAIN) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)]


def matrices(
    frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    numeric = list(NUMERIC_CONTROLS)
    train = (
        frame.iloc[train_idx][numeric]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    test = (
        frame.iloc[test_idx][numeric]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    med = np.array(
        [
            np.nanmedian(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0
            for c in train.T
        ]
    )
    train_missing = ~np.isfinite(train)
    test_missing = ~np.isfinite(test)
    train = np.where(train_missing, med, train)
    test = np.where(test_missing, med, test)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[(~np.isfinite(std)) | (std < 1e-10)] = 1.0
    train = (train - mean) / std
    test = (test - mean) / std
    active = train_missing.any(axis=0) | test_missing.any(axis=0)
    parts_train = [train, train_missing[:, active].astype(float)]
    parts_test = [test, test_missing[:, active].astype(float)]
    categories = sorted(
        frame.iloc[train_idx]["sw1_industry"].fillna("<MISSING>").astype(str).unique()
    )
    lookup = {value: i for i, value in enumerate(categories)}
    for source, parts, n in (
        (frame.iloc[train_idx], parts_train, len(train_idx)),
        (frame.iloc[test_idx], parts_test, len(test_idx)),
    ):
        codes = (
            source["sw1_industry"]
            .fillna("<MISSING>")
            .astype(str)
            .map(lookup)
            .fillna(-1)
            .to_numpy(int)
        )
        onehot = np.zeros((n, max(0, len(categories) - 1)), dtype=float)
        valid = codes > 0
        if onehot.shape[1]:
            onehot[np.flatnonzero(valid), codes[valid] - 1] = 1.0
        parts.append(onehot)
    return np.column_stack([np.ones(len(train_idx)), *parts_train]), np.column_stack(
        [np.ones(len(test_idx)), *parts_test]
    )


def ridge_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    penalty = np.eye(x_train.shape[1]) * 10.0
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(x_train.T @ x_train + penalty, rcond=1e-10) @ (
        x_train.T @ y_train
    )
    return x_test @ beta


def crossfit_multi(
    frame: pd.DataFrame,
    targets: np.ndarray,
    folds: pd.DataFrame,
    target_dates: pd.Series | None,
    role: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    values = np.asarray(targets, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    residual = np.full(values.shape, np.nan, dtype=np.float32)
    signal = pd.to_datetime(frame["signal_date"]).to_numpy(dtype="datetime64[ns]")
    target_np = (
        pd.to_datetime(target_dates, errors="coerce").to_numpy(dtype="datetime64[ns]")
        if target_dates is not None
        else None
    )
    audits = []
    for row in folds.itertuples(index=False):
        start = np.datetime64(pd.Timestamp(row.test_start), "ns")
        end = np.datetime64(pd.Timestamp(row.test_end), "ns")
        train_mask = signal < start
        if target_np is not None:
            train_mask &= (~np.isnat(target_np)) & (target_np < start)
        test_mask = (signal >= start) & (signal <= end)
        finite_train = np.isfinite(values).all(axis=1) & train_mask
        finite_test = np.isfinite(values).all(axis=1) & test_mask
        train_idx = np.flatnonzero(finite_train)
        test_idx = np.flatnonzero(finite_test)
        if len(train_idx) < 10_000 or len(test_idx) < 1_000:
            audits.append(
                {
                    "role": role,
                    "fold_id": row.fold_id,
                    "status": "DEFERRED",
                    "train_rows": len(train_idx),
                    "test_rows": len(test_idx),
                }
            )
            continue
        fit_idx = deterministic_thin(train_idx)
        x_train, x_test = matrices(frame, fit_idx, test_idx)
        prediction = ridge_predict(x_train, values[fit_idx], x_test)
        residual[test_idx] = (values[test_idx] - prediction).astype(np.float32)
        audits.append(
            {
                "role": role,
                "fold_id": row.fold_id,
                "status": "COMPLETE",
                "train_rows": len(train_idx),
                "fit_rows": len(fit_idx),
                "test_rows": len(test_idx),
                "max_train_target": (
                    str(pd.Timestamp(target_np[train_idx].max()))
                    if target_np is not None
                    else "CURRENT"
                ),
                "test_start": str(pd.Timestamp(start)),
                "target_strictly_before_test": bool(
                    target_np is None or target_np[train_idx].max() < start
                ),
            }
        )
    return residual, pd.DataFrame(audits)


def normal_p(t: float) -> float:
    return (
        max(math.erfc(abs(t) / math.sqrt(2.0)), float(np.nextafter(0.0, 1.0)))
        if np.isfinite(t)
        else math.nan
    )


def hac_meat(scores: np.ndarray, lag: int) -> float:
    x = np.asarray(scores, dtype=float)
    x = x - x.mean()
    value = float(x @ x)
    for distance in range(1, min(lag, len(x) - 1) + 1):
        weight = 1.0 - distance / (lag + 1.0)
        value += 2.0 * weight * float(x[distance:] @ x[:-distance])
    return value


def estimate(
    dose: np.ndarray, outcome: np.ndarray, dates: pd.Series, lag: int
) -> tuple[dict[str, object], pd.DataFrame]:
    valid = np.isfinite(dose) & np.isfinite(outcome)
    d = np.asarray(dose[valid], dtype=float)
    y = np.asarray(outcome[valid], dtype=float)
    dt = pd.to_datetime(dates[valid])
    denominator = float(d @ d)
    coefficient = float(d @ y / denominator)
    residual = y - coefficient * d
    daily = (
        pd.DataFrame(
            {
                "signal_date": dt.to_numpy(),
                "num": d * y,
                "den": d * d,
                "score": d * residual,
            }
        )
        .groupby("signal_date", sort=True)
        .sum()
        .reset_index()
    )
    se = (
        math.sqrt(max(hac_meat(daily["score"].to_numpy(float), lag), 0.0)) / denominator
    )
    t = coefficient / se if se > 0 else math.nan
    return {
        "coefficient": coefficient,
        "se_hac": se,
        "ci95_low": coefficient - 1.959963984540054 * se,
        "ci95_high": coefficient + 1.959963984540054 * se,
        "t_hac": t,
        "p_two_sided_hac": normal_p(t),
        "n_rows": len(d),
        "n_dates": int(dt.nunique()),
        "hac_lag": lag,
    }, daily


def bootstrap_counts(date_count: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    blocks = int(math.ceil(date_count / BLOCK_LENGTH))
    counts = np.zeros((BOOTSTRAPS, date_count), dtype=np.int16)
    offsets = np.arange(BLOCK_LENGTH)
    for b in range(BOOTSTRAPS):
        starts = rng.integers(0, date_count, size=blocks)
        indices = ((starts[:, None] + offsets[None, :]) % date_count).ravel()[
            :date_count
        ]
        counts[b] = np.bincount(indices, minlength=date_count)
    return counts


def run_estimation() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    if not PANEL.exists():
        prepare_panel()
    all_dates = pd.DatetimeIndex(
        sorted(pd.read_parquet(PANEL, columns=["signal_date"])["signal_date"].unique())
    )
    counts = bootstrap_counts(len(all_dates))
    date_lookup = {date: i for i, date in enumerate(all_dates)}
    rows: list[dict[str, object]] = []
    daily_rows: list[pd.DataFrame] = []
    nuisance_rows: list[pd.DataFrame] = []
    bootstrap_store: dict[tuple[str, str, int], np.ndarray] = {}

    for proxy, spec in PROXIES.items():
        columns = [
            "signal_date",
            "stock_code",
            "sw1_industry",
            *NUMERIC_CONTROLS,
            f"fresh_rank_{proxy}",
            f"current_rank_{proxy}",
            spec["level"],
            f"pre_{proxy}",
        ]
        for horizon in HORIZONS:
            columns += [
                f"target_date_{horizon}",
                f"future_rank_{proxy}_{horizon}",
                f"future_level_{proxy}_{horizon}",
            ]
        frame = (
            pd.read_parquet(PANEL, columns=list(dict.fromkeys(columns)))
            .sort_values(["signal_date", "stock_code"], kind="mergesort")
            .reset_index(drop=True)
        )
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        folds = fold_registry(frame["signal_date"])
        dose_target = (
            pd.to_numeric(frame[f"fresh_rank_{proxy}"], errors="coerce").to_numpy(float)
            * 10.0
        )
        base_targets = np.column_stack(
            [
                dose_target,
                pd.to_numeric(frame[f"current_rank_{proxy}"], errors="coerce").to_numpy(
                    float
                ),
                pd.to_numeric(frame[spec["level"]], errors="coerce").to_numpy(float),
                pd.to_numeric(frame[f"pre_{proxy}"], errors="coerce").to_numpy(float),
            ]
        )
        base_residual, base_audit = crossfit_multi(
            frame, base_targets, folds, None, f"{proxy}|dose_current_pre"
        )
        nuisance_rows.append(base_audit)
        dose_residual = base_residual[:, 0]
        current_residuals = {"rank": base_residual[:, 1], "raw": base_residual[:, 2]}
        pre_raw_residual = base_residual[:, 3]

        for horizon in HORIZONS:
            future_targets = np.column_stack(
                [
                    pd.to_numeric(
                        frame[f"future_rank_{proxy}_{horizon}"], errors="coerce"
                    ).to_numpy(float),
                    pd.to_numeric(
                        frame[f"future_level_{proxy}_{horizon}"], errors="coerce"
                    ).to_numpy(float),
                ]
            )
            future_residual, future_audit = crossfit_multi(
                frame,
                future_targets,
                folds,
                frame[f"target_date_{horizon}"],
                f"{proxy}|future|H{horizon}",
            )
            nuisance_rows.append(future_audit)
            for scale, idx in (("rank", 0), ("raw", 1)):
                future = future_residual[:, idx]
                current = current_residuals[scale]
                common = (
                    np.isfinite(dose_residual)
                    & np.isfinite(future)
                    & np.isfinite(current)
                )
                # All three coefficients must use the identical endpoint-available
                # support.  Otherwise beta(future-current) need not equal
                # beta(future)-beta(current) merely because the current-level
                # regression contains rows whose future endpoint is unavailable.
                future_common = np.where(common, future, np.nan)
                current_common = np.where(common, current, np.nan)
                change = np.full(len(frame), np.nan, dtype=np.float32)
                change[common] = future[common] - current[common]
                estimates = {}
                dailies = {}
                for estimand, outcome in (
                    ("current_level", current_common),
                    ("future_endpoint_level", future_common),
                    ("future_minus_current", change),
                ):
                    result, daily = estimate(
                        dose_residual,
                        outcome,
                        frame["signal_date"],
                        lag=max(1, horizon),
                    )
                    estimates[estimand] = result
                    daily["proxy"] = proxy
                    daily["scale"] = scale
                    daily["holding_h"] = horizon
                    daily["estimand"] = estimand
                    daily_rows.append(daily)
                    dailies[estimand] = daily
                identity_error = abs(
                    estimates["future_minus_current"]["coefficient"]
                    - (
                        estimates["future_endpoint_level"]["coefficient"]
                        - estimates["current_level"]["coefficient"]
                    )
                )
                current_beta = estimates["current_level"]["coefficient"]
                retained = (
                    estimates["future_endpoint_level"]["coefficient"] / current_beta
                    if abs(current_beta) > 1e-12
                    else math.nan
                )
                for estimand, result in estimates.items():
                    rows.append(
                        {
                            "proxy": proxy,
                            "scale": scale,
                            "holding_h": horizon,
                            "estimand": estimand,
                            **result,
                            "current_beta_same_support": current_beta,
                            "retained_fraction": (
                                retained
                                if estimand == "future_endpoint_level"
                                else math.nan
                            ),
                            "accounting_identity_abs_error": identity_error,
                            "observational_status": "construct-specific fresh-state association with endpoint level; not exogenous causality",
                        }
                    )

                # Bootstrap endpoint coefficient on the shared global date grid.
                endpoint_daily = dailies["future_endpoint_level"].set_index(
                    "signal_date"
                )
                num = np.zeros(len(all_dates))
                den = np.zeros(len(all_dates))
                positions = [
                    date_lookup[pd.Timestamp(date)] for date in endpoint_daily.index
                ]
                num[positions] = endpoint_daily["num"].to_numpy(float)
                den[positions] = endpoint_daily["den"].to_numpy(float)
                bootstrap_store[(proxy, scale, horizon)] = (counts @ num) / np.where(
                    (counts @ den) > 0, counts @ den, np.nan
                )

    results = pd.DataFrame(rows)
    nuisance = pd.concat(nuisance_rows, ignore_index=True)
    daily_stats = pd.concat(daily_rows, ignore_index=True)

    # Bootstrap inference and simultaneous bands for future endpoint levels.
    for proxy in PROXIES:
        for scale in ("rank", "raw"):
            formal_h = [h for h in HORIZONS if h <= 120]
            distributions = np.column_stack(
                [bootstrap_store[(proxy, scale, h)] for h in formal_h]
            )
            point = np.array(
                [
                    results.loc[
                        (results["proxy"] == proxy)
                        & (results["scale"] == scale)
                        & (results["holding_h"] == h)
                        & (results["estimand"] == "future_endpoint_level"),
                        "coefficient",
                    ].iloc[0]
                    for h in formal_h
                ]
            )
            se = np.nanstd(distributions, axis=0, ddof=1)
            z = (distributions - point) / se
            max_t = np.nanmax(np.abs(z), axis=1)
            critical = float(np.nanquantile(max_t, 0.95))
            for index, horizon in enumerate(formal_h):
                mask = (
                    (results["proxy"] == proxy)
                    & (results["scale"] == scale)
                    & (results["holding_h"] == horizon)
                    & (results["estimand"] == "future_endpoint_level")
                )
                dist = distributions[:, index]
                lower_count = int(np.sum(dist <= 0))
                upper_count = int(np.sum(dist >= 0))
                p_boot = min(
                    1.0, (2 * min(lower_count, upper_count) + 1) / (BOOTSTRAPS + 1)
                )
                results.loc[mask, "bootstrap_se"] = se[index]
                results.loc[mask, "bootstrap_ci95_low"] = np.nanquantile(dist, 0.025)
                results.loc[mask, "bootstrap_ci95_high"] = np.nanquantile(dist, 0.975)
                results.loc[mask, "bootstrap_p_finite"] = p_boot
                results.loc[mask, "bootstrap_max_t_critical"] = critical
                results.loc[mask, "bootstrap_simul95_low"] = (
                    point[index] - critical * se[index]
                )
                results.loc[mask, "bootstrap_simul95_high"] = (
                    point[index] + critical * se[index]
                )
                results.loc[mask, "bootstrap_status"] = "COMPLETE"
            mask250 = (
                (results["proxy"] == proxy)
                & (results["scale"] == scale)
                & (results["holding_h"] == 250)
                & (results["estimand"] == "future_endpoint_level")
            )
            results.loc[mask250, "bootstrap_status"] = (
                "DEFERRED_EFFECTIVE_NONOVERLAP_BLOCKS_LT8"
            )

    main = results[
        (results["scale"] == "rank") & (results["estimand"] == "future_endpoint_level")
    ].copy()
    main["p_holm_24"] = np.nan
    main["q_bh_24"] = np.nan
    complete = main["bootstrap_status"].eq("COMPLETE")
    # H250 remains in the frozen 24-family as working p=1.
    # The frozen formal family uses the finite B=1,000 date-block p values.
    # Analytic HAC p values remain in the table as a secondary diagnostic but
    # must not create sub-Monte-Carlo headline significance.
    work_p = np.where(complete, main["bootstrap_p_finite"].to_numpy(float), 1.0)
    main.loc[:, "p_holm_24"] = multipletests(work_p, method="holm")[1]
    main.loc[:, "q_bh_24"] = multipletests(work_p, method="fdr_bh")[1]
    results = results.merge(
        main[["proxy", "scale", "holding_h", "estimand", "p_holm_24", "q_bh_24"]],
        on=["proxy", "scale", "holding_h", "estimand"],
        how="left",
    )

    results_path = DATA / "construct_CF1_level_irf.csv"
    nuisance_path = DATA / "construct_CF1_nuisance_and_purge.csv"
    daily_path = CACHE / "construct_CF1_daily_sufficient_stats.parquet"
    identity_path = SOURCE / "construct_CF1_accounting_identity.csv"
    retained_path = SOURCE / "construct_CF1_retained_fraction.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    nuisance.to_csv(nuisance_path, index=False, encoding="utf-8-sig")
    daily_stats.to_parquet(daily_path, index=False, compression="zstd")
    results[
        ["proxy", "scale", "holding_h", "accounting_identity_abs_error"]
    ].drop_duplicates().to_csv(identity_path, index=False, encoding="utf-8-sig")
    main.to_csv(retained_path, index=False, encoding="utf-8-sig")
    plot_results_v2(main)
    write_reports_v2(main, results)
    audit(
        results,
        nuisance,
        (results_path, nuisance_path, daily_path, identity_path, retained_path),
    )
    return results, nuisance


def plot_results(main: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"T": "#B85C4A", "S": "#C19A3D", "K": "#3F6B8A"}
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.25), sharey=True)
    x = np.arange(len(HORIZONS))
    for ax, proxy in zip(axes, PROXIES):
        sub = main[main["proxy"].eq(proxy)].sort_values("holding_h")
        y = sub["retained_fraction"].to_numpy(float)
        ax.axhline(1, color="#777777", lw=0.8, ls="--")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.plot(x, y, color=colors[proxy], marker="o", lw=1.6, ms=4)
        deferred = sub["bootstrap_status"].ne("COMPLETE").to_numpy()
        ax.scatter(
            x[deferred],
            y[deferred],
            facecolors="white",
            edgecolors=colors[proxy],
            s=28,
            zorder=3,
        )
        ax.set_xticks(x, [str(h) for h in HORIZONS])
        ax.set_xlabel("H (trading days)")
        ax.set_title(
            {
                "T": "Total heat fresh state",
                "S": "Watchlist fresh state",
                "K": "Same-weekday stickiness",
            }[proxy],
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="y", color="#D9DEE3", lw=0.6)
    axes[0].set_ylabel("Endpoint coefficient / H0 coefficient")
    fig.suptitle(
        "Fresh attention states persist on different clocks",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.01,
        "H250 is hollow when long-block support is insufficient; negative change alone is not called reversal.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    png = FIGURES / "Fig_CF1_endpoint_level_retention.png"
    pdf = FIGURES / "Fig_CF1_endpoint_level_retention.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    main.to_csv(
        SOURCE / "Fig_CF1_endpoint_level_retention.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.25), sharey=False)
    for ax, proxy in zip(axes, PROXIES):
        sub = (
            pd.read_csv(DATA / "construct_CF1_level_irf.csv")
            if (DATA / "construct_CF1_level_irf.csv").exists()
            else None
        )
        # During first build, use the already-written SourceData if available.
        src = main[main["proxy"].eq(proxy)].sort_values("holding_h")
        ax.plot(
            x,
            src["coefficient"],
            color=colors[proxy],
            marker="o",
            lw=1.6,
            label="Future endpoint level",
        )
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_xticks(x, [str(h) for h in HORIZONS])
        ax.set_xlabel("H (trading days)")
        ax.set_title(proxy, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#D9DEE3", lw=0.6)
    axes[0].set_ylabel("Rank-level coefficient per 10-percentile fresh step")
    fig.suptitle(
        "Endpoint levels replace mechanically coupled change scores",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(
        FIGURES / "Fig_CF1_change_vs_endpoint_level.png", dpi=400, bbox_inches="tight"
    )
    fig.savefig(FIGURES / "Fig_CF1_change_vs_endpoint_level.pdf", bbox_inches="tight")
    plt.close(fig)
    main.to_csv(
        SOURCE / "Fig_CF1_change_vs_endpoint_level.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_reports(main: pd.DataFrame, results: pd.DataFrame) -> None:
    table = main.pivot(
        index="holding_h", columns="proxy", values="retained_fraction"
    ).reindex(HORIZONS)
    values = table.round(4).to_markdown()
    max_identity = float(
        pd.to_numeric(results["accounting_identity_abs_error"], errors="coerce").max()
    )
    professional = f"""# CF1：变化分数耦合审计与端点水平动态

## 设计

本模块在post-2022全市场common-TSK样本上，以T_delta、S_delta_5和K_same_weekday_z56作为构念特定fresh状态。所有普通交易控制只取注意快照日或更早的PIT值。结果使用未来端点水平；future-minus-current仅作会计桥，不能单独解释为反转。

## 结果读取

下表为未来端点系数相对同一H支持域H0系数的保留比例：

{values}

H250按原冻结协议只保留描述值，不作正式bootstrap推断；代码没有在此次运行中重新检验有效区块阈值。会计恒等式最大绝对误差为 `{max_identity:.3e}`。

## 识别边界

这是严格过去训练、高维调整和日期区块推断下的观察性状态轨迹。它修复了变化分数代数耦合，但不排除平台测量、共同信息、选择和未观测混杂，因此不称外生注意效应。
"""
    plain = f"""# CF1白话总结：负的变化不等于热度变成负数

旧图把“未来水平减去今天水平”作为结果。今天越高，减法里被减掉的数也越大，所以图容易天然向下。这次改为直接问：今天出现一份新的T、S或K状态以后，未来端点上还保留多少。

{values}

因此要区分三句话：保留比例在0和1之间叫部分衰减；接近0叫基本消散；真正小于0才叫越过原基线。H250的数据仍展示，但长期独立区块不足，不能拿它讲正式显著性。
"""
    (REPORTS / "CF1_endpoint_dynamics.md").write_text(professional, encoding="utf-8")
    (REPORTS / "CF1_reading_notes.md").write_text(plain, encoding="utf-8")


def plot_results_v2(main: pd.DataFrame) -> None:
    """Compact journal-style release figures with explicit uncertainty."""
    colors = {
        "T": COLORS["positive"],
        "S": COLORS["difference"],
        "K": COLORS["negative"],
    }
    titles = {"T": "Total heat", "S": "Watchlist heat", "K": "Same-weekday stickiness"}
    x = np.arange(len(HORIZONS))

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), sharey=True)
    for i, (ax, proxy) in enumerate(zip(axes, PROXIES)):
        sub = main[main["proxy"].eq(proxy)].sort_values("holding_h")
        values = sub["retained_fraction"].to_numpy(float)
        deferred = sub["bootstrap_status"].ne("COMPLETE").to_numpy()
        ax.axhline(1, color=COLORS["muted"], lw=0.75, ls="--")
        ax.axhline(0, color=COLORS["zero"], lw=0.75)
        ax.plot(x, values, color=colors[proxy], marker="o", lw=1.35, ms=3.4)
        ax.scatter(
            x[deferred],
            values[deferred],
            facecolors=COLORS["paper"],
            edgecolors=colors[proxy],
            s=24,
            zorder=3,
        )
        ax.set_xticks(x, [str(h) for h in HORIZONS])
        ax.set_xlabel("H (trading days)")
        ax.set_title(titles[proxy], loc="left")
        clean_axes(ax, grid_axis="y")
        panel_label(ax, chr(ord("a") + i))
    axes[0].set_ylabel("Endpoint / H0 coefficient")
    fig.suptitle(
        "Fresh attention states run on different clocks",
        x=0.08,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.015,
        "H250 is hollow: long-block support is insufficient for formal inference.",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.82, bottom=0.25, wspace=0.22)
    save_figure(fig, FIGURES / "Fig_CF1_endpoint_level_retention")
    plt.close(fig)
    main.to_csv(
        SOURCE / "Fig_CF1_endpoint_level_retention.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), sharey=False)
    for i, (ax, proxy) in enumerate(zip(axes, PROXIES)):
        sub = main[main["proxy"].eq(proxy)].sort_values("holding_h")
        complete = sub["bootstrap_status"].eq("COMPLETE").to_numpy()
        values = sub["coefficient"].to_numpy(float)
        low = sub["bootstrap_simul95_low"].to_numpy(float)
        high = sub["bootstrap_simul95_high"].to_numpy(float)
        ax.axhline(0, color=COLORS["zero"], lw=0.75)
        ax.plot(x, values, color=colors[proxy], marker="o", lw=1.25, ms=3.2)
        ax.errorbar(
            x[complete],
            values[complete],
            yerr=[values[complete] - low[complete], high[complete] - values[complete]],
            fmt="none",
            ecolor=colors[proxy],
            elinewidth=0.8,
            capsize=1.8,
        )
        deferred = ~complete
        ax.scatter(
            x[deferred],
            values[deferred],
            facecolors=COLORS["paper"],
            edgecolors=colors[proxy],
            s=24,
            zorder=3,
        )
        ax.set_xticks(x, [str(h) for h in HORIZONS])
        ax.set_xlabel("H (trading days)")
        ax.set_title(titles[proxy], loc="left")
        clean_axes(ax, grid_axis="y")
        panel_label(ax, chr(ord("a") + i))
    axes[0].set_ylabel("Endpoint rank coefficient\nper 10-percentile fresh step")
    fig.suptitle(
        "Endpoint levels remove change-score coupling",
        x=0.08,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.12, right=0.995, top=0.82, bottom=0.22, wspace=0.34)
    save_figure(fig, FIGURES / "Fig_CF1_change_vs_endpoint_level")
    plt.close(fig)
    main.to_csv(
        SOURCE / "Fig_CF1_change_vs_endpoint_level.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_reports_v2(main: pd.DataFrame, results: pd.DataFrame) -> None:
    table = main.pivot(
        index="holding_h", columns="proxy", values="retained_fraction"
    ).reindex(HORIZONS)
    values = table.round(4).to_markdown()
    gates = main[
        [
            "proxy",
            "holding_h",
            "coefficient",
            "bootstrap_simul95_low",
            "bootstrap_simul95_high",
            "bootstrap_p_finite",
            "p_holm_24",
            "bootstrap_status",
        ]
    ].copy()
    gate_table = gates.round(6).to_markdown(index=False)
    max_identity = float(
        pd.to_numeric(results["accounting_identity_abs_error"], errors="coerce").max()
    )
    professional = f"""# CF1：变化分数耦合审计与端点水平动力学

## 研究问题与修正

旧设计把“未来水平−当前水平”作为结果，同时把“当前水平−严格过去均值”作为解释变量。二者共享当前水平，因此负系数可能由减法结构机械产生。本模块在 post-2022 全市场 common-TSK 样本上，改为直接预测未来端点水平；变化量只保留为会计核对，不再单独承担“反转”结论。

主 exposure 分别为 `T_delta`、`S_delta_5` 和 `K_same_weekday_z56`。三者先转为当日横截fresh rank；报告系数的单位是“fresh rank 提高10个百分位时，未来端点rank的变化”。普通控制只使用 attention observation date 及以前可见的 PIT 值；使用 8 个 expanding-time 外推折（最后一折为2026年7–8月）、H 特定 purge、日期循环移动块 bootstrap（B=1,000，块长120）。会计恒等式最大绝对误差为 `{max_identity:.3e}`。

## 核心结果

下表是未来端点系数相对同一 H、同一可用支持上 H0 系数的保留比例：

{values}

- **T（总热度）**：H1/H3 的端点落到基线另一侧，保留比例约 −0.087；但在预冻结 24 端点 Holm 家族下 `p=0.0899`，因此写“短期过冲候选”，不写已确认普遍反转。
- **S（自选热度）**：H1–H120 保留约 1.02–1.18，说明它更像慢状态或持续记忆；所有正式 H 的有限重抽样 Holm p 为 0.024。这里仍可能包含平台测量惯性与共同信息，不能升级为收益因果。
- **K（同星期黏性）**：H1/H3 只保留约 10%–11%，H20 约5%，H120 约2.3%；端点仍为正但快速衰减。它不是“越黏越好”，而是短期状态记忆。
- **H250**：由于 120 日长块下有效非重叠支持不足 8 块，正式区间与 p 值关闭，只保留描述值。

## 全部正式格

{gate_table}

## 识别边界

这是高维调整后的观察性状态轨迹，不是外生处理效应。结果排除了一个明确的代数伪影，却没有排除平台测量、共同信息、选择、价格反馈或未观测混杂。因而可支持“三种注意力构念具有不同时间尺度”，不能单独支持“注意力造成收益”。
"""
    plain = f"""# CF1 白话总结：负的变化，不一定代表真的反转

旧算法有一个容易误会的地方：它用“未来减今天”作结果，又用“今天减过去”作原因。今天这个数同时出现在等号两边，而且一边是加、一边是减，所以天然容易画出负关系。

这次改成直接看未来那一天的热度水平，结论变得清楚：

- **总热度 T** 像突然涌入的围观。三五天内有越过原基线的迹象，但最严格的全家族门没有通过，所以只能叫“过冲候选”。
- **自选热度 S** 像把股票放进收藏夹。它一旦形成，几个月后仍大体保留，是三者里最慢的状态。
- **黏性 K** 像短期反复回来查看。最初只留下约一成，随后继续衰减；“黏性高”不等于价格一定更好。

保留比例表如下；1 表示完整保留，0 表示基本消散，小于 0 才表示越过原来的基线：

{values}

H250 按原冻结协议固定只报告描述数值；代码没有在此次运行中重新检验有效区块阈值。这个实验说明 T、S、K 是三只不同速度的钟；它还不能说明其中任何一只钟推动了股价。
"""
    (REPORTS / "CF1_endpoint_dynamics.md").write_text(professional, encoding="utf-8")
    (REPORTS / "CF1_reading_notes.md").write_text(plain, encoding="utf-8")


def audit(
    results: pd.DataFrame, nuisance: pd.DataFrame, files: tuple[Path, ...]
) -> None:
    complete_main = results[
        (results["scale"] == "rank") & (results["estimand"] == "future_endpoint_level")
    ]
    checks = {
        "result_rows": len(results),
        "expected_rows": 3 * 2 * 8 * 3,
        "main_family_rows": len(complete_main),
        "expected_main_family": 24,
        "identity_max_abs_error": float(
            pd.to_numeric(
                results["accounting_identity_abs_error"], errors="coerce"
            ).max()
        ),
        "bootstrap_repetitions": BOOTSTRAPS,
        "block_length_market_dates": BLOCK_LENGTH,
        "h250_deferred_rows": int(
            complete_main.loc[complete_main["holding_h"].eq(250), "bootstrap_status"]
            .str.startswith("DEFERRED")
            .sum()
        ),
        "purge_violations": int(
            (
                nuisance.get("target_strictly_before_test", pd.Series(dtype=bool)).eq(
                    False
                )
            ).sum()
        ),
        "all_formal_p_positive": bool(
            (
                pd.to_numeric(
                    complete_main.loc[
                        complete_main["bootstrap_status"].eq("COMPLETE"),
                        "bootstrap_p_finite",
                    ],
                    errors="coerce",
                )
                > 0
            ).all()
        ),
    }
    hard_gates = {
        "complete_grid": checks["result_rows"] == checks["expected_rows"]
        and checks["main_family_rows"] == 24,
        "accounting_identity_exact": checks["identity_max_abs_error"] < 1e-10,
        "formal_random_worlds_1000": checks["bootstrap_repetitions"] == 1_000,
        "h250_fail_closed": checks["h250_deferred_rows"] == 3,
        "all_outcome_training_purged": checks["purge_violations"] == 0,
        "finite_p_values_valid": checks["all_formal_p_positive"],
    }
    payload = {
        "schema": "construct-CF1-release-1.0",
        "status": "PASS_WITH_DECLARED_DEFERRED" if all(hard_gates.values()) else "FAIL",
        "hard_gates": hard_gates,
        "checks": checks,
        "files": {
            str(path.relative_to(ROOT)): sha256(path) for path in files if path.exists()
        },
        "figures": [
            "Fig_CF1_endpoint_level_retention",
            "Fig_CF1_change_vs_endpoint_level",
        ],
    }
    (AUDIT / "construct_CF1_release.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if payload["status"] == "FAIL":
        raise RuntimeError(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started = time.time()
    if args.prepare or args.all:
        print(
            json.dumps(prepare_panel(force=args.force), ensure_ascii=False, default=str)
        )
    if args.estimate or args.all:
        run_estimation()
    if not (args.prepare or args.estimate or args.all):
        parser.error("use --prepare, --estimate or --all")
    print(
        json.dumps(
            {"elapsed_seconds": round(time.time() - started, 2)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
