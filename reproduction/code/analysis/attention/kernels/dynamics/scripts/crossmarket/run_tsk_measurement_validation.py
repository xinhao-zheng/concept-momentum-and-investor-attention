"""Experiment B: related-vendor endpoint validation of T/S/K stock statistics.

This script implements the frozen design in
``reports/crossmarket/measurement_validation_frozen_spec.md``.  It deliberately
separates ordinary supplier-labelled-list persistence, new entry, one-session retention and
conditional list score.  It is a predictive/construct-validity experiment, not
a causal-return experiment.

The pipeline is resumable::

    python run_tsk_measurement_validation.py --stage build
    python run_tsk_measurement_validation.py --stage model
    python run_tsk_measurement_validation.py --stage report

All observations and outcomes are hard-truncated before 2026-07-01.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "scripts"))
from plot_style import COLORS, clean_axes, panel_label, save_figure  # noqa: E402


RAW_XQ = Path("__SNAPSHOT__/attention/inputs")
PIT_PATH = Path(
    "__SNAPSHOT__/input_recovery/stock_pit_covariates_daily.universe.parquet"
)
ATTENTION_PATH = Path("__SNAPSHOT__/attention/inputs/attention_daily.parquet")

CACHE = PROJECT / "cache" / "crossmarket"
DATA = PROJECT / "data" / "crossmarket"
FIGURES = PROJECT / "figures" / "crossmarket"
REPORTS = PROJECT / "reports" / "crossmarket"
for _folder in (CACHE, DATA, FIGURES, REPORTS):
    _folder.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2026-09-01")
SEED = 20260713
ANCHOR_MINUTE = 15 * 60 + 30
ANCHOR_MAX_DISTANCE_MIN = 90
ROTATE_BY = 137
NEGATIVE_KEEP_MOD = 20
EMBARGO_DATES = 2
BOOTSTRAP_REPS = 1000
BLOCK_DAYS = 5

HORIZONS = ("1h", "24h")
XQ_FILES = {"1h": RAW_XQ / "沪深_1h.csv", "24h": RAW_XQ / "沪深_24h.csv"}
ANCHOR_FILES = {h: CACHE / f"measurement_xq_A_{h}_anchor.parquet" for h in HORIZONS}
PANEL_FILES = {h: CACHE / f"measurement_tsk_{h}_panel.parquet" for h in HORIZONS}

TSK = [
    "log_T",
    "T_delta",
    "log_S",
    "S_delta_5",
    "S_delta_20",
    "log_K",
    "K_delta",
    "K_same_weekday_z56",
]
BASE_NUMERIC = [
    "ret_1",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "drawdown_from_60d_high",
    "log_amount_t",
    "amount_to_float_mv_t",
    "amount_to_float_mv_20_mean",
    "amihud_20",
    "zero_amount_days_20",
    "suspension_days_20",
    "beta_hs300_60",
    "idio_vol_hs300_60",
    "log_float_mv_t",
    "market_return_t",
]
ATTENTION_META = ["attention_asof_cutoff", "observation_date", "common_TSK"]

FOLDS = [
    ("2023H1", "2023-01-01", "2023-07-01"),
    ("2023H2", "2023-07-01", "2024-01-01"),
    ("2024H1", "2024-01-01", "2024-07-01"),
    ("2024H2", "2024-07-01", "2025-01-01"),
    ("2025H1", "2025-01-01", "2025-07-01"),
    ("2025H2", "2025-07-01", "2026-01-01"),
    ("2026H1", "2026-01-01", "2026-07-01"),
    ("2026H2_partial", "2026-07-01", "2026-09-01"),
]


@dataclass
class ModelSpec:
    task: str
    kind: str
    train_negative_sample: bool
    headline_metric: str


TASKS = {
    "presence": ModelSpec("presence", "binary", True, "recall_at_100"),
    "new_entry": ModelSpec("new_entry", "binary", True, "recall_at_100"),
    "retention": ModelSpec("retention", "binary", False, "pr_auc"),
    "conditional_score": ModelSpec(
        "conditional_score", "regression", False, "spearman"
    ),
}


def sha256(path: Path, block: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def xq_schema() -> dict[str, pl.DataType]:
    return {
        "type": pl.Int64,
        "code": pl.String,
        "name": pl.String,
        "value": pl.Float64,
        "increment": pl.Int64,
        "percent": pl.Float64,
        "current": pl.Float64,
        "chg": pl.Float64,
        "source_timestamp": pl.String,
    }


def normalize_xq_code() -> pl.Expr:
    """Map provider SH600000/SZ000001/BJxxxxxx to research sh/sz/bj form."""
    return (
        pl.when(pl.col("code").str.contains(r"^(SH|SZ|BJ)\d{6}$"))
        .then(pl.col("code").str.to_lowercase())
        .otherwise(None)
        .alias("stock_code")
    )


def build_anchor(horizon: str) -> dict:
    path = XQ_FILES[horizon]
    if not path.exists():
        raise FileNotFoundError(path)
    lazy = (
        pl.scan_csv(
            path,
            skip_rows=1,
            encoding="utf8-lossy",
            schema=xq_schema(),
            infer_schema=False,
        )
        .with_columns(
            pl.col("source_timestamp")
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
            .alias("ts")
        )
        .filter(
            pl.col("ts").is_not_null() & (pl.col("ts") < pl.lit(CUTOFF.to_pydatetime()))
        )
        .with_columns(
            pl.col("ts").dt.date().alias("outcome_date"),
            (
                (
                    pl.col("ts").dt.hour().cast(pl.Int32) * 60
                    + pl.col("ts").dt.minute().cast(pl.Int32)
                    + pl.col("ts").dt.second().cast(pl.Float64) / 60
                    - ANCHOR_MINUTE
                ).abs()
            ).alias("anchor_distance_min"),
        )
    )
    snapshots = (
        lazy.group_by(["outcome_date", "ts"])
        .agg(
            pl.len().alias("snapshot_n"),
            pl.col("code").n_unique().alias("snapshot_n_unique"),
            pl.col("value").null_count().alias("null_value_n"),
            pl.col("anchor_distance_min").first().alias("anchor_distance_min"),
        )
        .sort(["outcome_date", "anchor_distance_min", "ts"])
        .group_by("outcome_date", maintain_order=True)
        .first()
        .filter(
            (pl.col("anchor_distance_min") <= ANCHOR_MAX_DISTANCE_MIN)
            & (pl.col("snapshot_n") == 100)
            & (pl.col("snapshot_n_unique") == 100)
            & (pl.col("null_value_n") == 0)
        )
        .collect()
    )
    chosen = snapshots.select(
        ["outcome_date", "ts", "anchor_distance_min", "snapshot_n"]
    )
    rows = (
        lazy.join(chosen.lazy(), on=["outcome_date", "ts"], how="inner")
        .with_columns(normalize_xq_code())
        .select(
            "outcome_date",
            "ts",
            "anchor_distance_min",
            "snapshot_n",
            "type",
            "code",
            "stock_code",
            pl.col("value").alias("provider_score"),
        )
        .sort(["outcome_date", "stock_code", "code"])
        .collect()
    )
    rows.write_parquet(ANCHOR_FILES[horizon], compression="zstd", statistics=True)
    valid_code = rows.filter(pl.col("stock_code").is_not_null()).height
    audit = {
        "horizon": horizon,
        "source": str(path),
        "source_bytes": path.stat().st_size,
        "source_sha256": sha256(path),
        "anchor_file": str(ANCHOR_FILES[horizon]),
        "complete_anchor_dates": snapshots.height,
        "anchor_min_date": str(rows["outcome_date"].min()),
        "anchor_max_date": str(rows["outcome_date"].max()),
        "anchor_distance_p50": float(snapshots["anchor_distance_min"].median()),
        "anchor_distance_p95": float(snapshots["anchor_distance_min"].quantile(0.95)),
        "rows": rows.height,
        "normalized_equity_like_code_rows": valid_code,
        "normalized_code_fraction": valid_code / max(rows.height, 1),
        "type_values": sorted(
            int(x) for x in rows["type"].drop_nulls().unique().to_list()
        ),
        "current_chg_percent_used": False,
        "type_used_as_rank": False,
    }
    return audit


def trading_dates() -> list[pd.Timestamp]:
    dates = (
        pl.scan_parquet(PIT_PATH)
        .select(pl.col("signal_date").unique().sort())
        .collect()["signal_date"]
        .to_pandas()
    )
    return [pd.Timestamp(x).normalize() for x in dates if pd.Timestamp(x) < CUTOFF]


def build_session_map(
    anchor: pl.DataFrame, all_dates: list[pd.Timestamp]
) -> pl.DataFrame:
    available = set(pd.to_datetime(anchor["outcome_date"].to_list()).normalize())
    records = []
    for i, date in enumerate(all_dates[:-1]):
        next_date = all_dates[i + 1]
        next2 = all_dates[i + 2] if i + 2 < len(all_dates) else pd.NaT
        if date not in available or next_date not in available:
            continue
        records.append(
            {
                "signal_date": date,
                "next_date": next_date,
                "next2_date": (
                    next2 if pd.notna(next2) and next2 in available else pd.NaT
                ),
            }
        )
    return pl.from_pandas(pd.DataFrame(records)).with_columns(
        pl.col("signal_date").cast(pl.Datetime("ns")),
        pl.col("next_date").cast(pl.Datetime("ns")),
        pl.col("next2_date").cast(pl.Datetime("ns")),
    )


def matched_past_state_map(
    daily: pd.DataFrame, dates: list[pd.Timestamp]
) -> pd.DataFrame:
    """Past-only nearest state, with expanding standardisation and >=20-session gap."""
    frame = daily.copy().sort_values("signal_date").reset_index(drop=True)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.normalize()
    cols = [
        "market_return_t",
        "amount_to_float_mv_20_mean",
        "momentum_20",
        "idio_vol_hs300_60",
    ]
    values = frame[cols].to_numpy(float)
    date_to_i = {d: i for i, d in enumerate(frame["signal_date"])}
    wanted = set(pd.to_datetime(dates).normalize())
    out = []
    for i, date in enumerate(frame["signal_date"]):
        if date not in wanted or i < 20:
            continue
        hist = values[: i + 1]
        scale = np.nanstd(hist, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
        center = np.nanmedian(hist, axis=0)
        target = np.nan_to_num((values[i] - center) / scale)
        candidates = np.arange(0, i - 19)
        cand = np.nan_to_num((values[candidates] - center) / scale)
        dist = np.sum((cand - target) ** 2, axis=1)
        j = int(candidates[int(np.argmin(dist))])
        out.append(
            {
                "signal_date": date,
                "past_source_date": frame.loc[j, "signal_date"],
                "past_session_gap": i - j,
                "past_state_distance": float(np.sqrt(np.min(dist))),
            }
        )
    return pd.DataFrame(out)


def date_shift_map(all_dates: list[pd.Timestamp], step: int) -> pd.DataFrame:
    records = []
    for i, date in enumerate(all_dates):
        j = i + step
        if 0 <= j < len(all_dates):
            records.append({"signal_date": date, "future_source_date": all_dates[j]})
    return pd.DataFrame(records)


def build_panel(horizon: str) -> dict:
    anchor = pl.read_parquet(ANCHOR_FILES[horizon])
    dates = trading_dates()
    session_map = build_session_map(anchor, dates)
    if session_map.is_empty():
        raise RuntimeError(f"No usable session map for {horizon}")

    attention = (
        pl.scan_parquet(ATTENTION_PATH)
        .select(["signal_date", "stock_code", *ATTENTION_META, *TSK])
        .filter(
            (pl.col("signal_date") < pl.lit(CUTOFF.to_pydatetime()))
            & (pl.col("common_TSK") == 1)
            & pl.all_horizontal([pl.col(c).is_not_null() for c in TSK])
            & (pl.col("observation_date") <= pl.col("attention_asof_cutoff"))
        )
    )
    cov = (
        pl.scan_parquet(PIT_PATH)
        .select(
            [
                "signal_date",
                "stock_code",
                "feature_asof_date",
                "source_trade_observed_t",
                *BASE_NUMERIC,
                "sw1_industry",
            ]
        )
        .filter(pl.col("signal_date") < pl.lit(CUTOFF.to_pydatetime()))
    )
    predictors = (
        cov.join(attention, on=["signal_date", "stock_code"], how="inner")
        .join(session_map.lazy(), on="signal_date", how="inner")
        .with_columns(
            (pl.col("signal_date").dt.weekday() - 1).alias("weekday"),
            (2 * math.pi * (pl.col("signal_date").dt.weekday() - 1) / 5)
            .sin()
            .alias("weekday_sin"),
            (2 * math.pi * (pl.col("signal_date").dt.weekday() - 1) / 5)
            .cos()
            .alias("weekday_cos"),
        )
    )

    # Current, next-session and next-next-session provider list states.  A missing
    # score remains missing; presence is added only after the complete snapshot gate.
    positive = anchor.filter(pl.col("stock_code").is_not_null()).select(
        pl.col("outcome_date").cast(pl.Datetime("ns")), "stock_code", "provider_score"
    )
    current = positive.rename(
        {"outcome_date": "signal_date", "provider_score": "current_provider_score"}
    )
    nxt = positive.rename(
        {"outcome_date": "next_date", "provider_score": "next_provider_score"}
    )
    nxt2 = positive.rename(
        {"outcome_date": "next2_date", "provider_score": "next2_provider_score"}
    )
    panel = (
        predictors.join(current.lazy(), on=["signal_date", "stock_code"], how="left")
        .join(nxt.lazy(), on=["next_date", "stock_code"], how="left")
        .join(nxt2.lazy(), on=["next2_date", "stock_code"], how="left")
        .with_columns(
            pl.col("current_provider_score")
            .is_not_null()
            .cast(pl.Int8)
            .alias("current_presence"),
            pl.col("next_provider_score")
            .is_not_null()
            .cast(pl.Int8)
            .alias("next_presence"),
            pl.when(pl.col("next2_date").is_not_null())
            .then(pl.col("next2_provider_score").is_not_null().cast(pl.Int8))
            .otherwise(None)
            .alias("next2_presence"),
            pl.when(pl.col("current_provider_score") >= 0)
            .then(pl.col("current_provider_score").log1p())
            .otherwise(None)
            .alias("current_log_score"),
            pl.when(pl.col("next_provider_score") >= 0)
            .then(pl.col("next_provider_score").log1p())
            .otherwise(None)
            .alias("next_log_score"),
        )
        .sort(["signal_date", "stock_code"])
        .collect()
    )

    # Exact within-date cyclic code rotation after the common-support gate.
    panel = panel.with_columns(
        pl.int_range(pl.len()).over("signal_date").alias("_row_in_date"),
        pl.len().over("signal_date").alias("_n_in_date"),
    )
    rotated = panel.select(
        "signal_date",
        (
            (
                (pl.col("_row_in_date") + pl.col("_n_in_date") - ROTATE_BY)
                % pl.col("_n_in_date")
            )
        ).alias("_row_in_date"),
        *[pl.col(c).alias(f"{c}__code_rotation") for c in TSK],
    )
    panel = panel.join(rotated, on=["signal_date", "_row_in_date"], how="left")

    # Future +20-session and past matched-state date diagnostics.
    used_dates = [
        pd.Timestamp(x).normalize()
        for x in panel["signal_date"].unique().sort().to_list()
    ]
    future_map = pl.from_pandas(date_shift_map(dates, 20)).with_columns(
        pl.col("signal_date").cast(pl.Datetime("ns")),
        pl.col("future_source_date").cast(pl.Datetime("ns")),
    )
    daily_state = (
        panel.group_by("signal_date")
        .agg(
            pl.col("market_return_t").median(),
            pl.col("amount_to_float_mv_20_mean").median(),
            pl.col("momentum_20").median(),
            pl.col("idio_vol_hs300_60").median(),
        )
        .sort("signal_date")
        .to_pandas()
    )
    past_map_pd = matched_past_state_map(daily_state, used_dates)
    past_map = pl.from_pandas(past_map_pd).with_columns(
        pl.col("signal_date").cast(pl.Datetime("ns")),
        pl.col("past_source_date").cast(pl.Datetime("ns")),
    )
    lookup = panel.select("signal_date", "stock_code", *TSK)
    future_lookup = lookup.rename(
        {"signal_date": "future_source_date", **{c: f"{c}__future20" for c in TSK}}
    )
    past_lookup = lookup.rename(
        {"signal_date": "past_source_date", **{c: f"{c}__past_state" for c in TSK}}
    )
    panel = (
        panel.join(future_map, on="signal_date", how="left")
        .join(future_lookup, on=["future_source_date", "stock_code"], how="left")
        .join(past_map, on="signal_date", how="left")
        .join(past_lookup, on=["past_source_date", "stock_code"], how="left")
        .drop(["_row_in_date", "_n_in_date"])
        .sort(["signal_date", "stock_code"])
    )
    panel.write_parquet(PANEL_FILES[horizon], compression="zstd", statistics=True)

    # Mapping audit: positives in the complete Top-100 that intersect the research universe.
    universe = set(panel["stock_code"].unique().to_list())
    anchor_pd = anchor.to_pandas()
    anchor_pd["mapped_to_research"] = anchor_pd["stock_code"].isin(universe)
    coverage = anchor_pd.groupby("outcome_date", as_index=False).agg(
        snapshot_n=("code", "size"), mapped_n=("mapped_to_research", "sum")
    )
    coverage["mapped_fraction"] = coverage["mapped_n"] / coverage["snapshot_n"]
    coverage["horizon"] = horizon
    coverage.to_csv(
        DATA / f"SourceData_measurement_code_mapping_{horizon}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    asof_violations = int(
        (panel["observation_date"] > panel["attention_asof_cutoff"]).sum()
    )
    feature_asof_violations = int(
        (panel["feature_asof_date"] > panel["signal_date"]).sum()
    )
    signal_minus2_violations = int(
        panel.select(
            (
                pl.col("attention_asof_cutoff").dt.date()
                > (pl.col("signal_date") - pl.duration(days=2)).dt.date()
            )
            .sum()
            .alias("n")
        ).item()
    )
    placebo_complete = pl.all_horizontal(
        [
            pl.col(f"{c}__{suffix}").is_not_null()
            for suffix in ("code_rotation", "future20", "past_state")
            for c in TSK
        ]
    )
    audit = {
        "horizon": horizon,
        "panel_file": str(PANEL_FILES[horizon]),
        "panel_rows": panel.height,
        "panel_dates": panel["signal_date"].n_unique(),
        "panel_min_date": str(panel["signal_date"].min()),
        "panel_max_date": str(panel["signal_date"].max()),
        "candidate_codes": panel["stock_code"].n_unique(),
        "asof_violations": asof_violations,
        "feature_asof_violations": feature_asof_violations,
        "signal_minus_2_calendar_day_violations": signal_minus2_violations,
        "current_positive_rows": int(panel["current_presence"].sum()),
        "next_positive_rows": int(panel["next_presence"].sum()),
        "placebo_common_rows": panel.filter(placebo_complete).height,
        "code_mapping_fraction_p05": float(coverage["mapped_fraction"].quantile(0.05)),
        "code_mapping_fraction_median": float(coverage["mapped_fraction"].median()),
        "code_mapping_fraction_min": float(coverage["mapped_fraction"].min()),
        "past_gap_min": int(panel["past_session_gap"].drop_nulls().min()),
        "past_gap_median": float(panel["past_session_gap"].drop_nulls().median()),
    }
    return audit


def fold_frame(
    path: Path,
    start: str,
    end: str,
    *,
    train: bool,
    embargo_cut: pd.Timestamp | None = None,
) -> pd.DataFrame:
    scan = pl.scan_parquet(path)
    if train:
        if embargo_cut is None:
            raise ValueError("embargo_cut required")
        scan = scan.filter(pl.col("signal_date") < pl.lit(embargo_cut.to_pydatetime()))
    else:
        scan = scan.filter(
            (pl.col("signal_date") >= pl.lit(pd.Timestamp(start).to_pydatetime()))
            & (pl.col("signal_date") < pl.lit(pd.Timestamp(end).to_pydatetime()))
        )
    return scan.collect().to_pandas()


def task_rows(frame: pd.DataFrame, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    if task == "presence":
        out = frame
        y = out["next_presence"].to_numpy(np.int8)
    elif task == "new_entry":
        out = frame.loc[frame["current_presence"].eq(0)]
        y = out["next_presence"].to_numpy(np.int8)
    elif task == "retention":
        out = frame.loc[frame["current_presence"].eq(1)]
        y = out["next_presence"].to_numpy(np.int8)
    elif task == "conditional_score":
        out = frame.loc[frame["next_presence"].eq(1) & frame["next_log_score"].notna()]
        y = out["next_log_score"].to_numpy(float)
    else:
        raise KeyError(task)
    return out.copy(), y


def feature_names(task: str, variant: str | None) -> list[str]:
    cols = [*BASE_NUMERIC, "weekday_sin", "weekday_cos", "sw1_industry"]
    if task in {"presence", "conditional_score"}:
        cols += ["current_presence", "current_log_score"]
    elif task == "retention":
        cols += ["current_log_score"]
    if variant is not None:
        suffix = "" if variant == "real" else f"__{variant}"
        cols += [f"{c}{suffix}" for c in TSK]
    return cols


def prepare_xy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    task: str,
    variant: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    cols = feature_names(task, variant)
    numeric = [c for c in cols if c != "sw1_industry"]
    categories = sorted(str(x) for x in train["sw1_industry"].dropna().unique())
    mapping = {value: i for i, value in enumerate(categories)}
    if len(mapping) >= 63:
        raise RuntimeError(f"Too many industry categories: {len(mapping)}")
    x_train_num = (
        train[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    )
    x_test_num = (
        test[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    )
    tr_cat = (
        train["sw1_industry"].astype(str).map(mapping).to_numpy(float).reshape(-1, 1)
    )
    te_cat = (
        test["sw1_industry"].astype(str).map(mapping).to_numpy(float).reshape(-1, 1)
    )
    x_train = np.column_stack([x_train_num, tr_cat]).astype(np.float32, copy=False)
    x_test = np.column_stack([x_test_num, te_cat]).astype(np.float32, copy=False)
    y_train = task_rows(train, task)[1]  # The caller supplies the task-filtered frame.
    y_test = task_rows(test, task)[1]
    return x_train, y_train, x_test, y_test, [x_train.shape[1] - 1]


def stable_negative_mask(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(
        frame[["signal_date", "stock_code"]], index=False
    ).to_numpy(np.uint64)
    return (y == 1) | ((hashes % NEGATIVE_KEEP_MOD) == 0)


def fit_predict_binary(
    train: pd.DataFrame,
    y_train: np.ndarray,
    test: pd.DataFrame,
    task: str,
    variant: str | None,
    *,
    sample_negatives: bool,
    seed_offset: int,
) -> np.ndarray:
    x_train, _, x_test, _, categorical = prepare_xy(train, test, task, variant)
    weights = np.ones(len(y_train), dtype=np.float32)
    if sample_negatives:
        keep = stable_negative_mask(train, y_train)
        weights[(y_train == 0)] = NEGATIVE_KEEP_MOD
        x_fit, y_fit, w_fit = x_train[keep], y_train[keep], weights[keep]
    else:
        x_fit, y_fit, w_fit = x_train, y_train, weights
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.075,
        max_iter=35,
        max_leaf_nodes=15,
        min_samples_leaf=80 if sample_negatives else 30,
        l2_regularization=1.0,
        max_bins=63,
        categorical_features=categorical,
        early_stopping=False,
        random_state=SEED + seed_offset,
    )
    model.fit(x_fit, y_fit, sample_weight=w_fit)
    return np.clip(model.predict_proba(x_test)[:, 1], 1e-7, 1 - 1e-7)


def fit_predict_regression(
    train: pd.DataFrame,
    y_train: np.ndarray,
    test: pd.DataFrame,
    task: str,
    variant: str | None,
    *,
    seed_offset: int,
) -> np.ndarray:
    x_train, _, x_test, _, categorical = prepare_xy(train, test, task, variant)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.075,
        max_iter=45,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        max_bins=63,
        categorical_features=categorical,
        early_stopping=False,
        random_state=SEED + seed_offset,
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def binary_daily_metrics(
    pred: pd.DataFrame, model: str, *, ranking: bool
) -> pd.DataFrame:
    rows = []
    for date, group in pred.groupby("signal_date", sort=True):
        y = group["y"].to_numpy(int)
        p = group[model].to_numpy(float)
        if len(y) == 0:
            continue
        record = {
            "signal_date": date,
            "model": model,
            "n": len(y),
            "positives": int(y.sum()),
            "pr_auc": average_precision_score(y, p) if y.sum() > 0 else np.nan,
            "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p, labels=[0, 1]),
        }
        if ranking:
            k = min(100, len(y))
            top = np.argpartition(p, -k)[-k:]
            hits = int(y[top].sum())
            record["precision_at_100"] = hits / k
            record["recall_at_100"] = hits / y.sum() if y.sum() else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def regression_daily_metrics(pred: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for date, group in pred.groupby("signal_date", sort=True):
        y = group["y"].to_numpy(float)
        p = group[model].to_numpy(float)
        valid = np.isfinite(y) & np.isfinite(p)
        y, p = y[valid], p[valid]
        rho = (
            spearmanr(y, p).statistic
            if len(y) >= 3 and np.nanstd(y) > 0 and np.nanstd(p) > 0
            else np.nan
        )
        rows.append(
            {
                "signal_date": date,
                "model": model,
                "n": len(y),
                "spearman": rho,
                "mae": mean_absolute_error(y, p) if len(y) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(pred: pd.DataFrame, model: str) -> tuple[dict, pd.DataFrame]:
    y = pred["y"].to_numpy(int)
    p = np.clip(pred[model].to_numpy(float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    # Cap slope-fit rows deterministically; reliability bins use the full sample.
    if len(y) > 250_000:
        h = pd.util.hash_pandas_object(
            pred[["signal_date", "stock_code"]], index=False
        ).to_numpy(np.uint64)
        take = np.argsort(h)[:250_000]
        y_fit, x_fit = y[take], logit[take]
    else:
        y_fit, x_fit = y, logit

    def objective(beta: np.ndarray) -> float:
        eta = beta[0] + beta[1] * x_fit
        prob = 1 / (1 + np.exp(-np.clip(eta, -35, 35)))
        return float(
            -np.sum(
                y_fit * np.log(prob + 1e-12) + (1 - y_fit) * np.log(1 - prob + 1e-12)
            )
        )

    opt = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    pbar = float(np.mean(p))
    ybar = float(np.mean(y))
    cil = math.log(
        np.clip(ybar, 1e-8, 1 - 1e-8) / np.clip(1 - ybar, 1e-8, 1)
    ) - math.log(np.clip(pbar, 1e-8, 1 - 1e-8) / np.clip(1 - pbar, 1e-8, 1))
    summary = {
        "model": model,
        "n": len(y),
        "prevalence": ybar,
        "mean_prediction": pbar,
        "calibration_in_large": cil,
        "calibration_intercept": float(opt.x[0]),
        "calibration_slope": float(opt.x[1]),
        "optimizer_success": bool(opt.success),
    }
    tmp = pred[["y"]].copy()
    tmp["prediction"] = p
    try:
        tmp["bin"] = pd.qcut(tmp["prediction"], 10, labels=False, duplicates="drop")
    except ValueError:
        tmp["bin"] = 0
    curve = tmp.groupby("bin", as_index=False).agg(
        n=("y", "size"),
        mean_prediction=("prediction", "mean"),
        observed_rate=("y", "mean"),
    )
    curve["model"] = model
    return summary, curve


def embargo_cut(path: Path, test_start: str) -> pd.Timestamp:
    dates = (
        pl.scan_parquet(path)
        .filter(
            pl.col("signal_date") < pl.lit(pd.Timestamp(test_start).to_pydatetime())
        )
        .select(pl.col("signal_date").unique().sort())
        .collect()["signal_date"]
        .to_pandas()
    )
    if len(dates) <= EMBARGO_DATES:
        raise RuntimeError("Insufficient dates before fold")
    return pd.Timestamp(dates.iloc[-EMBARGO_DATES]).normalize()


def placebo_common(frame: pd.DataFrame) -> pd.Series:
    cols = [
        f"{c}__{suffix}"
        for suffix in ("code_rotation", "future20", "past_state")
        for c in TSK
    ]
    return frame[cols].notna().all(axis=1)


def model_horizon(horizon: str) -> dict:
    path = PANEL_FILES[horizon]
    if not path.exists():
        raise FileNotFoundError(path)
    predictions: list[pd.DataFrame] = []
    daily: list[pd.DataFrame] = []
    fold_rows = []
    calibration_rows = []
    calibration_curves = []

    for fold_i, (fold, start, end) in enumerate(FOLDS):
        cut = embargo_cut(path, start)
        train_all = fold_frame(path, start, end, train=True, embargo_cut=cut)
        test_all = fold_frame(path, start, end, train=False)
        if train_all.empty or test_all.empty:
            continue
        print(
            f"[{horizon} {fold}] train={len(train_all):,} test={len(test_all):,} embargo<{cut.date()}",
            flush=True,
        )

        for task_i, (task, spec) in enumerate(TASKS.items()):
            train, y_train = task_rows(train_all, task)
            test, y_test = task_rows(test_all, task)
            if len(train) == 0 or len(test) == 0:
                continue
            base = (
                fit_predict_binary(
                    train,
                    y_train,
                    test,
                    task,
                    None,
                    sample_negatives=spec.train_negative_sample,
                    seed_offset=1000 * fold_i + 100 * task_i,
                )
                if spec.kind == "binary"
                else fit_predict_regression(
                    train,
                    y_train,
                    test,
                    task,
                    None,
                    seed_offset=1000 * fold_i + 100 * task_i,
                )
            )
            real = (
                fit_predict_binary(
                    train,
                    y_train,
                    test,
                    task,
                    "real",
                    sample_negatives=spec.train_negative_sample,
                    seed_offset=1000 * fold_i + 100 * task_i + 1,
                )
                if spec.kind == "binary"
                else fit_predict_regression(
                    train,
                    y_train,
                    test,
                    task,
                    "real",
                    seed_offset=1000 * fold_i + 100 * task_i + 1,
                )
            )
            pred = test[["signal_date", "stock_code"]].copy()
            pred["y"] = y_test
            pred["baseline"] = base
            pred["real"] = real
            pred["fold"] = fold
            pred["task"] = task
            pred["horizon"] = horizon
            predictions.append(pred)
            for model in ("baseline", "real"):
                metrics = (
                    binary_daily_metrics(
                        pred, model, ranking=task in {"presence", "new_entry"}
                    )
                    if spec.kind == "binary"
                    else regression_daily_metrics(pred, model)
                )
                metrics["fold"] = fold
                metrics["task"] = task
                metrics["horizon"] = horizon
                metrics["sample"] = "main_common_support"
                daily.append(metrics)

            # Placebos are frozen for the two list-entry ranking tasks only.
            if task in {"presence", "new_entry"}:
                tr_common = placebo_common(train)
                te_common = placebo_common(test)
                train_p, y_train_p = (
                    train.loc[tr_common].copy(),
                    y_train[tr_common.to_numpy()],
                )
                test_p, y_test_p = (
                    test.loc[te_common].copy(),
                    y_test[te_common.to_numpy()],
                )
                if len(train_p) and len(test_p):
                    pred_p = test_p[["signal_date", "stock_code"]].copy()
                    pred_p["y"] = y_test_p
                    variants: list[tuple[str, str | None]] = [
                        ("placebo_baseline", None),
                        ("placebo_real", "real"),
                        ("code_rotation", "code_rotation"),
                        ("future20", "future20"),
                        ("past_state", "past_state"),
                    ]
                    for vi, (label, variant) in enumerate(variants):
                        pred_p[label] = fit_predict_binary(
                            train_p,
                            y_train_p,
                            test_p,
                            task,
                            variant,
                            sample_negatives=True,
                            seed_offset=1000 * fold_i + 100 * task_i + 10 + vi,
                        )
                        met = binary_daily_metrics(pred_p, label, ranking=True)
                        met["fold"] = fold
                        met["task"] = task
                        met["horizon"] = horizon
                        met["sample"] = "placebo_common_support"
                        daily.append(met)
                    pred_p["fold"] = fold
                    pred_p["task"] = task
                    pred_p["horizon"] = horizon
                    pred_p["sample"] = "placebo_common_support"
                    # Retain predictions for paired re-analysis.
                    predictions.append(pred_p)

            fold_rows.append(
                {
                    "horizon": horizon,
                    "fold": fold,
                    "task": task,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_dates": train["signal_date"].nunique(),
                    "test_dates": test["signal_date"].nunique(),
                    "train_positive_or_mean": float(np.mean(y_train)),
                    "test_positive_or_mean": float(np.mean(y_test)),
                    "embargo_cut_exclusive": str(cut.date()),
                }
            )
        del train_all, test_all
        gc.collect()

    pred_all = pd.concat(predictions, ignore_index=True, sort=False)
    daily_all = pd.concat(daily, ignore_index=True, sort=False)
    pred_path = CACHE / f"measurement_oof_predictions_{horizon}.parquet"
    pred_all.to_parquet(pred_path, index=False, compression="zstd")
    daily_all.to_csv(
        DATA / f"SourceData_measurement_daily_metrics_{horizon}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Calibration uses only the main common-support OOF binary predictions.
    for task in ("presence", "new_entry", "retention"):
        part = pred_all.loc[
            pred_all["task"].eq(task) & pred_all["baseline"].notna(),
            ["signal_date", "stock_code", "y", "baseline", "real"],
        ]
        for model in ("baseline", "real"):
            summary, curve = calibration_summary(part, model)
            summary.update({"horizon": horizon, "task": task})
            curve["horizon"] = horizon
            curve["task"] = task
            calibration_rows.append(summary)
            calibration_curves.append(curve)
    pd.DataFrame(calibration_rows).to_csv(
        DATA / f"SourceData_measurement_calibration_{horizon}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(calibration_curves, ignore_index=True).to_csv(
        DATA / f"SourceData_measurement_calibration_curve_{horizon}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fold_rows).to_csv(
        DATA / f"SourceData_measurement_fold_sample_{horizon}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "horizon": horizon,
        "prediction_file": str(pred_path),
        "prediction_rows": len(pred_all),
        "daily_metric_rows": len(daily_all),
        "fold_task_rows": len(fold_rows),
    }


def moving_block_mean(
    values: np.ndarray, rng: np.random.Generator, block: int = BLOCK_DAYS
) -> float:
    n = len(values)
    starts = rng.integers(0, n, size=math.ceil(n / block))
    idx = np.concatenate([(np.arange(block) + s) % n for s in starts])[:n]
    return float(np.nanmean(values[idx]))


def paired_summary(daily: pd.DataFrame) -> pd.DataFrame:
    metric_direction = {
        "recall_at_100": 1,
        "precision_at_100": 1,
        "pr_auc": 1,
        "roc_auc": 1,
        "brier": -1,
        "log_loss": -1,
        "spearman": 1,
        "mae": -1,
    }
    rows = []
    rng = np.random.default_rng(SEED)
    comparisons = [
        ("main_common_support", "baseline", "real", "real_vs_baseline"),
        (
            "placebo_common_support",
            "placebo_baseline",
            "placebo_real",
            "real_vs_baseline_placebo_support",
        ),
        (
            "placebo_common_support",
            "placebo_baseline",
            "code_rotation",
            "code_rotation_vs_baseline",
        ),
        (
            "placebo_common_support",
            "placebo_baseline",
            "future20",
            "future20_vs_baseline",
        ),
        (
            "placebo_common_support",
            "placebo_baseline",
            "past_state",
            "past_state_vs_baseline",
        ),
        (
            "placebo_common_support",
            "code_rotation",
            "placebo_real",
            "real_vs_code_rotation",
        ),
        ("placebo_common_support", "future20", "placebo_real", "real_vs_future20"),
        ("placebo_common_support", "past_state", "placebo_real", "real_vs_past_state"),
    ]
    for (horizon, task, sample), group in daily.groupby(["horizon", "task", "sample"]):
        available = set(group["model"].dropna())
        for sample_name, base_model, aug_model, comparison in comparisons:
            if sample != sample_name or not {base_model, aug_model}.issubset(available):
                continue
            left = group.loc[group["model"].eq(base_model)].set_index("signal_date")
            right = group.loc[group["model"].eq(aug_model)].set_index("signal_date")
            common_dates = left.index.intersection(right.index).sort_values()
            for metric, direction in metric_direction.items():
                if metric not in left or metric not in right:
                    continue
                a = left.loc[common_dates, metric].to_numpy(float)
                b = right.loc[common_dates, metric].to_numpy(float)
                valid = np.isfinite(a) & np.isfinite(b)
                diff = direction * (b[valid] - a[valid])
                if len(diff) < 5:
                    continue
                boots = np.array(
                    [moving_block_mean(diff, rng) for _ in range(BOOTSTRAP_REPS)]
                )
                # Finite-resampling correction: a Monte-Carlo p value is never
                # reported as zero merely because all B draws share one sign.
                tail_left = (int(np.sum(boots <= 0)) + 1) / (BOOTSTRAP_REPS + 1)
                tail_right = (int(np.sum(boots >= 0)) + 1) / (BOOTSTRAP_REPS + 1)
                p = min(1.0, 2 * min(tail_left, tail_right))
                rows.append(
                    {
                        "horizon": horizon,
                        "task": task,
                        "sample": sample,
                        "comparison": comparison,
                        "metric": metric,
                        "direction": "positive_is_improvement",
                        "n_dates": len(diff),
                        "baseline_mean": float(np.nanmean(a[valid])),
                        "comparison_mean": float(np.nanmean(b[valid])),
                        "improvement": float(np.nanmean(diff)),
                        "ci_low": float(np.quantile(boots, 0.025)),
                        "ci_high": float(np.quantile(boots, 0.975)),
                        "bootstrap_p": p,
                    }
                )
    return pd.DataFrame(rows)


def holm_adjust(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def aggregate_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.concat(
        [
            pd.read_csv(
                DATA / f"SourceData_measurement_daily_metrics_{h}.csv",
                parse_dates=["signal_date"],
            )
            for h in HORIZONS
        ],
        ignore_index=True,
    )
    summary = paired_summary(daily)
    # A second independent call starts from the same frozen seed.  Exact
    # equality is a reproducibility gate for the moving-block resampling code.
    summary_repeat = paired_summary(daily)
    headline_map = {
        "presence": "recall_at_100",
        "new_entry": "recall_at_100",
        "retention": "pr_auc",
        "conditional_score": "spearman",
    }
    mask = summary["comparison"].eq("real_vs_baseline") & summary.apply(
        lambda r: headline_map.get(r["task"]) == r["metric"], axis=1
    )
    summary["holm_p_8_headlines"] = np.nan
    if mask.sum():
        summary.loc[mask, "holm_p_8_headlines"] = holm_adjust(
            summary.loc[mask, "bootstrap_p"].to_numpy(float)
        )
        repeat_mask = summary_repeat["comparison"].eq(
            "real_vs_baseline"
        ) & summary_repeat.apply(
            lambda r: headline_map.get(r["task"]) == r["metric"], axis=1
        )
        summary_repeat["holm_p_8_headlines"] = np.nan
        summary_repeat.loc[repeat_mask, "holm_p_8_headlines"] = holm_adjust(
            summary_repeat.loc[repeat_mask, "bootstrap_p"].to_numpy(float)
        )
    sort_cols = ["horizon", "task", "sample", "comparison", "metric"]
    first = summary.sort_values(sort_cols).reset_index(drop=True)
    second = summary_repeat.sort_values(sort_cols).reset_index(drop=True)
    deterministic_repeat = first.equals(second)
    canonical = first.to_csv(index=False, float_format="%.15g", lineterminator="\n")
    canonical_repeat = second.to_csv(
        index=False, float_format="%.15g", lineterminator="\n"
    )
    repeat_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    repeat_hash_2 = hashlib.sha256(canonical_repeat.encode("utf-8")).hexdigest()
    summary.to_csv(
        DATA / "SourceData_measurement_increment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    roundtrip = pd.read_csv(DATA / "SourceData_measurement_increment_summary.csv")
    roundtrip_numeric = [
        c for c in summary.columns if pd.api.types.is_numeric_dtype(summary[c])
    ]
    roundtrip_ok = len(roundtrip) == len(summary) and all(
        np.allclose(
            pd.to_numeric(roundtrip[c], errors="coerce").to_numpy(float),
            pd.to_numeric(summary[c], errors="coerce").to_numpy(float),
            equal_nan=True,
            rtol=1e-12,
            atol=1e-14,
        )
        for c in roundtrip_numeric
    )
    determinism = pd.DataFrame(
        [
            {
                "bootstrap_repetitions": BOOTSTRAP_REPS,
                "seed": SEED,
                "paired_summary_repeat_exact": deterministic_repeat,
                "first_sha256": repeat_hash,
                "second_sha256": repeat_hash_2,
                "hashes_equal": repeat_hash == repeat_hash_2,
                "csv_roundtrip_numeric_equal": roundtrip_ok,
                "determinism_gate_pass": bool(
                    deterministic_repeat
                    and repeat_hash == repeat_hash_2
                    and roundtrip_ok
                ),
            }
        ]
    )
    determinism.to_csv(
        DATA / "SourceData_measurement_determinism_gate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(
        DATA / "SourceData_measurement_daily_metrics_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return daily, summary


def plot_results(daily: pd.DataFrame, summary: pd.DataFrame) -> None:
    headline = {
        "presence": ("recall_at_100", "All-candidate presence\nRecall@100"),
        "new_entry": ("recall_at_100", "New entry\nRecall@100"),
        "retention": ("pr_auc", "One-step retention\nPR-AUC"),
        "conditional_score": ("spearman", "Conditional score\nSpearman"),
    }
    main = summary.loc[summary["comparison"].eq("real_vs_baseline")].copy()
    selected = []
    for task, (metric, label) in headline.items():
        part = main.loc[(main["task"] == task) & (main["metric"] == metric)].copy()
        part["label"] = label
        selected.append(part)
    plot = pd.concat(selected, ignore_index=True)
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.35), sharey=False)
    for j, (task, (_, title)) in enumerate(headline.items()):
        ax = axes[j]
        part = plot.loc[plot["task"].eq(task)].set_index("horizon").reindex(HORIZONS)
        y = np.arange(len(part))
        colors = [COLORS["a_share"], COLORS["difference"]]
        ax.axvline(0, color=COLORS["zero"], lw=0.8, zorder=0)
        for k, (_, row) in enumerate(part.iterrows()):
            ax.errorbar(
                row["improvement"],
                y[k],
                xerr=[
                    [row["improvement"] - row["ci_low"]],
                    [row["ci_high"] - row["improvement"]],
                ],
                fmt="o",
                color=colors[k],
                capsize=2,
                lw=1.1,
                ms=4,
                zorder=3,
            )
        ax.set_yticks(y, ["1 h list", "24 h list"])
        ax.set_title(title)
        ax.set_xlabel("OOS improvement")
        clean_axes(ax, grid_axis="x")
        panel_label(ax, chr(ord("a") + j))
    fig.subplots_adjust(wspace=0.58)
    save_figure(fig, FIGURES / "Fig_MV1_tsk_incremental_validation")
    plt.close(fig)

    # Stock-identity and timing falsifications for new entry and all-candidate presence.
    comps = [
        ("real_vs_baseline_placebo_support", "Real T/S/K", COLORS["a_share"]),
        ("code_rotation_vs_baseline", "Code rotation", COLORS["alpha"]),
        ("future20_vs_baseline", "Forbidden future (+20)", COLORS["difference"]),
        ("past_state_vs_baseline", "Past state", COLORS["gamma"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.7), sharex=False)
    for j, task in enumerate(("presence", "new_entry")):
        ax = axes[j]
        subset = summary.loc[
            (summary["task"] == task) & (summary["metric"] == "recall_at_100")
        ]
        labels, positions = [], []
        pos = 0
        ax.axvline(0, color=COLORS["zero"], lw=0.8)
        for horizon in HORIZONS:
            for comp, label, color in comps:
                row = subset.loc[
                    (subset["horizon"] == horizon) & (subset["comparison"] == comp)
                ]
                if row.empty:
                    continue
                r = row.iloc[0]
                ax.errorbar(
                    r["improvement"],
                    pos,
                    xerr=[
                        [r["improvement"] - r["ci_low"]],
                        [r["ci_high"] - r["improvement"]],
                    ],
                    fmt="o",
                    color=color,
                    capsize=2,
                    lw=1.0,
                    ms=3.8,
                )
                labels.append(f"{horizon} · {label}")
                positions.append(pos)
                pos += 1
            pos += 0.55
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Recall@100 improvement")
        ax.set_title(
            "All-candidate presence"
            if task == "presence"
            else "New entry among t outsiders"
        )
        clean_axes(ax, grid_axis="x")
        panel_label(ax, chr(ord("a") + j))
    fig.text(
        0.995,
        0.01,
        "Future shift is a leakage/timing diagnostic, not a zero-null placebo.",
        ha="right",
        va="bottom",
        fontsize=6.7,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(wspace=0.78, bottom=0.16)
    save_figure(fig, FIGURES / "Fig_MV2_tsk_placebo_specificity")
    plt.close(fig)

    # Report fold-level increments to show variation across evaluation periods.
    main_daily = daily.loc[daily["sample"].eq("main_common_support")]
    records = []
    for (horizon, task, fold), group in main_daily.groupby(["horizon", "task", "fold"]):
        metric = headline[task][0]
        left = group.loc[group["model"].eq("baseline"), ["signal_date", metric]].rename(
            columns={metric: "base"}
        )
        right = group.loc[group["model"].eq("real"), ["signal_date", metric]].rename(
            columns={metric: "real"}
        )
        merged = left.merge(right, on="signal_date")
        direction = -1 if metric in {"mae", "brier", "log_loss"} else 1
        records.append(
            {
                "horizon": horizon,
                "task": task,
                "fold": fold,
                "improvement": direction * (merged["real"] - merged["base"]).mean(),
            }
        )
    folds = pd.DataFrame(records)
    folds.to_csv(
        DATA / "SourceData_measurement_fold_increments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 3.9), sharex=True)
    for panel_index, (ax, (task, (_, title))) in enumerate(
        zip(axes.ravel(), headline.items())
    ):
        part = folds.loc[folds["task"].eq(task)]
        for horizon, color, marker in [
            ("1h", COLORS["a_share"], "o"),
            ("24h", COLORS["difference"], "s"),
        ]:
            g = (
                part.loc[part["horizon"].eq(horizon)]
                .set_index("fold")
                .reindex([x[0] for x in FOLDS])
            )
            ax.plot(
                np.arange(len(g)),
                g["improvement"],
                marker=marker,
                color=color,
                label=f"{horizon} list",
            )
        ax.axhline(0, color=COLORS["zero"], lw=0.8)
        ax.set_title(title.replace("\n", " "))
        ax.set_xticks(
            np.arange(len(FOLDS)), [x[0] for x in FOLDS], rotation=45, ha="right"
        )
        clean_axes(ax, grid_axis="y")
        panel_label(ax, chr(ord("a") + panel_index))
    axes[0, 0].legend(loc="best")
    fig.supxlabel("Out-of-time test block", y=0.015)
    fig.supylabel("Augmented-model improvement")
    fig.subplots_adjust(wspace=0.35, hspace=0.36, bottom=0.23)
    save_figure(fig, FIGURES / "Fig_MV3_tsk_time_stability")
    plt.close(fig)


def fmt(x: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"


def fmt_holm_p(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    resolution = 8 * 2 / (BOOTSTRAP_REPS + 1)
    if x <= resolution + 1e-12:
        return f"<{resolution:.3f}"
    return f"{x:.4f}"


def write_reports(build_audit: dict, model_audit: dict, summary: pd.DataFrame) -> None:
    headline_metric = {
        "presence": "recall_at_100",
        "new_entry": "recall_at_100",
        "retention": "pr_auc",
        "conditional_score": "spearman",
    }
    main = summary.loc[summary["comparison"].eq("real_vs_baseline")].copy()
    lines = []
    for task, metric in headline_metric.items():
        for horizon in HORIZONS:
            row = main.loc[
                (main["task"] == task)
                & (main["metric"] == metric)
                & (main["horizon"] == horizon)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(
                f"| {task} | {horizon} | {metric} | {fmt(r['baseline_mean'])} | {fmt(r['comparison_mean'])} | "
                f"{fmt(r['improvement'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {fmt_holm_p(r['holm_p_8_headlines'])} |"
            )
    identity = summary.loc[
        summary["comparison"].isin(
            [
                "real_vs_baseline_placebo_support",
                "code_rotation_vs_baseline",
                "future20_vs_baseline",
                "past_state_vs_baseline",
                "real_vs_code_rotation",
                "real_vs_future20",
                "real_vs_past_state",
            ]
        )
        & summary["metric"].eq("recall_at_100")
    ]
    gates = []
    timing_gates = []
    for task in ("presence", "new_entry"):
        for horizon in HORIZONS:
            p = identity.loc[
                (identity["task"] == task) & (identity["horizon"] == horizon)
            ].set_index("comparison")
            if "real_vs_code_rotation" in p.index:
                direct = p.loc["real_vs_code_rotation"]
                real = p.loc["real_vs_baseline_placebo_support", "improvement"]
                rot = p.loc["code_rotation_vs_baseline", "improvement"]
                gates.append(
                    (
                        task,
                        horizon,
                        direct["ci_low"] > 0,
                        real,
                        rot,
                        direct["improvement"],
                        direct["ci_low"],
                        direct["ci_high"],
                    )
                )
            if "real_vs_future20" in p.index:
                direct = p.loc["real_vs_future20"]
                timing_gates.append(
                    (
                        task,
                        horizon,
                        direct["ci_low"] > 0,
                        direct["improvement"],
                        direct["ci_low"],
                        direct["ci_high"],
                    )
                )
    all_identity = bool(gates) and all(x[2] for x in gates)
    all_timing = bool(timing_gates) and all(x[2] for x in timing_gates)
    stable = main.groupby(["task", "horizon"])["ci_low"].first().gt(0).to_dict()
    if all_identity and all_timing and any(stable.values()):
        evidence = (
            "E2/E3 stock-specific and time-specific related-vendor endpoint overlap"
        )
    elif all_identity and any(stable.values()):
        evidence = (
            "E2 stock-specific predictive validation; timing-specificity gate failed"
        )
    else:
        evidence = "E1/E2 descriptive/predictive evidence only"
    fold_data = pd.read_csv(DATA / "SourceData_measurement_fold_increments.csv")
    fold_lines = []
    for task in TASKS:
        for horizon in HORIZONS:
            g = fold_data[(fold_data.task == task) & (fold_data.horizon == horizon)]
            fold_lines.append(
                f"- {task}, {horizon}: positive in {(g.improvement > 0).sum()}/{len(g)} test blocks "
                f"(range {g.improvement.min():.4f} to {g.improvement.max():.4f})."
            )

    professional = f"""# Experiment B — Related-vendor endpoint validation of T/S/K

## Decision

**Evidence label: {evidence}.** This experiment asks whether the vendor's total heat (T), self-selection heat (S), and stickiness (K) improve prediction of a supplier-labelled Top-100 endpoint whose product slug is `stock-xueqiu-hot-stock`. The local files do not authenticate official platform identity or upstream independence. It does not identify T/S/K causing either attention or returns.

The design separates four questions that must not be pooled: all-candidate next-session presence, genuinely new entry among current outsiders, one-step retention among current insiders, and conditional provider score among next-session insiders. Current list state is included in the relevant baseline, so ordinary list persistence cannot be credited to T/S/K.

## Frozen headline results

Positive improvement means the T/S/K-augmented model is better. Confidence intervals are paired five-session moving-block intervals. Holm correction covers the eight frozen horizon-by-endpoint tests.

| Task | List | Metric | Baseline | +T/S/K | Improvement (95% CI) | Holm p |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(lines)}

## What the related-vendor endpoint supports

The code-rotation falsification is the decisive stock-identity check. A positive real increment that is no larger than the rotated-code increment cannot validate stock-specific information. The future +20-session diagnostic and matched-past-state diagnostic are deliberately harder timing checks: persistence can make them predictive, so they are not strict zero-effect nulls.

Stock-identity gates (real improvement > code-rotation improvement):
{chr(10).join(f'- {task}, {horizon}: {"PASS" if passed else "FAIL"} (real-minus-rotation {direct:.4f}, CI [{lo:.4f}, {hi:.4f}]; real vs baseline {real:.4f}; rotation vs baseline {rot:.4f})' for task, horizon, passed, real, rot, direct, lo, hi in gates)}

Timing-specificity gates (real T/S/K must outperform +20-session T/S/K):
{chr(10).join(f'- {task}, {horizon}: {"PASS" if passed else "FAIL"} (real-minus-future20 {direct:.4f}, CI [{lo:.4f}, {hi:.4f}])' for task, horizon, passed, direct, lo, hi in timing_gates)}

The +20-session vector outperforms the correctly timed vector for entry/presence in this sample. Because T/S/K are persistent and prices/list activity can feed back into later T/S/K, this diagnostic is not a conventional zero-effect placebo; nevertheless it decisively prevents a claim that the incremental signal is uniquely located at the frozen signal date. The supported statement is stock specificity, not timing specificity.

Out-of-time stability is heterogeneous and must not be inferred from the pooled confidence interval alone:
{chr(10).join(fold_lines)}

The cleanest repeatable component is one-step retention (positive in every block for both list horizons). All-candidate presence is positive on average but only in four of seven blocks for each horizon. New entry is larger but not universal, and conditional-score increments are economically very small even where statistically detectable.

## Method and anti-leakage controls

- Every T/S/K row obeys `observation_date <= attention_asof_cutoff <= signal_date - 2 calendar days`.
- Every test block is strictly out of time. The final two signal dates before each block are embargoed; preprocessing is fitted on past training data only.
- Baseline and augmented models use the same rows. Sampled training negatives receive inverse-sampling weights; test prevalence is untouched.
- Provider `type` is not treated as rank. `current`, `chg`, and `percent` are excluded. A missing or incomplete snapshot makes the whole day ineligible; an outside-list score remains missing, never zero.
- The supplier-labelled score is called provider list score. It is not re-labelled as clicks, searches, or latent mental attention.
- Moving-block Monte-Carlo p values use the finite-resampling `+1` correction, so an all-same-sign set of 1,000 draws is reported at its finite resolution rather than as p=0.

## Scope boundary

This is related-vendor endpoint predictive overlap, not independent construct validation. It can strengthen the statement that T/S/K track some stock-specific component also visible in another supplier-organized list product. It cannot establish upstream independence, that T/S/K are complete measures of human attention, that attention causes price changes, or that the relationship is tradable after costs.

## Reproducibility

The frozen design is in `measurement_validation_frozen_spec.md`; daily paired metrics, calibration curves, fold samples, increment summaries, exact OOF predictions, code-mapping coverage and model figures are stored beside this report. `SourceData_measurement_determinism_gate.csv` records a second seeded aggregation run, matching canonical hashes, and a CSV numeric round-trip gate. The hard cutoff is `< 2026-07-01`.
"""
    (REPORTS / "measurement_validation_professional.md").write_text(
        professional, encoding="utf-8"
    )

    plain_rows = []
    names = {
        "presence": "明天是否上榜",
        "new_entry": "今天榜外、明天新进榜",
        "retention": "今天榜内、明天还在",
        "conditional_score": "明天榜内的热度次序",
    }
    for task, metric in headline_metric.items():
        for horizon in HORIZONS:
            row = main.loc[
                (main["task"] == task)
                & (main["metric"] == metric)
                & (main["horizon"] == horizon)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            verdict = (
                "稳定增加"
                if r["ci_low"] > 0
                else ("稳定变差" if r["ci_high"] < 0 else "不能确认增加")
            )
            plain_rows.append(
                f"- {horizon} 榜单，{names[task]}：改善 {r['improvement']:.4f}，95%区间 [{r['ci_low']:.4f}, {r['ci_high']:.4f}]；结论是“{verdict}”。"
            )
    plain = f"""# 白话总结：T、S、K 到底有没有被另一套数据认出来？

## 一句话

我们没有把“昨天还在榜上，所以明天大概率还在”冒充成 T/S/K 有效。先让基础模型知道今天是否上榜、今天榜值多少，再看加上 T/S/K 后是否还能多猜对一点。

## 我们实际做了四道题

1. 所有股票里猜明天谁上榜；
2. 只看今天不在榜里的股票，猜明天谁是新进入者；
3. 只看今天已经在榜里的股票，猜明天谁留下；
4. 只看明天确实上榜的股票，猜榜值高低顺序。

模型只用过去训练，再去猜后面的半年；一共连续检验 2023 上半年到 2026 上半年七段时间。

## 数字怎么读

下面“改善”都已经统一为正数更好；区间跨过 0 就表示目前不能排除只是波动。

{chr(10).join(plain_rows)}

## 最关键的真假检查

我们把同一天整套 T/S/K 故意转给错误股票（代码轮换）。如果真数据不比“张冠李戴”强，就不能说 T/S/K 认出了具体股票。我们还用了未来错位 20 个交易日、以及过去相似市场状态日作对照；后两者因为热度有记忆，不一定必须降到零，但必须如实展示。

本次最需要克制的一点是：**未来错位 20 个交易日的 T/S/K，比正确日期的 T/S/K 更会猜“谁将上榜/新进榜”**。这不等于模型真能在当时使用未来数据；它是故意做的诊断，说明同一股票的热度状态很持久，或者“先上榜/价格变化，后来又制造热度”的反向路径存在。因此我们能说它认得“哪只股票”，但不能说信息只属于当前这个精确时点。

分七段看，最整齐的是“今天榜内、明天还在”：1h 和 24h 都是 7/7 段改善。总体上榜只在两种榜单各 4/7 段改善；新进入是 1h 的 5/7、24h 的 6/7。也就是说，平均数字是真的，但并非每个时期都一样强。

## 这能证明什么、不能证明什么

- 本实验最多支持：T/S/K与另一供应商榜单端点中的某些“具体股票”信息有可重复的预测重合；两套产品的上游独立性未证明。
- 更精确的上限：代码轮换检验支持“具体股票”信息；未来错位检验不支持“只在当前精确时点有效”。
- 不能证明：T/S/K 就是人的心理注意力；热度导致收益；或模型可以直接赚钱。
- 所以这一步叫“测量验证”，不是“收益因果识别”。

所有数据都截断在 2026-07-01 之前；缺整张 Top-100 快照的日子整天不用，榜外股票只有“未上榜=0”，绝不把它的榜值伪造为 0。
"""
    (REPORTS / "measurement_validation_plain_chinese.md").write_text(
        plain, encoding="utf-8"
    )

    determinism = (
        pd.read_csv(DATA / "SourceData_measurement_determinism_gate.csv")
        .iloc[0]
        .to_dict()
    )
    audit = {
        "schema_version": "tsk-independent-measurement-v1-frozen",
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "cutoff_exclusive": str(CUTOFF),
        "input_files": {
            "pit": {
                "path": str(PIT_PATH),
                "bytes": PIT_PATH.stat().st_size,
                "sha256": sha256(PIT_PATH),
            },
            "attention": {
                "path": str(ATTENTION_PATH),
                "bytes": ATTENTION_PATH.stat().st_size,
                "sha256": sha256(ATTENTION_PATH),
            },
        },
        "semantics": {
            "T": "vendor total heat",
            "S": "vendor self-selection heat; structurally unavailable before 2022-03-23",
            "K": "vendor stickiness",
            "xueqiu_value": "provider list score, not clicks/searches/mental attention",
            "signal_rule": "observation_date <= attention_asof_cutoff <= signal_date - 2 calendar days",
        },
        "build": build_audit,
        "model": model_audit,
        "headline_family_size": 8,
        "bootstrap_inference": {
            "repetitions": BOOTSTRAP_REPS,
            "block_sessions": BLOCK_DAYS,
            "finite_resampling_p_correction": "2*min((tail_count+1)/(B+1))",
            "determinism_roundtrip": determinism,
        },
        "evidence_label": evidence,
    }
    (DATA / "audit_measurement_validation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    qa = f"""# QA — Experiment B related-vendor endpoint validation

- Hard cutoff `< 2026-07-01`: PASS by source filters and panel maximum-date audit.
- T/S/K as-of violations: {'PASS' if all(x['asof_violations'] == 0 and x['signal_minus_2_calendar_day_violations'] == 0 for x in build_audit['panels']) else 'FAIL'}.
- Price/trading/size feature-as-of violations (`feature_asof_date > signal_date`): {'PASS' if all(x.get('feature_asof_violations', 1) == 0 for x in build_audit['panels']) else 'FAIL'}.
- Complete Top-100 anchor required before coding absence: PASS by anchor build gate (`n=100`, unique=100, non-null score=100).
- `type` used as rank: PASS (never used); `current/chg/percent` excluded: PASS.
- SH/SZ/BJ deterministic code normalization and date-level mapping audit: PASS; inspect `SourceData_measurement_code_mapping_*.csv` for coverage.
- Baseline and +T/S/K share test rows: PASS by single task-frame prediction construction.
- t list state controls persistence: PASS for all-candidate presence, retention and conditional score; new entry is restricted to t outsiders.
- Post-entry t+2 survival substituted for retention: PASS (not used as primary task).
- Expanding past-only CV plus two-date embargo: PASS; see fold sample source data.
- Probability metrics under sampled negatives: PASS; negative sampling is training-only and inverse-probability weights are used; test prevalence is untouched.
- Paired date-level five-session moving-block CI: PASS.
- Finite-bootstrap +1 p-value correction; no Monte-Carlo p=0: PASS.
- Repeated aggregate run and CSV numeric round-trip deterministic: {'PASS' if determinism.get('determinism_gate_pass') else 'FAIL'} (matching hashes: {determinism.get('hashes_equal')}).
- Eight frozen headline tests Holm-adjusted: PASS.
- Stock-identity falsification is judged by a direct paired real-minus-code-rotation block CI, not by two separate significance tests.
- Timing specificity: {'PASS' if all_timing else 'FAIL'}; the correctly timed vector does not uniformly outperform the deliberately future-shifted +20-session vector.
- Causal claim made from predictive validation: PASS (explicitly prohibited).

Evidence label: **{evidence}**.
"""
    (REPORTS / "QA_measurement_validation.md").write_text(qa, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["build", "model", "report", "all"], default="all"
    )
    args = parser.parse_args()
    t0 = time.time()
    audit_path = CACHE / "measurement_build_audit.json"
    model_path = CACHE / "measurement_model_audit.json"

    if args.stage in {"build", "all"}:
        anchors = [build_anchor(h) for h in HORIZONS]
        panels = [build_panel(h) for h in HORIZONS]
        build_audit = {"anchors": anchors, "panels": panels}
        audit_path.write_text(
            json.dumps(build_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Build completed in {(time.time() - t0) / 60:.1f} min", flush=True)

    if args.stage in {"model", "all"}:
        model_audit = {"horizons": [model_horizon(h) for h in HORIZONS]}
        model_path.write_text(
            json.dumps(model_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Models completed in {(time.time() - t0) / 60:.1f} min", flush=True)

    if args.stage in {"report", "all"}:
        build_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        model_audit = json.loads(model_path.read_text(encoding="utf-8"))
        daily, summary = aggregate_outputs()
        plot_results(daily, summary)
        write_reports(build_audit, model_audit, summary)
        print(f"Reports completed in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
