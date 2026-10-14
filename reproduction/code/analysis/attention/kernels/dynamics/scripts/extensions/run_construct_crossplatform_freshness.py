"""CF2: split slow T/S/K state from current related-vendor endpoint information.

The frozen estimand is the paired OOF improvement X2-X1, where X1 contains
strictly past 5/20/60-session T/S/K vectors and X2 adds the current vector.
X3 replaces the current vector by a within-date code rotation.  X4 adds a
future +20-session vector and is a prohibited leakage/feedback diagnostic,
never a deployable model.  All source observations are < 2026-09-01.

Stages are resumable::

    python run_construct_crossplatform_freshness.py --stage build
    python run_construct_crossplatform_freshness.py --stage model
    python run_construct_crossplatform_freshness.py --stage report
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
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
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "scripts"))
from plot_style import COLORS, clean_axes, panel_label, save_figure  # noqa: E402


def load_measurement_module():
    path = PROJECT / "scripts" / "crossmarket" / "run_tsk_measurement_validation.py"
    spec = importlib.util.spec_from_file_location("measurement_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_measurement_module()
CUTOFF = pd.Timestamp("2026-09-01")
SEED = 20260714
LAGS = (5, 20, 60)
B = 1000
BLOCK_DAYS = 5
HORIZONS = ("1h", "24h")
TSK = list(BASE.TSK)
BASE_NUMERIC = list(BASE.BASE_NUMERIC)
FOLDS = list(BASE.FOLDS)
TASKS = BASE.TASKS

ATTENTION = Path("__SNAPSHOT__/attention/inputs/attention_daily.parquet")
PIT = Path("__SNAPSHOT__/input_recovery/stock_pit_covariates_daily.universe.parquet")
BASE_PANELS = {
    h: PROJECT / "cache" / "crossmarket" / f"measurement_tsk_{h}_panel.parquet"
    for h in HORIZONS
}

CACHE = PROJECT / "cache" / "extensions"
DATA = PROJECT / "data" / "extensions"
FIGURES = PROJECT / "figures" / "extensions"
REPORTS = PROJECT / "reports" / "extensions"
AUDIT = PROJECT / "audit" / "extensions"
SOURCE = PROJECT / "SourceData" / "extensions"
for folder in (CACHE, DATA, FIGURES, REPORTS, AUDIT, SOURCE):
    folder.mkdir(parents=True, exist_ok=True)

PANELS = {h: CACHE / f"CF2_crossplatform_freshness_{h}_panel.parquet" for h in HORIZONS}
PREDICTIONS = {
    h: CACHE / f"CF2_crossplatform_freshness_{h}_oof.parquet" for h in HORIZONS
}

MAIN_MODELS = ("X0", "X1", "X2", "X3")
FUTURE_MODELS = ("X1_future_support", "X4_future20")
PRIMARY_METRIC = {
    "presence": "recall_at_100",
    "new_entry": "recall_at_100",
    "retention": "pr_auc",
    "conditional_score": "spearman",
}
# Frozen practical-equivalence margins, in headline-metric units.  These are
# deliberately small absolute changes, not post-result standard deviations.
EQUIVALENCE_MARGIN = {
    "presence": 0.0025,
    "new_entry": 0.0025,
    "retention": 0.0025,
    "conditional_score": 0.0050,
}


def sha256(path: Path, block: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def trading_dates() -> list[pd.Timestamp]:
    out = (
        pl.scan_parquet(PIT)
        .filter(pl.col("signal_date") < pl.lit(CUTOFF.to_pydatetime()))
        .select(pl.col("signal_date").unique().sort())
        .collect()["signal_date"]
        .to_list()
    )
    return [pd.Timestamp(x).normalize() for x in out]


def lag_map(dates: Sequence[pd.Timestamp], lag: int) -> pl.DataFrame:
    records = [
        {"signal_date": dates[i], f"past{lag}_source_date": dates[i - lag]}
        for i in range(lag, len(dates))
    ]
    return pl.from_pandas(pd.DataFrame(records)).with_columns(
        pl.col("signal_date").cast(pl.Datetime("ns")),
        pl.col(f"past{lag}_source_date").cast(pl.Datetime("ns")),
    )


def build_panel(horizon: str, force: bool = False) -> dict:
    target = PANELS[horizon]
    if target.exists() and not force:
        panel = pl.scan_parquet(target)
        return {
            "horizon": horizon,
            "status": "REUSED",
            "rows": int(panel.select(pl.len()).collect().item()),
            "sha256": sha256(target),
        }

    dates = trading_dates()
    base_cols = [
        "signal_date",
        "stock_code",
        "feature_asof_date",
        "attention_asof_cutoff",
        "observation_date",
        "next_date",
        "next2_date",
        "current_presence",
        "next_presence",
        "next2_presence",
        "current_log_score",
        "next_log_score",
        *BASE_NUMERIC,
        "weekday_sin",
        "weekday_cos",
        "sw1_industry",
        *TSK,
        *[f"{c}__code_rotation" for c in TSK],
        *[f"{c}__future20" for c in TSK],
    ]
    panel = pl.scan_parquet(BASE_PANELS[horizon]).select(base_cols)
    lookup = (
        pl.scan_parquet(ATTENTION)
        .filter(
            (pl.col("signal_date") < pl.lit(CUTOFF.to_pydatetime()))
            & (pl.col("common_TSK") == 1)
            & (pl.col("observation_date") <= pl.col("attention_asof_cutoff"))
        )
        .select("signal_date", "stock_code", *TSK)
    )
    for lag in LAGS:
        mapping = lag_map(dates, lag).lazy()
        lagged = lookup.rename(
            {
                "signal_date": f"past{lag}_source_date",
                **{c: f"{c}__past{lag}" for c in TSK},
            }
        )
        panel = panel.join(mapping, on="signal_date", how="left").join(
            lagged, on=[f"past{lag}_source_date", "stock_code"], how="left"
        )

    past_cols = [f"{c}__past{lag}" for lag in LAGS for c in TSK]
    current_cols = [*TSK, *[f"{c}__code_rotation" for c in TSK]]
    future_cols = [f"{c}__future20" for c in TSK]
    main_complete = pl.all_horizontal(
        [pl.col(c).is_not_null() for c in [*past_cols, *current_cols]]
    )
    future_complete = pl.all_horizontal([pl.col(c).is_not_null() for c in future_cols])
    panel = (
        panel.with_columns(
            main_complete.cast(pl.Int8).alias("main_common_support"),
            (main_complete & future_complete)
            .cast(pl.Int8)
            .alias("future_common_support"),
        )
        .sort(["signal_date", "stock_code"])
        .collect()
    )
    panel.write_parquet(target, compression="zstd", statistics=True)

    audit = {
        "horizon": horizon,
        "status": "BUILT",
        "source_panel": str(BASE_PANELS[horizon]),
        "source_attention": str(ATTENTION),
        "rows": panel.height,
        "dates": panel["signal_date"].n_unique(),
        "codes": panel["stock_code"].n_unique(),
        "min_signal_date": str(panel["signal_date"].min()),
        "max_signal_date": str(panel["signal_date"].max()),
        "max_next_date": str(panel["next_date"].max()),
        "main_common_rows": int(panel["main_common_support"].sum()),
        "future_common_rows": int(panel["future_common_support"].sum()),
        "attention_asof_violations": int(
            (panel["observation_date"] > panel["attention_asof_cutoff"]).sum()
        ),
        "feature_asof_violations": int(
            (panel["feature_asof_date"] > panel["signal_date"]).sum()
        ),
        "sha256": sha256(target),
    }
    return audit


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


def base_features(task: str) -> list[str]:
    cols = [*BASE_NUMERIC, "weekday_sin", "weekday_cos", "sw1_industry"]
    if task in {"presence", "conditional_score"}:
        cols += ["current_presence", "current_log_score"]
    elif task == "retention":
        cols += ["current_log_score"]
    return cols


def features(task: str, model: str) -> list[str]:
    cols = base_features(task)
    past = [f"{c}__past{lag}" for lag in LAGS for c in TSK]
    if model != "X0":
        cols += past
    if model == "X2":
        cols += TSK
    elif model == "X3":
        cols += [f"{c}__code_rotation" for c in TSK]
    elif model == "X4_future20":
        cols += [f"{c}__future20" for c in TSK]
    return cols


def prepare_xy(train: pd.DataFrame, test: pd.DataFrame, task: str, model: str):
    cols = features(task, model)
    numeric = [c for c in cols if c != "sw1_industry"]
    categories = sorted(str(x) for x in train["sw1_industry"].dropna().unique())
    mapping = {value: i for i, value in enumerate(categories)}
    tr_num = train[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    te_num = test[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    tr_cat = (
        train["sw1_industry"].astype(str).map(mapping).to_numpy(float).reshape(-1, 1)
    )
    te_cat = (
        test["sw1_industry"].astype(str).map(mapping).to_numpy(float).reshape(-1, 1)
    )
    return (
        np.column_stack([tr_num, tr_cat]).astype(np.float32, copy=False),
        np.column_stack([te_num, te_cat]).astype(np.float32, copy=False),
        [len(numeric)],
    )


def stable_negative_mask(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(
        frame[["signal_date", "stock_code"]], index=False
    ).to_numpy(np.uint64)
    return (y == 1) | ((hashes % BASE.NEGATIVE_KEEP_MOD) == 0)


def fit_predict(
    train: pd.DataFrame,
    y_train: np.ndarray,
    test: pd.DataFrame,
    task: str,
    model: str,
    seed_offset: int,
) -> np.ndarray:
    x_train, x_test, categorical = prepare_xy(train, test, task, model)
    spec = TASKS[task]
    if spec.kind == "binary":
        weights = np.ones(len(y_train), dtype=np.float32)
        if spec.train_negative_sample:
            keep = stable_negative_mask(train, y_train)
            weights[y_train == 0] = BASE.NEGATIVE_KEEP_MOD
            x_fit, y_fit, w_fit = x_train[keep], y_train[keep], weights[keep]
        else:
            x_fit, y_fit, w_fit = x_train, y_train, weights
        estimator = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.075,
            max_iter=35,
            max_leaf_nodes=15,
            min_samples_leaf=80 if spec.train_negative_sample else 30,
            l2_regularization=1.0,
            max_bins=63,
            categorical_features=categorical,
            early_stopping=False,
            random_state=SEED + seed_offset,
        )
        estimator.fit(x_fit, y_fit, sample_weight=w_fit)
        return np.clip(estimator.predict_proba(x_test)[:, 1], 1e-7, 1 - 1e-7)
    estimator = HistGradientBoostingRegressor(
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
    estimator.fit(x_train, y_train)
    return estimator.predict(x_test)


def fold_frame(
    path: Path, start: str, end: str, support: str, train: bool, cut: pd.Timestamp
) -> pd.DataFrame:
    scan = pl.scan_parquet(path).filter(pl.col(support) == 1)
    if train:
        scan = scan.filter(pl.col("signal_date") < pl.lit(cut.to_pydatetime()))
    else:
        scan = scan.filter(
            (pl.col("signal_date") >= pl.lit(pd.Timestamp(start).to_pydatetime()))
            & (pl.col("signal_date") < pl.lit(pd.Timestamp(end).to_pydatetime()))
        )
    return scan.collect().to_pandas()


def embargo_cut(path: Path, start: str) -> pd.Timestamp:
    dates = (
        pl.scan_parquet(path)
        .filter(
            (pl.col("main_common_support") == 1)
            & (pl.col("signal_date") < pl.lit(pd.Timestamp(start).to_pydatetime()))
        )
        .select(pl.col("signal_date").unique().sort())
        .collect()["signal_date"]
        .to_list()
    )
    if len(dates) <= BASE.EMBARGO_DATES:
        raise RuntimeError(f"Insufficient history before {start}")
    return pd.Timestamp(dates[-BASE.EMBARGO_DATES]).normalize()


def binary_daily(pred: pd.DataFrame, model: str, ranking: bool) -> pd.DataFrame:
    rows = []
    for date, g in pred.groupby("signal_date", sort=True):
        y = g["y"].to_numpy(int)
        p = g[model].to_numpy(float)
        if not len(y):
            continue
        row = {
            "signal_date": date,
            "model": model,
            "n": len(y),
            "positives": int(y.sum()),
            "pr_auc": average_precision_score(y, p) if y.sum() else np.nan,
            "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p, labels=[0, 1]),
        }
        if ranking:
            k = min(100, len(y))
            top = np.argpartition(p, -k)[-k:]
            hits = int(y[top].sum())
            row["precision_at_100"] = hits / k
            row["recall_at_100"] = hits / y.sum() if y.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def regression_daily(pred: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for date, g in pred.groupby("signal_date", sort=True):
        y = g["y"].to_numpy(float)
        p = g[model].to_numpy(float)
        valid = np.isfinite(y) & np.isfinite(p)
        y, p = y[valid], p[valid]
        rows.append(
            {
                "signal_date": date,
                "model": model,
                "n": len(y),
                "spearman": (
                    spearmanr(y, p).statistic
                    if len(y) >= 3 and np.std(y) > 0 and np.std(p) > 0
                    else np.nan
                ),
                "mae": mean_absolute_error(y, p) if len(y) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_daily_metrics(
    daily: list[pd.DataFrame],
    pred: pd.DataFrame,
    models: Sequence[str],
    task: str,
    horizon: str,
    fold: str,
    sample: str,
) -> None:
    kind = TASKS[task].kind
    for model in models:
        out = (
            binary_daily(pred, model, task in {"presence", "new_entry"})
            if kind == "binary"
            else regression_daily(pred, model)
        )
        out["fold"] = fold
        out["task"] = task
        out["horizon"] = horizon
        out["sample"] = sample
        daily.append(out)


def model_horizon(horizon: str, force: bool = False) -> dict:
    if PREDICTIONS[horizon].exists() and not force:
        return {
            "horizon": horizon,
            "status": "REUSED",
            "sha256": sha256(PREDICTIONS[horizon]),
        }
    predictions: list[pd.DataFrame] = []
    daily: list[pd.DataFrame] = []
    samples = []
    path = PANELS[horizon]
    for fi, (fold, start, end) in enumerate(FOLDS):
        cut = embargo_cut(path, start)
        train_all = fold_frame(path, start, end, "main_common_support", True, cut)
        test_all = fold_frame(path, start, end, "main_common_support", False, cut)
        if train_all.empty or test_all.empty:
            continue
        print(
            f"[{horizon} {fold}] main train={len(train_all):,} test={len(test_all):,} cut<{cut.date()}",
            flush=True,
        )
        for ti, task in enumerate(TASKS):
            train, y_train = task_rows(train_all, task)
            test, y_test = task_rows(test_all, task)
            if train.empty or test.empty:
                continue
            pred = test[["signal_date", "stock_code"]].copy()
            pred["y"] = y_test
            for mi, model in enumerate(MAIN_MODELS):
                pred[model] = fit_predict(
                    train, y_train, test, task, model, 10000 * fi + 1000 * ti + mi
                )
            pred["fold"], pred["task"], pred["horizon"], pred["sample"] = (
                fold,
                task,
                horizon,
                "main_common_support",
            )
            predictions.append(pred)
            add_daily_metrics(
                daily, pred, MAIN_MODELS, task, horizon, fold, "main_common_support"
            )
            samples.append(
                {
                    "horizon": horizon,
                    "fold": fold,
                    "task": task,
                    "sample": "main_common_support",
                    "embargo_cut_exclusive": str(cut.date()),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_dates": train["signal_date"].nunique(),
                    "test_dates": test["signal_date"].nunique(),
                    "train_target_mean": float(np.mean(y_train)),
                    "test_target_mean": float(np.mean(y_test)),
                }
            )

        # X4 is kept on a separate, strictly paired support because the final
        # 20 sessions cannot have a legal future vector before the cutoff.
        train_future_all = train_all.loc[
            train_all["future_common_support"].eq(1)
        ].copy()
        test_future_all = test_all.loc[test_all["future_common_support"].eq(1)].copy()
        for ti, task in enumerate(TASKS):
            train, y_train = task_rows(train_future_all, task)
            test, y_test = task_rows(test_future_all, task)
            if train.empty or test.empty:
                continue
            pred = test[["signal_date", "stock_code"]].copy()
            pred["y"] = y_test
            pred["X1_future_support"] = fit_predict(
                train, y_train, test, task, "X1", 10000 * fi + 1000 * ti + 101
            )
            pred["X4_future20"] = fit_predict(
                train, y_train, test, task, "X4_future20", 10000 * fi + 1000 * ti + 102
            )
            pred["fold"], pred["task"], pred["horizon"], pred["sample"] = (
                fold,
                task,
                horizon,
                "future_common_support",
            )
            predictions.append(pred)
            add_daily_metrics(
                daily, pred, FUTURE_MODELS, task, horizon, fold, "future_common_support"
            )
            samples.append(
                {
                    "horizon": horizon,
                    "fold": fold,
                    "task": task,
                    "sample": "future_common_support",
                    "embargo_cut_exclusive": str(cut.date()),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_dates": train["signal_date"].nunique(),
                    "test_dates": test["signal_date"].nunique(),
                    "train_target_mean": float(np.mean(y_train)),
                    "test_target_mean": float(np.mean(y_test)),
                }
            )
        del train_all, test_all, train_future_all, test_future_all
        gc.collect()

    pred_all = pd.concat(predictions, ignore_index=True, sort=False)
    daily_all = pd.concat(daily, ignore_index=True, sort=False)
    pred_all.to_parquet(PREDICTIONS[horizon], index=False, compression="zstd")
    daily_path = DATA / f"CF2_crossplatform_freshness_{horizon}_daily_metrics.csv"
    sample_path = DATA / f"CF2_crossplatform_freshness_{horizon}_fold_samples.csv"
    daily_all.to_csv(daily_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(samples).to_csv(sample_path, index=False, encoding="utf-8-sig")
    return {
        "horizon": horizon,
        "status": "MODELED",
        "prediction_rows": len(pred_all),
        "daily_metric_rows": len(daily_all),
        "sha256": sha256(PREDICTIONS[horizon]),
    }


def moving_block_mean(values: np.ndarray, rng: np.random.Generator) -> float:
    n = len(values)
    starts = rng.integers(0, n, size=math.ceil(n / BLOCK_DAYS))
    idx = np.concatenate([(np.arange(BLOCK_DAYS) + start) % n for start in starts])[:n]
    return float(np.mean(values[idx]))


def holm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty(len(p), float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out


def paired_summary(daily: pd.DataFrame) -> pd.DataFrame:
    directions = {
        "recall_at_100": 1,
        "precision_at_100": 1,
        "pr_auc": 1,
        "roc_auc": 1,
        "brier": -1,
        "log_loss": -1,
        "spearman": 1,
        "mae": -1,
    }
    comparisons = [
        ("main_common_support", "X0", "X1", "slow_state_X1_minus_X0"),
        ("main_common_support", "X1", "X2", "current_increment_X2_minus_X1"),
        ("main_common_support", "X1", "X3", "rotation_X3_minus_X1"),
        ("main_common_support", "X3", "X2", "real_current_X2_minus_rotation_X3"),
        (
            "future_common_support",
            "X1_future_support",
            "X4_future20",
            "future_diagnostic_X4_minus_X1",
        ),
    ]
    rows = []
    rng = np.random.default_rng(SEED)
    for (horizon, task, sample), group in daily.groupby(["horizon", "task", "sample"]):
        available = set(group["model"])
        for wanted_sample, left_name, right_name, label in comparisons:
            if sample != wanted_sample or not {left_name, right_name}.issubset(
                available
            ):
                continue
            left = group.loc[group["model"].eq(left_name)].set_index("signal_date")
            right = group.loc[group["model"].eq(right_name)].set_index("signal_date")
            common = left.index.intersection(right.index).sort_values()
            for metric, direction in directions.items():
                if metric not in left or metric not in right:
                    continue
                a = left.loc[common, metric].to_numpy(float)
                b = right.loc[common, metric].to_numpy(float)
                valid = np.isfinite(a) & np.isfinite(b)
                diff = direction * (b[valid] - a[valid])
                if len(diff) < 5:
                    continue
                boots = np.array([moving_block_mean(diff, rng) for _ in range(B)])
                p = min(
                    1.0,
                    2
                    * min(
                        (np.sum(boots <= 0) + 1) / (B + 1),
                        (np.sum(boots >= 0) + 1) / (B + 1),
                    ),
                )
                fold_means = []
                fold_col = left.loc[common, "fold"] if "fold" in left else None
                if fold_col is not None:
                    for fold in sorted(pd.Series(fold_col[valid]).dropna().unique()):
                        mask = np.asarray(fold_col[valid] == fold)
                        if mask.any():
                            fold_means.append(float(np.mean(diff[mask])))
                rows.append(
                    {
                        "horizon": horizon,
                        "task": task,
                        "sample": sample,
                        "comparison": label,
                        "metric": metric,
                        "positive_is_improvement": True,
                        "n_dates": len(diff),
                        "left_mean": float(np.mean(a[valid])),
                        "right_mean": float(np.mean(b[valid])),
                        "improvement": float(np.mean(diff)),
                        "ci_low": float(np.quantile(boots, 0.025)),
                        "ci_high": float(np.quantile(boots, 0.975)),
                        "bootstrap_p": p,
                        "positive_fold_count": int(np.sum(np.asarray(fold_means) > 0)),
                        "available_fold_count": len(fold_means),
                    }
                )
    result = pd.DataFrame(rows)
    headline = result["comparison"].eq("current_increment_X2_minus_X1") & result.apply(
        lambda r: PRIMARY_METRIC.get(r["task"]) == r["metric"], axis=1
    )
    result["holm_p_8_headlines"] = np.nan
    result.loc[headline, "holm_p_8_headlines"] = holm(
        result.loc[headline, "bootstrap_p"].to_numpy(float)
    )
    result["equivalence_margin"] = result["task"].map(EQUIVALENCE_MARGIN)
    result["rotation_equivalent_to_zero"] = (
        result["comparison"].eq("rotation_X3_minus_X1")
        & (result["ci_low"] > -result["equivalence_margin"])
        & (result["ci_high"] < result["equivalence_margin"])
    )
    return result


def aggregate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.concat(
        [
            pd.read_csv(
                DATA / f"CF2_crossplatform_freshness_{h}_daily_metrics.csv",
                parse_dates=["signal_date"],
            )
            for h in HORIZONS
        ],
        ignore_index=True,
    )
    summary = paired_summary(daily)
    repeat = paired_summary(daily)
    sort_cols = ["horizon", "task", "sample", "comparison", "metric"]
    deterministic = (
        summary.sort_values(sort_cols)
        .reset_index(drop=True)
        .equals(repeat.sort_values(sort_cols).reset_index(drop=True))
    )
    primary = summary.loc[
        summary["comparison"].eq("current_increment_X2_minus_X1")
        & summary.apply(lambda r: PRIMARY_METRIC.get(r["task"]) == r["metric"], axis=1)
    ].copy()
    rotation = summary.loc[
        summary["comparison"].eq("real_current_X2_minus_rotation_X3")
        & summary.apply(lambda r: PRIMARY_METRIC.get(r["task"]) == r["metric"], axis=1),
        ["horizon", "task", "improvement", "ci_low", "ci_high"],
    ].rename(
        columns={
            "improvement": "X2_minus_X3",
            "ci_low": "X2_minus_X3_ci_low",
            "ci_high": "X2_minus_X3_ci_high",
        }
    )
    gates = primary.merge(rotation, on=["horizon", "task"], how="left")
    gates["passes_holm"] = gates["holm_p_8_headlines"] < 0.05
    gates["passes_fold_direction"] = gates["positive_fold_count"] >= 5
    gates["real_beats_rotation"] = gates["X2_minus_X3"] > 0
    gates["current_increment_confirmed"] = gates[
        ["passes_holm", "passes_fold_direction", "real_beats_rotation"]
    ].all(axis=1)
    summary.to_csv(
        DATA / "CF2_crossplatform_freshness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gates.to_csv(
        DATA / "CF2_crossplatform_freshness_headline_gates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(
        DATA / "CF2_crossplatform_freshness_daily_metrics_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        SOURCE / "SourceData_Figure_CF2.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {
                "B": B,
                "block_days": BLOCK_DAYS,
                "seed": SEED,
                "deterministic_repeat_exact": deterministic,
            }
        ]
    ).to_csv(AUDIT / "CF2_bootstrap_determinism.csv", index=False, encoding="utf-8-sig")
    return daily, summary, gates


def plot(summary: pd.DataFrame) -> None:
    primary = summary.loc[
        summary["metric"].eq(summary["task"].map(PRIMARY_METRIC))
        & summary["comparison"].isin(
            [
                "slow_state_X1_minus_X0",
                "current_increment_X2_minus_X1",
                "rotation_X3_minus_X1",
                "future_diagnostic_X4_minus_X1",
            ]
        )
    ].copy()
    order = list(PRIMARY_METRIC)
    labels = {
        "slow_state_X1_minus_X0": "Past state",
        "current_increment_X2_minus_X1": "Current T/S/K",
        "rotation_X3_minus_X1": "Rotated code",
        "future_diagnostic_X4_minus_X1": "Future +20 (forbidden)",
    }
    colors = {
        "slow_state_X1_minus_X0": COLORS["alpha"],
        "current_increment_X2_minus_X1": COLORS["a_share"],
        "rotation_X3_minus_X1": COLORS["zero"],
        "future_diagnostic_X4_minus_X1": COLORS["difference"],
    }
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 4.35), constrained_layout=False)
    for row, horizon in enumerate(HORIZONS):
        for col, task in enumerate(order):
            ax = axes[row, col]
            part = primary.loc[
                (primary["horizon"] == horizon) & (primary["task"] == task)
            ]
            comps = list(labels)
            y = np.arange(len(comps))
            ax.axvline(0, color=COLORS["zero"], lw=0.8, zorder=0)
            for i, comp in enumerate(comps):
                q = part.loc[part["comparison"].eq(comp)]
                if q.empty:
                    continue
                r = q.iloc[0]
                ax.errorbar(
                    r["improvement"],
                    i,
                    xerr=[
                        [r["improvement"] - r["ci_low"]],
                        [r["ci_high"] - r["improvement"]],
                    ],
                    fmt="o",
                    color=colors[comp],
                    ms=3.8,
                    capsize=2,
                    lw=1.0,
                )
            ax.set_yticks(y)
            ax.set_yticklabels([labels[x] for x in comps] if col == 0 else [])
            ax.set_title(f"{horizon} list · {task.replace('_', ' ')}", fontsize=8.5)
            clean_axes(ax)
            panel_label(ax, chr(ord("a") + row * 4 + col))
    fig.supxlabel("Paired OOF improvement (positive is better)", y=0.025, fontsize=8.5)
    fig.subplots_adjust(
        left=0.16, right=0.99, top=0.94, bottom=0.12, wspace=0.42, hspace=0.65
    )
    save_figure(fig, FIGURES / "Figure_CF2_crossplatform_freshness")
    plt.close(fig)


def write_reports(summary: pd.DataFrame, gates: pd.DataFrame) -> None:
    rows = []
    for _, r in gates.sort_values(["horizon", "task"]).iterrows():
        verdict = "通过" if r["current_increment_confirmed"] else "未通过"
        rows.append(
            f"| {r['horizon']} | {r['task']} | {r['improvement']:.5f} "
            f"[{r['ci_low']:.5f}, {r['ci_high']:.5f}] | {r['holm_p_8_headlines']:.4f} | "
            f"{int(r['positive_fold_count'])}/{int(r['available_fold_count'])} | {r['X2_minus_X3']:.5f} | {verdict} |"
        )
    table = "\n".join(rows)
    professional = f"""# CF2 — 慢状态与当前新增信息的相关供应商端点拆分

## 识别对象

这不是收益因果识别。它检验供应商T/S/K是否在价格、成交、规模、行业、星期效应和另一供应商榜单端点自身惯性之外，包含可重复的股票身份信息。该榜单产品slug为 `stock-xueqiu-hot-stock`，但本地文件未认证平台身份，两套产品的上游独立性也未被证明。主估计量是严格配对的样本外 `Metric(X2)-Metric(X1)`：X1只见过去5/20/60个交易日的T/S/K，X2才加入当前T/S/K。

- X0：普通市场特征和当前供应商榜单状态；
- X1：X0 + 严格过去 5/20/60 日 T/S/K；
- X2：X1 + 当前 T/S/K；
- X3：X1 + 同日股票代码轮换的当前 T/S/K（负控制）；
- X4：X1 + 未来 20 日 T/S/K，只作禁止使用的泄漏/反馈诊断。

## 预冻结确认结果

| 榜单 | 任务 | X2−X1 改善 [95% 日期块区间] | 8端点 Holm p | 同号外推折 | X2−X3 | 结论 |
|---|---|---:|---:|---:|---:|---|
{table}

“确认”同时要求 Holm<0.05、至少 5 个连续外推折同号（本轮共8折，保留原绝对门槛5）、且真实当前值胜过同日代码轮换。未通过不等于证明无信息；它只表示没有跨过预冻结门槛。X4 即使表现更强也不能进入可部署模型，只能说明未来信息/共同反馈能产生多大上界。

本轮各任务的确认状态与效应尺寸以上表为准。1h和24h分别表示榜单统计窗口，retention结果指下一交易日是否仍留在对应榜单。当前状态的增量不等于独立构念验证，也不能推出注意力对收益的因果效应。

## 推断边界

日期按5-session循环移动块进行B=1,000次重抽样，有限随机p值最小为1/1001；所有输入观测早于2026-09-01。该模块只支持“相关供应商数据生态中的股票身份/留存预测重合”或“共同慢状态”语言，不支持“独立平台构念验证”或“注意力导致收益”。
"""
    plain = f"""# CF2 白话版：旧状态和今天的新消息到底谁有用

把预测分成五步：先只看普通行情和供应商榜单端点自己昨天在不在榜；再把一两周到三个月前的T/S/K历史加进去；然后只多给模型看今天的T/S/K。这样今天带来的提升，才不是把长期黏性误当成新消息。

| 供应商榜单端点 | 要预测的事 | 今天 T/S/K 多带来的改善 | 严格门槛结论 |
|---|---|---:|---|
"""
    for _, r in gates.sort_values(["horizon", "task"]).iterrows():
        plain += f"| {r['horizon']} | {r['task']} | {r['improvement']:.5f} | {'通过' if r['current_increment_confirmed'] else '未通过'} |\n"
    plain += """

代码轮换相当于把甲股票今天的T/S/K塞给乙股票；如果真实值不能胜过它，就不能说模型认出了股票自己的当前信息。未来20日值是故意放进去的“作弊上界”，它表现好也不能用于真实预测。这里回答的是T/S/K与另一供应商榜单端点是否存在预测重合，不回答独立平台构念，更不回答它是否造成股价上涨或下跌。
"""
    (REPORTS / "CF2_crossplatform_freshness_professional.md").write_text(
        professional, encoding="utf-8"
    )
    (REPORTS / "CF2_crossplatform_freshness_plain.md").write_text(
        plain, encoding="utf-8"
    )


def audit_existing_artifacts() -> tuple[list[dict], list[dict], dict]:
    """Reconstruct release checks even when the resumable report stage runs alone."""
    build_rows: list[dict] = []
    model_rows: list[dict] = []
    for horizon in HORIZONS:
        panel = pl.scan_parquet(PANELS[horizon])
        stats = (
            panel.select(
                pl.len().alias("rows"),
                pl.col("signal_date").n_unique().alias("dates"),
                pl.col("stock_code").n_unique().alias("codes"),
                pl.col("signal_date").min().alias("min_signal_date"),
                pl.col("signal_date").max().alias("max_signal_date"),
                pl.col("next_date").max().alias("max_next_date"),
                pl.col("main_common_support").sum().alias("main_common_rows"),
                pl.col("future_common_support").sum().alias("future_common_rows"),
                (pl.col("observation_date") > pl.col("attention_asof_cutoff"))
                .sum()
                .alias("attention_asof_violations"),
                (pl.col("feature_asof_date") > pl.col("signal_date"))
                .sum()
                .alias("feature_asof_violations"),
            )
            .collect()
            .to_dicts()[0]
        )
        stats.update(
            {
                "horizon": horizon,
                "file": str(PANELS[horizon]),
                "sha256": sha256(PANELS[horizon]),
            }
        )
        for key in ("min_signal_date", "max_signal_date", "max_next_date"):
            stats[key] = str(stats[key])
        build_rows.append(stats)

        pred = pl.scan_parquet(PREDICTIONS[horizon])
        pred_stats = (
            pred.select(
                pl.len().alias("prediction_rows"),
                pl.col("signal_date").min().alias("min_signal_date"),
                pl.col("signal_date").max().alias("max_signal_date"),
                pl.col("fold").n_unique().alias("folds"),
                pl.col("task").n_unique().alias("tasks"),
                pl.col("sample").n_unique().alias("samples"),
            )
            .collect()
            .to_dicts()[0]
        )
        pred_stats.update(
            {
                "horizon": horizon,
                "file": str(PREDICTIONS[horizon]),
                "sha256": sha256(PREDICTIONS[horizon]),
            }
        )
        pred_stats["min_signal_date"] = str(pred_stats["min_signal_date"])
        pred_stats["max_signal_date"] = str(pred_stats["max_signal_date"])
        model_rows.append(pred_stats)

    sample = pd.concat(
        [
            pd.read_csv(DATA / f"CF2_crossplatform_freshness_{h}_fold_samples.csv")
            for h in HORIZONS
        ],
        ignore_index=True,
    )
    deterministic = (
        pd.read_csv(AUDIT / "CF2_bootstrap_determinism.csv").iloc[0].to_dict()
    )
    checks = {
        "fold_sample_rows": len(sample),
        "expected_fold_sample_rows": len(HORIZONS) * len(FOLDS) * len(TASKS) * 2,
        "main_fold_task_cells": int((sample["sample"] == "main_common_support").sum()),
        "future_fold_task_cells": int(
            (sample["sample"] == "future_common_support").sum()
        ),
        "all_train_test_nonempty": bool(
            (sample[["train_rows", "test_rows", "train_dates", "test_dates"]] > 0)
            .all()
            .all()
        ),
        "all_panels_before_cutoff": all(
            pd.Timestamp(x["max_signal_date"]) < CUTOFF for x in build_rows
        ),
        "all_outcomes_by_cutoff": all(
            pd.Timestamp(x["max_next_date"]) <= CUTOFF - pd.Timedelta(days=1)
            for x in build_rows
        ),
        "attention_asof_violations": int(
            sum(x["attention_asof_violations"] for x in build_rows)
        ),
        "feature_asof_violations": int(
            sum(x["feature_asof_violations"] for x in build_rows)
        ),
        "prediction_folds_each": {x["horizon"]: int(x["folds"]) for x in model_rows},
        "prediction_tasks_each": {x["horizon"]: int(x["tasks"]) for x in model_rows},
        "bootstrap_determinism": deterministic,
    }
    return build_rows, model_rows, checks


def final_audit(
    build_audits: list[dict], model_audits: list[dict], gates: pd.DataFrame
) -> dict:
    disk_build, disk_model, checks = audit_existing_artifacts()
    if not build_audits:
        build_audits = disk_build
    if not model_audits:
        model_audits = disk_model
    hard_gates = {
        "complete_8_headline_grid": len(gates) == 8,
        "all_panels_before_cutoff": checks["all_panels_before_cutoff"],
        "all_outcomes_by_cutoff": checks["all_outcomes_by_cutoff"],
        "attention_asof_clean": checks["attention_asof_violations"] == 0,
        "feature_asof_clean": checks["feature_asof_violations"] == 0,
        "complete_fold_sample_grid": checks["fold_sample_rows"]
        == checks["expected_fold_sample_rows"],
        "all_train_test_nonempty": checks["all_train_test_nonempty"],
        "all_registered_oof_folds_each": all(
            v == len(FOLDS) for v in checks["prediction_folds_each"].values()
        ),
        "four_tasks_each": all(
            v == 4 for v in checks["prediction_tasks_each"].values()
        ),
        "bootstrap_B_1000": B == 1000,
        "bootstrap_deterministic": bool(
            checks["bootstrap_determinism"]["deterministic_repeat_exact"]
        ),
        "finite_p_values_positive": bool(
            (pd.to_numeric(gates["bootstrap_p"], errors="coerce") > 0).all()
        ),
    }
    result = {
        "experiment": "CF2_crossplatform_freshness",
        "status": "PASS" if all(hard_gates.values()) else "FAIL_CLOSED",
        "cutoff_exclusive": str(CUTOFF),
        "B": B,
        "block_days": BLOCK_DAYS,
        "seed": SEED,
        "expected_headlines": 8,
        "observed_headlines": len(gates),
        "confirmed_headlines": (
            int(gates["current_increment_confirmed"].sum()) if len(gates) else 0
        ),
        "hard_gates": hard_gates,
        "checks": checks,
        "build": build_audits,
        "model": model_audits,
        "outputs": {
            "summary": str(DATA / "CF2_crossplatform_freshness_summary.csv"),
            "gates": str(DATA / "CF2_crossplatform_freshness_headline_gates.csv"),
            "figure_png": str(FIGURES / "Figure_CF2_crossplatform_freshness.png"),
            "figure_pdf": str(FIGURES / "Figure_CF2_crossplatform_freshness.pdf"),
            "source_data": str(SOURCE / "SourceData_Figure_CF2.csv"),
        },
    }
    (AUDIT / "CF2_crossplatform_freshness_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("build", "model", "report", "all"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_audits: list[dict] = []
    model_audits: list[dict] = []
    if args.stage in {"build", "all"}:
        for horizon in HORIZONS:
            audit = build_panel(horizon, args.force)
            build_audits.append(audit)
            print(json.dumps(audit, ensure_ascii=False), flush=True)
    if args.stage in {"model", "all"}:
        for horizon in HORIZONS:
            audit = model_horizon(horizon, args.force)
            model_audits.append(audit)
            print(json.dumps(audit, ensure_ascii=False), flush=True)
    if args.stage in {"report", "all"}:
        daily, summary, gates = aggregate()
        plot(summary)
        write_reports(summary, gates)
        audit = final_audit(build_audits, model_audits, gates)
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
