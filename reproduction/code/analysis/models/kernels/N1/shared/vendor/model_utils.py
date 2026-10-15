#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared expanding-time orthogonal-model and date-HAC utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


HORIZONS = (1, 3, 5, 10, 20, 60, 120, 250)
MAX_NUISANCE_TRAIN = 80_000
RIDGE_ALPHA = 10.0
MIN_POSITIVE_P = float(np.nextafter(0.0, 1.0))


def normal_two_sided_p(t_value: float) -> float:
    """Two-sided normal-reference p value, clipped away from numeric zero.

    This is an analytic normal-reference p value, not a Monte Carlo p value.
    The clipping only prevents a machine-underflow zero from being serialized.
    """

    if not np.isfinite(t_value):
        return math.nan
    return max(float(math.erfc(abs(t_value) / math.sqrt(2.0))), MIN_POSITIVE_P)


def deterministic_thin(
    indices: np.ndarray, maximum: int = MAX_NUISANCE_TRAIN
) -> np.ndarray:
    """Evenly thin chronological eligible rows without looking at values."""

    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
    return indices[positions]


def fold_masks(
    dates: pd.Series, fold_registry: pd.DataFrame
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Expanding-time folds for signal-date quantities only.

    This API is appropriate for features, treatments and doses that are fully
    known on ``dates``.  A forward outcome must instead use
    :func:`purged_fold_masks`, because its label can become known many trading
    days after the signal date.
    """

    registry = fold_registry.copy()
    registry["signal_date"] = pd.to_datetime(registry["signal_date"])
    outputs: list[tuple[int, np.ndarray, np.ndarray]] = []
    date_values = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    for fold in sorted(
        int(value)
        for value in registry.loc[registry["fold_id"] >= 0, "fold_id"].unique()
    ):
        row = registry.loc[registry["fold_id"].eq(fold)].iloc[0]
        test_start = np.datetime64(pd.Timestamp(row["test_start_date_for_fold"]), "ns")
        test_end = np.datetime64(pd.Timestamp(row["test_end_date_for_fold"]), "ns")
        train = np.flatnonzero(date_values < test_start)
        test = np.flatnonzero((date_values >= test_start) & (date_values <= test_end))
        if not len(train) or not len(test):
            raise RuntimeError(f"empty expanding fold {fold}")
        outputs.append((fold, train, test))
    return outputs


def purged_fold_masks(
    signal_dates: pd.Series,
    target_dates: pd.Series,
    fold_registry: pd.DataFrame,
    *,
    module: str,
    outcome: str,
    horizon: int,
    lens: str,
    target_date_role: str,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], pd.DataFrame]:
    """Build expanding-time folds purged by label availability.

    A row is eligible for an outcome-model training fold only when both its
    signal date and the date on which the complete label is known are strictly
    earlier than the fold's test start.  Rows with a missing target date are
    excluded. Test membership is defined by signal date; test outcomes are
    evaluated after they become observable.

    The returned audit has one row per fold and records the maximum retained
    target date. Callers must persist it and require
    ``target_strictly_before_test``.
    """

    if len(signal_dates) != len(target_dates):
        raise ValueError("signal_dates and target_dates must have equal length")
    registry = fold_registry.copy()
    registry["signal_date"] = pd.to_datetime(registry["signal_date"])
    signal_values = pd.to_datetime(signal_dates, errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    target_values = pd.to_datetime(target_dates, errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    if np.isnat(signal_values).any():
        raise ValueError("signal_dates contain missing values")

    outputs: list[tuple[int, np.ndarray, np.ndarray]] = []
    audit_rows: list[dict[str, object]] = []
    folds = sorted(
        int(value)
        for value in registry.loc[registry["fold_id"] >= 0, "fold_id"].unique()
    )
    for fold in folds:
        row = registry.loc[registry["fold_id"].eq(fold)].iloc[0]
        test_start = np.datetime64(pd.Timestamp(row["test_start_date_for_fold"]), "ns")
        test_end = np.datetime64(pd.Timestamp(row["test_end_date_for_fold"]), "ns")
        train_candidate = np.flatnonzero(signal_values < test_start)
        candidate_targets = target_values[train_candidate]
        known = ~np.isnat(candidate_targets)
        strictly_past = known & (candidate_targets < test_start)
        train = train_candidate[strictly_past]
        test = np.flatnonzero(
            (signal_values >= test_start) & (signal_values <= test_end)
        )
        if not len(train) or not len(test):
            raise RuntimeError(
                f"empty purged expanding fold {fold} for {module}|{lens}|{outcome}|H{horizon}"
            )
        maximum_target = target_values[train].max()
        maximum_signal = signal_values[train].max()
        strict = bool(maximum_target < test_start)
        if not strict:
            raise RuntimeError(
                f"horizon purge failed for {module}|{lens}|{outcome}|H{horizon}|fold{fold}: "
                f"max target {maximum_target} >= test start {test_start}"
            )
        outputs.append((fold, train, test))
        audit_rows.append(
            {
                "module": module,
                "lens": lens,
                "outcome": outcome,
                "holding_h": int(horizon),
                "fold_id": int(fold),
                "target_date_role": target_date_role,
                "test_start": pd.Timestamp(test_start),
                "test_end": pd.Timestamp(test_end),
                "train_candidate_rows": int(len(train_candidate)),
                "train_rows_after_purge": int(len(train)),
                "purged_target_missing_rows": int((~known).sum()),
                "purged_target_on_or_after_test_rows": int(
                    (known & ~strictly_past).sum()
                ),
                "max_train_signal_date": pd.Timestamp(maximum_signal),
                "max_train_target_date": pd.Timestamp(maximum_target),
                "target_strictly_before_test": strict,
                "test_rows": int(len(test)),
            }
        )
    audit = pd.DataFrame(audit_rows)
    if len(audit) != len(folds) or not bool(audit["target_strictly_before_test"].all()):
        raise RuntimeError(
            f"incomplete horizon-purge audit for {module}|{lens}|{outcome}|H{horizon}"
        )
    return outputs, audit


def _industry_matrix(values: pd.Series, categories: Sequence[str]) -> np.ndarray:
    lookup = {category: index for index, category in enumerate(categories)}
    codes = values.fillna("<MISSING>").astype(str).map(lookup).fillna(-1).to_numpy(int)
    if len(categories) <= 1:
        return np.empty((len(values), 0), dtype=np.float32)
    matrix = np.zeros((len(values), len(categories) - 1), dtype=np.float32)
    valid = codes > 0
    matrix[np.flatnonzero(valid), codes[valid] - 1] = 1.0
    return matrix


def control_matrices(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    numeric_columns: Sequence[str],
    industry_column: str | None = "sw1_industry",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_numeric = (
        frame.iloc[train_indices][list(numeric_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    test_numeric = (
        frame.iloc[test_indices][list(numeric_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    # Convert columns separately to handle the all-NaN pre-2022 S history.
    # Training columns with no finite values receive a zero median below.
    medians = np.asarray(
        [
            (
                np.nanmedian(column[np.isfinite(column)])
                if np.isfinite(column).any()
                else 0.0
            )
            for column in train_numeric.T
        ],
        dtype=float,
    )
    medians[~np.isfinite(medians)] = 0.0
    train_missing = ~np.isfinite(train_numeric)
    test_missing = ~np.isfinite(test_numeric)
    train_numeric = np.where(train_missing, medians, train_numeric)
    test_numeric = np.where(test_missing, medians, test_numeric)
    means = train_numeric.mean(axis=0)
    scales = train_numeric.std(axis=0, ddof=0)
    scales[~np.isfinite(scales) | (scales < 1e-10)] = 1.0
    train_numeric = ((train_numeric - means) / scales).astype(np.float32)
    test_numeric = ((test_numeric - means) / scales).astype(np.float32)
    active_missing = train_missing.any(axis=0) | test_missing.any(axis=0)
    train_parts = [train_numeric]
    test_parts = [test_numeric]
    if active_missing.any():
        train_parts.append(train_missing[:, active_missing].astype(np.float32))
        test_parts.append(test_missing[:, active_missing].astype(np.float32))
    industry_categories: list[str] = []
    if industry_column and industry_column in frame.columns:
        industry_categories = sorted(
            frame[industry_column].fillna("<MISSING>").astype(str).unique()
        )
        train_parts.append(
            _industry_matrix(
                frame.iloc[train_indices][industry_column], industry_categories
            )
        )
        test_parts.append(
            _industry_matrix(
                frame.iloc[test_indices][industry_column], industry_categories
            )
        )
    train = np.column_stack(train_parts).astype(np.float64, copy=False)
    test = np.column_stack(test_parts).astype(np.float64, copy=False)
    # Explicit intercept; it is not penalized in ridge_fit_predict.
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    meta = {
        "numeric_columns": list(numeric_columns),
        "industry_column": industry_column,
        "industry_categories": industry_categories,
        "matrix_columns": train.shape[1],
    }
    return train, test, meta


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float = RIDGE_ALPHA,
) -> np.ndarray:
    y = np.asarray(y_train, dtype=float)
    penalty = np.eye(x_train.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    gram = x_train.T @ x_train + penalty
    rhs = x_train.T @ y
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram, rcond=1e-10) @ rhs
    return x_test @ coefficients


def crossfit_residual(
    frame: pd.DataFrame,
    target: np.ndarray,
    folds: list[tuple[int, np.ndarray, np.ndarray]],
    numeric_controls: Sequence[str],
    industry_column: str | None = "sw1_industry",
    maximum_train: int = MAX_NUISANCE_TRAIN,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    target = np.asarray(target, dtype=float)
    residual = np.full(len(frame), np.nan, dtype=np.float32)
    diagnostics: list[dict[str, object]] = []
    for fold, train_all, test_all in folds:
        train_valid = train_all[np.isfinite(target[train_all])]
        test_valid = test_all[np.isfinite(target[test_all])]
        if len(train_valid) < 1_000 or len(test_valid) < 100:
            diagnostics.append(
                {
                    "fold_id": fold,
                    "train_rows": len(train_valid),
                    "test_rows": len(test_valid),
                    "r2": math.nan,
                    "status": "insufficient",
                }
            )
            continue
        train_fit = deterministic_thin(train_valid, maximum_train)
        x_train, x_test, meta = control_matrices(
            frame,
            train_fit,
            test_valid,
            numeric_controls,
            industry_column=industry_column,
        )
        prediction = ridge_fit_predict(x_train, target[train_fit], x_test)
        residual[test_valid] = (target[test_valid] - prediction).astype(np.float32)
        denominator = float(
            np.square(target[test_valid] - np.nanmean(target[test_valid])).sum()
        )
        r2 = (
            1.0 - float(np.square(target[test_valid] - prediction).sum()) / denominator
            if denominator > 0
            else math.nan
        )
        diagnostics.append(
            {
                "fold_id": fold,
                "train_eligible_rows": len(train_valid),
                "train_fit_rows": len(train_fit),
                "test_rows": len(test_valid),
                "r2": r2,
                "matrix_columns": meta["matrix_columns"],
                "status": "estimated",
            }
        )
    return residual, diagnostics


def within_date_standardized_dose(
    values: pd.Series,
    dates: pd.Series,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, float, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    central = numeric.where(numeric.between(lower, upper, inclusive="both"))
    centered = central - central.groupby(
        pd.to_datetime(dates), observed=True
    ).transform("mean")
    scale = float(centered.std(ddof=0))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.full(len(values), np.nan), math.nan, int(central.notna().sum())
    return (centered / scale).to_numpy(float), scale, int(central.notna().sum())


def hac_meat(scores: np.ndarray, lag: int) -> np.ndarray:
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    matrix = matrix - np.nanmean(matrix, axis=0, keepdims=True)
    meat = matrix.T @ matrix
    maximum = min(int(lag), len(matrix) - 1)
    for distance in range(1, maximum + 1):
        weight = 1.0 - distance / (maximum + 1.0)
        cross = matrix[distance:].T @ matrix[:-distance]
        meat += weight * (cross + cross.T)
    return meat


def orthogonal_univariate(
    dose_residual: np.ndarray,
    outcome_residual: np.ndarray,
    dates: pd.Series,
    hac_lag: int,
) -> dict[str, object]:
    d = np.asarray(dose_residual, dtype=float)
    y = np.asarray(outcome_residual, dtype=float)
    date_values = pd.to_datetime(dates)
    valid = np.isfinite(d) & np.isfinite(y)
    d = d[valid]
    y = y[valid]
    clustered_dates = date_values[valid]
    denominator = float(np.dot(d, d))
    if len(d) < 100 or denominator <= 1e-12:
        return {
            "coefficient": math.nan,
            "se_hac": math.nan,
            "n_rows": len(d),
            "n_dates": clustered_dates.nunique(),
        }
    coefficient = float(np.dot(d, y) / denominator)
    residual = y - coefficient * d
    row_score = d * residual
    daily_score = (
        pd.Series(row_score).groupby(clustered_dates.to_numpy(), sort=True).sum()
    )
    meat = float(hac_meat(daily_score.to_numpy(float), hac_lag)[0, 0])
    se = math.sqrt(max(meat, 0.0)) / denominator
    t_value = coefficient / se if se > 0 else math.nan
    p_value = normal_two_sided_p(t_value)
    effective_n = (
        denominator**2 / float(np.power(d, 4).sum())
        if np.power(d, 4).sum() > 0
        else math.nan
    )
    return {
        "coefficient": coefficient,
        "se_hac": se,
        "ci95_low": coefficient - 1.959963984540054 * se,
        "ci95_high": coefficient + 1.959963984540054 * se,
        "t_hac": t_value,
        "p_two_sided_hac": p_value,
        "n_rows": len(d),
        "n_dates": int(clustered_dates.nunique()),
        "hac_lag_sample_dates": int(hac_lag),
        "post_nuisance_effective_n": effective_n,
        "dose_residual_variance": float(np.var(d)),
    }


def orthogonal_multivariate(
    dose_residuals: np.ndarray,
    outcome_residual: np.ndarray,
    dates: pd.Series,
    hac_lag: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    d = np.asarray(dose_residuals, dtype=float)
    y = np.asarray(outcome_residual, dtype=float)
    date_values = pd.to_datetime(dates)
    valid = np.isfinite(y) & np.isfinite(d).all(axis=1)
    d = d[valid]
    y = y[valid]
    clustered_dates = date_values[valid]
    bread_inverse = d.T @ d
    bread = np.linalg.pinv(bread_inverse, rcond=1e-10)
    coefficient = bread @ (d.T @ y)
    residual = y - d @ coefficient
    row_scores = d * residual[:, None]
    daily_scores = (
        pd.DataFrame(row_scores)
        .groupby(clustered_dates.to_numpy(), sort=True)
        .sum()
        .to_numpy(float)
    )
    meat = hac_meat(daily_scores, hac_lag)
    covariance = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return (
        coefficient,
        se,
        {
            "n_rows": len(y),
            "n_dates": int(clustered_dates.nunique()),
            "hac_lag_sample_dates": int(hac_lag),
            "design_condition_number": float(np.linalg.cond(bread_inverse)),
        },
    )


def add_multiplicity(
    frame: pd.DataFrame,
    family_columns: Sequence[str],
    p_column: str = "p_two_sided_hac",
) -> pd.DataFrame:
    output = frame.copy()
    output["p_holm_family"] = np.nan
    output["q_bh_family"] = np.nan
    output["family_n"] = 0
    for _, indices in output.groupby(
        list(family_columns), dropna=False, sort=True
    ).groups.items():
        index = list(indices)
        p = pd.to_numeric(output.loc[index, p_column], errors="coerce").to_numpy(float)
        valid = np.isfinite(p)
        if not valid.any():
            continue
        output.loc[np.asarray(index)[valid], "p_holm_family"] = multipletests(
            p[valid], method="holm"
        )[1]
        output.loc[np.asarray(index)[valid], "q_bh_family"] = multipletests(
            p[valid], method="fdr_bh"
        )[1]
        output.loc[index, "family_n"] = int(valid.sum())
    return output


def add_multiplicity_fail_closed(
    frame: pd.DataFrame,
    family_columns: Sequence[str],
    *,
    status_column: str = "estimability_status",
    complete_label: str = "COMPLETE",
    p_column: str = "p_two_sided_hac",
) -> pd.DataFrame:
    """Adjust a frozen family while suppressing inference for deferred cells.

    Deferred hypotheses remain in their pre-declared family as working p=1,
    so the multiple-testing family never shrinks.  Their pre-gate estimates
    are retained only under ``diagnostic_*`` names; formally named standard
    errors, intervals, test statistics and p values are blanked afterwards.
    """
    output = frame.copy()
    complete = output[status_column].eq(complete_label)
    diagnostic_fields = (
        "coefficient",
        "se_hac",
        "ci95_low",
        "ci95_high",
        "t_hac",
        p_column,
    )
    for column in diagnostic_fields:
        if column in output:
            output[f"diagnostic_{column}"] = output[column].where(~complete)
    working = "_p_for_multiplicity"
    output[working] = pd.to_numeric(output[p_column], errors="coerce").where(
        complete, 1.0
    )
    output = add_multiplicity(output, family_columns, p_column=working)
    output["family_n_complete"] = output.groupby(
        list(family_columns), dropna=False, sort=False
    )[status_column].transform(lambda value: int(value.eq(complete_label).sum()))
    output["deferred_multiplicity_rule"] = (
        "deferred hypotheses enter adjustment as p=1; formal inference blank"
    )
    formal_fields = (
        "coefficient",
        "se_hac",
        "ci95_low",
        "ci95_high",
        "t_hac",
        p_column,
        "p_holm_family",
        "q_bh_family",
    )
    for column in formal_fields:
        if column in output:
            output.loc[~complete, column] = np.nan
    output.drop(columns=[working], inplace=True)
    return output


def horizon_hac_lag(horizon: int, stride: int = 5) -> int:
    # The sampled observations are spaced five eligible market dates apart.
    # Using ceil(H/stride), rather than treating them as adjacent daily rows,
    # preserves at least the original overlap horizon requested by the design.
    return max(1, int(math.ceil(horizon / stride)))
