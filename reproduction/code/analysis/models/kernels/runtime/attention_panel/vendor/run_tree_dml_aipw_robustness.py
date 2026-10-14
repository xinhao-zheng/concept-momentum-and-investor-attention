#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M2 tree-nuisance DML and low/high-state AIPW robustness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from model_utils import (
    HORIZONS,
    add_multiplicity_fail_closed,
    deterministic_thin,
    fold_masks,
    hac_meat,
    horizon_hac_lag,
    normal_two_sided_p,
    orthogonal_univariate,
    within_date_standardized_dose,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "mechanism"
DATA = ROOT / "data" / "mechanism"
AUDIT = ROOT / "audit" / "mechanism"
PANEL = CACHE / "mechanism_analysis_panel.parquet"
FOLDS = DATA / "mechanism_expanding_time_folds.csv"
FACTORIAL_SUPPORT = DATA / "mechanism_factorial_support.csv"
SEED = 20260714
FEATURES = ("T", "S", "K")
EXPOSURES = tuple(
    [f"own_{x}" for x in FEATURES] + [f"CategoryShare_{x}_loo" for x in FEATURES]
)
STATE_COLUMNS = {
    "T_high_vs_low": "T_delta_pct",
    "S_high_vs_low": "S_level",
    "K_high_vs_low": "K_level",
}
SIMULTANEOUS_Z = NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * len(HORIZONS)))
MAX_TRAIN = 30_000
MAX_AIPW_GROUP_TRAIN = 15_000

CONTROLS = (
    "ret_1",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "log_float_mv_t",
    "log_total_mv_t",
    "log_amount_t",
    "log_amount_20_mean",
    "amount_to_float_mv_20_mean",
    "amihud_20",
    "beta_hs300_60",
    "idio_vol_hs300_60",
    "drawdown_from_60d_high",
    "market_return_t",
    "suspension_days_20",
    "zero_amount_days_20",
    "institution_net_inflow_to_amount_20",
    "large_net_inflow_to_amount_20",
    "retail_net_inflow_to_amount_20",
    "member_hs300_t",
    "member_zz500_t",
    "member_zz1000_t",
    "score_F5_W10",
    "protocol_concept_degree",
    "network_member_n",
    "peer_count_all",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_x(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    numeric = (
        frame[list(CONTROLS)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    )
    numeric[~np.isfinite(numeric)] = np.nan
    industry = (
        frame["sw1_industry"]
        .fillna("<MISSING>")
        .astype("category")
        .cat.codes.to_numpy(np.float32)[:, None]
    )
    return np.column_stack([numeric, industry]), numeric.shape[1]


def regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=80,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        categorical_features=[len(CONTROLS)],
        random_state=seed,
    )


def classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=80,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        categorical_features=[len(CONTROLS)],
        random_state=seed,
    )


def purged_folds(
    frame: pd.DataFrame,
    base: list[tuple[int, np.ndarray, np.ndarray]],
    horizon: int,
    lens: str,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], list[dict[str, object]]]:
    end_column = (
        f"attention_target_date_{horizon}"
        if lens == "alpha"
        else f"open_exit_date_{horizon}"
    )
    end = pd.to_datetime(frame[end_column], errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    output = []
    audit = []
    for fold, train, test in base:
        start = np.datetime64(
            pd.to_datetime(frame.iloc[test]["signal_date"]).min(), "ns"
        )
        keep = (~pd.isna(end[train])) & (end[train] < start)
        clean = train[keep]
        maximum = end[clean].max() if len(clean) else np.datetime64("NaT")
        output.append((fold, clean, test))
        audit.append(
            {
                "holding_h": horizon,
                "fold_id": fold,
                "end_column": end_column,
                "test_start": str(start),
                "max_training_exit": str(maximum),
                "train_rows_before_purge": len(train),
                "train_rows_after_purge": len(clean),
                "target_strictly_before_test": bool(len(clean) and maximum < start),
            }
        )
    return output, audit


def tree_residual(
    x: np.ndarray,
    target: np.ndarray,
    folds: list[tuple[int, np.ndarray, np.ndarray]],
    seed_offset: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    target = np.asarray(target, float)
    residual = np.full(len(target), np.nan, dtype=np.float32)
    diagnostics = []
    for fold, train, test in folds:
        train = train[np.isfinite(target[train])]
        test = test[np.isfinite(target[test])]
        if len(train) < 1_000 or len(test) < 100:
            diagnostics.append(
                {
                    "fold_id": fold,
                    "status": "insufficient",
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
            )
            continue
        fit = deterministic_thin(train, MAX_TRAIN)
        model = regressor(SEED + seed_offset + fold)
        model.fit(x[fit], target[fit])
        prediction = model.predict(x[test])
        residual[test] = (target[test] - prediction).astype(np.float32)
        denominator = float(np.square(target[test] - np.mean(target[test])).sum())
        r2 = (
            1.0 - float(np.square(target[test] - prediction).sum()) / denominator
            if denominator > 0
            else math.nan
        )
        diagnostics.append(
            {
                "fold_id": fold,
                "status": "estimated",
                "train_rows": len(train),
                "train_fit_rows": len(fit),
                "test_rows": len(test),
                "r2": r2,
            }
        )
    return residual, diagnostics


def build_states(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    states: dict[str, np.ndarray] = {}
    component = {}
    for name, column in STATE_COLUMNS.items():
        # Reproduce the frozen factorial implementation exactly.  These
        # percentile fields are stored as float32; promoting them to float64
        # before comparing with Python's 1/3 moves the exact float32 boundary
        # just above 1/3 and incorrectly drops 21 frozen-support rows.
        values = pd.to_numeric(frame[column], errors="coerce")
        treatment = np.full(len(frame), np.nan)
        treatment[(values <= 1 / 3).to_numpy()] = 0.0
        treatment[(values >= 2 / 3).to_numpy()] = 1.0
        states[name] = treatment
        component[name[0]] = treatment
    joint = np.full(len(frame), np.nan)
    all_low = np.column_stack([component[x] == 0 for x in FEATURES]).all(axis=1)
    all_high = np.column_stack([component[x] == 1 for x in FEATURES]).all(axis=1)
    joint[all_low] = 0.0
    joint[all_high] = 1.0
    states["TSK_all_high_vs_all_low"] = joint
    return states


def state_alignment_checks(
    frame: pd.DataFrame, states: dict[str, np.ndarray], lens: str
) -> dict[str, object]:
    mismatches = 0
    components: dict[str, np.ndarray] = {}
    for name, column in STATE_COLUMNS.items():
        value = pd.to_numeric(frame[column], errors="coerce")
        reference = np.full(len(frame), np.nan)
        reference[(value <= 1 / 3).to_numpy()] = 0.0
        reference[(value >= 2 / 3).to_numpy()] = 1.0
        observed = states[name]
        mismatches += int(
            (
                ~((np.isnan(reference) & np.isnan(observed)) | (reference == observed))
            ).sum()
        )
        components[name[0]] = observed
    eligible = np.isfinite(np.column_stack([components[x] for x in FEATURES])).all(
        axis=1
    )
    labels = np.array(["" for _ in range(len(frame))], dtype=object)
    labels[eligible] = [
        f"T{int(t)}S{int(s)}K{int(k)}"
        for t, s, k in zip(
            components["T"][eligible],
            components["S"][eligible],
            components["K"][eligible],
        )
    ]
    actual = pd.Series(labels[eligible]).value_counts().to_dict()
    expected_frame = pd.read_csv(FACTORIAL_SUPPORT)
    expected_frame = expected_frame[expected_frame["lens"].eq(lens)]
    expected = expected_frame.set_index("factorial_cell")["rows"].astype(int).to_dict()
    cells = sorted(set(actual) | set(expected))
    cell_mismatches = {
        cell: {
            "actual": int(actual.get(cell, 0)),
            "expected": int(expected.get(cell, 0)),
        }
        for cell in cells
        if int(actual.get(cell, 0)) != int(expected.get(cell, 0))
    }
    joint = states["TSK_all_high_vs_all_low"]
    joint_reference = np.full(len(frame), np.nan)
    joint_reference[labels == "T0S0K0"] = 0.0
    joint_reference[labels == "T1S1K1"] = 1.0
    joint_mismatches = int(
        (
            ~(
                (np.isnan(joint_reference) & np.isnan(joint))
                | (joint_reference == joint)
            )
        ).sum()
    )
    return {
        "individual_state_row_mismatches": mismatches,
        "joint_state_row_mismatches": joint_mismatches,
        "factorial_cell_count_mismatches": cell_mismatches,
        "factorial_extreme_rows": int(eligible.sum()),
    }


def aipw_estimate(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    dates: pd.Series,
    folds: list[tuple[int, np.ndarray, np.ndarray]],
    horizon: int,
    seed_offset: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    psi = np.full(len(y), np.nan)
    diagnostics = []
    for fold, train, test in folds:
        train = train[np.isfinite(y[train]) & np.isfinite(w[train])]
        test = test[np.isfinite(y[test]) & np.isfinite(w[test])]
        train0 = train[w[train] == 0]
        train1 = train[w[train] == 1]
        test0 = test[w[test] == 0]
        test1 = test[w[test] == 1]
        if min(len(train0), len(train1), len(test0), len(test1)) < 1_000:
            diagnostics.append(
                {
                    "fold_id": fold,
                    "status": "insufficient",
                    "train0": len(train0),
                    "train1": len(train1),
                    "test0": len(test0),
                    "test1": len(test1),
                    "test_rows": len(test),
                }
            )
            continue
        fit0 = deterministic_thin(train0, MAX_AIPW_GROUP_TRAIN)
        fit1 = deterministic_thin(train1, MAX_AIPW_GROUP_TRAIN)
        fit = np.sort(np.concatenate([fit0, fit1]))
        # Group-specific capping must not turn the propensity training prior
        # into an artificial 50/50 sample.  Restore each original group mass
        # with inverse sampling-fraction weights.
        propensity_weight = np.where(
            w[fit] == 1,
            len(train1) / max(len(fit1), 1),
            len(train0) / max(len(fit0), 1),
        )
        propensity = classifier(SEED + seed_offset + fold)
        outcome0 = regressor(SEED + seed_offset + 100 + fold)
        outcome1 = regressor(SEED + seed_offset + 200 + fold)
        propensity.fit(x[fit], w[fit].astype(int), sample_weight=propensity_weight)
        outcome0.fit(x[fit0], y[fit0])
        outcome1.fit(x[fit1], y[fit1])
        raw_e = propensity.predict_proba(x[test])[:, 1]
        e = np.clip(raw_e, 0.05, 0.95)
        mu0 = outcome0.predict(x[test])
        mu1 = outcome1.predict(x[test])
        wt = w[test]
        yt = y[test]
        psi[test] = (
            mu1 - mu0 + wt * (yt - mu1) / e - (1.0 - wt) * (yt - mu0) / (1.0 - e)
        )
        diagnostics.append(
            {
                "fold_id": fold,
                "status": "estimated",
                "train0": len(train0),
                "train1": len(train1),
                "fit0": len(fit0),
                "fit1": len(fit1),
                "test0": len(test0),
                "test1": len(test1),
                "test_rows": len(test),
                "original_treated_fraction": float(len(train1) / len(train)),
                "sampled_treated_fraction": float(len(fit1) / len(fit)),
                "weighted_treated_fraction": float(
                    propensity_weight[w[fit] == 1].sum() / propensity_weight.sum()
                ),
                "weighted_prior_restore_error": float(
                    abs(
                        propensity_weight[w[fit] == 1].sum() / propensity_weight.sum()
                        - len(train1) / len(train)
                    )
                ),
                "propensity_clipped_fraction": float(
                    ((raw_e < 0.05) | (raw_e > 0.95)).mean()
                ),
                "propensity_clipped_rows": int(((raw_e < 0.05) | (raw_e > 0.95)).sum()),
                "propensity_p01": float(np.quantile(raw_e, 0.01)),
                "propensity_p99": float(np.quantile(raw_e, 0.99)),
            }
        )
    valid = np.isfinite(psi)
    values = psi[valid]
    date_values = pd.to_datetime(dates)[valid]
    coefficient = float(values.mean()) if len(values) else math.nan
    daily_score = (
        pd.Series(values - coefficient).groupby(date_values.to_numpy(), sort=True).sum()
    )
    meat = (
        float(hac_meat(daily_score.to_numpy(float), horizon_hac_lag(horizon))[0, 0])
        if len(daily_score)
        else math.nan
    )
    se = (
        math.sqrt(max(meat, 0.0)) / len(values)
        if len(values) and np.isfinite(meat)
        else math.nan
    )
    t = coefficient / se if se > 0 else math.nan
    estimated_diagnostics = [
        row for row in diagnostics if row.get("status") == "estimated"
    ]
    clipped = [row["propensity_clipped_fraction"] for row in estimated_diagnostics]
    total_test = sum(int(row["test_rows"]) for row in estimated_diagnostics)
    total_clipped = sum(
        int(row["propensity_clipped_rows"]) for row in estimated_diagnostics
    )
    estimate = {
        "coefficient": coefficient,
        "se_hac": se,
        "ci95_low": coefficient - 1.959963984540054 * se,
        "ci95_high": coefficient + 1.959963984540054 * se,
        "t_hac": t,
        "p_two_sided_hac": normal_two_sided_p(t),
        "n_rows": int(valid.sum()),
        "n_dates": int(date_values.nunique()),
        "hac_lag_sample_dates": horizon_hac_lag(horizon),
        "pooled_propensity_clipped_fraction": (
            float(total_clipped / total_test) if total_test else math.nan
        ),
        "max_fold_propensity_clipped_fraction": (
            float(np.max(clipped)) if clipped else math.nan
        ),
    }
    return estimate, diagnostics


def run_lens(lens: str) -> None:
    started = time.time()
    columns = {
        "lens",
        "signal_date",
        "stock_code",
        "sw1_industry",
        *CONTROLS,
        *EXPOSURES,
        *STATE_COLUMNS.values(),
        *[f"price_excess_{h}" for h in HORIZONS],
        *[f"open_exit_date_{h}" for h in HORIZONS],
        *[f"attention_target_date_{h}" for h in HORIZONS],
    }
    frame = pd.read_parquet(
        PANEL, columns=sorted(columns), filters=[("lens", "==", lens)]
    )
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame.sort_values(["signal_date", "stock_code"], inplace=True, kind="mergesort")
    frame.reset_index(drop=True, inplace=True)
    x, _ = model_x(frame)
    base_folds = fold_masks(frame["signal_date"], pd.read_csv(FOLDS))
    nuisance_rows = []
    purge_rows = []

    dose_residual: dict[str, np.ndarray] = {}
    dose_folds_complete: dict[str, bool] = {}
    for index, exposure in enumerate(EXPOSURES):
        values = pd.to_numeric(frame[exposure], errors="coerce")
        finite = values[np.isfinite(values)]
        p01, p99 = float(finite.quantile(0.01)), float(finite.quantile(0.99))
        dose, scale, central = within_date_standardized_dose(
            values, frame["signal_date"], p01, p99
        )
        residual, diagnostics = tree_residual(x, dose, base_folds, 1_000 * index)
        dose_residual[exposure] = residual
        dose_folds_complete[exposure] = len(diagnostics) == len(base_folds) and all(
            row.get("status") == "estimated" for row in diagnostics
        )
        for row in diagnostics:
            nuisance_rows.append(
                {
                    "lens": lens,
                    "module": "tree_dml",
                    "role": "dose",
                    "variable": exposure,
                    "holding_h": 0,
                    "dose_scale": scale,
                    "central_rows": central,
                    **row,
                }
            )

    outcome_residual = {}
    outcome_folds_complete = {}
    purged_by_h = {}
    for horizon in HORIZONS:
        folds, audit = purged_folds(frame, base_folds, horizon, lens)
        purged_by_h[horizon] = folds
        for row in audit:
            purge_rows.append({"lens": lens, "module": "tree_dml_and_aipw", **row})
        target = pd.to_numeric(
            frame[f"price_excess_{horizon}"], errors="coerce"
        ).to_numpy(float)
        residual, diagnostics = tree_residual(x, target, folds, 100_000 + horizon)
        outcome_residual[horizon] = residual
        outcome_folds_complete[horizon] = len(diagnostics) == len(folds) and all(
            row.get("status") == "estimated" for row in diagnostics
        )
        for row in diagnostics:
            nuisance_rows.append(
                {
                    "lens": lens,
                    "module": "tree_dml",
                    "role": "outcome",
                    "variable": "price_excess",
                    "holding_h": horizon,
                    **row,
                }
            )

    dml_rows = []
    for exposure in EXPOSURES:
        group = "own" if exposure.startswith("own_") else "CategoryShare"
        for horizon in HORIZONS:
            estimate = orthogonal_univariate(
                dose_residual[exposure],
                outcome_residual[horizon],
                frame["signal_date"],
                horizon_hac_lag(horizon),
            )
            dml_rows.append(
                {
                    "lens": lens,
                    "exposure_group": group,
                    "exposure": exposure,
                    "holding_h": horizon,
                    **estimate,
                    "estimability_status": (
                        "COMPLETE"
                        if int(estimate["n_rows"]) >= 10_000
                        and int(estimate["n_dates"]) >= 100
                        and dose_folds_complete[exposure]
                        and outcome_folds_complete[horizon]
                        else "DEFERRED_INCOMPLETE_TIME_FOLDS"
                    ),
                    "learner": "HistGradientBoosting nuisance + orthogonal linear score",
                    "observational_status": "tree-nuisance robustness under measured covariates; not exogenous causality",
                }
            )
    dml = pd.DataFrame(dml_rows)
    dml["multiplicity_family"] = (
        "tree_dml_3features_x_8H|" + dml["lens"] + "|" + dml["exposure_group"]
    )
    dml = add_multiplicity_fail_closed(dml, ["multiplicity_family"])
    dml["simultaneous_method"] = "Bonferroni 95% over 8 frozen horizons"
    dml_complete_mask = dml["estimability_status"].eq("COMPLETE")
    dml["simultaneous_z"] = np.where(dml_complete_mask, SIMULTANEOUS_Z, np.nan)
    dml["simul95_low"] = dml["coefficient"] - SIMULTANEOUS_Z * dml["se_hac"]
    dml["simul95_high"] = dml["coefficient"] + SIMULTANEOUS_Z * dml["se_hac"]
    dml["diagnostic_simul95_low"] = (
        dml["diagnostic_coefficient"] - SIMULTANEOUS_Z * dml["diagnostic_se_hac"]
    )
    dml["diagnostic_simul95_high"] = (
        dml["diagnostic_coefficient"] + SIMULTANEOUS_Z * dml["diagnostic_se_hac"]
    )

    states = build_states(frame)
    state_alignment = state_alignment_checks(frame, states, lens)
    aipw_rows = []
    for state_index, (state, treatment) in enumerate(states.items()):
        for horizon in HORIZONS:
            y = pd.to_numeric(
                frame[f"price_excess_{horizon}"], errors="coerce"
            ).to_numpy(float)
            estimate, diagnostics = aipw_estimate(
                x,
                y,
                treatment,
                frame["signal_date"],
                purged_by_h[horizon],
                horizon,
                1_000_000 + state_index * 10_000 + horizon,
            )
            for row in diagnostics:
                nuisance_rows.append(
                    {
                        "lens": lens,
                        "module": "tree_aipw",
                        "role": "joint_nuisance",
                        "variable": state,
                        "holding_h": horizon,
                        **row,
                    }
                )
            support_gate = (
                int(estimate["n_rows"]) >= 10_000 and int(estimate["n_dates"]) >= 100
            )
            overlap_gate = (
                np.isfinite(estimate["pooled_propensity_clipped_fraction"])
                and estimate["pooled_propensity_clipped_fraction"] <= 0.20
            )
            time_fold_gate = len(diagnostics) == len(purged_by_h[horizon]) and all(
                row.get("status") == "estimated" for row in diagnostics
            )
            complete = support_gate and overlap_gate and time_fold_gate
            aipw_rows.append(
                {
                    "lens": lens,
                    "state_contrast": state,
                    "holding_h": horizon,
                    **estimate,
                    "support_gate_pass": support_gate,
                    "overlap_gate_pass": overlap_gate,
                    "all_time_folds_pass": time_fold_gate,
                    "estimability_status": (
                        "COMPLETE"
                        if complete
                        else "DEFERRED_SUPPORT_OVERLAP_OR_TIME_FOLD_GATE"
                    ),
                    "learner": "HistGradientBoosting propensity and outcome regressions; cross-fitted AIPW",
                    "observational_status": "high-low state contrast under measured covariates; not exogenous causality",
                }
            )
    aipw = pd.DataFrame(aipw_rows)
    aipw["multiplicity_family"] = "tree_aipw_4states_x_8H|" + aipw["lens"]
    aipw = add_multiplicity_fail_closed(aipw, ["multiplicity_family"])
    aipw["simultaneous_method"] = "Bonferroni 95% over 8 frozen horizons"
    aipw_complete_mask = aipw["estimability_status"].eq("COMPLETE")
    aipw["simultaneous_z"] = np.where(aipw_complete_mask, SIMULTANEOUS_Z, np.nan)
    aipw["simul95_low"] = aipw["coefficient"] - SIMULTANEOUS_Z * aipw["se_hac"]
    aipw["simul95_high"] = aipw["coefficient"] + SIMULTANEOUS_Z * aipw["se_hac"]
    aipw["diagnostic_simul95_low"] = (
        aipw["diagnostic_coefficient"] - SIMULTANEOUS_Z * aipw["diagnostic_se_hac"]
    )
    aipw["diagnostic_simul95_high"] = (
        aipw["diagnostic_coefficient"] + SIMULTANEOUS_Z * aipw["diagnostic_se_hac"]
    )

    dml_path = DATA / f"mechanism_tree_dml_{lens}.csv"
    aipw_path = DATA / f"mechanism_tree_aipw_{lens}.csv"
    nuisance_path = DATA / f"mechanism_tree_nuisance_{lens}.csv"
    purge_path = DATA / f"mechanism_tree_purge_{lens}.csv"
    dml.to_csv(dml_path, index=False, encoding="utf-8-sig")
    aipw.to_csv(aipw_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(nuisance_rows).to_csv(nuisance_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(purge_rows).drop_duplicates().to_csv(
        purge_path, index=False, encoding="utf-8-sig"
    )
    checks = {
        "dml_rows": len(dml),
        "expected_dml_rows": 48,
        "aipw_rows": len(aipw),
        "expected_aipw_rows": 32,
        "dml_complete": int(dml["estimability_status"].eq("COMPLETE").sum()),
        "aipw_complete": int(aipw["estimability_status"].eq("COMPLETE").sum()),
        "dml_deferred": int((~dml["estimability_status"].eq("COMPLETE")).sum()),
        "aipw_deferred": int((~aipw["estimability_status"].eq("COMPLETE")).sum()),
        "positive_p_complete": int(
            (
                pd.to_numeric(
                    dml.loc[
                        dml["estimability_status"].eq("COMPLETE"), "p_two_sided_hac"
                    ],
                    errors="coerce",
                )
                > 0
            ).sum()
            + (
                pd.to_numeric(
                    aipw.loc[
                        aipw["estimability_status"].eq("COMPLETE"), "p_two_sided_hac"
                    ],
                    errors="coerce",
                )
                > 0
            ).sum()
        ),
        "purge_violations": int(
            (~pd.DataFrame(purge_rows)["target_strictly_before_test"]).sum()
        ),
        "dml_family_sizes": sorted(
            dml.loc[dml["family_n"] > 0, "family_n"].astype(int).unique().tolist()
        ),
        "aipw_family_sizes": sorted(
            aipw.loc[aipw["family_n"] > 0, "family_n"].astype(int).unique().tolist()
        ),
        "dml_family_complete_sizes": sorted(
            dml["family_n_complete"].astype(int).unique().tolist()
        ),
        "aipw_family_complete_sizes": sorted(
            aipw["family_n_complete"].astype(int).unique().tolist()
        ),
        "deferred_formal_inference_nonmissing": int(
            dml.loc[
                ~dml_complete_mask,
                [
                    "coefficient",
                    "se_hac",
                    "ci95_low",
                    "ci95_high",
                    "t_hac",
                    "p_two_sided_hac",
                    "p_holm_family",
                    "q_bh_family",
                    "simultaneous_z",
                    "simul95_low",
                    "simul95_high",
                ],
            ]
            .notna()
            .sum()
            .sum()
            + aipw.loc[
                ~aipw_complete_mask,
                [
                    "coefficient",
                    "se_hac",
                    "ci95_low",
                    "ci95_high",
                    "t_hac",
                    "p_two_sided_hac",
                    "p_holm_family",
                    "q_bh_family",
                    "simultaneous_z",
                    "simul95_low",
                    "simul95_high",
                ],
            ]
            .notna()
            .sum()
            .sum()
        ),
        "temporary_multiplicity_columns": [
            column for column in list(dml) + list(aipw) if column.startswith("_p_for_")
        ],
        "max_weighted_prior_restore_error": float(
            pd.to_numeric(
                pd.DataFrame(nuisance_rows).loc[
                    lambda value: value["module"].eq("tree_aipw"),
                    "weighted_prior_restore_error",
                ],
                errors="coerce",
            ).max()
        ),
        "max_pooled_propensity_clipped_fraction": float(
            pd.to_numeric(
                aipw["pooled_propensity_clipped_fraction"], errors="coerce"
            ).max()
        ),
        "max_single_fold_propensity_clipped_fraction": float(
            pd.to_numeric(
                aipw["max_fold_propensity_clipped_fraction"], errors="coerce"
            ).max()
        ),
        **state_alignment,
    }
    hard_gates = {
        "complete_tree_dml_grid": checks["dml_rows"] == checks["expected_dml_rows"],
        "complete_tree_aipw_grid": checks["aipw_rows"] == checks["expected_aipw_rows"],
        "all_cells_classified": checks["dml_complete"] + checks["dml_deferred"] == 48
        and checks["aipw_complete"] + checks["aipw_deferred"] == 32,
        "all_complete_cells_have_positive_p": checks["positive_p_complete"]
        == checks["dml_complete"] + checks["aipw_complete"],
        "aipw_sampling_weight_restores_treatment_prior": checks[
            "max_weighted_prior_restore_error"
        ]
        <= 1e-12,
        "aipw_states_match_frozen_factorial_definition": checks[
            "individual_state_row_mismatches"
        ]
        == 0
        and checks["joint_state_row_mismatches"] == 0
        and not checks["factorial_cell_count_mismatches"],
        "aipw_status_matches_support_overlap_and_time_fold_gates": bool(
            (
                aipw["estimability_status"].eq("COMPLETE")
                == (
                    aipw["support_gate_pass"].astype(bool)
                    & aipw["overlap_gate_pass"].astype(bool)
                    & aipw["all_time_folds_pass"].astype(bool)
                )
            ).all()
        ),
        "horizon_aware_purge": checks["purge_violations"] == 0,
        "dml_family_is_24": checks["dml_family_sizes"] == [24],
        "aipw_family_is_32": checks["aipw_family_sizes"] == [32],
        "complete_family_counts_are_declared": checks["dml_family_complete_sizes"]
        == [14]
        and checks["aipw_family_complete_sizes"] == [14],
        "deferred_formal_inference_is_fail_closed": checks[
            "deferred_formal_inference_nonmissing"
        ]
        == 0
        and not checks["temporary_multiplicity_columns"],
    }
    if all(hard_gates.values()):
        status = (
            "PASS"
            if checks["dml_complete"] == 48 and checks["aipw_complete"] == 32
            else "PASS_WITH_DEFERRED"
        )
    else:
        status = "FAIL"
    audit = {
        "schema": "mechanism-tree-dml-aipw-1.0",
        "status": status,
        "lens": lens,
        "hard_gates": hard_gates,
        "checks": checks,
        "hyperparameters": {
            "max_iter": 80,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 100,
            "l2": 1.0,
            "max_train": MAX_TRAIN,
            "max_aipw_group_train": MAX_AIPW_GROUP_TRAIN,
        },
        "files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (dml_path, aipw_path, nuisance_path, purge_path)
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (AUDIT / f"mechanism_tree_dml_aipw_{lens}.json").write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        ),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, default=str))
    if status == "FAIL":
        raise SystemExit(1)


def combine() -> None:
    files = {}
    for stem in (
        "mechanism_tree_dml",
        "mechanism_tree_aipw",
        "mechanism_tree_nuisance",
        "mechanism_tree_purge",
    ):
        combined = pd.concat(
            [
                pd.read_csv(DATA / f"{stem}_{lens}.csv")
                for lens in ("alpha", "beta", "gamma")
            ],
            ignore_index=True,
        )
        path = DATA / f"{stem}.csv"
        combined.to_csv(path, index=False, encoding="utf-8-sig")
        files[str(path.relative_to(ROOT))] = sha256(path)
    dml = pd.read_csv(DATA / "mechanism_tree_dml.csv")
    aipw = pd.read_csv(DATA / "mechanism_tree_aipw.csv")
    audits = [
        json.loads(
            (AUDIT / f"mechanism_tree_dml_aipw_{lens}.json").read_text(encoding="utf-8")
        )
        for lens in ("alpha", "beta", "gamma")
    ]
    acceptable = {"PASS", "PASS_WITH_DEFERRED"}
    checks = {
        "dml_rows": len(dml),
        "aipw_rows": len(aipw),
        "component_statuses": [item["status"] for item in audits],
    }
    gates = {
        "all_components_structurally_pass": all(
            item["status"] in acceptable for item in audits
        ),
        "dml_144_rows": len(dml) == 144,
        "aipw_96_rows": len(aipw) == 96,
    }
    if all(gates.values()):
        status = (
            "PASS"
            if all(item["status"] == "PASS" for item in audits)
            else "PASS_WITH_DEFERRED"
        )
    else:
        status = "FAIL"
    audit = {
        "schema": "mechanism-tree-dml-aipw-combined-1.0",
        "status": status,
        "hard_gates": gates,
        "checks": checks,
        "files": files,
    }
    (AUDIT / "mechanism_tree_dml_aipw.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))
    if status == "FAIL":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", choices=("alpha", "beta", "gamma"))
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()
    if args.lens:
        run_lens(args.lens)
    elif args.combine:
        combine()
    else:
        parser.error("provide --lens or --combine")


if __name__ == "__main__":
    main()
