#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Freeze measurement, support, overlap and expanding-time model gates.

This script runs before any nuisance/outcome model is fit.  Gate decisions use
feature availability, exact endpoint availability, central-dose support and
within-date variation, never coefficient signs or p-values.
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
CACHE = ROOT / "cache" / "mechanism"
DATA = ROOT / "data" / "mechanism"
AUDIT = ROOT / "audit" / "mechanism"
PANEL = CACHE / "mechanism_analysis_panel.parquet"
REGISTRY = DATA / "mechanism_support_gate_registry.csv"
MEASUREMENT = DATA / "mechanism_measurement_support.csv"
FACTORIAL = DATA / "mechanism_factorial_support.csv"
FOLDS = DATA / "mechanism_expanding_time_folds.csv"
ALIGNMENT = DATA / "mechanism_beta_gamma_alignment.csv"
CORRELATIONS = DATA / "mechanism_measurement_correlations.csv"
SELECTION_MISSINGNESS = DATA / "mechanism_selection_missingness.csv"
AUDIT_PATH = AUDIT / "mechanism_support_gates.json"

HORIZONS = (1, 3, 5, 10, 20, 60, 120, 250)
LENSES = ("alpha", "beta", "gamma")
EXPOSURES = (
    "own_T",
    "own_S",
    "own_K",
    "G_T_mean",
    "G_S_mean",
    "G_K_mean",
    "G_T_median",
    "G_S_median",
    "G_K_median",
    "CategoryShare_T_loo",
    "CategoryShare_S_loo",
    "CategoryShare_K_loo",
    "WithinShare_T",
    "WithinShare_S",
    "WithinShare_K",
)
BASE_CONTROLS = (
    "ret_1",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "log_float_mv_t",
    "log_amount_20_mean",
    "amihud_20",
    "beta_hs300_60",
    "idio_vol_hs300_60",
    "drawdown_from_60d_high",
)
MIN_ROWS = 50_000
MIN_DATES = 150
MIN_COVERAGE = 0.50
MIN_DATE_VARIATION_SHARE = 0.90
MIN_TAIL_ROWS = 2_000
CENTRAL_LOWER = 0.01
CENTRAL_UPPER = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def make_folds(dates: list[pd.Timestamp]) -> pd.DataFrame:
    n_dates = len(dates)
    initial = int(math.floor(0.40 * n_dates))
    remaining = n_dates - initial
    boundaries = np.linspace(initial, n_dates, 7, dtype=int)
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        fold = -1
        train_end: pd.Timestamp | None = None
        test_start: pd.Timestamp | None = None
        test_end: pd.Timestamp | None = None
        for candidate in range(6):
            start = int(boundaries[candidate])
            stop = int(boundaries[candidate + 1])
            if start <= index < stop:
                fold = candidate
                train_end = dates[start - 1]
                test_start = dates[start]
                test_end = dates[stop - 1]
                break
        rows.append(
            {
                "signal_date": date,
                "sample_date_position": index,
                "market_date_stride": 5,
                "fold_id": fold,
                "training_only_no_oof_prediction": fold < 0,
                "train_end_date_for_fold": train_end,
                "test_start_date_for_fold": test_start,
                "test_end_date_for_fold": test_end,
                "train_uses_only_strictly_earlier_dates": True,
            }
        )
    return pd.DataFrame(rows)


def measurement_support(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, object]]]:
    rows: list[dict[str, object]] = []
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for lens in LENSES:
        subset = panel[panel["lens"].eq(lens)]
        total = len(subset)
        for exposure in EXPOSURES:
            values = pd.to_numeric(subset[exposure], errors="coerce")
            valid = values.notna() & np.isfinite(values)
            observed = subset.loc[valid, ["signal_date", exposure]].copy()
            n = int(valid.sum())
            dates = int(observed["signal_date"].nunique())
            if n:
                quantiles = values[valid].quantile([0.01, 0.10, 0.50, 0.90, 0.99])
                p01, p10, median, p90, p99 = [
                    float(quantiles.loc[q]) for q in (0.01, 0.10, 0.50, 0.90, 0.99)
                ]
                within_sd = observed.groupby("signal_date", observed=True)[
                    exposure
                ].std()
                variation_share = float((within_sd > 1e-10).mean())
                lower_tail = int((values[valid] <= p10).sum())
                upper_tail = int((values[valid] >= p90).sum())
                central_n = int(values[valid].between(p01, p99, inclusive="both").sum())
            else:
                p01 = p10 = median = p90 = p99 = variation_share = math.nan
                lower_tail = upper_tail = central_n = 0
            support_pass = bool(
                n >= MIN_ROWS
                and dates >= MIN_DATES
                and n / total >= MIN_COVERAGE
                and variation_share >= MIN_DATE_VARIATION_SHARE
                and lower_tail >= MIN_TAIL_ROWS
                and upper_tail >= MIN_TAIL_ROWS
                and np.isfinite(p01)
                and np.isfinite(p99)
                and p99 > p01
            )
            record = {
                "lens": lens,
                "exposure": exposure,
                "total_rows": total,
                "observed_rows": n,
                "coverage": n / total if total else math.nan,
                "observed_dates": dates,
                "p01": p01,
                "p10": p10,
                "median": median,
                "p90": p90,
                "p99": p99,
                "central_98pct_rows": central_n,
                "lower_decile_rows": lower_tail,
                "upper_decile_rows": upper_tail,
                "positive_within_date_sd_share": variation_share,
                "support_gate_pass": support_pass,
                "continuous_estimand_support": "central 1st-99th percentile; coefficient scaled per one within-lens SD",
            }
            rows.append(record)
            lookup[(lens, exposure)] = record
    return pd.DataFrame(rows), lookup


def factorial_support(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lens in LENSES:
        subset = panel[panel["lens"].eq(lens)].copy()
        values = {
            "T": pd.to_numeric(subset["T_delta_pct"], errors="coerce"),
            "S": pd.to_numeric(subset["S_level"], errors="coerce"),
            "K": pd.to_numeric(subset["K_level"], errors="coerce"),
        }
        eligible = np.ones(len(subset), dtype=bool)
        for feature, series in values.items():
            eligible &= series.notna().to_numpy()
            eligible &= ((series <= 1 / 3) | (series >= 2 / 3)).to_numpy()
            subset[f"{feature}_high"] = (series >= 2 / 3).astype("Int8")
        extreme = subset.loc[eligible].copy()
        if extreme.empty:
            continue
        extreme["factorial_cell"] = (
            "T"
            + extreme["T_high"].astype(str)
            + "S"
            + extreme["S_high"].astype(str)
            + "K"
            + extreme["K_high"].astype(str)
        )
        for cell, group in extreme.groupby("factorial_cell", observed=True, sort=True):
            rows.append(
                {
                    "lens": lens,
                    "factorial_cell": cell,
                    "rows": len(group),
                    "dates": group["signal_date"].nunique(),
                    "cell_share_within_extreme_sample": len(group) / len(extreme),
                    "threshold_rule": "low <= 1/3; high >= 2/3; middle excluded",
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        lens_gate = result.groupby("lens", observed=True).agg(
            minimum_cell_rows=("rows", "min"),
            minimum_cell_dates=("dates", "min"),
            cells=("factorial_cell", "nunique"),
        )
        lens_gate["factorial_gate_pass"] = (
            (lens_gate["minimum_cell_rows"] >= 1_000)
            & (lens_gate["minimum_cell_dates"] >= 100)
            & (lens_gate["cells"] == 8)
        )
        result = result.merge(lens_gate.reset_index(), on="lens", how="left")
    return result


def beta_gamma_alignment() -> pd.DataFrame:
    beta = pd.read_parquet(
        CACHE / "mechanism_features_beta_preoutcome.parquet",
        columns=[
            "signal_date",
            "stock_code",
            "winning_concept",
            "G_T_mean",
            "G_S_mean",
            "G_K_mean",
            "CategoryShare_T_loo",
            "CategoryShare_S_loo",
            "CategoryShare_K_loo",
        ],
    ).rename(
        columns=lambda value: (
            value if value in {"signal_date", "stock_code"} else f"beta_{value}"
        )
    )
    gamma = pd.read_parquet(
        CACHE / "mechanism_features_gamma_preoutcome.parquet",
        columns=[
            "signal_date",
            "stock_code",
            "winning_concept",
            "G_T_mean",
            "G_S_mean",
            "G_K_mean",
            "CategoryShare_T_loo",
            "CategoryShare_S_loo",
            "CategoryShare_K_loo",
        ],
    ).rename(
        columns=lambda value: (
            value if value in {"signal_date", "stock_code"} else f"gamma_{value}"
        )
    )
    merged = beta.merge(
        gamma, on=["signal_date", "stock_code"], how="inner", validate="one_to_one"
    )
    rows: list[dict[str, object]] = [
        {
            "metric": "winning_concept_exact_match_share",
            "value": float(
                merged["beta_winning_concept"]
                .eq(merged["gamma_winning_concept"])
                .mean()
            ),
            "paired_rows": len(merged),
        }
    ]
    for feature in (
        "G_T_mean",
        "G_S_mean",
        "G_K_mean",
        "CategoryShare_T_loo",
        "CategoryShare_S_loo",
        "CategoryShare_K_loo",
    ):
        paired = merged[[f"beta_{feature}", f"gamma_{feature}"]].dropna()
        rows.append(
            {
                "metric": f"paired_pearson_{feature}",
                "value": (
                    float(paired.corr().iloc[0, 1]) if len(paired) > 2 else math.nan
                ),
                "paired_rows": len(paired),
            }
        )
    return pd.DataFrame(rows)


def measurement_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "T_delta_pct",
        "S_level",
        "K_level",
        "G_T_mean",
        "G_S_mean",
        "G_K_mean",
        "score_F5_W10",
        "ret_1",
        "momentum_20",
        "log_float_mv_t",
        "log_amount_20_mean",
        "amihud_20",
    ]
    rows: list[dict[str, object]] = []
    for lens in LENSES:
        subset = panel[panel["lens"].eq(lens)][
            ["signal_date", "stock_code", *columns]
        ].copy()
        # Deterministic diagnostic thinning, independent of values and outcomes.
        subset = subset.iloc[::10].copy()
        correlation = subset[columns].corr(method="spearman", min_periods=1_000)
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1 :]:
                rows.append(
                    {
                        "lens": lens,
                        "left": left,
                        "right": right,
                        "spearman": correlation.loc[left, right],
                        "diagnostic_rows": len(subset),
                        "sampling_rule": "every 10th row after deterministic lens/date/code sort",
                    }
                )
    return pd.DataFrame(rows)


def selection_missingness(panel: pd.DataFrame) -> pd.DataFrame:
    """Outcome observability by pre-outcome heat group, never by outcome value."""

    rows: list[dict[str, object]] = []
    for lens in LENSES:
        subset = panel[panel["lens"].eq(lens)].copy()
        for exposure in ("own_T", "own_S", "own_K"):
            values = pd.to_numeric(subset[exposure], errors="coerce")
            pct = values.groupby(subset["signal_date"], observed=True).rank(
                pct=True, method="average"
            )
            valid_exposure = values.notna() & np.isfinite(values)
            high = valid_exposure & pct.ge(0.80)
            rest = valid_exposure & ~high
            for horizon in HORIZONS:
                missing = pd.to_numeric(
                    subset[f"price_excess_{horizon}"], errors="coerce"
                ).isna()
                high_rate = float(missing[high].mean()) if high.any() else math.nan
                rest_rate = float(missing[rest].mean()) if rest.any() else math.nan
                rows.append(
                    {
                        "lens": lens,
                        "exposure": exposure,
                        "holding_h": horizon,
                        "high_rule": "within-date top 20% among observed exposure",
                        "high_rows": int(high.sum()),
                        "rest_rows": int(rest.sum()),
                        "high_missing_rate": high_rate,
                        "rest_missing_rate": rest_rate,
                        "high_minus_rest_missing_rate": high_rate - rest_rate,
                        "outcome_values_used": False,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parse_args()
    started = time.time()
    for directory in (DATA, AUDIT):
        directory.mkdir(parents=True, exist_ok=True)
    if not PANEL.exists():
        raise FileNotFoundError(PANEL)
    required = {
        "lens",
        "signal_date",
        "stock_code",
        "market_calendar_index",
        *EXPOSURES,
        *BASE_CONTROLS,
        *[f"price_excess_{h}" for h in HORIZONS],
        "T_delta_pct",
        "S_level",
        "K_level",
        "score_F5_W10",
        "S_delta_5",
        "K_delta",
        "log_T",
        "log_S",
        "log_K",
        *[
            f"future_attention_change_{feature}_{h}"
            for feature in ("T", "S", "K")
            for h in HORIZONS
        ],
    }
    columns = set(pq.ParquetFile(PANEL).schema_arrow.names)
    missing = required - columns
    if missing:
        raise ValueError(f"analysis panel missing support columns {sorted(missing)}")
    panel = pd.read_parquet(PANEL, columns=sorted(required))
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    panel = panel.sort_values(
        ["lens", "signal_date", "stock_code"], kind="mergesort"
    ).reset_index(drop=True)

    dates = sorted(pd.Timestamp(value) for value in panel["signal_date"].unique())
    folds = make_folds(dates)
    folds.to_csv(FOLDS, index=False, encoding="utf-8-sig")
    measurement, support_lookup = measurement_support(panel)
    measurement.to_csv(MEASUREMENT, index=False, encoding="utf-8-sig")
    factorial = factorial_support(panel)
    factorial.to_csv(FACTORIAL, index=False, encoding="utf-8-sig")
    alignment = beta_gamma_alignment()
    alignment.to_csv(ALIGNMENT, index=False, encoding="utf-8-sig")
    correlations = measurement_correlations(panel)
    correlations.to_csv(CORRELATIONS, index=False, encoding="utf-8-sig")
    selection = selection_missingness(panel)
    selection.to_csv(SELECTION_MISSINGNESS, index=False, encoding="utf-8-sig")

    complete_controls = (
        panel[list(BASE_CONTROLS)]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
    )
    registry_rows: list[dict[str, object]] = []
    for lens in LENSES:
        lens_mask = panel["lens"].eq(lens)
        for exposure in EXPOSURES:
            support = support_lookup[(lens, exposure)]
            exposure_values = pd.to_numeric(panel[exposure], errors="coerce")
            for horizon in HORIZONS:
                outcome = pd.to_numeric(
                    panel[f"price_excess_{horizon}"], errors="coerce"
                )
                valid = (
                    lens_mask
                    & complete_controls
                    & exposure_values.notna()
                    & outcome.notna()
                )
                n = int(valid.sum())
                dates_n = int(panel.loc[valid, "signal_date"].nunique())
                pass_gate = bool(
                    support["support_gate_pass"] and n >= MIN_ROWS and dates_n >= 100
                )
                registry_rows.append(
                    {
                        "estimand_family": "continuous_heat_to_price",
                        "lens": lens,
                        "exposure": exposure,
                        "outcome": f"price_excess_{horizon}",
                        "holding_h": horizon,
                        "complete_rows": n,
                        "complete_dates": dates_n,
                        "support_gate_pass": pass_gate,
                        "gate_reason": (
                            "PASS: measurement, tails, within-date variation, controls and endpoint support"
                            if pass_gate
                            else "FAIL: one or more frozen support thresholds not met"
                        ),
                        "central_dose_lower": support["p01"],
                        "central_dose_upper": support["p99"],
                        "gate_frozen_before_model_fit": True,
                    }
                )
    factorial_gate = (
        factorial.groupby("lens", observed=True)["factorial_gate_pass"]
        .first()
        .to_dict()
        if not factorial.empty
        else {}
    )
    for lens in LENSES:
        for horizon in HORIZONS:
            registry_rows.append(
                {
                    "estimand_family": "own_TSK_factorial",
                    "lens": lens,
                    "exposure": "T_high*S_high*K_high_full_2x2x2",
                    "outcome": f"price_excess_{horizon}",
                    "holding_h": horizon,
                    "complete_rows": int(
                        factorial.loc[factorial["lens"].eq(lens), "rows"].sum()
                    ),
                    "complete_dates": int(
                        factorial.loc[factorial["lens"].eq(lens), "dates"].max()
                    ),
                    "support_gate_pass": bool(factorial_gate.get(lens, False)),
                    "gate_reason": "all 8 extreme cells require >=1000 rows and >=100 dates",
                    "central_dose_lower": math.nan,
                    "central_dose_upper": math.nan,
                    "gate_frozen_before_model_fit": True,
                }
            )

    # Direction and stock-flow gates are frozen on the beta-reference feature
    # panel because own T/S/K do not depend on concept membership timing.
    direction = panel[panel["lens"].eq("beta")].copy()
    direction_controls = (
        direction[list(BASE_CONTROLS)]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
    )
    price_exposure = pd.to_numeric(direction["ret_1"], errors="coerce")
    for feature in ("T", "S", "K"):
        for horizon in HORIZONS:
            outcome_name = f"future_attention_change_{feature}_{horizon}"
            attention_outcome = pd.to_numeric(direction[outcome_name], errors="coerce")
            valid = (
                direction_controls & price_exposure.notna() & attention_outcome.notna()
            )
            n = int(valid.sum())
            dates_n = int(direction.loc[valid, "signal_date"].nunique())
            registry_rows.append(
                {
                    "estimand_family": "price_to_future_attention",
                    "lens": "beta_reference",
                    "exposure": "ret_1",
                    "outcome": outcome_name,
                    "holding_h": horizon,
                    "complete_rows": n,
                    "complete_dates": dates_n,
                    "support_gate_pass": bool(n >= MIN_ROWS and dates_n >= 100),
                    "gate_reason": "price and future attention endpoint support only; outcome values not inspected",
                    "central_dose_lower": float(price_exposure.quantile(0.01)),
                    "central_dose_upper": float(price_exposure.quantile(0.99)),
                    "gate_frozen_before_model_fit": True,
                }
            )
    stock_flow_exposure = {"T": "own_T", "S": "S_delta_5", "K": "K_delta"}
    for feature, exposure in stock_flow_exposure.items():
        dose = pd.to_numeric(direction[exposure], errors="coerce")
        for horizon in HORIZONS:
            outcome_name = f"future_attention_change_{feature}_{horizon}"
            attention_outcome = pd.to_numeric(direction[outcome_name], errors="coerce")
            valid = direction_controls & dose.notna() & attention_outcome.notna()
            n = int(valid.sum())
            dates_n = int(direction.loc[valid, "signal_date"].nunique())
            registry_rows.append(
                {
                    "estimand_family": "stock_flow_dynamics",
                    "lens": "beta_reference",
                    "exposure": exposure,
                    "outcome": outcome_name,
                    "holding_h": horizon,
                    "complete_rows": n,
                    "complete_dates": dates_n,
                    "support_gate_pass": bool(n >= MIN_ROWS and dates_n >= 100),
                    "gate_reason": "current innovation plus current level/lags are required in the model; endpoint support frozen here",
                    "central_dose_lower": float(dose.quantile(0.01)),
                    "central_dose_upper": float(dose.quantile(0.99)),
                    "gate_frozen_before_model_fit": True,
                }
            )
    registry = pd.DataFrame(registry_rows)
    registry.to_csv(REGISTRY, index=False, encoding="utf-8-sig")

    checks = {
        "panel_rows": len(panel),
        "dates": len(dates),
        "folds": int(folds.loc[folds["fold_id"] >= 0, "fold_id"].nunique()),
        "fold_chronology_violations": int(
            (
                folds.loc[folds["fold_id"] >= 0, "train_end_date_for_fold"]
                >= folds.loc[folds["fold_id"] >= 0, "test_start_date_for_fold"]
            ).sum()
        ),
        "measurement_rows": len(measurement),
        "continuous_registry_rows": int(
            registry["estimand_family"].eq("continuous_heat_to_price").sum()
        ),
        "continuous_pass_rows": int(
            registry.loc[
                registry["estimand_family"].eq("continuous_heat_to_price"),
                "support_gate_pass",
            ].sum()
        ),
        "factorial_cells": (
            int(factorial["factorial_cell"].nunique()) if not factorial.empty else 0
        ),
        "direction_registry_rows": int(
            registry["estimand_family"].eq("price_to_future_attention").sum()
        ),
        "stock_flow_registry_rows": int(
            registry["estimand_family"].eq("stock_flow_dynamics").sum()
        ),
        "selection_missingness_rows": len(selection),
        "future_outcome_values_used_for_gate": False,
        "outcome_missingness_used_for_gate": True,
    }
    hard_gates = {
        "six_expanding_time_folds": checks["folds"] == 6,
        "strict_train_before_test": checks["fold_chronology_violations"] == 0,
        "all_15_exposures_x_3_lenses_measured": checks["measurement_rows"]
        == len(EXPOSURES) * len(LENSES),
        "continuous_registry_complete_15x3x8": checks["continuous_registry_rows"]
        == len(EXPOSURES) * len(LENSES) * len(HORIZONS),
        "at_least_one_continuous_estimand_passes": checks["continuous_pass_rows"] > 0,
        "all_eight_factorial_cells_registered": checks["factorial_cells"] == 8,
        "direction_registry_complete_3x8": checks["direction_registry_rows"]
        == 3 * len(HORIZONS),
        "stock_flow_registry_complete_3x8": checks["stock_flow_registry_rows"]
        == 3 * len(HORIZONS),
        "selection_missingness_complete_3x3x8": checks["selection_missingness_rows"]
        == 3 * 3 * len(HORIZONS),
        "gate_registry_frozen_before_model_fit": bool(
            registry["gate_frozen_before_model_fit"].all()
        ),
        "coefficient_sign_or_p_used_for_gate": False,
    }
    status = (
        "PASS"
        if all(
            value
            for key, value in hard_gates.items()
            if key != "coefficient_sign_or_p_used_for_gate"
        )
        else "FAIL"
    )
    payload = {
        "schema": "current-mechanism-support-gates-1.0",
        "status": status,
        "hard_gates": hard_gates,
        "checks": checks,
        "thresholds": {
            "minimum_rows": MIN_ROWS,
            "minimum_dates_measurement": MIN_DATES,
            "minimum_dates_endpoint": 100,
            "minimum_coverage": MIN_COVERAGE,
            "minimum_positive_within_date_sd_share": MIN_DATE_VARIATION_SHARE,
            "minimum_each_decile_tail_rows": MIN_TAIL_ROWS,
            "continuous_central_support": [CENTRAL_LOWER, CENTRAL_UPPER],
            "factorial": "T/S/K low <=1/3, high >=2/3; each of 8 cells >=1000 rows and >=100 dates",
        },
        "fold_rule": "40% initial train; six successive 10% OOF test blocks; every training set strictly precedes its test block",
        "sample_spacing_rule": "one sample date per five eligible market dates; HAC lags later use original H converted by ceil(H/5)",
        "outputs": {
            "registry": str(REGISTRY),
            "measurement": str(MEASUREMENT),
            "factorial": str(FACTORIAL),
            "folds": str(FOLDS),
            "beta_gamma_alignment": str(ALIGNMENT),
            "correlations": str(CORRELATIONS),
            "selection_missingness": str(SELECTION_MISSINGNESS),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    AUDIT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": status, "checks": checks, "output": str(REGISTRY)},
            ensure_ascii=False,
            default=str,
        )
    )
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
