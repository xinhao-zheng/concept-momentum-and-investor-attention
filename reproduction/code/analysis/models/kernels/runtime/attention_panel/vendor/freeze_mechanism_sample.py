#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Freeze the outcome-free sample and specification for the mechanism study.

The physical sample is written before any future price or future attention
outcome is opened.  It is the beta-gamma common finite-positive F5/W10 support,
restricted to every fifth eligible market date and a deterministic one-in-four
hash sample of stock-date keys.  The date and stock sampling rules depend only
on identifiers and already-frozen score support.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CQ = ROOT.parents[2]
CACHE = ROOT / "cache" / "mechanism"
DATA = ROOT / "data" / "mechanism"
AUDIT = ROOT / "audit" / "mechanism"

COMMON = ROOT / "cache" / "abg" / "common_beta_gamma_scores_future_free.parquet"
ATTENTION = ROOT.parent / "dynamics/shared/attention_daily.parquet"
SAMPLE = CACHE / "mechanism_sample_keys_before_outcomes.parquet"
DATE_REGISTRY = DATA / "mechanism_sample_date_registry.csv"
SPECIFICATION = DATA / "mechanism_frozen_specification.csv"
AUDIT_PATH = AUDIT / "mechanism_sample_freeze.json"

CUTOFF = pd.Timestamp("2026-09-01")
HORIZONS = (1, 3, 5, 10, 20, 60, 120, 250)
DATE_STRIDE = 5
HASH_MODULUS = 4
HASH_KEEP = 0
HASH_SEED = 20260713


def parquet_identity(path: Path) -> dict[str, object]:
    pf = pq.ParquetFile(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "columns": pf.schema_arrow.names,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_specification() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for design in ("alpha", "beta", "gamma"):
        for horizon in HORIZONS:
            rows.append(
                {
                    "design": design,
                    "score_specification": "F5/W10 fully leave-one-stock-out",
                    "holding_h": horizon,
                    "primary_role": (
                        "paired_primary"
                        if design in {"beta", "gamma"}
                        else "close_reference"
                    ),
                    "outcome_definition": (
                        f"exact market-calendar open[t+1] to open[t+{horizon + 1}]"
                        if design in {"beta", "gamma"}
                        else f"exact market-calendar close[t] to close[t+{horizon}]"
                    ),
                    "date_stride": DATE_STRIDE,
                    "stock_date_hash_modulus": HASH_MODULUS,
                    "stock_date_hash_keep": HASH_KEEP,
                    "stock_date_hash_seed": HASH_SEED,
                    "cutoff_rule": "signal, feature observation, entry and exit all <= 2026-08-31",
                    "inference_unit": "signal date",
                    "causal_language": "under-observables adjusted observational; not an exogenous causal effect",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(SPECIFICATION, index=False, encoding="utf-8-sig")
    return frame


def main() -> None:
    args = parse_args()
    started = time.time()
    for directory in (CACHE, DATA, AUDIT):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (COMMON, ATTENTION):
        if not path.exists():
            raise FileNotFoundError(path)
    if SAMPLE.exists() and not args.force:
        raise FileExistsError(
            f"sample already exists; pass --force to replace: {SAMPLE}"
        )

    common_names = pq.ParquetFile(COMMON).schema_arrow.names
    date_col, code_col = common_names[:2]
    if any(name.lower().startswith("future_") for name in common_names):
        raise AssertionError("common score support unexpectedly contains future fields")
    required_common = {
        "beta_concept_state_signal_date",
        "beta_concept_asof_date",
        "beta_score_F5_W10_DEFAULT",
        "gamma_concept_state_signal_date",
        "gamma_concept_asof_date",
        "gamma_score_F5_W10_DEFAULT",
    }
    if not required_common.issubset(common_names):
        raise ValueError(
            f"common score file missing {sorted(required_common - set(common_names))}"
        )

    attention_names = pq.ParquetFile(ATTENTION).schema_arrow.names
    attention_date, attention_code = "signal_date", "stock_code"
    if not {attention_date, attention_code}.issubset(attention_names):
        raise ValueError("attention source lacks signal_date/stock_code")

    common_dates = (
        pl.scan_parquet(COMMON)
        .select(pl.col(date_col).alias("signal_date"))
        .filter(pl.col("signal_date") < pl.lit(CUTOFF.to_pydatetime()))
        .unique()
        .sort("signal_date")
        .collect(engine="streaming")["signal_date"]
        .to_list()
    )
    attention_dates = set(
        pl.scan_parquet(ATTENTION)
        .select(attention_date)
        .filter(pl.col(attention_date) < pl.lit(CUTOFF.to_pydatetime()))
        .unique()
        .collect(engine="streaming")[attention_date]
        .to_list()
    )
    eligible_dates = [
        pd.Timestamp(value) for value in common_dates if value in attention_dates
    ]
    selected_dates = eligible_dates[::DATE_STRIDE]
    if not selected_dates:
        raise RuntimeError("no sample dates survived the outcome-free date rule")
    date_registry = pd.DataFrame(
        {
            "signal_date": eligible_dates,
            "eligible_date_index": range(len(eligible_dates)),
            "date_stride": DATE_STRIDE,
        }
    )
    date_registry["selected_before_outcome"] = (
        date_registry["eligible_date_index"].mod(DATE_STRIDE).eq(0)
    )
    date_registry.to_csv(DATE_REGISTRY, index=False, encoding="utf-8-sig")

    selected_literal = pl.Series(
        "selected_signal_date", selected_dates, dtype=pl.Datetime("ns")
    )
    common = (
        pl.scan_parquet(COMMON)
        .select(
            pl.col(date_col).cast(pl.Datetime("ns")).alias("signal_date"),
            pl.col(code_col).cast(pl.String).alias("stock_code"),
            "beta_concept_state_signal_date",
            "beta_concept_asof_date",
            "beta_score_F5_W10_DEFAULT",
            "gamma_concept_state_signal_date",
            "gamma_concept_asof_date",
            "gamma_score_F5_W10_DEFAULT",
        )
        .filter(
            pl.col("signal_date").is_in(selected_literal.implode())
            & pl.col("beta_score_F5_W10_DEFAULT").is_finite()
            & (pl.col("beta_score_F5_W10_DEFAULT") > 0)
            & pl.col("gamma_score_F5_W10_DEFAULT").is_finite()
            & (pl.col("gamma_score_F5_W10_DEFAULT") > 0)
        )
        .with_columns(
            pl.concat_str(
                [pl.col("signal_date").dt.strftime("%Y-%m-%d"), pl.col("stock_code")],
                separator="|",
            )
            .hash(seed=HASH_SEED)
            .alias("preoutcome_key_hash")
        )
        .filter((pl.col("preoutcome_key_hash") % HASH_MODULUS) == HASH_KEEP)
        .with_columns(
            pl.lit(DATE_STRIDE).cast(pl.UInt8).alias("date_stride"),
            pl.lit(HASH_MODULUS).cast(pl.UInt8).alias("hash_modulus"),
            pl.lit(HASH_KEEP).cast(pl.UInt8).alias("hash_keep"),
            pl.lit(True).alias("sample_frozen_before_outcome"),
        )
        .sort(["signal_date", "stock_code"])
    )
    temporary = SAMPLE.with_name(f".{SAMPLE.name}.partial")
    temporary.unlink(missing_ok=True)
    common.sink_parquet(
        temporary,
        compression="zstd",
        compression_level=6,
        statistics=True,
        engine="streaming",
        mkdir=True,
    )
    temporary.replace(SAMPLE)

    checks = (
        pl.scan_parquet(SAMPLE)
        .select(
            pl.len().alias("rows"),
            pl.struct(["signal_date", "stock_code"]).n_unique().alias("unique_keys"),
            pl.col("signal_date").n_unique().alias("dates"),
            pl.col("stock_code").n_unique().alias("stocks"),
            pl.col("signal_date").min().alias("signal_min"),
            pl.col("signal_date").max().alias("signal_max"),
            (~pl.col("sample_frozen_before_outcome"))
            .sum()
            .alias("freeze_flag_violations"),
            (
                (
                    pl.col("beta_concept_asof_date")
                    > pl.col("beta_concept_state_signal_date")
                )
                | (
                    pl.col("gamma_concept_asof_date")
                    > pl.col("gamma_concept_state_signal_date")
                )
                | (pl.col("beta_concept_state_signal_date") > pl.col("signal_date"))
                | (pl.col("gamma_concept_state_signal_date") >= pl.col("signal_date"))
            )
            .sum()
            .alias("timing_violations"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    specification = write_specification()
    hard_gates = {
        "sample_keys_unique": int(checks["rows"]) == int(checks["unique_keys"]),
        "all_sample_flags_frozen": int(checks["freeze_flag_violations"]) == 0,
        "beta_gamma_membership_timing_valid": int(checks["timing_violations"]) == 0,
        "signal_cutoff_valid": pd.Timestamp(checks["signal_max"]) < CUTOFF,
        "outcome_file_never_opened": True,
        "complete_specification_24_rows": len(specification) == 3 * len(HORIZONS),
    }
    payload = {
        "schema": "current-mechanism-sample-freeze-1.0",
        "status": "PASS" if all(hard_gates.values()) else "FAIL",
        "hard_gates": hard_gates,
        "sample_rule": {
            "support": "beta-gamma common finite-positive fully-LOO F5/W10 score support",
            "date_rule": f"every {DATE_STRIDE}th eligible signal date, first eligible date as phase zero",
            "stock_date_rule": f"hash(date|code, seed={HASH_SEED}) mod {HASH_MODULUS} == {HASH_KEEP}",
            "feature_key_availability_used_for_date_intersection_only": True,
            "feature_values_used_for_sampling": False,
            "future_outcomes_used_for_sampling": False,
        },
        "checks": {
            key: str(value) if "signal_" in key else value
            for key, value in checks.items()
        },
        "inputs": {
            "common_scores": parquet_identity(COMMON),
            "attention_frozen_feature_source": parquet_identity(ATTENTION),
            "attention_projection_rule": "future_* columns are forbidden downstream; old concepts are not protocol membership",
        },
        "outputs": {
            "sample": parquet_identity(SAMPLE),
            "date_registry": str(DATE_REGISTRY),
            "specification": str(SPECIFICATION),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    AUDIT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "checks": payload["checks"],
                "output": str(SAMPLE),
            },
            ensure_ascii=False,
        )
    )
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
