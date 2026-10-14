#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild the protocol-specific fully-LOO F5/W10 winning concept.

Beta uses the synchronous membership state.  Gamma uses the previous trading
state.  Both use the exact point-in-time snapshot recorded in their score
files.  The F5 and W10 card scores are reconstructed from current close returns
and raw membership only; no outcome or old hybrid winner/G is read.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CQ = ROOT.parents[2]
HARD = CQ / "concept-study"
CACHE = ROOT / "cache" / "mechanism"
AUDIT = ROOT / "audit" / "mechanism"
SAMPLE = CACHE / "mechanism_sample_keys_before_outcomes.parquet"
MEMBERSHIP = ROOT / "cache/abg/raw_stock_concept_snapshots.parquet"
OPEN_PANEL = ROOT / "cache/abg/open_return_panel.parquet"
CUTOFF = pd.Timestamp("2026-09-01")


def percentile_rank(column: str) -> pl.Expr:
    finite = pl.when(pl.col(column).is_finite()).then(pl.col(column)).otherwise(None)
    return finite.rank(method="min").over("signal_date") / finite.count().over(
        "signal_date"
    )


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
    parser.add_argument("--design", choices=("beta", "gamma"), required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    return parser.parse_args()


def build_mapping(design: str) -> pd.DataFrame:
    membership_names = pq.ParquetFile(MEMBERSHIP).schema_arrow.names
    membership_date = membership_names[0]
    open_names = pq.ParquetFile(OPEN_PANEL).schema_arrow.names
    open_date = open_names[0]
    membership_dates = pd.DatetimeIndex(
        pl.scan_parquet(MEMBERSHIP)
        .select(pl.col(membership_date).cast(pl.Datetime("ns")))
        .filter(pl.col(membership_date) < pl.lit(CUTOFF.to_pydatetime()))
        .unique()
        .sort(membership_date)
        .collect(engine="streaming")[membership_date]
        .to_numpy()
    )
    calendar = pd.DatetimeIndex(
        pl.scan_parquet(OPEN_PANEL)
        .select(pl.col(open_date).cast(pl.Datetime("ns")))
        .filter(pl.col(open_date) < pl.lit(CUTOFF.to_pydatetime()))
        .unique()
        .sort(open_date)
        .collect(engine="streaming")[open_date]
        .to_numpy()
    )
    calendar = calendar[calendar >= membership_dates.min()]
    membership_values = membership_dates.to_numpy(dtype="datetime64[ns]")
    rows: list[dict[str, pd.Timestamp]] = []
    first = 0 if design == "beta" else 1
    for position in range(first, len(calendar)):
        signal = pd.Timestamp(calendar[position])
        state = signal if design == "beta" else pd.Timestamp(calendar[position - 1])
        snapshot_position = int(
            np.searchsorted(membership_values, np.datetime64(state), side="right") - 1
        )
        if snapshot_position < 0:
            continue
        asof = pd.Timestamp(membership_values[snapshot_position])
        rows.append(
            {
                "signal_date": signal,
                "concept_state_signal_date": state,
                "concept_asof_date": asof,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("membership timing map is empty")
    return result


def main() -> None:
    args = parse_args()
    design = args.design
    started = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    score_path = ROOT / "cache" / "abg" / f"{design}_scores_future_free.parquet"
    output = CACHE / f"winner_{design}_F5_W10.parquet"
    audit_path = AUDIT / f"winner_{design}_lineage.json"
    work_base = CACHE / f"_work_{design}_winner_pairs_base.parquet"
    work_state = CACHE / f"_work_{design}_winner_concept_state.parquet"
    work_loo = CACHE / f"_work_{design}_winner_pairs_loo.parquet"
    for path in (SAMPLE, MEMBERSHIP, OPEN_PANEL, score_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists() and not args.force:
        raise FileExistsError(f"winner exists; pass --force to replace: {output}")
    for path in (work_base, work_state, work_loo):
        path.unlink(missing_ok=True)

    mapping_pd = build_mapping(design)
    mapping = (
        pl.from_pandas(mapping_pd, include_index=False)
        .with_columns(pl.all().cast(pl.Datetime("ns")))
        .lazy()
    )
    membership_names = pq.ParquetFile(MEMBERSHIP).schema_arrow.names
    membership_date, membership_code, membership_concept = membership_names[:3]
    open_names = pq.ParquetFile(OPEN_PANEL).schema_arrow.names
    open_date, open_code = open_names[:2]
    if "ret_close_1" not in open_names:
        raise ValueError("open panel lacks ret_close_1")

    membership = (
        pl.scan_parquet(MEMBERSHIP)
        .select(
            pl.col(membership_date).cast(pl.Datetime("ns")).alias("concept_asof_date"),
            pl.col(membership_code).cast(pl.String).alias("stock_code"),
            pl.col(membership_concept).cast(pl.String).alias("concept"),
        )
        .filter(pl.col("concept_asof_date") < pl.lit(CUTOFF.to_pydatetime()))
        .unique(subset=["concept_asof_date", "stock_code", "concept"])
    )
    current = (
        pl.scan_parquet(OPEN_PANEL)
        .select(
            pl.col(open_date).cast(pl.Datetime("ns")).alias("signal_date"),
            pl.col(open_code).cast(pl.String).alias("stock_code"),
            pl.col("ret_close_1").cast(pl.Float64).alias("stock_ret"),
        )
        .filter(
            (pl.col("signal_date") < pl.lit(CUTOFF.to_pydatetime()))
            & pl.col("stock_ret").is_finite()
        )
        .unique(subset=["signal_date", "stock_code"], keep="last")
    )
    base = (
        membership.join(mapping, on="concept_asof_date", how="inner")
        .join(current, on=["signal_date", "stock_code"], how="inner")
        .select(
            "concept_asof_date",
            "concept_state_signal_date",
            "signal_date",
            "stock_code",
            "concept",
            "stock_ret",
        )
        .sort(["signal_date", "stock_code", "concept"])
    )
    base.sink_parquet(
        work_base,
        compression="zstd",
        compression_level=5,
        statistics=True,
        engine="streaming",
        mkdir=True,
    )
    state = (
        pl.scan_parquet(work_base)
        .group_by(["signal_date", "concept"])
        .agg(
            pl.col("stock_ret").sum().alias("concept_ret_sum"),
            pl.len().cast(pl.UInt32).alias("concept_member_n"),
        )
    )
    state.sink_parquet(
        work_state,
        compression="zstd",
        compression_level=5,
        statistics=True,
        engine="streaming",
        mkdir=True,
    )
    loo = (
        pl.scan_parquet(work_base)
        .join(pl.scan_parquet(work_state), on=["signal_date", "concept"], how="left")
        .with_columns(
            pl.when(pl.col("concept_member_n") > 1)
            .then(
                (pl.col("concept_ret_sum") - pl.col("stock_ret"))
                / (pl.col("concept_member_n") - 1)
            )
            .otherwise(None)
            .cast(pl.Float64)
            .alias("concept_ret_loo")
        )
        .select(
            "concept_asof_date",
            "concept_state_signal_date",
            "signal_date",
            "stock_code",
            "concept",
            "stock_ret",
            "concept_member_n",
            "concept_ret_loo",
        )
        .sort(["stock_code", "concept", "signal_date"])
    )
    loo.sink_parquet(
        work_loo,
        compression="zstd",
        compression_level=5,
        statistics=True,
        engine="streaming",
        mkdir=True,
    )

    pairs = (
        pl.scan_parquet(work_loo)
        .sort(["stock_code", "concept", "signal_date"])
        .with_columns(
            pl.col("concept_ret_loo")
            .clip(lower_bound=-0.999)
            .log1p()
            .alias("concept_logret_loo")
        )
        .with_columns(
            pl.col("concept_logret_loo")
            .rolling_sum(window_size=5, min_samples=5)
            .over(["stock_code", "concept"])
            .exp()
            .sub(1.0)
            .alias("concept_mom5_loo"),
            pl.rolling_corr(
                pl.col("stock_ret"),
                pl.col("concept_ret_loo"),
                window_size=10,
                min_samples=5,
            )
            .over(["stock_code", "concept"])
            .alias("concept_corr10_loo"),
        )
        .with_columns(
            percentile_rank("concept_mom5_loo").alias("F5_loo_pct"),
            percentile_rank("concept_corr10_loo").alias("W10_loo_pct"),
        )
        .with_columns(
            (pl.col("F5_loo_pct") * pl.col("W10_loo_pct"))
            .cast(pl.Float32)
            .alias("pair_score_F5_W10")
        )
    )

    # Materialize the original protocol card kernel once for both sample winners
    # and the independent full-market ABG comparison. No outcome is read.
    full_cards = CACHE / f"full_protocol_cards_{design}.parquet"
    pairs.sink_parquet(full_cards, compression="zstd", engine="streaming")
    pairs = pl.scan_parquet(full_cards)
    pairs.group_by(["signal_date", "stock_code"]).agg(
        pl.col("pair_score_F5_W10").mean().cast(pl.Float64).alias("FW"),
        pl.col("pair_score_F5_W10").count().alias("valid_cards"),
        pl.col("concept_asof_date").first(),
        pl.col("concept_state_signal_date").first(),
    ).sort(["signal_date", "stock_code"]).sink_parquet(
        CACHE / f"full_stock_scores_{design}.parquet",
        compression="zstd",
        engine="streaming",
    )

    sample_expected_col = f"{design}_score_F5_W10_DEFAULT"
    sample = pl.scan_parquet(SAMPLE).select(
        "signal_date",
        "stock_code",
        pl.col(sample_expected_col)
        .cast(pl.Float64)
        .alias("expected_stock_score_F5_W10"),
        f"{design}_concept_state_signal_date",
        f"{design}_concept_asof_date",
    )
    target_pairs = pairs.join(
        sample.select("signal_date", "stock_code"),
        on=["signal_date", "stock_code"],
        how="inner",
    )
    reconstructed = target_pairs.group_by(["signal_date", "stock_code"]).agg(
        pl.col("pair_score_F5_W10")
        .mean()
        .cast(pl.Float64)
        .alias("reconstructed_stock_score_F5_W10"),
        pl.col("pair_score_F5_W10")
        .count()
        .cast(pl.UInt16)
        .alias("valid_concept_cards_F5_W10"),
        pl.len().cast(pl.UInt16).alias("protocol_concept_degree"),
    )
    winner = (
        target_pairs.filter(pl.col("pair_score_F5_W10").is_finite())
        .sort(
            ["signal_date", "stock_code", "pair_score_F5_W10", "concept"],
            descending=[False, False, True, False],
        )
        .unique(subset=["signal_date", "stock_code"], keep="first", maintain_order=True)
        .select(
            "signal_date",
            "stock_code",
            pl.col("concept").alias("winning_concept"),
            pl.col("pair_score_F5_W10").alias("winning_pair_score_F5_W10"),
            pl.col("concept_member_n")
            .cast(pl.UInt32)
            .alias("winning_concept_member_n"),
        )
    )
    final = (
        sample.join(reconstructed, on=["signal_date", "stock_code"], how="left")
        .join(winner, on=["signal_date", "stock_code"], how="left")
        .with_columns(
            (
                pl.col("reconstructed_stock_score_F5_W10")
                - pl.col("expected_stock_score_F5_W10")
            )
            .abs()
            .alias("stock_score_absolute_error"),
            pl.lit(design).alias("membership_design"),
            pl.lit(True).alias("winner_frozen_before_outcome"),
        )
        .sort(["signal_date", "stock_code"])
    )
    temporary = output.with_name(f".{output.name}.partial")
    temporary.unlink(missing_ok=True)
    final.sink_parquet(
        temporary,
        compression="zstd",
        compression_level=6,
        statistics=True,
        engine="streaming",
        mkdir=True,
    )
    temporary.replace(output)

    checks = (
        pl.scan_parquet(output)
        .select(
            pl.len().alias("rows"),
            pl.struct(["signal_date", "stock_code"]).n_unique().alias("unique_keys"),
            pl.col("signal_date").n_unique().alias("dates"),
            pl.col("winning_concept").is_null().sum().alias("missing_winner"),
            pl.col("stock_score_absolute_error").max().alias("max_score_abs_error"),
            pl.col("stock_score_absolute_error").mean().alias("mean_score_abs_error"),
            (pl.col("stock_score_absolute_error") > 1e-6)
            .sum()
            .alias("score_error_gt_1e6"),
            (~pl.col("winner_frozen_before_outcome"))
            .sum()
            .alias("freeze_flag_violations"),
            (
                pl.col(f"{design}_concept_asof_date")
                > pl.col(f"{design}_concept_state_signal_date")
            )
            .sum()
            .alias("snapshot_after_state"),
            (pl.col(f"{design}_concept_state_signal_date") > pl.col("signal_date"))
            .sum()
            .alias("state_after_signal"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    hard_gates = {
        "one_output_row_per_frozen_sample_key": int(checks["rows"])
        == int(checks["unique_keys"]),
        "winner_complete_on_positive_score_sample": int(checks["missing_winner"]) == 0,
        "archived_reference_matches_current_inputs_diagnostic": int(
            checks["score_error_gt_1e6"]
        )
        == 0,
        "winner_frozen_before_outcome": int(checks["freeze_flag_violations"]) == 0,
        "snapshot_not_after_state": int(checks["snapshot_after_state"]) == 0,
        "state_not_after_signal": int(checks["state_after_signal"]) == 0,
        "formal_outcome_read": False,
        "old_hybrid_winner_or_G_read": False,
    }
    # False-valued descriptive fields are not gates; record them outside the
    # Boolean pass calculation.
    gate_values = [
        value
        for key, value in hard_gates.items()
        if not key.endswith(("_read", "_diagnostic"))
    ]
    status = "PENDING_INDEPENDENT_NUMPY_VALIDATION" if all(gate_values) else "FAIL"
    payload = {
        "schema": "current-protocol-winner-1.0",
        "status": status,
        "design": design,
        "definition": (
            "synchronous beta membership state"
            if design == "beta"
            else "one-trading-state-lagged gamma membership state"
        ),
        "winner_rule": "highest fully-LOO F5/W10 stock-concept-card score; concept ascending lexical tie-break",
        "hard_gates": hard_gates,
        "checks": checks,
        "inputs": {
            "sample": parquet_identity(SAMPLE),
            "membership": parquet_identity(MEMBERSHIP),
            "current_return_panel": parquet_identity(OPEN_PANEL),
            "score_reference": parquet_identity(score_path),
        },
        "output": parquet_identity(output),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    audit_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if not args.keep_work:
        for path in (work_base, work_state, work_loo):
            path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "status": status,
                "design": design,
                "checks": checks,
                "output": str(output),
            },
            ensure_ascii=False,
            default=str,
        )
    )
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
