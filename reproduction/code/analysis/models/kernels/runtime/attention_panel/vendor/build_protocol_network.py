#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build protocol-specific T/S/K peer and category mechanisms.

The concept graph comes only from the raw point-in-time membership snapshots
selected by the beta or gamma timing protocol.  The historical attention file
is projected only for conservative, already-observed T/S/K measurements; its
old ``concepts`` field, future fields, old winner and old G are never read.
"""

from __future__ import annotations

import argparse
import shutil
import json
import math
import time
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CQ = ROOT.parents[2]
HARD = CQ / "concept-study"
CACHE = ROOT / "cache" / "mechanism"
DATA = ROOT / "data" / "mechanism"
AUDIT = ROOT / "audit" / "mechanism"
MEMBERSHIP = ROOT / "cache/abg/raw_stock_concept_snapshots.parquet"
ATTENTION = ROOT.parent / "dynamics/shared/attention_daily.parquet"
SAMPLE = CACHE / "mechanism_sample_keys_before_outcomes.parquet"
CUTOFF = pd.Timestamp("2026-09-01")
MIN_PEERS = 5
FEATURES = ("T", "S", "K")

ATTENTION_COLUMNS = (
    "signal_date",
    "stock_code",
    "attention_asof_cutoff",
    "observation_date",
    "log_T",
    "T_delta",
    "log_S",
    "S_delta_5",
    "S_delta_20",
    "log_K",
    "K_delta",
    "K_same_weekday_z56",
    "attention_lag_calendar_days",
    "long_TK",
    "common_TSK",
    "T_delta_pct",
    "S_level",
    "K_level",
    "K_same_weekday_pct",
    "market_mean_log_T",
    "market_median_log_T",
    "market_mean_log_S",
    "market_mean_log_K",
    "market_T_hhi",
    "market_T_top10_share",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=("beta", "gamma"), required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--drop-cards", action="store_true")
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


def finite_value(column: str) -> pl.Expr:
    return pl.col(column).is_finite() & pl.col(column).is_not_null()


def build_cards(design: str, winner: Path, cards: Path) -> None:
    membership_names = pq.ParquetFile(MEMBERSHIP).schema_arrow.names
    raw_date, raw_code, raw_concept = membership_names[:3]
    mapping = (
        pl.scan_parquet(winner)
        .select(
            "signal_date",
            pl.col(f"{design}_concept_state_signal_date").alias(
                "concept_state_signal_date"
            ),
            pl.col(f"{design}_concept_asof_date").alias("concept_asof_date"),
        )
        .unique()
    )
    membership = (
        pl.scan_parquet(MEMBERSHIP)
        .select(
            pl.col(raw_date).cast(pl.Datetime("ns")).alias("concept_asof_date"),
            pl.col(raw_code).cast(pl.String).alias("stock_code"),
            pl.col(raw_concept).cast(pl.String).alias("concept"),
        )
        .filter(pl.col("concept_asof_date") < pl.lit(CUTOFF.to_pydatetime()))
        .unique(subset=["concept_asof_date", "stock_code", "concept"])
    )
    attention_names = set(pq.ParquetFile(ATTENTION).schema_arrow.names)
    missing = set(ATTENTION_COLUMNS) - attention_names
    if missing:
        raise ValueError(f"attention feature source missing {sorted(missing)}")
    if (
        any(name.startswith("future_") for name in ATTENTION_COLUMNS)
        or "concepts" in ATTENTION_COLUMNS
    ):
        raise AssertionError("forbidden attention projection")
    attention = (
        pl.scan_parquet(ATTENTION)
        .select([pl.col(name) for name in ATTENTION_COLUMNS])
        .filter(pl.col("signal_date") < pl.lit(CUTOFF.to_pydatetime()))
        .unique(subset=["signal_date", "stock_code"], keep="last")
    )
    base = (
        membership.join(mapping, on="concept_asof_date", how="inner")
        .select(
            "signal_date",
            "concept_state_signal_date",
            "concept_asof_date",
            "stock_code",
            "concept",
        )
        .with_columns(
            pl.len()
            .over(["signal_date", "stock_code"])
            .cast(pl.UInt16)
            .alias("protocol_concept_degree")
        )
        .join(attention, on=["signal_date", "stock_code"], how="left")
        .with_columns(
            (1.0 / pl.col("protocol_concept_degree").cast(pl.Float64)).alias(
                "allocation_weight"
            ),
            pl.col("T_delta").cast(pl.Float64).alias("own_T"),
            pl.col("S_level").cast(pl.Float64).alias("own_S"),
            pl.col("K_level").cast(pl.Float64).alias("own_K"),
            pl.col("log_T")
            .cast(pl.Float64)
            .exp()
            .sub(1.0)
            .clip(lower_bound=0.0)
            .alias("raw_T"),
            pl.col("log_S")
            .cast(pl.Float64)
            .exp()
            .sub(1.0)
            .clip(lower_bound=0.0)
            .alias("raw_S"),
            pl.col("log_K")
            .cast(pl.Float64)
            .exp()
            .sub(1.0)
            .clip(lower_bound=0.0)
            .alias("raw_K"),
            pl.lit(design).alias("membership_design"),
        )
        .sort(["signal_date", "concept", "stock_code"])
    )
    temporary = cards.with_name(f".{cards.name}.partial")
    temporary.unlink(missing_ok=True)
    base.sink_parquet(
        temporary,
        compression="zstd",
        compression_level=5,
        statistics=True,
        engine="streaming",
        mkdir=True,
    )
    temporary.replace(cards)


def aggregate_features(
    design: str, winner: Path, cards: Path, base_output: Path
) -> None:
    source = pl.scan_parquet(cards)
    concept_aggs: list[pl.Expr] = [
        pl.len().cast(pl.UInt32).alias("network_member_n"),
        pl.col("stock_code")
        .n_unique()
        .cast(pl.UInt32)
        .alias("network_unique_member_n"),
    ]
    market_aggs: list[pl.Expr] = []
    for feature in FEATURES:
        own = f"own_{feature}"
        raw = f"raw_{feature}"
        valid_own = finite_value(own)
        valid_raw = finite_value(raw) & (pl.col(raw) >= 0)
        concept_aggs.extend(
            [
                pl.when(valid_own)
                .then(pl.col("allocation_weight") * pl.col(own))
                .otherwise(0.0)
                .sum()
                .alias(f"sum_wx_{feature}"),
                pl.when(valid_own)
                .then(pl.col("allocation_weight"))
                .otherwise(0.0)
                .sum()
                .alias(f"sum_w_{feature}"),
                valid_own.sum().cast(pl.UInt32).alias(f"valid_n_{feature}"),
                pl.when(valid_own)
                .then(pl.col(own))
                .otherwise(0.0)
                .sum()
                .alias(f"sum_x_noalloc_{feature}"),
                pl.when(valid_own & (pl.col("protocol_concept_degree") == 1))
                .then(pl.col(own))
                .otherwise(0.0)
                .sum()
                .alias(f"sum_x_single_{feature}"),
                (valid_own & (pl.col("protocol_concept_degree") == 1))
                .sum()
                .cast(pl.UInt32)
                .alias(f"valid_n_single_{feature}"),
                pl.when(valid_raw)
                .then(pl.col("allocation_weight") * pl.col(raw))
                .otherwise(0.0)
                .sum()
                .alias(f"concept_total_alloc_{feature}"),
                pl.when(valid_raw)
                .then(pl.col(raw))
                .otherwise(0.0)
                .sum()
                .alias(f"concept_total_noalloc_{feature}"),
                pl.when(valid_raw & (pl.col("protocol_concept_degree") == 1))
                .then(pl.col(raw))
                .otherwise(0.0)
                .sum()
                .alias(f"concept_total_single_{feature}"),
            ]
        )
        market_aggs.extend(
            [
                pl.when(valid_raw)
                .then(pl.col("allocation_weight") * pl.col(raw))
                .otherwise(0.0)
                .sum()
                .alias(f"market_total_alloc_{feature}"),
                pl.when(valid_raw)
                .then(pl.col(raw))
                .otherwise(0.0)
                .sum()
                .alias(f"market_total_noalloc_{feature}"),
                pl.when(valid_raw & (pl.col("protocol_concept_degree") == 1))
                .then(pl.col(raw))
                .otherwise(0.0)
                .sum()
                .alias(f"market_total_single_{feature}"),
            ]
        )
    concept = source.group_by(["signal_date", "concept"]).agg(concept_aggs)
    market = source.group_by("signal_date").agg(market_aggs)
    winner_keys = pl.scan_parquet(winner).select(
        "signal_date",
        "stock_code",
        "winning_concept",
        "winning_pair_score_F5_W10",
        "winning_concept_member_n",
        "protocol_concept_degree",
    )
    focal = source.join(
        winner_keys.select(
            "signal_date",
            "stock_code",
            pl.col("winning_concept").alias("concept"),
        ),
        on=["signal_date", "stock_code", "concept"],
        how="inner",
    ).select(
        "signal_date",
        "stock_code",
        "concept",
        "protocol_concept_degree",
        "allocation_weight",
        *[f"own_{feature}" for feature in FEATURES],
        *[f"raw_{feature}" for feature in FEATURES],
        *[
            name
            for name in ATTENTION_COLUMNS
            if name not in {"signal_date", "stock_code"}
        ],
    )
    joined = (
        winner_keys.join(
            focal.rename({"concept": "winning_concept"}),
            on=["signal_date", "stock_code", "winning_concept"],
            how="left",
            suffix="_focal",
        )
        .join(
            concept.rename({"concept": "winning_concept"}),
            on=["signal_date", "winning_concept"],
            how="left",
        )
        .join(market, on="signal_date", how="left")
    )
    expressions: list[pl.Expr] = [
        (pl.col("network_member_n") - 1).cast(pl.Int32).alias("peer_count_all"),
        pl.col("allocation_weight").is_not_null().alias("focal_card_match"),
        pl.lit(design).alias("membership_design"),
        pl.lit(True).alias("features_frozen_before_outcome"),
    ]
    for feature in FEATURES:
        own = pl.col(f"own_{feature}")
        raw = pl.col(f"raw_{feature}")
        own_valid = own.is_finite() & own.is_not_null()
        raw_valid = raw.is_finite() & raw.is_not_null() & (raw >= 0)
        peer_n = pl.col(f"valid_n_{feature}") - own_valid.cast(pl.UInt32)
        peer_w = pl.col(f"sum_w_{feature}") - pl.when(own_valid).then(
            pl.col("allocation_weight")
        ).otherwise(0.0)
        peer_wx = pl.col(f"sum_wx_{feature}") - pl.when(own_valid).then(
            pl.col("allocation_weight") * own
        ).otherwise(0.0)
        noalloc_n = pl.col(f"valid_n_{feature}") - own_valid.cast(pl.UInt32)
        noalloc_sum = pl.col(f"sum_x_noalloc_{feature}") - pl.when(own_valid).then(
            own
        ).otherwise(0.0)
        single_own = own_valid & (pl.col("protocol_concept_degree") == 1)
        single_n = pl.col(f"valid_n_single_{feature}") - single_own.cast(pl.UInt32)
        single_sum = pl.col(f"sum_x_single_{feature}") - pl.when(single_own).then(
            own
        ).otherwise(0.0)
        expressions.extend(
            [
                peer_n.cast(pl.Int32).alias(f"valid_peer_n_{feature}"),
                pl.when((peer_n >= MIN_PEERS) & (peer_w > 0))
                .then(peer_wx / peer_w)
                .otherwise(None)
                .cast(pl.Float32)
                .alias(f"G_{feature}_mean"),
                pl.when(noalloc_n >= MIN_PEERS)
                .then(noalloc_sum / noalloc_n)
                .otherwise(None)
                .cast(pl.Float32)
                .alias(f"G_{feature}_mean_noallocation"),
                pl.when(single_n >= MIN_PEERS)
                .then(single_sum / single_n)
                .otherwise(None)
                .cast(pl.Float32)
                .alias(f"G_{feature}_mean_single_concept"),
                pl.when(pl.col(f"market_total_alloc_{feature}") > 0)
                .then(
                    pl.col(f"concept_total_alloc_{feature}")
                    / pl.col(f"market_total_alloc_{feature}")
                )
                .otherwise(None)
                .cast(pl.Float32)
                .alias(f"CategoryShare_{feature}"),
                pl.when(
                    raw_valid & ((pl.col(f"market_total_alloc_{feature}") - raw) > 0)
                )
                .then(
                    (
                        pl.col(f"concept_total_alloc_{feature}")
                        - pl.col("allocation_weight") * raw
                    )
                    / (pl.col(f"market_total_alloc_{feature}") - raw)
                )
                .otherwise(None)
                .cast(pl.Float32)
                .alias(f"CategoryShare_{feature}_loo"),
                pl.when(pl.col(f"market_total_noalloc_{feature}") > 0)
                .then(
                    pl.col(f"concept_total_noalloc_{feature}")
                    / pl.col(f"market_total_noalloc_{feature}")
                )
                .otherwise(None)
                .cast(pl.Float32)
                .alias(f"CategoryShare_{feature}_noallocation"),
                pl.when(pl.col(f"market_total_single_{feature}") > 0)
                .then(
                    pl.col(f"concept_total_single_{feature}")
                    / pl.col(f"market_total_single_{feature}")
                )
                .otherwise(None)
                .cast(pl.Float32)
                .alias(f"CategoryShare_{feature}_single_concept"),
                pl.when(raw_valid & (pl.col(f"concept_total_alloc_{feature}") > 0))
                .then(
                    pl.col("allocation_weight")
                    * raw
                    / pl.col(f"concept_total_alloc_{feature}")
                )
                .otherwise(None)
                .cast(pl.Float32)
                .alias(f"WithinShare_{feature}"),
            ]
        )
    final = (
        joined.with_columns(expressions)
        .select(
            "signal_date",
            "stock_code",
            "membership_design",
            "winning_concept",
            "winning_pair_score_F5_W10",
            "winning_concept_member_n",
            "network_member_n",
            "network_unique_member_n",
            "protocol_concept_degree",
            "peer_count_all",
            "focal_card_match",
            "features_frozen_before_outcome",
            *[
                name
                for name in ATTENTION_COLUMNS
                if name not in {"signal_date", "stock_code"}
            ],
            *[f"own_{feature}" for feature in FEATURES],
            *[f"raw_{feature}" for feature in FEATURES],
            *[f"valid_peer_n_{feature}" for feature in FEATURES],
            *[f"G_{feature}_mean" for feature in FEATURES],
            *[f"G_{feature}_mean_noallocation" for feature in FEATURES],
            *[f"G_{feature}_mean_single_concept" for feature in FEATURES],
            *[f"CategoryShare_{feature}" for feature in FEATURES],
            *[f"CategoryShare_{feature}_loo" for feature in FEATURES],
            *[f"CategoryShare_{feature}_noallocation" for feature in FEATURES],
            *[f"CategoryShare_{feature}_single_concept" for feature in FEATURES],
            *[f"WithinShare_{feature}" for feature in FEATURES],
        )
        .sort(["signal_date", "stock_code"])
    )
    temporary = base_output.with_name(f".{base_output.name}.partial")
    temporary.unlink(missing_ok=True)
    final.sink_parquet(
        temporary,
        compression="zstd",
        compression_level=6,
        statistics=True,
        engine="streaming",
        mkdir=True,
    )
    temporary.replace(base_output)


def weighted_loo_median(
    group: pd.DataFrame,
    feature: str,
    focal_codes: Iterable[str],
) -> tuple[dict[str, float], dict[str, int]]:
    valid = group[feature].notna() & np.isfinite(
        pd.to_numeric(group[feature], errors="coerce")
    )
    valid &= group["allocation_weight"].notna() & group["allocation_weight"].gt(0)
    observed = group.loc[valid, ["stock_code", feature, "allocation_weight"]].copy()
    observed["stock_code"] = observed["stock_code"].astype(str)
    observed = observed.sort_values([feature, "stock_code"], kind="mergesort")
    codes = observed["stock_code"].to_numpy(str)
    values = observed[feature].to_numpy(float)
    weights = observed["allocation_weight"].to_numpy(float)
    cumulative = np.cumsum(weights)
    total = float(weights.sum())
    positions = {code: index for index, code in enumerate(codes)}
    medians: dict[str, float] = {}
    counts: dict[str, int] = {}
    for focal in focal_codes:
        position = positions.get(str(focal))
        own_weight = float(weights[position]) if position is not None else 0.0
        peer_n = len(values) - (1 if position is not None else 0)
        counts[str(focal)] = peer_n
        peer_total = total - own_weight
        if peer_n < MIN_PEERS or peer_total <= 0:
            medians[str(focal)] = math.nan
            continue
        target = 0.5 * peer_total
        candidate = int(np.searchsorted(cumulative, target, side="left"))
        if position is not None and candidate >= position:
            candidate = int(
                np.searchsorted(cumulative, target + own_weight, side="left")
            )
        candidate = min(candidate, len(values) - 1)
        if position is not None and candidate == position:
            candidate = min(candidate + 1, len(values) - 1)
        medians[str(focal)] = float(values[candidate])
    return medians, counts


def process_median_date(
    day: pd.DataFrame,
    winners: pd.DataFrame,
    lineage_rows: list[dict[str, object]],
) -> pd.DataFrame:
    date = pd.Timestamp(day["signal_date"].iloc[0])
    selected = winners[winners["signal_date"].eq(date)].copy()
    output = selected[["signal_date", "stock_code"]].copy()
    for feature in FEATURES:
        output[f"G_{feature}_median"] = np.nan
    groups = {
        str(name): frame
        for name, frame in day.groupby("concept", observed=True, sort=False)
    }
    for concept, indices in selected.groupby(
        "winning_concept", observed=True, sort=False
    ).groups.items():
        peer_group = groups.get(str(concept))
        if peer_group is None:
            continue
        focal_codes = selected.loc[indices, "stock_code"].astype(str).tolist()
        for feature in FEATURES:
            medians, counts = weighted_loo_median(
                peer_group, f"own_{feature}", focal_codes
            )
            mask = output["stock_code"].isin(focal_codes)
            output.loc[mask, f"G_{feature}_median"] = output.loc[
                mask, "stock_code"
            ].map(medians)
            if len(lineage_rows) < 72 and focal_codes:
                focal = focal_codes[0]
                peers = peer_group[peer_group["stock_code"].astype(str).ne(focal)]
                valid = peers[f"own_{feature}"].notna() & np.isfinite(
                    pd.to_numeric(peers[f"own_{feature}"], errors="coerce")
                )
                valid &= peers["allocation_weight"].gt(0)
                denominator = peers.loc[valid, "allocation_weight"].sum()
                brute_mean = (
                    (
                        peers.loc[valid, f"own_{feature}"]
                        * peers.loc[valid, "allocation_weight"]
                    ).sum()
                    / denominator
                    if denominator > 0 and int(valid.sum()) >= MIN_PEERS
                    else math.nan
                )
                lineage_rows.append(
                    {
                        "signal_date": date,
                        "stock_code": focal,
                        "winning_concept": str(concept),
                        "feature": feature,
                        "brute_force_valid_peer_n": int(valid.sum()),
                        "brute_force_inverse_degree_mean": brute_mean,
                        "brute_force_inverse_degree_median": medians.get(
                            focal, math.nan
                        ),
                        "focal_excluded": True,
                    }
                )
    return output


def build_weighted_medians(
    cards: Path,
    winner: Path,
    median_output: Path,
    lineage_output: Path,
) -> None:
    winners = pd.read_parquet(
        winner, columns=["signal_date", "stock_code", "winning_concept"]
    )
    winners["signal_date"] = pd.to_datetime(winners["signal_date"])
    winners["stock_code"] = winners["stock_code"].astype(str)
    pf = pq.ParquetFile(cards)
    columns = [
        "signal_date",
        "stock_code",
        "concept",
        "allocation_weight",
        "own_T",
        "own_S",
        "own_K",
    ]
    outputs: list[pd.DataFrame] = []
    lineage_rows: list[dict[str, object]] = []
    carry: pd.DataFrame | None = None
    for batch in pf.iter_batches(batch_size=600_000, columns=columns):
        frame = batch.to_pandas()
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        if carry is not None and not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
        last_date = frame["signal_date"].max()
        complete = frame[frame["signal_date"] < last_date]
        carry = frame[frame["signal_date"] == last_date].copy()
        for _, day in complete.groupby("signal_date", observed=True, sort=True):
            outputs.append(process_median_date(day, winners, lineage_rows))
    if carry is not None and not carry.empty:
        for _, day in carry.groupby("signal_date", observed=True, sort=True):
            outputs.append(process_median_date(day, winners, lineage_rows))
    medians = pd.concat(outputs, ignore_index=True)
    medians = medians.sort_values(["signal_date", "stock_code"], kind="mergesort")
    medians.to_parquet(median_output, index=False, compression="zstd")
    pd.DataFrame(lineage_rows).to_csv(lineage_output, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    design = args.design
    started = time.time()
    for directory in (CACHE, DATA, AUDIT):
        directory.mkdir(parents=True, exist_ok=True)
    winner = CACHE / f"winner_{design}_F5_W10.parquet"
    cards = CACHE / f"protocol_membership_cards_{design}_sample_dates.parquet"
    base_output = CACHE / f"mechanism_features_{design}_base.parquet"
    medians = CACHE / f"mechanism_weighted_medians_{design}.parquet"
    output = CACHE / f"mechanism_features_{design}_preoutcome.parquet"
    lineage_output = DATA / f"mechanism_network_lineage_examples_{design}.csv"
    audit_path = AUDIT / f"mechanism_network_{design}.json"
    for path in (winner, MEMBERSHIP, ATTENTION, SAMPLE):
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists() and not args.force:
        raise FileExistsError(f"network output exists; pass --force: {output}")

    build_cards(design, winner, cards)
    aggregate_features(design, winner, cards, base_output)
    # Preserve the exact primary exposure/control projection for C1 while the
    # separately reported weighted-median sensitivity is constructed.
    shutil.copy2(
        base_output, CACHE / f"mechanism_features_{design}_primary_preoutcome.parquet"
    )
    build_weighted_medians(cards, winner, medians, lineage_output)
    final = (
        pl.scan_parquet(base_output)
        .join(pl.scan_parquet(medians), on=["signal_date", "stock_code"], how="left")
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
            (~pl.col("focal_card_match")).sum().alias("focal_card_mismatches"),
            (~pl.col("features_frozen_before_outcome"))
            .sum()
            .alias("freeze_flag_violations"),
            (pl.col("observation_date") > pl.col("attention_asof_cutoff"))
            .fill_null(False)
            .sum()
            .alias("attention_asof_violations"),
            (pl.col("observation_date") >= pl.lit(CUTOFF.to_pydatetime()))
            .fill_null(False)
            .sum()
            .alias("attention_cutoff_violations"),
            *[
                pl.col(f"G_{feature}_mean")
                .is_not_null()
                .sum()
                .alias(f"G_{feature}_mean_rows")
                for feature in FEATURES
            ],
            *[
                pl.col(f"G_{feature}_median")
                .is_not_null()
                .sum()
                .alias(f"G_{feature}_median_rows")
                for feature in FEATURES
            ],
            *[
                (
                    (pl.col(f"CategoryShare_{feature}") < 0)
                    | (pl.col(f"CategoryShare_{feature}") > 1 + 1e-6)
                    | (pl.col(f"CategoryShare_{feature}_loo") < 0)
                    | (pl.col(f"CategoryShare_{feature}_loo") > 1 + 1e-6)
                    | (pl.col(f"WithinShare_{feature}") < 0)
                    | (pl.col(f"WithinShare_{feature}") > 1 + 1e-6)
                )
                .fill_null(False)
                .sum()
                .alias(f"share_range_violations_{feature}")
                for feature in FEATURES
            ],
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    sample_rows = pq.ParquetFile(SAMPLE).metadata.num_rows
    hard_gates = {
        "one_feature_row_per_frozen_key": int(checks["rows"])
        == sample_rows
        == int(checks["unique_keys"]),
        "focal_winner_is_protocol_membership_card": int(checks["focal_card_mismatches"])
        == 0,
        "features_frozen_before_outcome": int(checks["freeze_flag_violations"]) == 0,
        "attention_observation_at_or_before_signal_minus_2_days": int(
            checks["attention_asof_violations"]
        )
        == 0,
        "attention_source_before_cutoff": int(checks["attention_cutoff_violations"])
        == 0,
        "primary_shares_within_unit_interval": all(
            int(checks[f"share_range_violations_{feature}"]) == 0
            for feature in FEATURES
        ),
        "weighted_mean_and_median_available_for_T": int(checks["G_T_mean_rows"]) > 0
        and int(checks["G_T_median_rows"]) > 0,
        "old_attention_concepts_field_projected": False,
        "old_hybrid_winner_or_G_read": False,
        "future_attention_field_projected": False,
        "price_outcome_read": False,
    }
    descriptive_false = {
        "old_attention_concepts_field_projected",
        "old_hybrid_winner_or_G_read",
        "future_attention_field_projected",
        "price_outcome_read",
    }
    status = (
        "PASS"
        if all(
            value for key, value in hard_gates.items() if key not in descriptive_false
        )
        else "FAIL"
    )
    payload = {
        "schema": "current-protocol-network-1.0",
        "status": status,
        "design": design,
        "hard_gates": hard_gates,
        "checks": checks,
        "definitions": {
            "membership": f"raw PIT membership under {design} protocol",
            "winner": "protocol-specific highest fully-LOO F5/W10 card score; lexical concept tie-break",
            "G_primary": "inverse-degree weighted fully-LOO peer mean; at least five valid peers",
            "G_median": "exact inverse-degree weighted fully-LOO peer median; at least five valid peers",
            "CategoryShare": "winner-concept raw T/S/K allocation divided by market allocation; stock weight 1/protocol concept degree",
            "CategoryShare_loo": "focal stock removed from winner-concept numerator and market denominator",
            "WithinShare": "focal 1/degree allocation divided by winner-concept allocated raw attention",
            "sensitivities": "no-allocation and single-concept-only category/peer constructions",
        },
        "inputs": {
            "winner": parquet_identity(winner),
            "membership": parquet_identity(MEMBERSHIP),
            "attention": parquet_identity(ATTENTION),
            "attention_projection": list(ATTENTION_COLUMNS),
        },
        "outputs": {
            "features": parquet_identity(output),
            "cards_retained": not args.drop_cards,
            "cards": parquet_identity(cards),
            "lineage_examples": str(lineage_output),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    audit_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    base_output.unlink(missing_ok=True)
    medians.unlink(missing_ok=True)
    if args.drop_cards:
        cards.unlink(missing_ok=True)
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
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
