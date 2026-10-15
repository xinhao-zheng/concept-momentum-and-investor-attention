"""Rebuild daily attention values from raw observations on a declared sample index.

The index fixes sample membership; no reference values enter construction.
Future attention changes are outcome fields, not signal-time predictors.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import argparse
import datetime
import hashlib
import json
import os
import re
import time
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from runtime_config import load_config

DATE = "signal_date"
CODE = "stock_code"
HARD_CUTOFF = pd.Timestamp("2026-09-01")
LAST_ALLOWED = pd.Timestamp("2026-08-31")
SELF_START = pd.Timestamp("2022-03-23")
HORIZONS = (1, 3, 5, 10, 20, 60, 120, 250)
ENCODING = "gb18030"
ROLLING_A = 56
MIN_HISTORY = 20
POPULAR_ROOT = Path(".")
FUTURE_TARGETS = {}


def stock_output_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field(DATE, pa.timestamp("ns"), nullable=False),
        pa.field(CODE, pa.string(), nullable=False),
        pa.field("attention_asof_cutoff", pa.timestamp("ns"), nullable=False),
        pa.field("observation_date", pa.timestamp("ns")),
        pa.field("log_T", pa.float32()),
        pa.field("T_delta", pa.float32()),
        pa.field("log_S", pa.float32()),
        pa.field("S_delta_5", pa.float32()),
        pa.field("S_delta_20", pa.float32()),
        pa.field("log_K", pa.float32()),
        pa.field("K_delta", pa.float32()),
        pa.field("K_same_weekday_z56", pa.float32()),
        pa.field("concepts", pa.string()),
        pa.field("attention_lag_calendar_days", pa.float32()),
        pa.field("concept_degree", pa.uint16(), nullable=False),
        pa.field("long_TK", pa.int8(), nullable=False),
        pa.field("common_TSK", pa.int8(), nullable=False),
    ]
    for horizon in HORIZONS:
        for short in ("T", "S", "K"):
            fields.append(
                pa.field(f"future_own_{short}_change_h{horizon}", pa.float32())
            )
    return pa.schema(fields)


def _safe_log1p(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return np.log1p(numeric.where(numeric >= 0))


def _prior_z(
    values: pd.Series, window: int = ROLLING_A, minimum: int = MIN_HISTORY
) -> pd.Series:
    history = values.shift(1).rolling(window, min_periods=minimum)
    sd = history.std(ddof=1)
    return (values - history.mean()) / sd.where(sd > 1e-12)


def _exact_diff(values: pd.Series, dates: pd.Series, days: int) -> pd.Series:
    valid = dates.diff(days).dt.days.eq(days)
    return values.diff(days).where(valid)


def _same_weekday_robust_z(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Prior 56-calendar-day robust z using the prior eight same weekdays."""

    answer = pd.Series(np.nan, index=values.index, dtype="float64")
    weekdays = dates.dt.weekday
    for weekday in range(7):
        idx = values.index[weekdays.eq(weekday)]
        if len(idx) == 0:
            continue
        series = values.loc[idx]
        prior = series.shift(1)
        median = prior.rolling(8, min_periods=4).median()
        q25 = prior.rolling(8, min_periods=4).quantile(0.25)
        q75 = prior.rolling(8, min_periods=4).quantile(0.75)
        robust_scale = (q75 - q25) / 1.349
        answer.loc[idx] = (series - median) / robust_scale.where(robust_scale > 1e-12)
    return answer


def normalize_concepts(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    for separator in (",", ";", "/", "|", "\uff0c", "\uff1b", "\u3002"):
        text = text.replace(separator, "\u3001")
    labels = sorted(
        {x.strip() for x in re.sub(r"\s+", "", text).split("\u3001") if x.strip()}
    )
    return "|".join(labels) if labels else None


def split_concepts(value: Any) -> list[str]:
    return (
        []
        if value is None or pd.isna(value) or not str(value)
        else str(value).split("|")
    )


def date_maps(
    calendar: pd.DatetimeIndex,
) -> tuple[dict[pd.Timestamp, int], dict[int, dict[pd.Timestamp, pd.Timestamp]]]:
    position = {pd.Timestamp(value): i for i, value in enumerate(calendar)}
    targets: dict[int, dict[pd.Timestamp, pd.Timestamp]] = {h: {} for h in HORIZONS}
    for signal, index in position.items():
        for horizon in HORIZONS:
            exit_index = index + horizon + 1
            if exit_index < len(calendar):
                exit_date = pd.Timestamp(calendar[exit_index])
                if exit_date <= LAST_ALLOWED:
                    targets[horizon][signal] = exit_date
    return position, targets


@dataclass(frozen=True)
class StockTask:
    stock_code: str
    signal_dates_ns: tuple[int, ...]
    source_path: str | None


def prepare_raw_attention(path: str | None) -> pd.DataFrame:
    output_columns = [
        "observation_date",
        "log_T",
        "T_delta",
        "log_S",
        "S_delta_5",
        "S_delta_20",
        "log_K",
        "K_delta",
        "K_same_weekday_z56",
        "concepts",
    ]
    if path is None:
        return pd.DataFrame(columns=output_columns)
    try:
        raw = pd.read_csv(
            path, encoding=ENCODING, skiprows=1, low_memory=False, on_bad_lines="skip"
        )
    except (OSError, UnicodeError, pd.errors.ParserError):
        return pd.DataFrame(columns=output_columns)
    if len(raw.columns) < 7:
        return pd.DataFrame(columns=output_columns)
    date_col, total_col, self_col, stick_col, concept_col = raw.columns[2:7]
    raw["observation_date"] = pd.to_datetime(
        raw[date_col], errors="coerce"
    ).dt.normalize()
    raw = raw[
        raw["observation_date"].notna() & (raw["observation_date"] < HARD_CUTOFF)
    ].copy()
    raw = raw.sort_values("observation_date", kind="mergesort").drop_duplicates(
        "observation_date", keep="last"
    )
    raw = raw.reset_index(drop=True)
    raw["log_T"] = _safe_log1p(raw[total_col])
    t_change = _exact_diff(raw["log_T"], raw["observation_date"], 1)
    raw["T_delta"] = _prior_z(t_change)
    raw["log_S"] = _safe_log1p(raw[self_col])
    raw["S_delta_5"] = _exact_diff(raw["log_S"], raw["observation_date"], 5)
    raw["S_delta_20"] = _exact_diff(raw["log_S"], raw["observation_date"], 20)
    pre_self = raw["observation_date"] < SELF_START
    raw.loc[pre_self, ["log_S", "S_delta_5", "S_delta_20"]] = np.nan
    raw["log_K"] = _safe_log1p(raw[stick_col])
    k_change = _exact_diff(raw["log_K"], raw["observation_date"], 1)
    raw["K_delta"] = _prior_z(k_change)
    raw["K_same_weekday_z56"] = _same_weekday_robust_z(
        raw["log_K"], raw["observation_date"]
    )
    raw["concepts"] = raw[concept_col].map(normalize_concepts)
    return raw[output_columns]


def process_stock(task: StockTask) -> pd.DataFrame:
    signal_dates = pd.DatetimeIndex(
        pd.to_datetime(np.asarray(task.signal_dates_ns, dtype=np.int64))
    ).normalize()
    result = pd.DataFrame({DATE: signal_dates, CODE: task.stock_code})
    result["attention_asof_cutoff"] = signal_dates - pd.Timedelta(days=2)
    raw = prepare_raw_attention(task.source_path)
    mapped_index = np.full(len(result), -1, dtype=int)
    if not raw.empty:
        observations = raw["observation_date"].to_numpy(dtype="datetime64[ns]")
        mapped_index = (
            np.searchsorted(
                observations,
                result["attention_asof_cutoff"].to_numpy(dtype="datetime64[ns]"),
                side="right",
            )
            - 1
        )
        safe = np.maximum(mapped_index, 0)
        mapped = raw.iloc[safe].reset_index(drop=True)
        mapped.loc[mapped_index < 0, :] = np.nan
        for column in raw.columns:
            result[column] = mapped[column].to_numpy()
    else:
        for column in raw.columns:
            if column == "observation_date":
                result[column] = pd.NaT
            elif column == "concepts":
                result[column] = None
            else:
                result[column] = np.nan
    result["attention_lag_calendar_days"] = (
        result[DATE] - pd.to_datetime(result["observation_date"])
    ).dt.days
    result["concept_degree"] = (
        result["concepts"].map(lambda x: len(split_concepts(x))).astype("uint16")
    )
    result["long_TK"] = (result["log_T"].notna() & result["log_K"].notna()).astype(
        "int8"
    )
    result["common_TSK"] = (
        result["log_T"].notna()
        & result["log_S"].notna()
        & result["log_K"].notna()
        & (pd.to_datetime(result["observation_date"]) >= SELF_START)
    ).astype("int8")

    # Future attention fields are diagnostic outcomes only.  They are measured
    # at the same trading-date endpoint as the price H exit.
    if not raw.empty:
        obs_int = raw["observation_date"].to_numpy(dtype="datetime64[ns]")
        for horizon in HORIZONS:
            target_dates = [
                FUTURE_TARGETS.get(horizon, {}).get(pd.Timestamp(d), pd.NaT)
                for d in signal_dates
            ]
            target_np = pd.DatetimeIndex(target_dates).to_numpy(dtype="datetime64[ns]")
            idx = np.searchsorted(obs_int, target_np, side="left")
            valid = (idx < len(obs_int)) & ~pd.isna(target_dates)
            safe = np.minimum(idx, max(len(obs_int) - 1, 0))
            valid &= obs_int[safe] == target_np
            for short, source in (("T", "log_T"), ("S", "log_S"), ("K", "log_K")):
                target_value = np.full(len(result), np.nan, dtype=float)
                target_value[valid] = raw[source].to_numpy(dtype=float)[safe[valid]]
                result[f"future_own_{short}_change_h{horizon}"] = (
                    target_value
                    - pd.to_numeric(result[source], errors="coerce").to_numpy(float)
                )
    else:
        for horizon in HORIZONS:
            for short in ("T", "S", "K"):
                result[f"future_own_{short}_change_h{horizon}"] = np.nan
    return result


def make_tasks(keys: pd.DataFrame) -> list[StockTask]:
    source = {path.stem.lower(): str(path) for path in POPULAR_ROOT.glob("*.csv")}
    tasks: list[StockTask] = []
    for code, group in keys.groupby(CODE, observed=True, sort=True):
        dates = np.sort(group[DATE].to_numpy(dtype="datetime64[ns]"))
        tasks.append(
            StockTask(
                str(code),
                tuple(dates.view("int64").tolist()),
                source.get(str(code).lower()),
            )
        )
    return tasks


def rank_expr(column: str, group: str = DATE) -> pl.Expr:
    finite = pl.when(pl.col(column).is_finite()).then(pl.col(column)).otherwise(None)
    return (
        finite.rank(method="average").over(group) / finite.count().over(group)
    ).cast(pl.Float32)


def finalize_stock_daily(raw_path: Path, output: Path) -> None:
    lazy = pl.scan_parquet(raw_path).with_columns(
        [
            rank_expr("T_delta").alias("T_delta_pct"),
            rank_expr("log_S", "observation_date").alias("S_level"),
            rank_expr("log_K", "observation_date").alias("K_level"),
            rank_expr("K_same_weekday_z56").alias("K_same_weekday_pct"),
            (pl.col("log_T").exp() - 1.0).clip(lower_bound=0).alias("T_raw_scale"),
        ]
    )
    market = (
        lazy.group_by("observation_date")
        .agg(
            [
                pl.col("log_T").mean().alias("market_mean_log_T"),
                pl.col("log_T").median().alias("market_median_log_T"),
                pl.col("log_S").mean().alias("market_mean_log_S"),
                pl.col("log_K").mean().alias("market_mean_log_K"),
                pl.col("T_raw_scale").sum().alias("market_T_sum"),
                (pl.col("T_raw_scale") ** 2).sum().alias("market_T_sumsq"),
                pl.col("T_raw_scale").top_k(10).sum().alias("market_T_top10_sum"),
            ]
        )
        .with_columns(
            [
                (pl.col("market_T_sumsq") / (pl.col("market_T_sum") ** 2)).alias(
                    "market_T_hhi"
                ),
                (pl.col("market_T_top10_sum") / pl.col("market_T_sum")).alias(
                    "market_T_top10_share"
                ),
            ]
        )
    )
    final = (
        lazy.join(market, on="observation_date", how="left")
        .drop(["T_raw_scale", "market_T_sum", "market_T_sumsq", "market_T_top10_sum"])
        .sort([DATE, CODE])
    )
    final.sink_parquet(output, compression="zstd", maintain_order=True, mkdir=True)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    global POPULAR_ROOT, FUTURE_TARGETS, ENCODING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--raw-attention", type=Path)
    parser.add_argument("--observation-index", type=Path)
    parser.add_argument(
        "--calendar",
        type=Path,
        help="Market CSV with candle_end_time or date, or date parquet",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoding", default="gb18030")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.workers < 1:
        parser.error("--workers must be positive")
    POPULAR_ROOT = (
        args.raw_attention or cfg.path("raw_snapshot/stock-popular-concept-detail")
    ).resolve()
    if not POPULAR_ROOT.is_dir() or not any(POPULAR_ROOT.glob("*.csv")):
        raise FileNotFoundError(
            "Raw attention directory is missing or contains no CSV files: "
            + str(POPULAR_ROOT)
        )
    index = (
        args.observation_index
        or Path(__file__).resolve().parents[2]
        / "data/attention-observation-index.parquet"
    ).resolve()
    calendar_path = (
        args.calendar or cfg.path("raw_snapshot/stock-main-index-data/sh000300.csv")
    ).resolve()
    output = (
        args.output or cfg.output("attention-inputs/attention_daily.parquet")
    ).resolve()
    if output.exists():
        raise FileExistsError("Use a new output path: " + str(output))
    if output.is_relative_to(POPULAR_ROOT):
        raise ValueError("Output must not be placed inside the raw input folder")
    ENCODING = args.encoding
    if calendar_path.suffix == ".parquet":
        calendar = pq.read_table(calendar_path, columns=["date"]).to_pandas()["date"]
    else:
        frame = pd.read_csv(calendar_path)
        date_column = "candle_end_time" if "candle_end_time" in frame else "date"
        calendar = frame[date_column]
    calendar = (
        pd.to_datetime(calendar, errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    calendar = pd.DatetimeIndex(
        calendar[(calendar >= pd.Timestamp("2016-11-24")) & (calendar <= LAST_ALLOWED)]
    ).normalize()
    if calendar.empty or calendar[-1] != LAST_ALLOWED:
        raise ValueError("Calendar must reach the published cutoff")
    _, FUTURE_TARGETS = date_maps(calendar)
    keys = pq.read_table(index, columns=[DATE, CODE]).to_pandas()
    keys[DATE] = pd.to_datetime(keys[DATE])
    keys[CODE] = keys[CODE].astype(str)
    if keys.duplicated([DATE, CODE]).any() or keys[DATE].isna().any():
        raise ValueError("Invalid observation index")
    if not keys[DATE].isin(calendar).all():
        raise ValueError("Observation dates must belong to the declared calendar")
    keys = keys.sort_values([CODE, DATE], kind="mergesort").reset_index(drop=True)
    tasks = make_tasks(keys)
    if not any(task.source_path for task in tasks):
        raise ValueError("No observation-index security matches a raw input filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_suffix(".raw.parquet")
    staging = output.with_suffix(".staging.parquet")
    if raw_output.exists() or staging.exists():
        raise FileExistsError(
            "Incomplete prior construction exists; choose a new output path"
        )
    schema = stock_output_schema()
    sources = []
    rows = 0
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(
        json.dumps(
            {
                "status": "STARTED",
                "stocks": len(tasks),
                "rows": len(keys),
                "workers": args.workers,
            }
        ),
        flush=True,
    )
    with pq.ParquetWriter(raw_output, schema, compression="zstd") as writer:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (task, frame) in enumerate(
                zip(tasks, pool.map(process_stock, tasks)), 1
            ):
                writer.write_table(
                    pa.Table.from_pandas(
                        frame, schema=schema, preserve_index=False, safe=False
                    )
                )
                rows += len(frame)
                sources.append(
                    {
                        "stock_code": task.stock_code,
                        "source_file": (
                            Path(task.source_path).name if task.source_path else None
                        ),
                        "sha256": sha(task.source_path) if task.source_path else None,
                        "rows": len(frame),
                        "mapped_observations": int(
                            frame["observation_date"].notna().sum()
                        ),
                    }
                )
                if i % 250 == 0:
                    print("RAW_STOCKS", i, "/", len(tasks), "ROWS", rows, flush=True)
    if rows != len(keys):
        raise AssertionError("Constructed rows differ from the declared index")
    print("FINALIZING_RANKS_AND_MARKET_SUMMARIES", flush=True)
    finalize_stock_daily(raw_output, staging)
    audit = (
        pl.scan_parquet(staging)
        .select(
            pl.len().alias("rows"),
            pl.col(DATE).max().alias("last_signal"),
            pl.col("observation_date").max().alias("last_observation"),
            (pl.col("observation_date") > pl.col("attention_asof_cutoff"))
            .sum()
            .alias("clock_violations"),
        )
        .collect()
        .to_dicts()[0]
    )
    if (
        audit["rows"] != len(keys)
        or audit["clock_violations"]
        or pd.Timestamp(audit["last_signal"]) > LAST_ALLOWED
    ):
        raise AssertionError("Attention output clock or row contract failed")
    os.replace(staging, output)
    receipt = {
        "started_utc": started,
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Raw attention values reconstructed on a preserved date/security sample index; this does not reconstruct the original historical sample-selection process or certify original publication timestamps.",
        "cutoff": cfg.cutoff,
        "attention_asof": "latest raw observation at or before signal_date minus 2 calendar days",
        "future_fields": "Outcome variables at exact market date t+H+1; not predictors",
        "raw_prefix_before_transforms": True,
        "calendar_sha256": sha(calendar_path),
        "observation_index_sha256": sha(index),
        "script_sha256": sha(__file__),
        "output_sha256": sha(output),
        "output_file": output.name,
        "audit": audit,
        "raw_sources": sources,
        "versions": {m.__name__: m.__version__ for m in [np, pd, pl, pa]},
    }
    output.with_suffix(".construction.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf8",
    )
    print(
        json.dumps(
            {
                "status": "CONSTRUCTED_NOT_YET_COMPARED",
                "rows": rows,
                "sha256": receipt["output_sha256"],
                "clock_violations": audit["clock_violations"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
