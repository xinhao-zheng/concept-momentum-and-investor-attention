"""Build covariates from separately obtained daily observations.

Compute features for every stock in the input directory from its CSV records
and the market calendar. Historical sample selection is applied separately
using the published observation index.

CSV columns follow the accompanying schema contract. Encoding and metadata-row
settings are explicit; --skip-metadata-rows=1 skips the first line without
inspection. Source files are read-only.
"""

from __future__ import annotations
import os

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ[_key] = "1"
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def pit_stock(
    path,
    calendar,
    market_return,
    cutoff,
    csv_encoding="gb18030",
    skip_metadata_rows=1,
    history_start=None,
    missing_amount_is_zero=False,
):
    raw = pd.read_csv(
        path, encoding=csv_encoding, skiprows=skip_metadata_rows, low_memory=False
    )
    raw["date"] = pd.to_datetime(raw["交易日期"], errors="coerce")
    raw = (
        raw[raw.date.notna() & (raw.date <= cutoff)]
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )
    if history_start is not None:
        raw = raw.loc[raw.index >= pd.Timestamp(history_start)]
    if raw.empty:
        return None
    cal = calendar[calendar >= raw.index.min()]
    d = raw.reindex(cal)
    observed = d["收盘价"].notna()
    close = d["收盘价"].ffill()
    amount = pd.to_numeric(d["成交额"], errors="coerce")
    fmv = pd.to_numeric(d["流通市值"], errors="coerce").ffill()
    r = (
        pd.to_numeric(d["收盘价"], errors="coerce")
        / pd.to_numeric(d["前收盘价"], errors="coerce")
        - 1
    )
    market_ret = market_return.reindex(cal)
    v = pd.DataFrame(index=cal)
    v["signal_date"] = cal
    v["stock_code"] = path.stem.lower()
    v["feature_asof_date"] = cal
    v["source_last_trade_date"] = pd.Series(
        np.where(observed, cal.values, np.datetime64("NaT")), index=cal
    ).ffill()
    v["source_trade_observed_t"] = observed.astype("int8")
    v["log_amount_t"] = np.log1p(amount.where(amount >= 0))
    v["amount_to_float_mv_t"] = amount / fmv.where(fmv > 0)
    v["log_amount_20_mean"] = v.log_amount_t.rolling(20, min_periods=10).mean()
    v["amount_to_float_mv_20_mean"] = v.amount_to_float_mv_t.rolling(
        20, min_periods=10
    ).mean()
    v["amihud_20"] = (
        (r.abs() / (amount / 1e8).where(amount > 0)).rolling(20, min_periods=10).mean()
    )
    v["suspension_days_20"] = (~observed).astype(float).rolling(20, min_periods=1).sum()
    zero = observed & ((amount.fillna(0) if missing_amount_is_zero else amount) <= 0)
    v["zero_amount_days_20"] = zero.astype(float).rolling(20, min_periods=1).sum()
    v["observed_trade_days_20"] = (
        observed.astype(float).rolling(20, min_periods=1).sum()
    )
    v["zero_amount_t"] = zero.astype("int8")
    name = d["股票名称"].ffill().astype(str)
    threshold = np.full(len(cal), 0.095)
    if path.stem.lower().startswith(("sh688", "sh689")):
        threshold = np.full(len(cal), 0.195)
    if path.stem.lower().startswith(("sz300", "sz301", "sz302")):
        threshold = np.full(len(cal), 0.195)
    if path.stem.lower().startswith("bj"):
        threshold = np.full(len(cal), 0.295)
    threshold = np.where(
        name.str.contains("ST", case=False, regex=False), 0.048, threshold
    )
    up = r >= threshold
    down = r <= -threshold
    one = (d["最高价"] == d["最低价"]) & (up | down)
    for label, val in [
        ("limit_up_proxy", up),
        ("limit_down_proxy", down),
        ("one_price_limit_proxy", one),
    ]:
        v[label + "_t"] = val.astype("int8")
        v[label + "_days_20"] = val.astype(float).rolling(20, min_periods=1).sum()
    v["market_return_t"] = market_ret
    v["ret_1"] = r
    for h in (5, 20, 60):
        v[f"momentum_{h}"] = close / close.shift(h) - 1
    v["drawdown_from_60d_high"] = close / close.rolling(60, min_periods=20).max() - 1
    cov = r.rolling(60, min_periods=40).cov(market_ret)
    vm = market_ret.where(r.notna()).rolling(60, min_periods=40).var()
    vr = r.rolling(60, min_periods=40).var()
    v["beta_hs300_60"] = cov / vm.where(vm > 0)
    v["idio_vol_hs300_60"] = np.sqrt((vr - cov**2 / vm.where(vm > 0)).clip(lower=0))
    v["log_float_mv_t"] = np.log(fmv.where(fmv > 0))
    tmv = pd.to_numeric(d["总市值"], errors="coerce").ffill()
    v["log_total_mv_t"] = np.log(tmv.where(tmv > 0))
    for tag, col in [
        ("hs300", "沪深300成分股"),
        ("sz50", "上证50成分股"),
        ("zz500", "中证500成分股"),
        ("cyb", "创业板指成分股"),
        ("zz1000", "中证1000成分股"),
        ("zz2000", "中证2000成分股"),
    ]:
        v[f"member_{tag}_t"] = d[col].where(observed).fillna("").eq("Y").astype("int8")
    for k, c in enumerate(
        ("新版申万一级行业名称", "新版申万二级行业名称", "新版申万三级行业名称"), 1
    ):
        v[f"sw{k}_industry"] = d[c].ffill()
    for tag, ch in [
        ("institution", "机构"),
        ("large", "大户"),
        ("mid", "中户"),
        ("retail", "散户"),
    ]:
        flow = (
            pd.to_numeric(d[ch + "资金买入额"], errors="coerce")
            - pd.to_numeric(d[ch + "资金卖出额"], errors="coerce")
        ) * 10000
        v[f"{tag}_net_inflow_to_amount_t"] = flow / amount.where(amount > 0)
        for h, mp in [(5, 3), (20, 10)]:
            v[f"{tag}_net_inflow_to_amount_{h}"] = flow.rolling(
                h, min_periods=mp
            ).sum() / amount.where(flow.notna()).rolling(h, min_periods=mp).sum().where(
                lambda x: x > 0
            )
    return v.reset_index(drop=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-prices", type=Path, required=True)
    parser.add_argument("--market-index", type=Path, required=True)
    parser.add_argument("--overlays", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-08-31")
    parser.add_argument(
        "--output-start",
        default=None,
        help="Filter emitted dates only; retain earlier history during calculation.",
    )
    parser.add_argument(
        "--row-policy", choices=("calendar", "observed"), default="calendar"
    )
    parser.add_argument(
        "--preprocessing-contract",
        choices=("full-history", "reconstructed-legacy"),
        default="full-history",
        help="Explicit reconstructed prefix contract; the historical generator source was not recovered.",
    )
    parser.add_argument("--csv-encoding", default="gb18030")
    parser.add_argument("--skip-metadata-rows", type=int, default=1)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--codes",
        nargs="*",
        help="Optional diagnostic subset; omitted for a full directory build.",
    )
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.cutoff)
    output_start = pd.Timestamp(args.output_start) if args.output_start else None
    market = pd.read_csv(args.market_index)
    market["date"] = pd.to_datetime(market["candle_end_time"])
    market = (
        market.loc[market.date <= cutoff]
        .sort_values("date")
        .drop_duplicates("date")
        .set_index("date")
    )
    calendar = pd.DatetimeIndex(market.index)
    market_return = market.close.pct_change(fill_method=None)
    originals = sorted(args.daily_prices.glob("*.csv"))
    if args.codes:
        requested = set(args.codes)
        originals = [p for p in originals if p.stem.lower() in requested]
        assert requested == {
            p.stem.lower() for p in originals
        }, "Requested security is absent"
    assert originals and len({p.stem.lower() for p in originals}) == len(originals)
    args.output.mkdir(parents=True, exist_ok=True)
    stocks = args.output / "by-stock"
    stocks.mkdir(exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError(
            "Output manifest already exists; choose a new run directory"
        )
    started = time.time()
    records = []
    total = 0

    def construct(original):
        overlay = args.overlays / original.name if args.overlays else None
        source = overlay if overlay is not None and overlay.is_file() else original
        frame = pit_stock(
            source,
            calendar,
            market_return,
            cutoff,
            args.csv_encoding,
            args.skip_metadata_rows,
        )
        if args.preprocessing_contract == "reconstructed-legacy" and frame is not None:
            legacy = pit_stock(
                source,
                calendar,
                market_return,
                cutoff,
                args.csv_encoding,
                args.skip_metadata_rows,
                history_start="2016-07-13",
                missing_amount_is_zero=True,
            )
            # The preserved panel joined an older prefix to a separately built
            # July/August extension. This reconstructed contract makes both
            # definitions explicit; it substitutes no archived feature values.
            prefix_end = pd.Timestamp("2026-06-30")
            prefix = (
                legacy.loc[legacy.signal_date <= prefix_end]
                if legacy is not None
                else frame.iloc[:0]
            )
            appended = frame.loc[frame.signal_date > prefix_end]
            frame = pd.concat([prefix, appended], ignore_index=True)
        if frame is None:
            return dict(
                security_id=original.stem.lower(),
                rows=0,
                source_sha256=sha256(source),
                overlay_used=source != original,
            )
        calculated_rows = len(frame)
        if output_start is not None:
            frame = frame.loc[frame.signal_date >= output_start]
        if args.row_policy == "observed":
            frame = frame.loc[frame.source_trade_observed_t.eq(1)]
        assert not frame.duplicated(["signal_date", "stock_code"]).any()
        assert frame.empty or frame.signal_date.max() <= cutoff
        target = stocks / (original.stem.lower() + ".parquet")
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, target, compression="zstd")
        return dict(
            security_id=original.stem.lower(),
            source_sha256=sha256(source),
            overlay_used=source != original,
            calculated_calendar_rows=calculated_rows,
            rows=len(frame),
            observed_rows=int(frame.source_trade_observed_t.sum()),
            first_signal=str(frame.signal_date.min().date()) if len(frame) else None,
            last_signal=str(frame.signal_date.max().date()) if len(frame) else None,
            output_file="by-stock/" + target.name,
            output_sha256=sha256(target),
            columns=table.num_columns,
        )

    # Bounded live results: executor.map over the whole directory may retain
    # many completed frames when an early stock is unusually slow.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for offset in range(0, len(originals), args.workers):
            batch = originals[offset : offset + args.workers]
            for record in executor.map(construct, batch):
                records.append(record)
                total += record["rows"]
            if len(records) % 100 == 0 or len(records) == len(originals):
                progress = dict(
                    status="RUNNING",
                    files=len(records),
                    total_files=len(originals),
                    rows=total,
                    elapsed_seconds=round(time.time() - started, 2),
                )
                write_json(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    manifest = dict(
        schema="raw-covariate-build/1",
        status="BUILT_FROM_RAW",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        constructor_sha256=sha256(__file__),
        market_index_sha256=sha256(args.market_index),
        cutoff=args.cutoff,
        output_start=args.output_start,
        pre_output_history_used=True,
        row_policy=args.row_policy,
        preprocessing_contract=args.preprocessing_contract,
        reconstructed_legacy_definition=(
            dict(
                prefix_end="2026-06-30",
                prefix_history_start="2016-07-13",
                prefix_zero_indicator_missing_amount_as_zero=True,
                appended_dates_use_full_history=True,
                original_generator_source_recovered=False,
            )
            if args.preprocessing_contract == "reconstructed-legacy"
            else None
        ),
        csv_encoding=args.csv_encoding,
        skip_metadata_rows=args.skip_metadata_rows,
        files=len(records),
        rows=total,
        source_observed_rows=sum(r.get("observed_rows", 0) for r in records),
        workers=args.workers,
        versions=dict(
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__,
            pyarrow=pa.__version__,
        ),
        cached_covariate_values_read=False,
        sample_admission_recovered=False,
        semantics="Calendar covariates for supplied raw securities; feature dates do not establish source publication time.",
        elapsed_seconds=round(time.time() - started, 2),
        records=records,
    )
    write_json(args.output / "manifest.json", manifest)
    write_json(
        args.output / "progress.json",
        dict(
            status="COMPLETE",
            files=len(records),
            rows=total,
            elapsed_seconds=manifest["elapsed_seconds"],
        ),
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "status",
                    "files",
                    "rows",
                    "source_observed_rows",
                    "elapsed_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
