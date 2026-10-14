"""Construct research prices, calendar and memberships from source observations."""

from pathlib import Path
import argparse, csv, hashlib, json, re, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

START = "2016-11-24"
CUTOFF = "2026-08-31"


def read_table(path, columns=None):
    required = set(columns or ["股票代码", "交易日期"])
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with Path(path).open(encoding=encoding, newline="") as f:
                for skip, row in enumerate(csv.reader(f)):
                    if required.issubset(row):
                        return pd.read_csv(
                            path,
                            encoding=encoding,
                            skiprows=skip,
                            usecols=columns,
                            low_memory=False,
                        )
                    if skip >= 15:
                        break
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Required input fields are absent: {sorted(required)}")


def price_frame(path, overlay):
    alternate = overlay / path.name if overlay else None
    src = alternate if alternate is not None and alternate.exists() else path
    x = read_table(
        src,
        ["股票代码", "股票名称", "交易日期", "开盘价", "收盘价", "前收盘价", "成交额"],
    )
    x["date"] = pd.to_datetime(x["交易日期"], errors="coerce")
    x = (
        x[x.date.le(CUTOFF)]
        .sort_values("date", kind="stable")
        .drop_duplicates("date", keep="last")
        .copy()
    )
    ratio = x["收盘价"] / x["前收盘价"]
    ratio = ratio.where(np.isfinite(ratio) & ratio.gt(0))
    x["ret"] = ratio - 1
    x["close_adj"] = ratio.cumprod()
    x["open_adj"] = x.close_adj * x["开盘价"] / x["收盘价"]
    x = x[x.date.ge(START)].rename(
        columns={"股票代码": "code", "股票名称": "name", "成交额": "amount"}
    )
    return x[["date", "code", "name", "open_adj", "close_adj", "ret", "amount"]]


def member_frame(raw):
    x = pd.read_csv(
        raw, encoding="utf-8", usecols=["股票代码", "交易日期", "所属概念"]
    ).rename(columns={"股票代码": "code", "交易日期": "asof", "所属概念": "concept"})
    x["asof"] = pd.to_datetime(x["asof"], errors="coerce")
    x = x[x["asof"].between(START, CUTOFF) & x.concept.notna()].copy()
    x["concept"] = x.concept.astype(str).map(
        lambda value: re.sub(r"\s+", "", re.sub(r"[,;/|，；。]", "、", value)).split(
            "、"
        )
    )
    x = x.explode("concept")
    x = x[x.concept.ne("")].drop_duplicates(["asof", "code", "concept"])
    return x[["asof", "code", "concept"]]


def parquet_stream(frames, target, schema):
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with pq.ParquetWriter(target, schema, compression="zstd") as writer:
        for i, frame in enumerate(frames, 1):
            writer.write_table(
                pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
            )
            rows += len(frame)
            if i % 500 == 0:
                print(target.name, i, rows, flush=True)
    return rows


def build_prices(raw, overlay, output, workers=2):
    schema = pa.schema(
        [
            ("date", pa.timestamp("ns")),
            ("code", pa.string()),
            ("name", pa.string()),
            ("open_adj", pa.float64()),
            ("close_adj", pa.float64()),
            ("ret", pa.float64()),
            ("amount", pa.float64()),
        ]
    )
    files = sorted(raw.glob("*.csv"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return parquet_stream(
            pool.map(lambda p: price_frame(p, overlay), files),
            output / "prices_latest.parquet",
            schema,
        )


def build_membership(archive, output):
    import io

    schema = pa.schema(
        [("asof", pa.timestamp("ns")), ("code", pa.string()), ("concept", pa.string())]
    )
    with zipfile.ZipFile(archive) as z:
        return parquet_stream(
            (
                member_frame(io.BytesIO(z.read(name)))
                for name in sorted(z.namelist())
                if name.endswith(".csv")
            ),
            output / "membership_latest.parquet",
            schema,
        )


def build_calendar(source, output):
    x = read_table(source, ["candle_end_time"])
    d = pd.to_datetime(x.candle_end_time, errors="raise")
    d = pd.DatetimeIndex(d[(d >= START) & (d <= CUTOFF)].sort_values().unique())
    assert not d.hasnans and d.is_unique
    out = pd.DataFrame({"date": d, "idx": np.arange(len(d), dtype=np.int32)})
    out.to_parquet(output / "calendar.parquet", index=False, compression="zstd")
    return len(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-prices", type=Path, required=True)
    p.add_argument("--price-overlay", type=Path)
    p.add_argument("--membership-archive", type=Path, required=True)
    p.add_argument("--calendar-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument(
        "--components",
        nargs="+",
        choices=["calendar", "prices", "membership"],
        default=["calendar", "prices", "membership"],
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    if "calendar" in a.components:
        counts["calendar"] = build_calendar(a.calendar_source, a.output)
    if "prices" in a.components:
        counts["prices"] = build_prices(
            a.raw_prices, a.price_overlay, a.output, a.workers
        )
    if "membership" in a.components:
        counts["membership"] = build_membership(a.membership_archive, a.output)
    receipt = {
        "scope": "Source reconstruction; no saved feature or score input",
        "start": START,
        "cutoff": CUTOFF,
        "rows": counts,
        "inputs": {
            "price_files": len(list(a.raw_prices.glob("*.csv"))),
            "overlay_files": (
                len(list(a.price_overlay.glob("*.csv"))) if a.price_overlay else 0
            ),
            "membership_archive_sha256": hashlib.sha256(
                a.membership_archive.read_bytes()
            ).hexdigest(),
        },
        "outputs": {
            f.name: hashlib.file_digest(f.open("rb"), "sha256").hexdigest()
            for f in a.output.glob("*.parquet")
        },
    }
    (a.output / "construction.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
