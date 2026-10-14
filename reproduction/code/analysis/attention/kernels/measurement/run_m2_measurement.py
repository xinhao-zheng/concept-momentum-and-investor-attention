"""M2 frozen descriptive rank-persistence audit; reads raw sources, writes here only.

Protocol: README.md (frozen before new statistics). No return/causal estimator.
Run with the project 00_环境/.venv/Scripts/python.exe -X utf8.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))
import cutoff as CFG  # noqa: E402

RAW = Path("__SNAPSHOT__/raw_snapshot")
CUTOFF = pd.Timestamp(CFG.OBSERVATION_CUTOFF)
HORIZONS = [1, 3, 5, 10, 20, 60, 120, 250]
TRANSFORMS = {"level": 0, "delta1": 1, "delta5": 5, "delta20": 20}
NAMES = ["T", "S", "K"]
FIELDS = ["总热度(万)", "自选热度(万)", "关注粘性(万)"]
BOOT = 10000
SEED = 20260922
MIN_STOCKS = 30
MIN_BLOCKS = 20
OUTPUT = HERE / "results"
FIGURES = HERE / "figures"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_source(path: Path):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during read: {path}")
    try:
        text = raw.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = raw.decode("gb18030")
        encoding = "gb18030"
    lines = text.splitlines()
    skip = 0 if lines[0].startswith("股票代码,") else 1
    df = pd.read_csv(
        io.StringIO(text),
        skiprows=skip,
        usecols=["股票代码", "交易日期"] + FIELDS,
        dtype={"股票代码": "string"},
    )
    df["交易日期"] = pd.to_datetime(df["交易日期"], errors="raise").dt.normalize()
    codes = df["股票代码"].dropna().unique().tolist()
    if len(codes) != 1:
        raise RuntimeError(f"Expected one stock code in file, got {codes}: {path}")
    if df.duplicated(["股票代码", "交易日期"]).any():
        raise RuntimeError(f"Duplicate stock/date keys: {path}")
    for field in FIELDS:
        df[field] = pd.to_numeric(df[field], errors="raise")
    manifest = {
        "file": str(path),
        "sha256": sha256(raw),
        "bytes": len(raw),
        "mtime_ns": before.st_mtime_ns,
        "encoding": encoding,
        "rows": len(df),
        "code": codes[0],
        "first_date": str(df["交易日期"].min().date()),
        "last_date": str(df["交易日期"].max().date()),
        "source_monotone_dates": bool(df["交易日期"].is_monotonic_increasing),
    }
    for name, field in zip(NAMES, FIELDS):
        good = np.isfinite(df[field].to_numpy(dtype=float))
        valid = df.loc[good, "交易日期"]
        manifest[f"{name}_valid_rows"] = int(good.sum())
        manifest[f"{name}_first"] = str(valid.min().date()) if len(valid) else None
        manifest[f"{name}_last"] = str(valid.max().date()) if len(valid) else None
    return df, manifest


def ingest():
    calendar_path = RAW / "stock-main-index-data" / "sh000001.csv"
    raw_calendar = calendar_path.read_bytes()
    cal_frame = pd.read_csv(io.BytesIO(raw_calendar), encoding="utf-8-sig")
    calendar = pd.DatetimeIndex(
        pd.to_datetime(cal_frame["candle_end_time"]).dt.normalize()
    )
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise RuntimeError("Market calendar is duplicated or unsorted")
    calendar = calendar[(calendar >= "2016-01-01") & (calendar <= CUTOFF)]
    if calendar[-1] != CUTOFF:
        raise RuntimeError("Proposed cutoff is not the latest calendar date")
    files = sorted((RAW / "stock-popular-concept-detail").glob("*.csv"))
    values = np.full((len(calendar), 3, len(files)), np.nan, dtype=np.float64)
    manifests, codes = [], []
    seen = set()
    excluded_nonmarket, excluded_after = 0, 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        for col, (df, record) in enumerate(executor.map(read_source, files)):
            code = record["code"]
            if code in seen:
                raise RuntimeError(f"Stock code occurs in multiple files: {code}")
            seen.add(code)
            codes.append(code)
            dates = pd.DatetimeIndex(df["交易日期"])
            positions = calendar.get_indexer(dates)
            take = positions >= 0
            excluded_after += int((dates > CUTOFF).sum())
            excluded_nonmarket += int(((dates <= CUTOFF) & (positions < 0)).sum())
            values[positions[take], :, col] = df.loc[take, FIELDS].to_numpy(
                dtype=np.float64
            )
            manifests.append(record)
            if (col + 1) % 500 == 0:
                print(f"Read {col + 1}/{len(files)} raw files", flush=True)
    valid = np.isfinite(values).all(axis=1)
    values = np.where(valid[:, None, :], values, np.nan)
    ranks = np.full_like(values, np.nan)
    for t in range(len(calendar)):
        mask = valid[t]
        if int(mask.sum()) >= MIN_STOCKS:
            ranks[t][:, mask] = (
                rankdata(values[t][:, mask], axis=1, method="average") / mask.sum()
            ).astype(np.float64)
    records = pd.DataFrame(manifests)
    records.to_csv(OUTPUT / "input_checksums.csv", index=False, encoding="utf-8-sig")
    coverage = pd.DataFrame({"date": calendar, "n_common_TSK": valid.sum(axis=1)})
    coverage.to_csv(
        OUTPUT / "daily_sample_coverage.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "calendar_file": str(calendar_path),
        "calendar_sha256": sha256(raw_calendar),
        "source_files": len(files),
        "unique_stock_codes": len(codes),
        "source_rows": int(records.rows.sum()),
        "source_first": records.first_date.min(),
        "source_last": records.last_date.max(),
        "cutoff": str(CUTOFF.date()),
        "calendar_rows": len(calendar),
        "common_TSK_stock_days": int(valid.sum()),
        "first_common_date": str(calendar[np.where(valid.any(axis=1))[0][0]].date()),
        "last_common_date": str(calendar[np.where(valid.any(axis=1))[0][-1]].date()),
        "ignored_after_cutoff_rows": excluded_after,
        "ignored_nonmarket_or_before_calendar_rows": excluded_nonmarket,
        "protocol_sha256": sha256((HERE / "README.md").read_bytes()),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "estimator": "Daily cross-sectional Spearman, x=current rank or rank change; y=future rank level",
        "H": HORIZONS,
        "transforms": TRANSFORMS,
        "B": BOOT,
        "seed": SEED,
        "min_blocks": MIN_BLOCKS,
        "min_stocks": MIN_STOCKS,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "stock_day_not_independent_inference": True,
        "causal_estimate": False,
        "conditional_prediction": False,
        "numeric_correction": "Raw float64; rank changes ordered by exact scaled half-rank cross-products, not floating subtraction",
    }
    del values
    return calendar, ranks, metadata


def spearman_rows(x, y):
    a = rankdata(x, axis=1, method="average")
    b = rankdata(y, axis=1, method="average")
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / den


def calculate_daily(calendar, ranks):
    rows = []
    daily_n = np.isfinite(ranks[:, 0, :]).sum(axis=1)
    for transform, lag in TRANSFORMS.items():
        for h in HORIZONS:
            for t in range(lag, len(calendar) - h):
                now = ranks[t]
                future = ranks[t + h]
                # Average ranks are half-integers. The common denominator
                # 2*n_t*n_lag does not affect cross-sectional Spearman order.
                # This avoids breaking exact ties through float cancellation.
                x = (
                    now
                    if lag == 0
                    else (
                        np.rint(now * 2 * daily_n[t]) * daily_n[t - lag]
                        - np.rint(ranks[t - lag] * 2 * daily_n[t - lag]) * daily_n[t]
                    )
                )
                valid = np.isfinite(x).all(axis=0) & np.isfinite(future).all(axis=0)
                n = int(valid.sum())
                if n < MIN_STOCKS:
                    continue
                ic = spearman_rows(x[:, valid], future[:, valid])
                if not np.isfinite(ic).all():
                    continue
                rows.append(
                    {
                        "date": calendar[t],
                        "future_date": calendar[t + h],
                        "transform": transform,
                        "H": h,
                        "n_stocks": n,
                        "T": ic[0],
                        "S": ic[1],
                        "K": ic[2],
                        "S_minus_T": ic[1] - ic[0],
                        "S_minus_K": ic[1] - ic[2],
                        "T_minus_K": ic[0] - ic[2],
                    }
                )
            print(f"Computed {transform} H{h}", flush=True)
    daily = pd.DataFrame(rows)
    daily.to_csv(OUTPUT / "daily_correlations.csv", index=False, encoding="utf-8-sig")
    return daily


def holm_full_family(frame, mask):
    p = frame.loc[mask, "p_raw"].fillna(1).to_numpy(float)
    order = np.argsort(p, kind="stable")
    sorted_adjusted = np.minimum(
        1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    )
    adjusted = np.ones_like(p)
    adjusted[order] = sorted_adjusted
    frame.loc[mask, "p_holm_96"] = adjusted


def summarize(daily):
    rows = []
    series_names = NAMES + ["S_minus_T", "S_minus_K", "T_minus_K"]
    for transform in TRANSFORMS:
        for h in HORIZONS:
            g = daily[(daily["transform"] == transform) & (daily.H == h)].sort_values(
                "date"
            )
            length = max(20, h)
            blocks = len(g) // length
            full = blocks * length
            inf = g.iloc[:full]
            if blocks >= MIN_BLOCKS:
                block_means = (
                    inf[series_names].to_numpy().reshape(blocks, length, 6).mean(axis=1)
                )
                rng = np.random.default_rng(SEED + h)
                indices = rng.integers(0, blocks, size=(BOOT, blocks))
                boot = block_means[indices].mean(axis=1)
                theta = block_means.mean(axis=0)
                low, high = np.quantile(boot, [0.025, 0.975], axis=0)
                p = (1 + (np.abs(boot - theta) >= np.abs(theta)).sum(axis=0)) / (
                    BOOT + 1
                )
            else:
                theta = (
                    inf[series_names].mean().to_numpy() if full else np.full(6, np.nan)
                )
                low, high, p = [np.full(6, np.nan) for _ in range(3)]
            for j, name in enumerate(series_names):
                rows.append(
                    {
                        "transform": transform,
                        "H": h,
                        "series": name,
                        "kind": "metric" if name in NAMES else "paired_difference",
                        "n_dates_all": len(g),
                        "n_stocks_median": g.n_stocks.median(),
                        "mean_all": g[name].mean(),
                        "mean_inference": theta[j],
                        "ci95_low": low[j],
                        "ci95_high": high[j],
                        "p_raw": p[j],
                        "block_length": length,
                        "n_complete_blocks": blocks,
                        "n_dates_inference": full,
                        "n_tail_dates_excluded": len(g) - full,
                        "first_signal_date": (
                            str(g.date.min().date()) if len(g) else None
                        ),
                        "last_signal_date": (
                            str(g.date.max().date()) if len(g) else None
                        ),
                        "inference_last_date": (
                            str(inf.date.max().date()) if full else None
                        ),
                        "status": (
                            "formal_eligible"
                            if blocks >= MIN_BLOCKS
                            else "descriptive_only"
                        ),
                    }
                )
    summary = pd.DataFrame(rows)
    summary["p_holm_96"] = np.nan
    for kind in ["metric", "paired_difference"]:
        mask = summary.kind == kind
        assert int(mask.sum()) == 96
        holm_full_family(summary, mask)
    summary["confirmed_nonzero"] = (summary.status == "formal_eligible") & (
        summary.p_holm_96 < 0.05
    )
    summary.to_csv(OUTPUT / "horizon_inference.csv", index=False, encoding="utf-8-sig")
    summary[summary.kind == "paired_difference"].to_csv(
        OUTPUT / "paired_family_inference.csv", index=False, encoding="utf-8-sig"
    )
    segments = {
        "historical_to_20260701": (None, "2026-07-01"),
        "viewed_extension_20260702_20260831": ("2026-07-02", "2026-08-31"),
    }
    seg_rows = []
    for segment, (start, end) in segments.items():
        d = daily[daily.date <= end]
        if start:
            d = d[d.date >= start]
        for transform in TRANSFORMS:
            for h in HORIZONS:
                g = d[(d["transform"] == transform) & (d.H == h)]
                for name in series_names:
                    seg_rows.append(
                        {
                            "segment": segment,
                            "transform": transform,
                            "H": h,
                            "series": name,
                            "n_dates": len(g),
                            "mean": g[name].mean(),
                            "n_stocks_median": g.n_stocks.median(),
                            "inference": "not_tested_segment_descriptive_only",
                        }
                    )
    pd.DataFrame(seg_rows).to_csv(
        OUTPUT / "period_descriptions.csv", index=False, encoding="utf-8-sig"
    )
    return summary


def figures(summary):
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 240,
            "figure.dpi": 120,
        }
    )
    colors = {"T": "#C47E40", "S": "#507C9B", "K": "#6B9384"}
    labels = {"T": "总热度 T", "S": "自选热度 S", "K": "关注黏性 K"}
    titles = {
        "level": "a  当前关注位置",
        "delta1": "b  过去 1 日的位置变化",
        "delta5": "c  过去 5 日的位置变化",
        "delta20": "d  过去 20 日的位置变化",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharex=True, sharey=True)
    for ax, transform in zip(axes.flat, TRANSFORMS):
        for name in NAMES:
            g = summary[
                (summary["transform"] == transform) & (summary.series == name)
            ].sort_values("H")
            ax.plot(g.H, g.mean_all, color=colors[name], lw=1.8, label=labels[name])
            formal = g.status == "formal_eligible"
            ax.scatter(
                g.loc[formal, "H"],
                g.loc[formal, "mean_all"],
                c=colors[name],
                s=27,
                zorder=3,
            )
            ax.scatter(
                g.loc[~formal, "H"],
                g.loc[~formal, "mean_all"],
                facecolor="white",
                edgecolor=colors[name],
                s=35,
                zorder=3,
            )
            ax.fill_between(
                g.loc[formal, "H"],
                g.loc[formal, "ci95_low"],
                g.loc[formal, "ci95_high"],
                color=colors[name],
                alpha=0.13,
            )
        ax.set_title(titles[transform], loc="left", fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
        ax.axhline(0, color="#79828A", lw=0.7)
        ax.grid(axis="y", color="#E5E9ED", lw=0.6)
        ax.set_ylim(-0.35, 1.05)
        ax.set_ylabel("与未来自身关注位置的秩相关")
        ax.set_xlabel("未来 H（市场交易日）")
    axes[0, 0].legend(loc="lower left", frameon=False)
    fig.suptitle(
        "相同测量方法下，三类热度是否仍呈现不同的持续性？",
        x=0.07,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.025,
        "同日共同股票样本；未来端始终是 level，没有减去当前值。点是描述均值；阴影是完整日期块样本的95%区间。\n空心点：不足20个完整时间块，仅描述。持续性不等于新增预测信息，更不等于收益或因果。",
        fontsize=9,
        color="#59616A",
    )
    fig.tight_layout(rect=(0.02, 0.085, 1, 0.94))
    fig.savefig(FIGURES / "M2A_measurement_persistence.png", bbox_inches="tight")
    fig.savefig(FIGURES / "M2A_measurement_persistence.pdf", bbox_inches="tight")
    plt.close(fig)
    pairs = ["S_minus_T", "S_minus_K", "T_minus_K"]
    pair_titles = ["自选热度 − 总热度", "自选热度 − 黏性", "总热度 − 黏性"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1), sharey=True)
    maxabs = max(
        0.05, float(summary[summary.kind == "paired_difference"].mean_all.abs().max())
    )
    for ax, name, title in zip(axes, pairs, pair_titles):
        g = summary[summary.series == name]
        pivot = g.pivot(index="transform", columns="H", values="mean_all").reindex(
            index=list(TRANSFORMS), columns=HORIZONS
        )
        im = ax.imshow(
            pivot.to_numpy(), vmin=-maxabs, vmax=maxabs, cmap="RdBu_r", aspect="auto"
        )
        for i, transform in enumerate(TRANSFORMS):
            for j, h in enumerate(HORIZONS):
                row = g[(g["transform"] == transform) & (g.H == h)].iloc[0]
                star = "*" if row.confirmed_nonzero else ""
                marker = "†" if row.status == "descriptive_only" else star
                color = "white" if abs(row.mean_all) > 0.6 * maxabs else "#20272E"
                ax.text(
                    j,
                    i,
                    f"{row.mean_all:.2f}{marker}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=color,
                )
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(8), HORIZONS)
        ax.set_yticks(range(4), ["当前位置", "1日变化", "5日变化", "20日变化"])
        ax.set_xlabel("未来 H（交易日）")
    fig.subplots_adjust(left=0.09, right=0.89, bottom=0.25, top=0.76, wspace=0.18)
    cax = fig.add_axes([0.92, 0.25, 0.015, 0.51])
    fig.colorbar(im, cax=cax, label="同日秩相关之差")
    fig.suptitle(
        "直接比较三类热度，而不是比较谁显著、谁不显著",
        x=0.09,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.06,
        "正值：标题前者的持续性相关更高；负值：后者更高。*：96项配对差家族 Holm 校正后 p<0.05；†：仅描述。\n这些是相关系数之差，不是收益差、因果效应，也不是‘某类热度更值得买’。",
        fontsize=9,
        color="#59616A",
    )
    fig.savefig(FIGURES / "M2B_paired_measures.png", bbox_inches="tight")
    fig.savefig(FIGURES / "M2B_paired_measures.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    started = time.time()
    OUTPUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    calendar, ranks, metadata = ingest()
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    daily = calculate_daily(calendar, ranks)
    summary = summarize(daily)
    figures(summary)
    metadata["elapsed_seconds"] = time.time() - started
    metadata["daily_statistic_rows"] = len(daily)
    metadata["paired_formal_eligible"] = int(
        (
            (summary.kind == "paired_difference")
            & (summary.status == "formal_eligible")
        ).sum()
    )
    metadata["paired_confirmed_nonzero"] = int(
        ((summary.kind == "paired_difference") & summary.confirmed_nonzero).sum()
    )
    (OUTPUT / "run_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "M2 completed",
        json.dumps(
            {
                k: metadata[k]
                for k in [
                    "elapsed_seconds",
                    "paired_formal_eligible",
                    "paired_confirmed_nonzero",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
