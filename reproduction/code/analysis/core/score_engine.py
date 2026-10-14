"""Rebuild LOO pairs, card scores and ranks independently of preserved QA.
All numerical differences are retained, not replaced by legacy doubles.
"""

import os
from pathlib import Path
from runtime_config import load_config

CONFIG = load_config()
OUT = CONFIG.output("core")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tmp").mkdir(exist_ok=True)
os.environ.update(
    {
        k: str(CONFIG.threads)
        for k in [
            "POLARS_MAX_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ]
    }
)
os.environ["TEMP"] = os.environ["TMP"] = str(OUT / "tmp")
import json, time, gc
import numpy as np
import pandas as pd
import polars as pl

EXP = Path(__file__).resolve().parent / "kernels"
CACHE = OUT / "cache"
OLD = CONFIG.path("core/cache")


def save(n, x):
    (OUT / n).write_text(
        json.dumps(x, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def log(*a):
    print(*a, flush=True)


def pairs(design):
    cal = pl.read_parquet(CACHE / "calendar.parquet").sort("idx")
    snap = (
        pl.scan_parquet(CACHE / "membership_latest.parquet")
        .select("asof")
        .unique()
        .collect()
        .sort("asof")["asof"]
        .to_numpy()
    )
    days = cal["date"].to_numpy()
    pos = np.searchsorted(snap, days, side="right") - 1
    if design == "gamma":
        pos = np.r_[-1, pos[:-1]]
    good = pos >= 0
    mp = pl.DataFrame(
        {
            "date": days[good],
            "idx": np.arange(len(days), dtype=np.int32)[good],
            "asof": snap[pos[good]],
        }
    )
    p = (
        pl.scan_parquet(CACHE / "membership_latest.parquet")
        .join(mp.lazy(), on="asof")
        .join(
            pl.scan_parquet(CACHE / "prices_latest.parquet")
            .select("date", "code", "ret")
            .filter(pl.col("ret").is_finite()),
            on=["date", "code"],
        )
    )
    # Sort contributions deterministically before group reduction; then LOO.
    p = p.sort(["date", "concept", "code"])
    totals = p.group_by("date", "concept").agg(
        pl.col("ret").sum().alias("sr"), pl.len().alias("n")
    )
    p = (
        p.join(totals, on=["date", "concept"])
        .filter(pl.col("n") > 1)
        .with_columns(((pl.col("sr") - pl.col("ret")) / (pl.col("n") - 1)).alias("loo"))
        .select("date", "idx", "code", "concept", "ret", "loo")
    )
    target = CACHE / f"{design}_past_pairs.parquet"
    p.sink_parquet(target, compression="zstd", engine="streaming")
    log("PAIR COMPLETE", design)
    if not (OLD / target.name).is_file():
        save(
            f"{design}_pairs_comparison.json",
            {
                "status": "REFERENCE_TABLES_NOT_SUPPLIED",
                "scope": "Fresh pairs were calculated; no archived pair table is required or substituted.",
            },
        )
        return
    key = ["date", "idx", "code", "concept"]
    a = pl.scan_parquet(target)
    b = pl.scan_parquet(OLD / target.name)
    j = a.join(b, on=key, suffix="_saved")
    info = (
        j.select(
            pl.len().alias("joined_rows"),
            (pl.col("loo") - pl.col("loo_saved")).abs().max().alias("loo_max_abs"),
            (pl.col("ret") - pl.col("ret_saved")).abs().max().alias("ret_max_abs"),
        )
        .collect()
        .to_dicts()[0]
    )
    info.update(
        fresh_rows=a.select(pl.len()).collect().item(),
        saved_rows=b.select(pl.len()).collect().item(),
    )
    save(f"{design}_pairs_comparison.json", info)


def score(design, clock, f=5, w=10, outname=None):
    def rolling(e, length, minn):
        e = (
            e.rolling_sum_by("idx", window_size=f"{length}i", min_samples=minn)
            if clock == "calendar"
            else e.rolling_sum(window_size=length, min_samples=minn)
        )
        return e.over(["code", "concept"])

    wm = w if w <= 5 else max(5, w // 2)
    expr = [
        (rolling(pl.col("loo").clip(lower_bound=-0.999).log1p(), f, f).exp() - 1).alias(
            "Fraw"
        )
    ]
    vals = {
        "n": pl.col("ret") * 0 + 1,
        "sx": pl.col("ret"),
        "sy": pl.col("loo"),
        "sxx": pl.col("ret").pow(2),
        "syy": pl.col("loo").pow(2),
        "sxy": pl.col("ret") * pl.col("loo"),
    }
    xvar = pl.col("sxx") - pl.col("sx").pow(2) / pl.col("n")
    yvar = pl.col("syy") - pl.col("sy").pow(2) / pl.col("n")
    cov = pl.col("sxy") - pl.col("sx") * pl.col("sy") / pl.col("n")
    partitions = OUT / "score_partitions"
    partitions.mkdir(exist_ok=True)
    partpaths = []
    for bucket in range(8):
        targetpart = partitions / f"{design}_{clock}_F{f}W{w}_{bucket}.parquet"
        partpaths.append(targetpart)
        if targetpart.exists():
            continue
        p = (
            pl.scan_parquet(CACHE / f"{design}_past_pairs.parquet")
            .filter(pl.col("code").hash(seed=19) % 8 == bucket)
            .sort(["code", "concept", "idx"])
        )
        p = p.with_columns(expr + [rolling(e, w, wm).alias(k) for k, e in vals.items()])
        p = p.with_columns(
            pl.when((xvar > 1e-15) & (yvar > 1e-15))
            .then(cov / (xvar * yvar).sqrt())
            .alias("Wraw")
        ).select("date", "idx", "code", "Fraw", "Wraw")
        p.sink_parquet(targetpart, compression="zstd", engine="streaming")
        log("ROLLING PART", design, clock, f, w, bucket)
        gc.collect()
    p = pl.scan_parquet(partpaths)
    # Percentiles use all finite cards per component BEFORE joint eligibility.
    for v, r in [("Fraw", "F"), ("Wraw", "W")]:
        finite = pl.when(pl.col(v).is_finite()).then(pl.col(v))
        p = p.with_columns(
            finite.rank(method="min").over("date").cast(pl.UInt64).alias(r + "_rank"),
            finite.count().over("date").cast(pl.UInt64).alias(r + "_count"),
        )
    p = p.filter(pl.col("F_rank").is_not_null() & pl.col("W_rank").is_not_null())
    s = p.group_by(["date", "idx", "code"]).agg(
        (pl.col("F_rank") * pl.col("W_rank")).sum().alias("FW_num"),
        pl.col("F_rank").sum().alias("F_only_num"),
        pl.col("W_rank").sum().alias("W_only_num"),
        pl.col("F_count").first(),
        pl.col("W_count").first(),
        pl.len().cast(pl.UInt64).alias("valid_cards"),
    )
    s = s.with_columns(
        (pl.col("F_count") * pl.col("W_count") * pl.col("valid_cards")).alias("FW_den"),
        (pl.col("F_count") * pl.col("valid_cards")).alias("F_only_den"),
        (pl.col("W_count") * pl.col("valid_cards")).alias("W_only_den"),
    )
    s = s.with_columns(
        *[
            (pl.col(m + "_num") / pl.col(m + "_den")).alias(m)
            for m in ["FW", "F_only", "W_only"]
        ]
    )
    target = CACHE / (outname or f"{design}_{clock}_scores.parquet")
    s.sink_parquet(target, compression="zstd", engine="streaming")
    log("SCORE COMPLETE", target.name)
    if (OLD / target.name).exists():
        a = pl.scan_parquet(target)
        b = pl.scan_parquet(OLD / target.name)
        j = a.join(b, on=["date", "idx", "code"], suffix="_saved")
        info = (
            j.select(
                pl.len().alias("joined_rows"),
                *[
                    (pl.col(k) - pl.col(k + "_saved")).abs().max().alias(k + "_max_abs")
                    for k in ["FW", "F_only", "W_only", "valid_cards"]
                ],
            )
            .collect()
            .to_dicts()[0]
        )
        info.update(
            fresh_rows=a.select(pl.len()).collect().item(),
            saved_rows=b.select(pl.len()).collect().item(),
        )
        save(target.stem + "_comparison.json", info)


def ranks():
    from functools import cmp_to_key

    c = pl.read_parquet(CACHE / "calendar.parquet")
    p = (
        pl.scan_parquet(CACHE / "prices_latest.parquet")
        .select("date", "code", "close_adj")
        .join(c.lazy(), on="date")
    )
    x = p
    for h in [5, 20]:
        x = x.join(
            p.select(
                "code",
                (pl.col("idx") + h).alias("idx"),
                pl.col("close_adj").alias(f"past{h}"),
            ),
            on=["idx", "code"],
            how="left",
        ).with_columns((pl.col("close_adj") / pl.col(f"past{h}") - 1).alias(f"mom{h}"))
    x = x.with_columns((-pl.col("mom5")).alias("rev5")).select(
        "date", "code", "mom5", "mom20", "rev5"
    )
    valuecols = ["mom5", "mom20", "rev5"]
    for d in ["gamma", "beta"]:
        for clock in ["calendar", "observations"]:
            cols = {m: f"{d}_{clock}_{m}" for m in ["FW", "F_only", "W_only"]}
            valuecols += list(cols.values())
            for m in ["FW", "F_only", "W_only"]:
                for z in ["num", "den"]:
                    cols[m + "_" + z] = f"{d}_{clock}_{m}_{z}"
            x = x.join(
                pl.scan_parquet(CACHE / f"{d}_{clock}_scores.parquet")
                .select("date", "code", *cols)
                .rename(cols),
                on=["date", "code"],
            )
    x = x.filter(pl.all_horizontal(pl.col(valuecols).is_finite()))
    x.sink_parquet(
        CACHE / "frozen_common_scores.parquet", compression="zstd", engine="streaming"
    )
    x = pl.read_parquet(CACHE / "frozen_common_scores.parquet")
    rows = []
    checks = []

    def exact_order(g, n, d):
        nums = [int(v) for v in g[n]]
        dens = [int(v) for v in g[d]]
        codes = g["code"].tolist()
        floats = g["score"].to_numpy()
        order = np.argsort(-floats, kind="stable").tolist()

        def compare(i, j):
            a = nums[i] * dens[j]
            b = nums[j] * dens[i]
            if a != b:
                return -1 if a > b else 1
            return -1 if codes[i] < codes[j] else 1 if codes[i] > codes[j] else 0

        # Approximate sorting is a speed aid. Adjacent clusters are exactly
        # ordered; an exhaustive adjacent integer check proves final order.
        lo = 0
        for k in range(1, len(order) + 1):
            if k == len(order) or abs(floats[order[k - 1]] - floats[order[k]]) > 1e-12:
                if k - lo > 1:
                    order[lo:k] = sorted(order[lo:k], key=cmp_to_key(compare))
                lo = k
        if any(compare(order[k], order[k + 1]) > 0 for k in range(len(order) - 1)):
            order = sorted(order, key=cmp_to_key(compare))
        assert all(compare(order[k], order[k + 1]) <= 0 for k in range(len(order) - 1))
        return order

    for d in ["gamma", "beta"]:
        for clock in ["calendar", "observations"]:
            for m in ["FW", "F_only", "W_only", "mom5", "rev5", "mom20"]:
                col = m if m in ["mom5", "rev5", "mom20"] else f"{d}_{clock}_{m}"
                if m in ["mom5", "rev5", "mom20"]:
                    part = (
                        x.select("date", "code", pl.col(col).alias("score"))
                        .sort(
                            ["date", "score", "code"], descending=[False, True, False]
                        )
                        .with_columns(
                            pl.col("code").cum_count().over("date").alias("rank")
                        )
                        .filter(pl.col("rank") <= 100)
                    )
                else:
                    n = col + "_num"
                    de = col + "_den"
                    p = x.select(
                        "date", "code", pl.col(col).alias("score"), n, de
                    ).to_pandas()
                    out = []
                    for date, g in p.groupby("date", sort=True):
                        g = g.reset_index(drop=True)
                        idx = exact_order(g, n, de)[:100]
                        q = g.iloc[idx][["date", "code", "score"]].copy()
                        q["rank"] = np.arange(1, len(q) + 1, dtype=np.uint32)
                        out.append(q)
                    part = pl.from_pandas(
                        pd.concat(out, ignore_index=True)
                    ).with_columns(
                        pl.col("rank").cast(pl.UInt32),
                        pl.col("date").cast(pl.Datetime("ns")),
                    )
                part = part.with_columns(
                    pl.lit(d).alias("design"),
                    pl.lit(clock).alias("clock"),
                    pl.lit(m).alias("method"),
                )
                rows.append(part)
                checks.append(
                    {
                        "design": d,
                        "clock": clock,
                        "method": m,
                        "rows": part.height,
                        "exact_integer_order_checked": m
                        not in ["mom5", "rev5", "mom20"],
                    }
                )
                log("RANK COMPLETE", checks[-1])
    pl.concat(rows).write_parquet(
        CACHE / "frozen_top100_ranks.parquet", compression="zstd"
    )
    save(
        "exact_rank_validation.json",
        {
            "comparisons": "integer cross-multiplication, ascending stock-code exact ties",
            "all_daily_adjacent_orders_checked": True,
            "checks": checks,
        },
    )
    save(
        "independent_common_support_comparison.json",
        {
            "rows": x.height,
            "dates": x["date"].n_unique(),
            "stocks": x["code"].n_unique(),
            "min_date": x["date"].min(),
            "max_date": x["date"].max(),
            "outcomes_used_to_choose_order": False,
        },
    )


if __name__ == "__main__":
    started = time.time()
    for d in ["beta", "gamma"]:
        if not (CACHE / f"{d}_past_pairs.parquet").exists():
            pairs(d)
        for clock in ["calendar", "observations"]:
            if not (CACHE / f"{d}_{clock}_scores.parquet").exists():
                score(d, clock)
            gc.collect()
    ranks()
    save(
        "score_rebuild_completion.json",
        {"elapsed_seconds": time.time() - started, "legacy_doubles_substituted": False},
    )
