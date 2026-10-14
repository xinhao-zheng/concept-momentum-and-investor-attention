"""Post-hoc literature-inspired benchmarks; frozen selection precedes endpoints."""

from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os

os.environ.setdefault("POLARS_MAX_THREADS", "4")
import numpy as np
import polars as pl

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))
import cutoff as CFG

SRC = CFG.NEW_S2_CACHE
CUT = pl.datetime(2026, 8, 31)
METHODS = ["CM5", "CM20", "BC10", "BC60"]
NS = [5, 20, 100]
HS = [1, 3, 5, 10, 20, 60, 120, 250]
B = 100000
SEED = 230923


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(x)
    return h.hexdigest()


def dump(name, obj):
    (HERE / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def evaluate():
    assert (
        sha(HERE / "frozen_selections.parquet")
        == json.loads((HERE / "selection_freeze.json").read_text(encoding="utf-8"))[
            "sha256"
        ]
    )
    cal = pl.read_parquet(SRC / "calendar.parquet")
    maxidx = cal["idx"].max()
    sel = pl.read_parquet(HERE / "frozen_selections.parquet").join(cal, on="date")
    prices = pl.read_parquet(
        SRC / "prices_latest.parquet", columns=["date", "code", "open_adj"]
    ).join(cal, on="date")
    entry = prices.select(
        "code", (pl.col("idx") - 1).alias("idx"), pl.col("open_adj").alias("entry")
    )
    rows = []
    pools = []
    for h in HS:
        exit_ = prices.select(
            "code",
            (pl.col("idx") - h - 1).alias("idx"),
            pl.col("open_adj").alias("exit"),
        )
        target = (
            entry.join(exit_, on=["code", "idx"], how="inner")
            .with_columns(
                pl.when(
                    pl.col("entry").is_finite()
                    & (pl.col("entry") > 0)
                    & pl.col("exit").is_finite()
                    & (pl.col("exit") > 0)
                )
                .then(pl.col("exit") / pl.col("entry") - 1)
                .alias("r")
            )
            .select("idx", "code", "r")
        )
        z = sel.filter(pl.col("idx") + h + 1 <= maxidx).join(
            target, on=["idx", "code"], how="left"
        )
        z.write_parquet(HERE / f"endpoints_H{h}.parquet", compression="zstd")
        for n in NS:
            r = (
                z.filter(pl.col("rank") <= n)
                .group_by(["date", "design", "method"])
                .agg(
                    pl.len().alias("intended"),
                    pl.col("r").count().alias("observed"),
                    pl.col("r").mean().alias("complete"),
                    pl.col("r").fill_null(0).mean().alias("zero"),
                )
                .with_columns(pl.lit(n).alias("N"), pl.lit(h).alias("H"))
            )
            assert r.select((pl.col("intended") == pl.col("N")).all()).item()
            rows.append(r)
        for d in ["beta", "gamma"]:
            pool = (
                pl.read_parquet(HERE / f"{d}_pool.parquet")
                .join(cal, on="date")
                .filter(pl.col("idx") + h + 1 <= maxidx)
                .join(target, on=["idx", "code"], how="left")
            )
            pools.append(
                pool.group_by("date")
                .agg(
                    pl.len().alias("intended"),
                    pl.col("r").count().alias("observed"),
                    pl.col("r").mean().alias("complete"),
                    pl.col("r").fill_null(0).mean().alias("zero"),
                )
                .with_columns(pl.lit(d).alias("design"), pl.lit(h).alias("H"))
            )
        print("endpoints", h, flush=True)
    daily = pl.concat(rows).sort(["design", "H", "N", "method", "date"])
    daily.write_parquet(HERE / "daily_returns.parquet", compression="zstd")
    pl.concat(pools).write_parquet(
        HERE / "daily_pool_returns.parquet", compression="zstd"
    )


def bootstrap(x, L, seed, b=B):
    n, k = x.shape
    v = np.isfinite(x)
    a = np.where(v, x, 0)
    c = np.vstack([np.zeros((1, k)), np.cumsum(np.vstack([a, a[:L]]), axis=0)])
    d = np.vstack([np.zeros((1, k)), np.cumsum(np.vstack([v, v[:L]]), axis=0)])
    s = np.arange(n)
    q, r = divmod(n, L)
    bs = c[s + L] - c[s]
    bc = d[s + L] - d[s]
    rs = c[s + r] - c[s]
    rc = d[s + r] - d[s]
    rng = np.random.default_rng(seed)
    draws = np.empty((b, k))
    for j in range(0, b, 200):
        m = min(200, b - j)
        ix = rng.integers(0, n, (m, q))
        num = bs[ix].sum(axis=1)
        den = bc[ix].sum(axis=1)
        if r:
            tail = rng.integers(0, n, m)
            num += rs[tail]
            den += rc[tail]
        draws[j : j + m] = np.divide(
            num, den, out=np.full_like(num, np.nan), where=den > 0
        )
    mean = np.nanmean(x, axis=0)
    lo, hi = np.nanquantile(draws, [0.025, 0.975], axis=0)
    pv = (1 + (np.abs(draws - mean) >= np.abs(mean)).sum(axis=0)) / (b + 1)
    return mean, lo, hi, pv


def holm(p):
    a = np.asarray(p)
    ix = np.argsort(a, kind="stable")
    out = np.empty(len(a))
    out[ix] = np.minimum(1, np.maximum.accumulate(a[ix] * (len(a) - np.arange(len(a)))))
    return out


def stats():
    daily = pl.read_parquet(HERE / "daily_returns.parquet")
    rows = []
    diffs = []
    sensitivity = []
    for di, d in enumerate(["beta", "gamma"]):
        for h in HS:
            mats = []
            meta = []
            for n in NS:
                dd = daily.filter(
                    (pl.col("design") == d) & (pl.col("H") == h) & (pl.col("N") == n)
                )
                fw = dd.filter(pl.col("method") == "FW").select(
                    "date",
                    pl.col("complete").alias("FW_complete"),
                    pl.col("zero").alias("FW_zero"),
                )
                for m in METHODS:
                    z = (
                        dd.filter(pl.col("method") == m)
                        .join(fw, on="date")
                        .sort("date")
                    )
                    for missing in ["complete", "zero"]:
                        x = z.select(
                            (pl.col("FW_" + missing) - pl.col(missing)).alias("delta")
                        )["delta"].to_numpy()
                        mats.append(x)
                        meta.append((n, m, missing))
                        diffs.append(
                            z.select("date").with_columns(
                                pl.Series("delta", x),
                                pl.lit(d).alias("design"),
                                pl.lit(h).alias("H"),
                                pl.lit(n).alias("N"),
                                pl.lit(m).alias("baseline"),
                                pl.lit(missing).alias("missing"),
                            )
                        )
            x = np.column_stack(mats)
            L = max(20, h)
            eligible = len(x) // L >= 20
            if eligible:
                mean, lo, hi, p = bootstrap(x, L, SEED + di * 1000 + h)
            else:
                mean = np.nanmean(x, axis=0)
                lo = hi = np.full(len(meta), np.nan)
                p = np.ones(len(meta))
            for j, (n, m, missing) in enumerate(meta):
                rows.append(
                    dict(
                        design=d,
                        H=h,
                        N=n,
                        baseline=m,
                        missing=missing,
                        n_dates=len(x),
                        n_finite=int(np.isfinite(x[:, j]).sum()),
                        L=L,
                        supported=eligible,
                        delta_pp=100 * mean[j],
                        lo_pp=100 * lo[j],
                        hi_pp=100 * hi[j],
                        p_raw=p[j] if eligible else None,
                        p_holm_input=p[j],
                    )
                )
            if d == "gamma" and h == 3:
                jj = [
                    i
                    for i, (n, m, missing) in enumerate(meta)
                    if n == 20 and missing == "complete"
                ]
                for ll in [40, 60]:
                    mm, llow, hhi, pp = bootstrap(x[:, jj], ll, SEED + ll)
                    for k, j in enumerate(jj):
                        sensitivity.append(
                            dict(
                                L=ll,
                                baseline=meta[j][1],
                                delta_pp=mm[k] * 100,
                                lo_pp=llow[k] * 100,
                                hi_pp=hhi[k] * 100,
                                p_raw=pp[k],
                                n_dates=len(x),
                            )
                        )
            print("inference", d, h, eligible, flush=True)
    assert len(rows) == 384
    for r, p in zip(rows, holm([r["p_holm_input"] for r in rows])):
        r["p_holm_384"] = p
        r["pass"] = bool(r["supported"] and p < 0.05)
    pl.DataFrame(rows).write_csv(HERE / "comparison_384.csv")
    pl.concat(diffs).write_parquet(
        HERE / "daily_differences.parquet", compression="zstd"
    )
    pl.DataFrame(sensitivity).write_csv(HERE / "primary_block_sensitivity.csv")
    pool = pl.read_parquet(HERE / "daily_pool_returns.parquet")
    desc = daily.join(pool, on=["date", "H", "design"], suffix="_pool").with_columns(
        ((pl.col("complete") - pl.col("complete_pool")) * 100).alias("pool_excess_pp")
    )
    desc.group_by(["design", "H", "N", "method"]).agg(
        pl.len().alias("n_dates"),
        pl.col("complete").mean().mul(100).alias("gross_pp"),
        pl.col("pool_excess_pp").mean(),
    ).sort(["design", "H", "N", "method"]).write_csv(HERE / "descriptive_returns.csv")
    print(
        pl.DataFrame(rows)
        .group_by("baseline")
        .agg(
            pl.col("pass").sum(),
            pl.col("supported").sum(),
            (pl.col("delta_pp") > 0).sum().alias("positive"),
        )
        .to_dicts(),
        flush=True,
    )
