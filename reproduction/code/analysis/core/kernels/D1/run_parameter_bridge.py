"""Six-pair formation-window comparisons using frozen selections and exact mature endpoints through 2026-08-31."""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("POLARS_MAX_THREADS", "4")
import numpy as np
import polars as pl
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))
import cutoff as CFG  # noqa: E402

S2 = CFG.NEW_S2
SRC = S2 / "cache"
FW = [(3, 5), (3, 20), (5, 10), (10, 20), (20, 10), (20, 50)]
NS = [5, 10, 20, 30, 50, 100]
HS = [3, 5, 10]
METHODS = ["FW", "F_only", "W_only", "A", "P"]
BASES = ["pool", "F_only", "W_only", "A", "P"]
CUTOFF = CFG.OBSERVATION_CUTOFF
B, SEED = 100000, 20260924
# Use the last eligible signal date for the fourth identity check.
IDENTITY_DATES = ["2017-01-03", "2020-06-30", "2023-01-03", "2026-08-31"]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(x)
    return h.hexdigest()


def dump(name, obj):
    p = HERE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def now():
    return datetime.now(timezone.utc).isoformat()


def tag(f, w):
    return f"F{f}_W{w}"


def evaluate():
    frozen = json.loads((HERE / "selection_freeze.json").read_text(encoding="utf-8"))
    assert sha(HERE / "frozen_top100_selections.parquet") == frozen["ranks_sha256"]
    dump(
        "evaluation_start.json",
        {"utc": now(), "selection_freeze_sha256": sha(HERE / "selection_freeze.json")},
    )
    cal = pl.read_parquet(SRC / "calendar.parquet")
    maxidx = cal["idx"].max()
    assert cal["date"].max().strftime("%Y-%m-%d") == CUTOFF
    prices = pl.read_parquet(
        SRC / "prices_latest.parquet", columns=["date", "code", "open_adj"]
    ).join(cal, on="date")
    sel = pl.read_parquet(HERE / "frozen_top100_selections.parquet").join(
        cal, on="date"
    )
    entry = prices.select(
        "code", (pl.col("idx") - 1).alias("idx"), pl.col("open_adj").alias("entry")
    )
    daily = []
    pool_daily = []
    endpoint_audit = []
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
        z.write_parquet(HERE / "cache" / f"endpoints_H{h}.parquet", compression="zstd")
        endpoint_audit.append(
            {
                "H": h,
                "rows": z.height,
                "missing": z["r"].null_count(),
                "max_exit_idx": z["idx"].max() + h + 1,
                "calendar_max_idx": maxidx,
            }
        )
        for n in NS:
            d = (
                z.filter(pl.col("rank") <= n)
                .group_by(["date", "F", "W", "method"])
                .agg(
                    pl.len().alias("intended"),
                    pl.col("r").count().alias("observed"),
                    pl.col("r").mean().alias("complete"),
                    pl.col("r").fill_null(0).mean().alias("zero"),
                )
                .with_columns(pl.lit(h).alias("H"), pl.lit(n).alias("N"))
            )
            assert d.select((pl.col("intended") == pl.col("N")).all()).item()
            daily.append(d)
        for f, w in FW:
            pool = (
                pl.read_parquet(HERE / "cache" / f"{tag(f,w)}_pool.parquet")
                .join(cal, on="date")
                .filter(pl.col("idx") + h + 1 <= maxidx)
                .join(target, on=["idx", "code"], how="left")
            )
            pool_daily.append(
                pool.group_by("date")
                .agg(
                    pl.len().alias("intended"),
                    pl.col("r").count().alias("observed"),
                    pl.col("r").mean().alias("complete"),
                    pl.col("r").fill_null(0).mean().alias("zero"),
                )
                .with_columns(
                    pl.lit(f).alias("F"), pl.lit(w).alias("W"), pl.lit(h).alias("H")
                )
            )
        print("EVALUATE H", h, flush=True)
    pl.concat(daily).sort(["F", "W", "H", "N", "method", "date"]).write_parquet(
        HERE / "daily_returns.parquet", compression="zstd"
    )
    pl.concat(pool_daily).write_parquet(
        HERE / "daily_pool_returns.parquet", compression="zstd"
    )
    dump("endpoint_audit.json", endpoint_audit)


def holm(p):
    p = np.asarray(p)
    ix = np.argsort(p, kind="stable")
    out = np.empty(len(p))
    out[ix] = np.minimum(1, np.maximum.accumulate(p[ix] * (len(p) - np.arange(len(p)))))
    return out


def bootstrap(x, L, seed, b=B):
    n, k = x.shape
    v = np.isfinite(x)
    a = np.where(v, x, 0.0)
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
    p = (1 + (np.abs(draws - mean) >= np.abs(mean)).sum(axis=0)) / (b + 1)
    return mean, lo, hi, p


def stats():
    daily = pl.read_parquet(HERE / "daily_returns.parquet")
    pools = pl.read_parquet(HERE / "daily_pool_returns.parquet")
    rows = []
    diffs = []
    sensitivity = []
    desc = []
    for j, (f, w) in enumerate(FW):
        for h in HS:
            mats = []
            meta = []
            pool = pools.filter(
                (pl.col("F") == f) & (pl.col("W") == w) & (pl.col("H") == h)
            )
            for n in NS:
                dd = daily.filter(
                    (pl.col("F") == f)
                    & (pl.col("W") == w)
                    & (pl.col("H") == h)
                    & (pl.col("N") == n)
                )
                fw = dd.filter(pl.col("method") == "FW").select(
                    "date",
                    pl.col("complete").alias("FW_complete"),
                    pl.col("zero").alias("FW_zero"),
                )
                for m in BASES:
                    base = pool if m == "pool" else dd.filter(pl.col("method") == m)
                    z = fw.join(
                        base.select("date", "complete", "zero"), on="date"
                    ).sort("date")
                    for missing in ["complete", "zero"]:
                        x = (z["FW_" + missing] - z[missing]).to_numpy()
                        mats.append(x)
                        meta.append((n, m, missing))
                        diffs.append(
                            z.select("date").with_columns(
                                pl.Series("delta", x),
                                pl.lit(f).alias("F"),
                                pl.lit(w).alias("W"),
                                pl.lit(h).alias("H"),
                                pl.lit(n).alias("N"),
                                pl.lit(m).alias("baseline"),
                                pl.lit(missing).alias("missing"),
                            )
                        )
                zz = dd.join(
                    pool.select("date", pl.col("complete").alias("pool_complete")),
                    on="date",
                )
                desc.extend(
                    zz.group_by("method")
                    .agg(
                        pl.len().alias("n_dates"),
                        pl.col("complete").mean().mul(100).alias("gross_pp"),
                        ((pl.col("complete") - pl.col("pool_complete")) * 100)
                        .mean()
                        .alias("pool_excess_pp"),
                        (1 - pl.col("observed") / pl.col("intended"))
                        .mean()
                        .alias("mean_missing_fraction"),
                    )
                    .with_columns(
                        pl.lit(f).alias("F"),
                        pl.lit(w).alias("W"),
                        pl.lit(h).alias("H"),
                        pl.lit(n).alias("N"),
                    )
                    .to_dicts()
                )
            x = np.column_stack(mats)
            mean, lo, hi, p = bootstrap(x, 20, SEED + j * 1000 + h)
            for k, (n, m, missing) in enumerate(meta):
                nf = int(np.isfinite(x[:, k]).sum())
                supported = nf // 20 >= 20
                rows.append(
                    dict(
                        F=f,
                        W=w,
                        H=h,
                        N=n,
                        baseline=m,
                        missing=missing,
                        n_dates=len(x),
                        n_finite=nf,
                        L=20,
                        supported=supported,
                        delta_pp=mean[k] * 100,
                        lo_pp=lo[k] * 100 if supported else None,
                        hi_pp=hi[k] * 100 if supported else None,
                        p_raw=p[k] if supported else None,
                        p_for_Holm=p[k] if supported else 1.0,
                    )
                )
            if h == 3:
                ix = [
                    k
                    for k, (n, m, missing) in enumerate(meta)
                    if n == 20 and missing == "complete"
                ]
                for L in [40, 60]:
                    mm, ll, hh, pp = bootstrap(x[:, ix], L, SEED + j * 1000 + L)
                    for k, i in enumerate(ix):
                        nf = int(np.isfinite(x[:, i]).sum())
                        supported = nf // L >= 20
                        sensitivity.append(
                            dict(
                                F=f,
                                W=w,
                                H=3,
                                N=20,
                                baseline=meta[i][1],
                                L=L,
                                n_finite=nf,
                                supported=supported,
                                delta_pp=mm[k] * 100,
                                lo_pp=ll[k] * 100 if supported else None,
                                hi_pp=hh[k] * 100 if supported else None,
                                p_raw=pp[k] if supported else None,
                                p_for_Holm=pp[k] if supported else 1.0,
                            )
                        )
            print("STATS", tag(f, w), h, flush=True)
    for missing in ["complete", "zero"]:
        family = [r for r in rows if r["missing"] == missing]
        assert len(family) == 540
        for r, p in zip(family, holm([r["p_for_Holm"] for r in family])):
            r["p_Holm540"] = p
            r["pass_Holm540"] = bool(r["supported"] and p < 0.05)
    for L in [40, 60]:
        family = [r for r in sensitivity if r["L"] == L]
        assert len(family) == 30
        for r, p in zip(family, holm([r["p_for_Holm"] for r in family])):
            r["p_Holm30_sensitivity"] = p
    pl.DataFrame(rows).sort(["missing", "F", "W", "H", "N", "baseline"]).write_csv(
        HERE / "comparison_1080.csv"
    )
    pl.DataFrame(sensitivity).write_csv(HERE / "block_sensitivity_60.csv")
    pl.concat(diffs).write_parquet(
        HERE / "daily_differences.parquet", compression="zstd"
    )
    pl.DataFrame(desc).write_csv(HERE / "descriptive_returns.csv")
    summary = (
        pl.DataFrame(rows)
        .group_by(["missing", "baseline"])
        .agg(
            pl.len().alias("tests"),
            (pl.col("delta_pp") > 0).sum().alias("positive"),
            pl.col("pass_Holm540").sum().alias("pass_Holm540"),
            pl.col("supported").sum().alias("supported"),
        )
        .sort(["missing", "baseline"])
    )
    summary.write_csv(HERE / "summary_by_baseline.csv")
    print(summary, flush=True)
