"""Frozen D4 matched-control experiment. Writes only beside this script."""

from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import math
import os
import time

os.environ.setdefault("POLARS_MAX_THREADS", "4")
import numpy as np
import polars as pl

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))
import cutoff as CFG

S2 = CFG.NEW_S2_CACHE
B2 = CFG.RUNTIME.output("models/B2/cache")
D1 = CFG.EXP / "D1/cache"
EXPECTED_POOL = json.loads(
    (CFG.NEW_S2_AUDIT / "selection_freeze.json").read_text(encoding="utf-8")
)["rows"]
CUT_Y, CUT_M, CUT_D = map(int, CFG.OBSERVATION_CUTOFF.split("-"))
SOURCES = {
    "scores": D1 / "F5_W10_authoritative_scores.parquet",
    "common": S2 / "frozen_common_scores.parquet",
    "prices": S2 / "prices_latest.parquet",
    "calendar": S2 / "calendar.parquet",
    "industry": B2 / "industry_labels.parquet",
    "features": B2 / "feature_panel.parquet",
}
NS = [5, 10, 20]
BASES = ["pool", "industry", "size", "industry_x_size"]
B = 1000
BOOT = 10000
SEED = 20260924
clock = time.perf_counter()


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(name, obj):
    (HERE / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def csvout(name, rows):
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def say(*a):
    print(*a, flush=True)


assert (HERE / "frozen_protocol.json").exists()
manifest = {
    k: {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
    for k, p in SOURCES.items()
}
dump("input_manifest_before.json", manifest)
say("READ AND MATCH HISTORICAL POOL")
common = pl.read_parquet(
    SOURCES["common"], columns=["date", "code", "gamma_calendar_FW"]
)
scores = pl.read_parquet(
    SOURCES["scores"], columns=["date", "idx", "code", "FW", "FW_num", "FW_den"]
)
pool = common.join(scores, on=["date", "code"], how="left")
assert pool.height == EXPECTED_POOL
assert pool["FW"].null_count() == 0
score_error = pool.select(
    (pl.col("FW") - pl.col("gamma_calendar_FW")).abs().max()
).item()
assert score_error == 0, score_error
pool = pool.drop("gamma_calendar_FW")
features = pl.read_parquet(SOURCES["features"], columns=["date", "code", "float_cap"])
labels = pl.read_parquet(SOURCES["industry"]).with_columns(
    pl.col("date").cast(pl.Datetime("ns"))
)
pool = pool.join(features, on=["date", "code"], how="left").join(
    labels, on=["date", "code"], how="left"
)
assert pool.height == common.height, "duplicate join keys"
coverage = {
    "original_frozen_stock_dates": common.height,
    "original_frozen_dates": common["date"].n_unique(),
    "score_max_abs_difference_from_S2": score_error,
    "missing_or_nonpositive_floatcap": pool.filter(
        pl.col("float_cap").is_null()
        | ~pl.col("float_cap").is_finite()
        | (pl.col("float_cap") <= 0)
    ).height,
    "missing_or_blank_industry": pool.filter(
        pl.col("industry").is_null() | pl.col("industry").str.strip_chars().eq("")
    ).height,
}
pool = pool.filter(
    pl.col("float_cap").is_finite()
    & (pl.col("float_cap") > 0)
    & pl.col("industry").is_not_null()
    & pl.col("industry").str.strip_chars().ne("")
    & pl.col("FW").is_finite()
)
del common, scores, features, labels
pool = pool.sort(["date", "float_cap", "code"]).with_columns(
    (((pl.col("code").cum_count().over("date") - 1) * 10 // pl.len().over("date")) + 1)
    .cast(pl.Int8)
    .alias("size_decile")
)
coverage.update(
    eligible_stock_dates=pool.height,
    eligible_dates=pool["date"].n_unique(),
    eligible_stocks=pool["code"].n_unique(),
    excluded_stock_dates=coverage["original_frozen_stock_dates"] - pool.height,
    smallest_pool=int(pool.group_by("date").len()["len"].min()),
    largest_pool=int(pool.group_by("date").len()["len"].max()),
)
assert coverage["smallest_pool"] >= 20
from exact_rank_util import rank_table

frozen = (
    rank_table(pool, "FW", "FW", {})
    .filter(pl.col("rank") <= 20)
    .select("date", "code", "rank")
    .join(pool, on=["date", "code"])
    .sort(["date", "rank"])
)
frozen.write_parquet(HERE / "frozen_top20_selections.parquet")
pool.write_parquet(HERE / "frozen_eligible_pool.parquet")
dump(
    "selection_freeze.json",
    {
        "frozen_at": dt.datetime.now().isoformat(),
        "pool_rows": pool.height,
        "selection_rows": frozen.height,
        "selection_sha256": sha(HERE / "frozen_top20_selections.parquet"),
        "pool_sha256": sha(HERE / "frozen_eligible_pool.parquet"),
        "future_outcome_join_has_not_started": True,
        "random_seed": SEED,
        "random_B": B,
        "random_sampling_algorithm": "Sequential uniform integer rejection per row to sample without replacement, only group size/count used; nested random prefixes for N.",
    },
)
say("TOP LISTS FROZEN", coverage)

calendar = pl.read_parquet(SOURCES["calendar"])
prices = pl.read_parquet(SOURCES["prices"], columns=["date", "code", "open_adj"]).join(
    calendar, on="date", how="inner"
)
maxidx = calendar.filter(pl.col("date") <= pl.datetime(CUT_Y, CUT_M, CUT_D))[
    "idx"
].max()
entry = prices.select(
    "code", (pl.col("idx") - 1).alias("idx"), pl.col("open_adj").alias("entry_open")
)
exitp = prices.select(
    "code", (pl.col("idx") - 4).alias("idx"), pl.col("open_adj").alias("exit_open")
)
evalpool = (
    pool.filter(pl.col("idx") <= maxidx - 4)
    .join(entry, on=["idx", "code"], how="left")
    .join(exitp, on=["idx", "code"], how="left")
)
assert evalpool.height == pool.filter(pl.col("idx") <= maxidx - 4).height
evalpool = evalpool.with_columns(
    pl.when(
        pl.col("entry_open").is_finite()
        & (pl.col("entry_open") > 0)
        & pl.col("exit_open").is_finite()
        & (pl.col("exit_open") > 0)
    )
    .then(pl.col("exit_open") / pl.col("entry_open") - 1)
    .otherwise(None)
    .cast(pl.Float64)
    .alias("r3")
)
del prices, entry, exitp, pool
coverage.update(
    mature_dates=evalpool["date"].n_unique(),
    mature_stock_dates=evalpool.height,
    missing_pool_outcomes=evalpool["r3"].null_count(),
    first_signal=str(evalpool["date"].min()),
    last_mature_signal=str(evalpool["date"].max()),
    last_price_cutoff=CFG.OBSERVATION_CUTOFF,
    ret_dtype=str(evalpool.schema["r3"]),
)
selected_map = {
    int(d["idx"][0]): d.sort("rank")["code"].to_list()
    for d in frozen.partition_by("idx", maintain_order=True)
}
evalpool = evalpool.sort(["idx", "code"])
daily = []
groupcoverage = []
selected_outcomes = []
repsums = {(n, b): np.zeros(B) for n in NS for b in BASES}
repcounts = {(n, b): np.zeros(B, dtype=np.int32) for n in NS for b in BASES}
repzeros = {(n, b): np.zeros(B) for n in NS for b in BASES}
repdays = {(n, b): 0 for n in NS for b in BASES}
repzerodays = {(n, b): 0 for n in NS for b in BASES}
rng = np.random.default_rng(SEED)
draw_hash = hashlib.sha256()


def sample_indices(m, k):
    if k == m:
        # Full group required: any ordering is equivalent for current k; randomize
        # order so smaller-N nested prefixes also remain uniform.
        u = rng.random((B, m))
        return np.argsort(u, axis=1)
    a = np.empty((B, k), dtype=np.int32)
    for j in range(k):
        a[:, j] = rng.integers(0, m, size=B)
        if j:
            bad = (a[:, :j] == a[:, j, None]).any(axis=1)
            while bad.any():
                a[bad, j] = rng.integers(0, m, size=int(bad.sum()))
                bad = (a[:, :j] == a[:, j, None]).any(axis=1)
    return a


parts = evalpool.partition_by("idx", maintain_order=True)
del evalpool
for di, day in enumerate(parts):
    idx = int(day["idx"][0])
    date = day["date"][0].strftime("%Y-%m-%d")
    codes = day["code"].to_list()
    codepos = {c: i for i, c in enumerate(codes)}
    top = np.array([codepos[c] for c in selected_map[idx]], dtype=np.int32)
    y = day["r3"].to_numpy()
    y0 = np.nan_to_num(y, nan=0.0)
    finite = np.isfinite(y)
    industry = np.asarray(day["industry"].to_list())
    size = day["size_decile"].to_numpy()
    labels = {
        "pool": np.zeros(len(y), dtype=np.int32),
        "industry": np.unique(industry, return_inverse=True)[1],
        "size": size,
        "industry_x_size": np.unique(
            np.char.add(np.char.add(industry.astype(str), "|"), size.astype(str)),
            return_inverse=True,
        )[1],
    }
    for n in NS:
        for i in top[:n]:
            selected_outcomes.append(
                {
                    "date": date,
                    "idx": idx,
                    "N": n,
                    "code": codes[i],
                    "r3": float(y[i]) if finite[i] else None,
                }
            )
    for base, g in labels.items():
        groups = {label: np.flatnonzero(g == label) for label in np.unique(g[top])}
        sums = {n: np.zeros(B) for n in NS}
        cnts = {n: np.zeros(B, dtype=np.int16) for n in NS}
        analytic = {n: 0.0 for n in NS}
        analytic0 = {n: 0.0 for n in NS}
        valid_analytic = {n: True for n in NS}
        g_sizes = {n: [] for n in NS}
        g_counts = {n: [] for n in NS}
        for label, positions in groups.items():
            ks = {n: int((g[top[:n]] == label).sum()) for n in NS}
            draw = sample_indices(len(positions), ks[20])
            draw_hash.update(draw.tobytes())
            chosen = positions[draw]
            value = y0[chosen]
            observed = finite[chosen]
            groupmean = (
                float(np.mean(y[positions][finite[positions]]))
                if finite[positions].any()
                else np.nan
            )
            groupzero = float(np.mean(y0[positions]))
            for n, k in ks.items():
                if k == 0:
                    continue
                sums[n] += value[:, :k].sum(axis=1)
                cnts[n] += observed[:, :k].sum(axis=1)
                analytic[n] += k / n * groupmean
                analytic0[n] += k / n * groupzero
                valid_analytic[n] &= np.isfinite(groupmean)
                g_sizes[n].extend([len(positions)] * k)
                g_counts[n].append(k)
        for n in NS:
            actual = (
                float(np.mean(y[top[:n]][finite[top[:n]]]))
                if finite[top[:n]].any()
                else np.nan
            )
            actual0 = float(np.sum(y0[top[:n]]) / n)
            random = np.divide(
                sums[n], cnts[n], out=np.full(B, np.nan), where=cnts[n] > 0
            )
            random0 = sums[n] / n
            paired = bool(np.isfinite(actual) and valid_analytic[n])
            if paired:
                repsums[n, base] += np.nan_to_num(random, nan=0.0)
                repcounts[n, base] += np.isfinite(random)
                repdays[n, base] += 1
            repzeros[n, base] += random0
            repzerodays[n, base] += 1
            daily.append(
                {
                    "date": date,
                    "idx": idx,
                    "N": n,
                    "benchmark": base,
                    "pool_n": len(y),
                    "selected_observed_n": int(finite[top[:n]].sum()),
                    "selected_missing_n": int((~finite[top[:n]]).sum()),
                    "actual": actual,
                    "group_mean": analytic[n] if valid_analytic[n] else None,
                    "delta": actual - analytic[n] if paired else None,
                    "actual_zero": actual0,
                    "group_mean_zero": analytic0[n],
                    "delta_zero": actual0 - analytic0[n],
                    "random_mean": float(np.nanmean(random)),
                    "random_median": float(np.nanmedian(random)),
                    "random_zero_mean": float(np.mean(random0)),
                    "random_zero_median": float(np.median(random0)),
                    "paired_day": paired,
                    "random_all_missing_draws": int((cnts[n] == 0).sum()),
                    "selected_weighted_group_size_mean": float(np.mean(g_sizes[n])),
                    "selected_group_size_min": min(g_sizes[n]),
                    "selected_group_size_max": max(g_sizes[n]),
                    "selected_singleton_fraction": float(
                        np.mean(np.array(g_sizes[n]) == 1)
                    ),
                    "distinct_selected_groups": len(g_counts[n]),
                }
            )
    if di % 200 == 0:
        say(
            "RANDOM MATCHING",
            di + 1,
            "/",
            len(parts),
            "elapsed",
            round(time.perf_counter() - clock, 1),
        )
csvout("daily_matched_controls.csv", daily)
pl.DataFrame(selected_outcomes).write_parquet(
    HERE / "frozen_selection_outcomes.parquet"
)
dump("coverage.json", coverage)


def bootstrap(x, length=20, B=10000):
    n, p = x.shape
    valid = np.isfinite(x)
    val = np.where(valid, x, 0)
    v = np.vstack([val, val[:length]])
    f = np.vstack([valid, valid[:length]])
    cs = np.vstack([np.zeros((1, p)), np.cumsum(v, axis=0)])
    cc = np.vstack([np.zeros((1, p)), np.cumsum(f, axis=0)])
    pos = np.arange(n)
    s = cs[pos + length] - cs[pos]
    c = cc[pos + length] - cc[pos]
    q, r = divmod(n, length)
    rs = cs[pos + r] - cs[pos]
    rc = cc[pos + r] - cc[pos]
    rg = np.random.default_rng(SEED + 901)
    draws = np.empty((B, p))
    for a in range(0, B, 100):
        k = min(100, B - a)
        ix = rg.integers(0, n, size=(k, q))
        num = s[ix].sum(axis=1)
        den = c[ix].sum(axis=1)
        if r:
            j = rg.integers(0, n, size=k)
            num += rs[j]
            den += rc[j]
        draws[a : a + k] = num / den
    mu = np.nanmean(x, axis=0)
    lo, hi = np.nanquantile(draws, [0.025, 0.975], axis=0)
    p = (1 + np.sum(np.abs(draws - mu) >= np.abs(mu), axis=0)) / (B + 1)
    return mu, lo, hi, p


df = pl.DataFrame(daily).with_columns(pl.col("paired_day").cast(pl.Boolean))
keys = [(n, b) for n in NS for b in BASES if b != "pool"]
matrix = np.column_stack(
    [
        df.filter((pl.col("N") == n) & (pl.col("benchmark") == b))
        .sort("idx")["delta"]
        .to_numpy()
        for n, b in keys
    ]
)
mean, lo, hi, pval = bootstrap(matrix, B=BOOT)
order = np.argsort(pval)
adj = np.empty(9)
adj[order] = np.minimum(1, np.maximum.accumulate(pval[order] * (9 - np.arange(9))))
tests = {
    key: {
        "mean_delta": float(mean[j]),
        "ci_low": float(lo[j]),
        "ci_high": float(hi[j]),
        "p_raw": float(pval[j]),
        "p_holm9": float(adj[j]),
    }
    for j, key in enumerate(keys)
}
summary = []
rep_rows = []
for n in NS:
    for b in BASES:
        sub = df.filter((pl.col("N") == n) & (pl.col("benchmark") == b))
        paired = sub.filter(pl.col("paired_day"))
        dist = repsums[n, b] / repcounts[n, b]
        dist0 = repzeros[n, b] / repzerodays[n, b]
        actual = paired["actual"].mean()
        actual0 = sub["actual_zero"].mean()
        row = {
            "N": n,
            "H": 3,
            "benchmark": b,
            "dates": paired.height,
            "zero_fill_dates": sub.height,
            "actual_mean": actual,
            "analytic_group_mean": paired["group_mean"].mean(),
            "actual_minus_group_mean": paired["delta"].mean(),
            "actual_zero_mean": sub["actual_zero"].mean(),
            "analytic_group_zero_mean": sub["group_mean_zero"].mean(),
            "actual_minus_group_mean_zero": sub["delta_zero"].mean(),
            "conditional_random_median": float(np.median(dist)),
            "conditional_random_q025": float(np.quantile(dist, 0.025)),
            "conditional_random_q975": float(np.quantile(dist, 0.975)),
            "actual_minus_conditional_random_median": float(actual - np.median(dist)),
            "conditional_random_zero_median": float(np.median(dist0)),
            "actual_minus_conditional_random_zero_median": float(
                actual0 - np.median(dist0)
            ),
            "selected_missing_total": int(sub["selected_missing_n"].sum()),
            "selected_weighted_group_size_mean": sub[
                "selected_weighted_group_size_mean"
            ].mean(),
            "selected_group_size_min": sub["selected_group_size_min"].min(),
            "selected_singleton_fraction": sub["selected_singleton_fraction"].mean(),
            "random_all_missing_draws": int(sub["random_all_missing_draws"].sum()),
            "random_replicate_min_dates": int(repcounts[n, b].min()),
            "random_replicate_max_dates": int(repcounts[n, b].max()),
            "ci_low": None,
            "ci_high": None,
            "p_raw": None,
            "p_holm9": None,
        }
        if (n, b) in tests:
            row.update({k: v for k, v in tests[n, b].items() if k != "mean_delta"})
        summary.append(row)
        for z in range(B):
            rep_rows.append(
                {
                    "N": n,
                    "benchmark": b,
                    "replicate": z,
                    "random_mean": float(dist[z]),
                    "random_zero_mean": float(dist0[z]),
                    "date_count": int(repcounts[n, b][z]),
                }
            )
csvout("matched_control_summary.csv", summary)
csvout("conditional_random_replicates.csv", rep_rows)
dump("summary.json", summary)
after = {
    k: {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
    for k, p in SOURCES.items()
}
assert after == manifest, "Source changed during experiment"
dump("input_manifest_after.json", after)
dump(
    "qa_results.json",
    {
        "completed_at": dt.datetime.now().isoformat(),
        "elapsed_seconds": time.perf_counter() - clock,
        "source_unchanged": after == manifest,
        "score_exact_to_S2": score_error == 0,
        "future_outcomes_used_for_selection": False,
        "endpoint_definition": "open_adj at exact idx+4 divided by exact idx+1 minus1, both float64",
        "random_draw_checksum": draw_hash.hexdigest(),
        "random_B": B,
        "time_bootstrap_B": BOOT,
        "block": 20,
        "holm_family": 9,
        "all_random_draws_without_replacement_by_construction": True,
        "zero_fill_inference": "point sensitivity only",
        "selected_lists_physically_frozen_before_outcome_join": True,
    },
)
say("FINISHED", json.dumps(summary, ensure_ascii=False))
