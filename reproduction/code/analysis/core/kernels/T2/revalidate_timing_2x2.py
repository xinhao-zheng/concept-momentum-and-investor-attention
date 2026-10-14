"""Fixed current-calendar F5/W10/H3/Top20 timing sensitivity; read-only sources.

Run with Python -B to avoid writing bytecode into archived dependency paths.
"""

from pathlib import Path
import os, sys, json, hashlib, datetime, time, math, platform

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))
import cutoff as CFG

OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("POLARS_MAX_THREADS", "8")
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

S2 = CFG.NEW_S2
CACHE = S2 / "cache"
AUDIT = S2 / "audit"
CUTOFF = datetime.datetime.fromisoformat(CFG.OBSERVATION_CUTOFF)
PROTOCOL = HERE / "protocol.md"
CELLS = {
    "alpha": ("beta", "close"),
    "beta": ("beta", "open"),
    "gamma": ("gamma", "open"),
    "delta": ("gamma", "close"),
}
CONTRASTS = [
    "membership_at_open_beta_minus_gamma",
    "membership_at_close_alpha_minus_delta",
    "clock_sync_alpha_minus_beta",
    "clock_lag_delta_minus_gamma",
    "interaction_alpha_minus_beta_minus_delta_plus_gamma",
]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(name, obj):
    (OUT / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def csv(name, d):
    d.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def manifest(paths):
    return [
        {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths
    ]


def say(*args):
    print(*args, flush=True)


def holm(p):
    p = np.asarray(p, float)
    idx = np.argsort(p, kind="stable")
    q = np.ones(len(p))
    q[idx] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[idx]))
    return q


def freeze():
    say("Freezing current-pool identities and original beta/gamma calendar lists")
    original = json.loads((AUDIT / "selection_freeze.json").read_text(encoding="utf-8"))
    assert (
        sha(CACHE / "frozen_common_scores.parquet") == original["common_scores_sha256"]
    )
    assert sha(CACHE / "frozen_top100_ranks.parquet") == original["ranks_sha256"]
    cols = ["date", "code", "beta_calendar_FW", "gamma_calendar_FW"] + [
        d + "_calendar_FW_" + v for d in ["beta", "gamma"] for v in ["num", "den"]
    ]
    pool = pl.read_parquet(CACHE / "frozen_common_scores.parquet", columns=cols)
    assert pool.select(pl.struct("date", "code").n_unique()).item() == pool.height
    assert pool.select(pl.all_horizontal(pl.col(cols[2:]).is_finite()).all()).item()
    cal = pl.read_parquet(CACHE / "calendar.parquet").sort("idx")
    assert np.array_equal(cal["idx"].to_numpy(), np.arange(cal.height))
    assert cal["date"].is_sorted() and cal["date"].n_unique() == cal.height
    pool = pool.join(cal, on="date", how="inner")
    assert pool.height == original["rows"]
    assert pool.group_by("date").len()["len"].min() >= 20
    native_errors = {}
    rank_errors = {}
    ranks = []
    for design in ["beta", "gamma"]:
        native = pl.read_parquet(
            CACHE / f"{design}_calendar_scores.parquet", columns=["date", "code", "FW"]
        )
        joined = pool.select("date", "code", f"{design}_calendar_FW").join(
            native, on=["date", "code"], how="inner"
        )
        assert joined.height == pool.height
        err = joined.select(
            (pl.col(f"{design}_calendar_FW") - pl.col("FW")).abs().max()
        ).item()
        assert err == 0
        native_errors[design] = err
        del native, joined
        from exact_rank_util import rank_table

        r = (
            rank_table(pool, f"{design}_calendar_FW", "FW", {"design": design})
            .filter(pl.col("rank") <= 20)
            .drop("method")
            .join(cal, on="date")
        )
        old = (
            pl.scan_parquet(CACHE / "frozen_top100_ranks.parquet")
            .filter(
                (pl.col("design") == design)
                & (pl.col("clock") == "calendar")
                & (pl.col("method") == "FW")
                & (pl.col("rank") <= 20)
            )
            .select("date", "code", "rank")
            .collect()
        )
        k = ["date", "code", "rank"]
        bad = (
            r.select(k).join(old, on=k, how="anti").height
            + old.join(r.select(k), on=k, how="anti").height
        )
        assert bad == 0
        rank_errors[design] = bad
        ranks.append(r)
    rank = pl.concat(ranks).sort(["design", "date", "rank"])
    pool.select("date", "idx", "code").write_parquet(
        OUT / "frozen_common_pool_keys.parquet", compression="zstd"
    )
    rank.write_parquet(OUT / "frozen_beta_gamma_top20.parquet", compression="zstd")
    frozen = {
        "frozen_at_utc": now(),
        "pool_rows": pool.height,
        "pool_dates": pool["date"].n_unique(),
        "pool_stocks": pool["code"].n_unique(),
        "rank_rows_two_lists": rank.height,
        "original_score_max_abs_errors": native_errors,
        "original_rank_disagreements": rank_errors,
        "unit_to_list_and_clock": CELLS,
        "no_future_endpoint_materialized_before_this_freeze": True,
        "pool_keys_sha256": sha(OUT / "frozen_common_pool_keys.parquet"),
        "ranks_sha256": sha(OUT / "frozen_beta_gamma_top20.parquet"),
        "note": "Original research outcomes had already been examined; this flag describes execution order only.",
    }
    dump("selection_freeze.json", frozen)
    return pool.select("date", "idx", "code"), rank, cal


def endpoints(pool, rank, cal):
    say("Materializing exact close and open H3 endpoints after rank freeze")
    maxidx = cal.filter(pl.col("date") <= CUTOFF)["idx"].max()
    maturity = (
        pool.group_by(["date", "idx"])
        .len()
        .sort("idx")
        .with_columns((pl.col("idx") <= maxidx - 4).alias("both_clocks_mature"))
    )
    maturity.write_csv(OUT / "date_maturity_screen.csv")
    eligible = pool.filter(pl.col("idx") <= maxidx - 4)
    prices = (
        pl.read_parquet(
            CACHE / "prices_latest.parquet",
            columns=["date", "code", "open_adj", "close_adj"],
        )
        .filter(pl.col("date") <= CUTOFF)
        .join(cal, on="date", how="inner")
    )
    assert prices.select(pl.struct("date", "code").n_unique()).item() == prices.height
    q = eligible
    for name, col, offset in [
        ("c0", "close_adj", 0),
        ("c3", "close_adj", 3),
        ("o1", "open_adj", 1),
        ("o4", "open_adj", 4),
    ]:
        ep = prices.select(
            "code", (pl.col("idx") - offset).alias("idx"), pl.col(col).alias(name)
        )
        q = q.join(ep, on=["idx", "code"], how="left")
    for clock, a, b in [("close", "c0", "c3"), ("open", "o1", "o4")]:
        ok = (
            pl.col(a).is_finite()
            & (pl.col(a) > 0)
            & pl.col(b).is_finite()
            & (pl.col(b) > 0)
        )
        q = q.with_columns(
            pl.when(ok)
            .then(pl.col(b) / pl.col(a) - 1)
            .otherwise(None)
            .alias(clock + "_return")
        )
    del prices, pool
    poolmeans = q.group_by(["date", "idx"]).agg(
        pl.len().alias("pool_intended"),
        *[
            e
            for clock in ["close", "open"]
            for e in [
                pl.col(clock + "_return").count().alias(clock + "_pool_observed"),
                pl.col(clock + "_return").mean().alias(clock + "_pool_complete"),
                pl.col(clock + "_return")
                .fill_null(0)
                .mean()
                .alias(clock + "_pool_zero"),
            ]
        ],
    )
    selected = rank.join(q, on=["date", "idx", "code"], how="inner").sort(
        ["design", "date", "rank"]
    )
    selected.write_parquet(OUT / "selected_exact_endpoints.parquet", compression="zstd")
    selected.filter(
        pl.col("close_return").is_null() | pl.col("open_return").is_null()
    ).write_csv(OUT / "missing_selected_endpoints.csv")
    rows = []
    for cell, (design, clock) in CELLS.items():
        r = (
            selected.filter(pl.col("design") == design)
            .group_by(["date", "idx"])
            .agg(
                pl.len().alias("selected_intended"),
                pl.col(clock + "_return").count().alias("selected_observed"),
                pl.col(clock + "_return").mean().alias("selected_complete"),
                pl.col(clock + "_return").fill_null(0).mean().alias("selected_zero"),
            )
        )
        r = r.join(
            poolmeans.select(
                "date",
                "idx",
                "pool_intended",
                pl.col(clock + "_pool_observed").alias("pool_observed"),
                pl.col(clock + "_pool_complete").alias("pool_complete"),
                pl.col(clock + "_pool_zero").alias("pool_zero"),
            ),
            on=["date", "idx"],
        )
        r = r.with_columns(
            pl.lit(cell).alias("cell"),
            pl.lit(design).alias("membership"),
            pl.lit(clock).alias("return_clock"),
            (pl.col("selected_complete") - pl.col("pool_complete")).alias(
                "spread_complete"
            ),
            (pl.col("selected_zero") - pl.col("pool_zero")).alias("spread_zero"),
        )
        rows.append(r)
    d = pl.concat(rows).sort(["date", "cell"])
    support = (
        d.group_by(["date", "idx"])
        .agg(
            pl.len().alias("n_cells"),
            pl.col("spread_complete").is_finite().sum().alias("finite_cells"),
        )
        .with_columns(
            ((pl.col("n_cells") == 4) & (pl.col("finite_cells") == 4)).alias(
                "common_estimable"
            )
        )
    )
    support.sort("idx").write_csv(OUT / "four_cell_date_support.csv")
    d = d.join(support.select("date", "common_estimable"), on="date")
    d.write_csv(OUT / "daily_cells_all_mature.csv")
    d = d.filter(pl.col("common_estimable"))
    d.write_csv(OUT / "daily_cells_common_support.csv")
    qas = {
        "mature_pool_rows": q.height,
        "mature_dates": q["date"].n_unique(),
        "common_dates": d["date"].n_unique(),
        "excluded_mature_dates": support.filter(~pl.col("common_estimable")).height,
        "first_common_signal": str(d["date"].min()),
        "last_common_signal": str(d["date"].max()),
        "source_cutoff": str(CUTOFF.date()),
        "pool_missing_close": q["close_return"].null_count(),
        "pool_missing_open": q["open_return"].null_count(),
        "close_alpha_delta_use_beta_gamma_lists": True,
        "no_selected_replacement": True,
        "minimum_selected_intended": d["selected_intended"].min(),
        "maximum_selected_intended": d["selected_intended"].max(),
    }
    return d.to_pandas(), selected.to_pandas(), qas


def infer(d):
    say(
        "Five paired contrasts, two conventions, shared circular blocks, one Holm20 family"
    )
    frames = []
    levels = []
    labels = []
    mat = []
    for convention in ["complete", "zero"]:
        s = d.pivot(
            index=["date", "idx"], columns="cell", values="spread_" + convention
        ).sort_index()
        a, b, g, e = [
            s[c].to_numpy() * 100 for c in ["alpha", "beta", "gamma", "delta"]
        ]
        x = np.column_stack([b - g, a - e, a - b, e - g, (a - b) - (e - g)])
        assert np.max(np.abs(x[:, 4] - (x[:, 1] - x[:, 0]))) < 1e-12
        for i, c in enumerate(CONTRASTS):
            mat.append(x[:, i])
            labels.append((convention, c))
        out = pd.DataFrame(x, columns=CONTRASTS, index=s.index).reset_index()
        out["missing_convention"] = convention
        frames.append(out)
        for cell in CELLS:
            q = d[d.cell.eq(cell)]
            levels.append(
                {
                    "cell": cell,
                    "membership": CELLS[cell][0],
                    "return_clock": CELLS[cell][1],
                    "missing_convention": convention,
                    "n_dates": len(q),
                    "selected_return_pct": q["selected_" + convention].mean() * 100,
                    "pool_return_pct": q["pool_" + convention].mean() * 100,
                    "spread_pp": q["spread_" + convention].mean() * 100,
                    "selected_intended_total": int(q.selected_intended.sum()),
                    "selected_observed_total": int(q.selected_observed.sum()),
                    "pool_intended_total": int(q.pool_intended.sum()),
                    "pool_observed_total": int(q.pool_observed.sum()),
                }
            )
    csv("daily_paired_contrasts.csv", pd.concat(frames, ignore_index=True))
    csv("cell_levels.csv", pd.DataFrame(levels))
    X = np.column_stack(mat)
    n = len(X)
    mu = X.mean(0)
    rows = []
    for L in [20, 60]:
        B = 20000
        rng = np.random.default_rng(2026092502 + L)
        m = math.ceil(n / L)
        tail = n - (m - 1) * L
        ext = np.vstack([X, X[:L]])
        cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(ext, axis=0)])
        sums = cs[np.arange(n) + L] - cs[np.arange(n)]
        tails = cs[np.arange(n) + tail] - cs[np.arange(n)]
        starts = rng.integers(0, n, size=(B, m))
        means = np.empty((B, X.shape[1]))
        for j in range(X.shape[1]):
            means[:, j] = (sums[starts[:, :-1], j].sum(1) + tails[starts[:, -1], j]) / n
        ci = np.quantile(means, [0.025, 0.975], axis=0)
        p = (1 + (np.abs(means - mu) >= np.abs(mu)).sum(0)) / (B + 1)
        for j, (convention, c) in enumerate(labels):
            rows.append(
                {
                    "contrast": c,
                    "missing_convention": convention,
                    "block_length": L,
                    "replicates": B,
                    "n_dates": n,
                    "nominal_blocks": n // L,
                    "mean_pp": mu[j],
                    "lower95_pp": ci[0, j],
                    "upper95_pp": ci[1, j],
                    "p_raw": p[j],
                    "p_for_family": p[j] if n // L >= 20 else 1.0,
                    "formally_supported": n // L >= 20,
                    "unit": "percentage_points",
                }
            )
    inf = pd.DataFrame(rows)
    inf["p_Holm20"] = holm(inf.p_for_family)
    inf["confirmed_positive"] = (
        inf.formally_supported & inf.mean_pp.gt(0) & inf.p_Holm20.lt(0.05)
    )
    inf["confirmed_negative"] = (
        inf.formally_supported & inf.mean_pp.lt(0) & inf.p_Holm20.lt(0.05)
    )
    csv("paired_inference_Holm20.csv", inf)
    return inf, pd.DataFrame(levels)


def crosschecks(d, selected, cal):
    reference = pd.read_parquet(S2 / "data/daily_portfolio_results.parquet")
    reference = reference[
        reference.design.eq("gamma")
        & reference.clock.eq("calendar")
        & reference.method.eq("FW")
        & reference.top_n.eq(20)
        & reference.H.eq(3)
    ].copy()
    reference["date"] = pd.to_datetime(reference["date"])
    reference = reference.rename(
        columns={"excess_complete": "delta", "excess_zero": "delta_zero"}
    )
    q = d[d.cell.eq("gamma")].merge(
        reference[["date", "delta", "delta_zero"]], on="date", validate="one_to_one"
    )
    errors = {
        "complete": float(np.max(np.abs(q.spread_complete - q.delta))),
        "zero": float(np.max(np.abs(q.spread_zero - q.delta_zero))),
    }
    assert len(q) == len(d[d.cell.eq("gamma")]) and max(errors.values()) < 1e-12
    dates = sorted(d.date.unique())
    picked = [dates[i] for i in [0, len(dates) // 2, len(dates) - 1]]
    calmap = dict(zip(cal["idx"].to_list(), pd.to_datetime(cal["date"].to_list())))
    rawrows = []
    rawman = []
    for date in picked:
        s = selected[selected.design.eq("gamma") & selected.date.eq(date)].sort_values(
            "rank"
        )
        s = s[
            np.isfinite(s[["c0", "c3", "o1", "o4"]]).all(axis=1)
            & s[["c0", "c3", "o1", "o4"]].gt(0).all(axis=1)
        ]
        assert len(s) > 0
        r = s.iloc[0]
        p = CFG.SOURCE_ROOT / "stock-trading-data" / (r.code + ".csv")
        before = sha(p)
        raw = pd.read_csv(
            p,
            encoding="gb18030",
            skiprows=1,
            usecols=["交易日期", "开盘价", "收盘价", "前收盘价"],
        )
        raw["date"] = pd.to_datetime(raw["交易日期"])
        raw = raw.sort_values("date").drop_duplicates("date", keep="last")
        ratio = raw["收盘价"] / raw["前收盘价"]
        raw["c"] = ratio.where(np.isfinite(ratio) & ratio.gt(0)).cumprod()
        raw["o"] = raw.c * raw["开盘价"] / raw["收盘价"]
        raw = raw.set_index("date")
        rr = {
            "close": raw.loc[calmap[r.idx + 3], "c"] / raw.loc[calmap[r.idx], "c"] - 1,
            "open": raw.loc[calmap[r.idx + 4], "o"] / raw.loc[calmap[r.idx + 1], "o"]
            - 1,
        }
        for clock in ["close", "open"]:
            cached = r[clock + "_return"]
            rawrows.append(
                {
                    "date": date,
                    "code": r.code,
                    "rank": int(r["rank"]),
                    "clock": clock,
                    "cached_return": cached,
                    "raw_return": rr[clock],
                    "abs_error": abs(rr[clock] - cached),
                }
            )
        after = sha(p)
        assert before == after
        rawman.append({"path": str(p), "sha256_before": before, "sha256_after": after})
    rawdf = pd.DataFrame(rawrows)
    csv("raw_price_endpoint_checks.csv", rawdf)
    dump("raw_price_input_manifest.json", rawman)
    return {
        "gamma_current_main_daily_reconciliation": errors,
        "gamma_reconciled_dates": len(q),
        "raw_windows": 3,
        "raw_clock_returns_checked": len(rawdf),
        "raw_max_abs_error": float(rawdf.abs_error.max()),
    }


def main():
    started = time.time()
    paths = [
        CACHE / n
        for n in [
            "frozen_common_scores.parquet",
            "frozen_top100_ranks.parquet",
            "beta_calendar_scores.parquet",
            "gamma_calendar_scores.parquet",
            "calendar.parquet",
            "prices_latest.parquet",
        ]
    ] + [
        AUDIT / "selection_freeze.json",
        AUDIT / "frozen_specification.json",
        S2 / "s2_prepare.py",
        PROTOCOL,
        S2 / "data/daily_portfolio_results.parquet",
    ]
    before = manifest(paths)
    dump("input_manifest_before.json", before)
    dump(
        "execution_environment.json",
        {
            "started_utc": now(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "polars": pl.__version__,
            "script_sha256": sha(__file__),
            "protocol_sha256": sha(PROTOCOL),
            "bytecode_writes_disabled": sys.dont_write_bytecode,
            "observation_cutoff": CFG.OBSERVATION_CUTOFF,
        },
    )
    pool, rank, cal = freeze()
    daily, selected, qa = endpoints(pool, rank, cal)
    inf, levels = infer(daily)
    qa.update(crosschecks(daily, selected, cal))
    after = manifest(paths)
    dump("input_manifest_after.json", after)
    assert before == after
    dates = daily[["date", "idx"]].drop_duplicates().sort_values("idx")
    qa.update(
        {
            "source_hashes_unchanged": True,
            "calendar_gaps_on_common_dates": int(dates.idx.diff().dropna().gt(1).sum()),
            "input_manifest_rows": len(before),
            "inference_family_size": len(inf),
            "confirmed_rows": int(
                (inf.confirmed_positive | inf.confirmed_negative).sum()
            ),
            "completed_utc": now(),
            "elapsed_seconds": time.time() - started,
            "script_sha256": sha(__file__),
        }
    )
    dump("summary.json", qa)
    say(
        levels[
            [
                "cell",
                "missing_convention",
                "n_dates",
                "selected_return_pct",
                "pool_return_pct",
                "spread_pp",
            ]
        ].to_string(index=False)
    )
    say(
        inf[
            [
                "contrast",
                "missing_convention",
                "block_length",
                "mean_pp",
                "lower95_pp",
                "upper95_pp",
                "p_Holm20",
            ]
        ].to_string(index=False)
    )
    say("DONE", time.time() - started)
