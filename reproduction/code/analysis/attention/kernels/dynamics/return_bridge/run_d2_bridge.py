"""Historical observational return bridge with read-only inputs."""

from __future__ import annotations
import os

for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[k] = "2"
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
M3 = HERE.parent
C2 = Path("__SNAPSHOT__/models/C2")
OLD = ROOT.parent / "attention-study"
DATA = HERE / "data"
AUDIT = HERE / "audit"
SAMPLE = C2 / "cache/mechanism/mechanism_sample_keys_before_outcomes.parquet"
CF = M3 / "shared/construct_CF1_level_panel.parquet"
PIT = Path("__SNAPSHOT__/input_recovery/stock_pit_covariates_daily.universe.parquet")
CARDS = C2 / "cache/mechanism/protocol_membership_cards_beta_sample_dates.parquet"
CONTROLS = [
    "slow_T",
    "slow_S",
    "slow_K",
    "ret_1",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "log_float_mv_t",
    "log_amount_t",
    "amount_to_float_mv_20_mean",
    "amihud_20",
    "beta_hs300_60",
    "idio_vol_hs300_60",
    "drawdown_from_60d_high",
]
PROXIES = ["T", "S", "K"]
WORLDS = 50
B = 2000
BLOCK = 8
SPLIT_REFERENCE = False


def log(s):
    print(datetime.now(timezone.utc).isoformat(), s, flush=True)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for z in iter(lambda: f.read(8 << 20), b""):
            h.update(z)
    return h.hexdigest()


def savej(p, d):
    Path(p).write_text(
        json.dumps(d, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def freeze():
    if (AUDIT / "frozen_design.json").exists():
        raise FileExistsError("Frozen design already exists")
    savej(
        AUDIT / "frozen_design.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "posthoc_historical_observational_bridge_not_preregistered",
            "H": 20,
            "worlds": WORLDS,
            "B": B,
            "block_sample_dates": BLOCK,
            "seed": 2026092402,
            "family_n": 9,
            "split_reference": SPLIT_REFERENCE,
            "protocol_sha256": sha(HERE / "PROTOCOL.md"),
            "script_sha256": sha(__file__),
            "inputs": {str(p): sha(p) for p in [SAMPLE, CF, PIT, CARDS]},
            "runtime": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "polars": pl.__version__,
            },
        },
    )
    log("design frozen before bridge outcomes")


def peer(A, R):
    good = np.isfinite(R)
    total = A.T @ np.nan_to_num(R, nan=0.0)
    count = A.T @ good.astype(float)
    usable = count >= 6
    inv = np.divide(1.0, count - 1, out=np.zeros_like(count), where=usable)
    denom = A @ usable.astype(float)
    num = A @ (total * inv) - R * (A @ inv)
    return np.divide(num, denom, out=np.full_like(R, np.nan), where=denom > 0)


def corr(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    n = good.sum(1)
    x = np.where(good, x, 0.0)
    y = np.where(good, y, 0.0)
    sx = x.sum(1)
    sy = y.sum(1)
    xx = (x * x).sum(1) - sx * sx / np.maximum(n, 1)
    yy = (y * y).sum(1) - sy * sy / np.maximum(n, 1)
    xy = (x * y).sum(1) - sx * sy / np.maximum(n, 1)
    return np.divide(
        xy,
        np.sqrt(np.maximum(xx * yy, 0)),
        out=np.full(len(n), np.nan),
        where=(n >= 16) & (xx > 1e-12) & (yy > 1e-12),
    )


def fit_date(f):
    exposures = f[[f"fresh_rank_{q}" for q in PROXIES]].to_numpy(float) - 0.5
    cs = []
    for c in CONTROLS:
        v = pd.to_numeric(f[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        cs.append(v.rank(pct=True).fillna(0.5).to_numpy() - 0.5)
        if v.isna().any():
            cs.append(v.isna().to_numpy(float))
    ind = pd.get_dummies(
        f.sw1_industry.fillna("MISSING"), drop_first=True, dtype=float
    ).to_numpy()
    X = np.column_stack([np.ones(len(f)), exposures, *cs, ind])
    Y = f[
        ["future_real_minus_pseudo", "past_real_minus_pseudo", "future_minus_past"]
    ].to_numpy()
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    if ok.sum() < 200:
        return None
    beta = np.linalg.lstsq(X[ok], Y[ok], rcond=1e-9)[0][1:4, :] * 0.1
    return beta, int(ok.sum())


def run():
    if (DATA / "bridge_results.csv").exists():
        raise FileExistsError("Existing complete results are protected")
    st = time.time()
    start = pd.Timestamp("2022-06-01")
    keys = (
        pl.scan_parquet(SAMPLE)
        .select("signal_date", "stock_code")
        .filter(pl.col("signal_date") >= start)
        .unique()
    )
    cols = [
        "signal_date",
        "stock_code",
        "observation_date",
        "pit_date",
        "sw1_industry",
        *[f"fresh_rank_{q}" for q in PROXIES],
        *CONTROLS,
    ]
    f = (
        pl.scan_parquet(CF)
        .select(cols)
        .join(keys, on=["signal_date", "stock_code"], how="semi")
        .collect()
        .to_pandas()
    )
    input_rows = len(f)
    f = f.dropna(
        subset=["observation_date", "pit_date", *[f"fresh_rank_{q}" for q in PROXIES]]
    ).sort_values(["signal_date", "stock_code"])
    if SPLIT_REFERENCE:
        numeric_code = f.stock_code.str.extract(r"(\d+)", expand=False).astype(int)
        f = f[numeric_code % 2 == 1].copy()
    savej(
        AUDIT / "preoutcome_input_gate.json",
        {
            "input_rows": input_rows,
            "rows_after_known_attention_and_pit_date": len(f),
            "exclusion": "Missing attention rank, observation date or PIT control timestamp; before computing bridge outcomes.",
            "calendar_rule": "Past endpoint is strictly before observation date, including non-trading observation dates via searchsorted(left).",
            "implementation_note": "Initial assertion caught NaT PIT dates before outcomes; added explicit eligibility exclusion and calendar-safe weekend mapping. Design and scientific endpoints unchanged.",
            "executed_script_sha256": sha(__file__),
        },
    )
    assert (f.pit_date <= f.observation_date).all()
    pitcols = [
        "signal_date",
        "stock_code",
        "ret_1",
        "source_trade_observed_t",
        "sw1_industry",
        "log_float_mv_t",
        "log_amount_t",
    ]
    pit = (
        pl.scan_parquet(PIT)
        .select(pitcols)
        .filter(pl.col("signal_date") >= pd.Timestamp("2022-03-01"))
        .collect()
        .to_pandas()
    )
    assert not pit.duplicated(["signal_date", "stock_code"]).any()
    pit.loc[~pit.source_trade_observed_t.astype(bool), "ret_1"] = np.nan
    if SPLIT_REFERENCE:
        code_number = pit.stock_code.str.extract(r"(\d+)", expand=False).astype(int)
        reference = pit[code_number % 2 == 0]
        factors = reference.groupby(
            ["signal_date", "sw1_industry"], dropna=False
        ).ret_1.agg(["mean", "count"])
        factors.loc[factors["count"] < 20, "mean"] = np.nan
        pit = pit.join(
            factors["mean"].rename("reference_industry_return"),
            on=["signal_date", "sw1_industry"],
        )
        pit["residual_return"] = pit.ret_1 - pit.reference_industry_return
        factors.reset_index().to_parquet(
            AUDIT / "split_reference_industry_factors.parquet", index=False
        )
        savej(
            AUDIT / "split_reference_gate.json",
            {
                "reference_rule": "numeric stock code even",
                "focal_rule": "numeric stock code odd",
                "reference_and_focal_disjoint": True,
                "minimum_reference_stocks_per_industry_date": 20,
                "focal_sample_rows": len(f),
                "context": "Preliminary conventional industry demeaning can reintroduce focal through industry benchmark; this post-result implementation correction closes that path. Preliminary results retained separately; no claim of preregistration.",
            },
        )
    else:
        pit["residual_return"] = pit.ret_1 - pit.groupby(
            ["signal_date", "sw1_industry"], dropna=False
        ).ret_1.transform("mean")
    codes = np.array(sorted(pit.stock_code.unique()))
    lookup = {v: i for i, v in enumerate(codes)}
    dates = np.array(sorted(pit.signal_date.unique()), dtype="datetime64[ns]")
    date_lookup = {pd.Timestamp(v): i for i, v in enumerate(dates)}
    R = (
        pit.pivot(index="stock_code", columns="signal_date", values="residual_return")
        .reindex(index=codes, columns=dates)
        .to_numpy(float)
    )
    log(
        f"loaded {len(f):,} attention sample rows; return universe {len(codes)} stocks, {len(dates)} dates"
    )
    cards = (
        pl.scan_parquet(CARDS)
        .select("signal_date", "stock_code", "concept", "concept_asof_date")
        .filter(pl.col("signal_date") >= pd.Timestamp("2022-03-01"))
        .collect()
        .to_pandas()
    )
    card_dates = np.array(sorted(cards.signal_date.unique()), dtype="datetime64[ns]")
    card_groups = {
        pd.Timestamp(d): v for d, v in cards.groupby("signal_date", sort=False)
    }
    pit_groups = {
        pd.Timestamp(d): v.set_index("stock_code")
        for d, v in pit.groupby("signal_date", sort=False)
    }
    audits = []
    results = []
    allrows = []
    rng = np.random.default_rng(2026092402)
    for di, (sd, g0) in enumerate(f.groupby("signal_date", sort=True)):
        g = g0.copy()
        si = date_lookup.get(sd, -1)
        if si < 0 or si + 20 >= len(dates):
            continue
        obsmin = g.observation_date.min()
        prior = card_dates[card_dates < np.datetime64(obsmin)]
        if not len(prior):
            continue
        gd = pd.Timestamp(prior[-1])
        cg = card_groups[gd]
        cg = cg[cg.stock_code.isin(lookup)].drop_duplicates(["stock_code", "concept"])
        concepts = {v: i for i, v in enumerate(sorted(cg.concept.unique()))}
        ii = cg.stock_code.map(lookup).to_numpy()
        jj = cg.concept.map(concepts).to_numpy()
        A = csr_matrix((np.ones(len(cg)), (ii, jj)), shape=(len(codes), len(concepts)))
        degree = np.asarray(A.sum(1)).ravel()
        attrs = pit_groups[gd].reindex(codes)
        attrs["degree"] = degree
        attrs["size_bin"] = pd.qcut(
            attrs.log_float_mv_t.rank(method="first"),
            3,
            labels=False,
            duplicates="drop",
        )
        attrs["degree_bin"] = pd.qcut(
            attrs.degree.rank(method="first"), 3, labels=False, duplicates="drop"
        )
        attrs["ix"] = np.arange(len(codes))
        blocks = [
            v.ix.to_numpy()
            for _, v in attrs.groupby(
                ["sw1_industry", "size_bin", "degree_bin"], dropna=False
            )
            if len(v) >= 3
        ]
        exchangeable = sum(map(len, blocks)) / len(codes)
        g = g[g.stock_code.isin(lookup)].copy()
        focal = g.stock_code.map(lookup).to_numpy()
        g["obsi"] = np.searchsorted(
            dates, g.observation_date.to_numpy(dtype="datetime64[ns]"), side="left"
        )
        g = g[g.obsi.notna() & (g.obsi >= 20)].copy()
        focal = g.stock_code.map(lookup).to_numpy()
        obsi = g.obsi.to_numpy(int)
        first = int(obsi.min() - 20)
        last = si + 21
        rr = R[:, first:last]
        fw = np.arange(si + 1, si + 21) - first
        pw = obsi[:, None] - np.arange(20, 0, -1)[None, :] - first

        def outcomes(P):
            yf = corr(rr[focal][:, fw], P[focal][:, fw])
            yp = corr(
                np.take_along_axis(rr[focal], pw, axis=1),
                np.take_along_axis(P[focal], pw, axis=1),
            )
            return np.column_stack([yf, yp])

        real = outcomes(peer(A, rr))
        worlds = []
        dist = []
        numattrs = attrs[["log_float_mv_t", "degree", "log_amount_t"]].to_numpy(float)
        scale = np.nanstd(numattrs, axis=0)
        for w in range(WORLDS):
            perm = np.arange(len(codes))
            for b in blocks:
                perm[b] = rng.permutation(b)
            worlds.append(outcomes(peer(A[perm, :], rr)))
            dist.append(np.nanmean(np.abs(numattrs - numattrs[perm]) / scale, axis=0))
        ww = np.stack(worlds)
        count = np.isfinite(ww).sum(0)
        pseudo = np.divide(
            np.nansum(ww, 0), count, out=np.full_like(real, np.nan), where=count >= 40
        )
        distance = np.mean(dist, axis=0)
        g["future_real_corr"] = real[:, 0]
        g["past_real_corr"] = real[:, 1]
        g["future_pseudo_corr"] = pseudo[:, 0]
        g["past_pseudo_corr"] = pseudo[:, 1]
        g["future_real_minus_pseudo"] = real[:, 0] - pseudo[:, 0]
        g["past_real_minus_pseudo"] = real[:, 1] - pseudo[:, 1]
        g["future_minus_past"] = g.future_real_minus_pseudo - g.past_real_minus_pseudo
        g["graph_date"] = gd
        g["future_first_date"] = pd.Timestamp(dates[si + 1])
        g["future_last_date"] = pd.Timestamp(dates[si + 20])
        gate = bool(exchangeable >= 0.6 and (distance <= 0.1).all())
        g["matching_gate_pass"] = gate
        audits.append(
            {
                "signal_date": sd,
                "graph_date": gd,
                "graph_asof_max": cg.concept_asof_date.max(),
                "min_observation_date": obsmin,
                "max_pit_date": g.pit_date.max(),
                "n_rows": len(g),
                "exchangeable_fraction": exchangeable,
                "mean_absolute_standardized_size_distance": distance[0],
                "mean_absolute_standardized_degree_distance": distance[1],
                "mean_absolute_standardized_liquidity_distance": distance[2],
                "matching_gate_pass": gate,
                "future_first_date": pd.Timestamp(dates[si + 1]),
                "future_last_date": pd.Timestamp(dates[si + 20]),
                "graph_before_all_observations": gd < obsmin,
                "source_asof_before_all_observations": cg.concept_asof_date.max()
                < obsmin,
            }
        )
        fit = fit_date(g)
        if fit:
            beta, n = fit
            for q, proxy in enumerate(PROXIES):
                for y, outcome in enumerate(
                    [
                        "future_real_minus_pseudo",
                        "past_real_minus_pseudo",
                        "future_minus_past",
                    ]
                ):
                    results.append(
                        {
                            "signal_date": sd,
                            "proxy": proxy,
                            "outcome": outcome,
                            "coefficient": beta[q, y],
                            "n_rows": n,
                            "matching_gate_pass": gate,
                        }
                    )
        allrows.append(
            g[
                [
                    "signal_date",
                    "stock_code",
                    "observation_date",
                    "pit_date",
                    "graph_date",
                    *[f"fresh_rank_{q}" for q in PROXIES],
                    "future_real_corr",
                    "past_real_corr",
                    "future_pseudo_corr",
                    "past_pseudo_corr",
                    "future_real_minus_pseudo",
                    "past_real_minus_pseudo",
                    "future_minus_past",
                    "matching_gate_pass",
                ]
            ]
        )
        if di % 20 == 0:
            log(
                f"dates={di+1} elapsed={time.time()-st:.1f}s; matching={gate}; paired distance={distance.round(3)}"
            )
    ad = pd.DataFrame(audits)
    dd = pd.DataFrame(results)
    ad.to_csv(
        AUDIT / "graph_timing_and_matching.csv", index=False, encoding="utf-8-sig"
    )
    dd.to_csv(DATA / "daily_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.concat(allrows, ignore_index=True).to_parquet(
        DATA / "bridge_stock_date_outcomes.parquet", index=False
    )
    wide = dd.pivot(
        index="signal_date", columns=["proxy", "outcome"], values="coefficient"
    ).dropna()
    n = len(wide)
    x = wide.to_numpy()
    mean = x.mean(0)
    brng = np.random.default_rng(2026092403)
    inds = (
        brng.integers(0, n, size=(B, int(np.ceil(n / BLOCK)), 1))
        + np.arange(BLOCK)[None, None, :]
    ) % n
    inds = inds.reshape(B, -1)[:, :n]
    boot = x[inds].mean(1)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    p = (1 + (np.abs(boot - mean) >= np.abs(mean)).sum(0)) / (B + 1)
    holm = multipletests(p, method="holm")[1]
    support = n >= 40 and n // BLOCK >= 8
    matching = bool(ad.matching_gate_pass.all())
    out = []
    for j, (proxy, y) in enumerate(wide.columns):
        out.append(
            {
                "proxy": proxy,
                "outcome": y,
                "holding_h": 20,
                "coefficient": mean[j],
                "diagnostic_ci95_low": low[j],
                "diagnostic_ci95_high": high[j],
                "diagnostic_p_bootstrap": p[j],
                "diagnostic_p_holm_9": holm[j],
                "n_dates": n,
                "independent_blocks_floor": n // BLOCK,
                "support_pass": support,
                "all_date_matching_gate_pass": matching,
                "formal_p_holm_9": holm[j] if support and matching else np.nan,
                "classification": (
                    "OBSERVATIONAL_TIME_SPECIFICITY"
                    if support and matching
                    else "DESCRIPTION_ONLY_MATCHING_GATE_FAIL"
                ),
                "unit": "correlation difference per +10 percentile points of fresh attention",
                "causal_claim_allowed": False,
            }
        )
    pd.DataFrame(out).to_csv(
        DATA / "bridge_results.csv", index=False, encoding="utf-8-sig"
    )
    freezej = json.loads((AUDIT / "frozen_design.json").read_text(encoding="utf-8"))
    savej(
        AUDIT / "completion.json",
        {
            "status": (
                "COMPLETE_WITH_MATCHING_BOUNDARY"
                if not matching
                else "COMPLETE_OBSERVATIONAL"
            ),
            "elapsed_seconds": time.time() - st,
            "date_count": n,
            "stock_date_rows": sum(len(v) for v in allrows),
            "matching_dates_pass": int(ad.matching_gate_pass.sum()),
            "matching_dates_total": len(ad),
            "timing_violations": int(
                (
                    ~ad.graph_before_all_observations
                    | ~ad.source_asof_before_all_observations
                ).sum()
            ),
            "source_hashes_unchanged": {
                p: sha(p) == v for p, v in freezej["inputs"].items()
            },
            "note": "Bootstrap diagnostic values are retained even when matching gates fail; formal inference blank. No external causal identification.",
        },
    )
    log("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--split-reference", action="store_true")
    a = parser.parse_args()
    if a.split_reference:
        SPLIT_REFERENCE = True
        DATA = HERE / "primary_split_reference/data"
        AUDIT = HERE / "primary_split_reference/audit"
    DATA.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    if a.freeze:
        freeze()
    if a.run:
        run()
