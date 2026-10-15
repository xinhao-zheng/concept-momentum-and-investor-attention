"""Independent paired list-width diagnostic. Frozen local inputs only; no fits."""

from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
import pyarrow
from scipy.stats import beta


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p, kind="stable")
    a = np.empty(len(p))
    a[order] = np.minimum(
        1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    )
    return a


def circular_means(x, L, B, rng):
    n = len(x)
    q, r = divmod(n, L)
    # Every bootstrap path has n observations: q full blocks and one partial block.
    doubled = np.concatenate([x, x], axis=0)
    prefix = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(doubled, axis=0)])
    full = prefix[np.arange(n) + L] - prefix[np.arange(n)]
    tail = prefix[np.arange(n) + r] - prefix[np.arange(n)]
    out = np.empty((B, x.shape[1]))
    for i in range(0, B, 1024):
        k = min(1024, B - i)
        starts = rng.integers(0, n, size=(k, q + (r > 0)))
        a = full[starts[:, :q]].sum(axis=1)
        if r:
            a += tail[starts[:, -1]]
        out[i : i + k] = a / n
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    inp = root / "work/root_R2_inputs"
    out = (a.output or root / "work/root_R2_results").resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    manifest = json.loads((inp / "manifest.json").read_text(encoding="utf-8"))
    for f in manifest["files"]:
        source = root / f["local"]
        assert sha(source) == f["sha256"], source
    plan = root / "protocols/R2_portfolio_width.md"
    assert sha(plan) == manifest["plan_sha256"]
    cal = pd.read_parquet(Path("__CORE_CACHE__/calendar.parquet"))
    cal = cal.loc[cal.date <= pd.Timestamp("2026-08-31")].sort_values("idx")
    assert cal.idx.diff().dropna().eq(1).all()
    ci = cal.set_index("idx").date
    di = cal.set_index("date").idx
    daily = pd.concat(
        [pd.read_parquet(inp / f"daily_H5_{y}.parquet") for y in range(2020, 2027)],
        ignore_index=True,
    )
    daily = daily.loc[daily.method.isin(["ridge_ordinary", "ridge_concept"])].copy()
    assert not daily.duplicated(["date", "method", "top_n"]).any()
    assert daily.H.eq(5).all() and daily.n_intended.eq(daily.top_n).all()
    idx = daily.date.map(di)
    assert idx.notna().all() and (idx + 6 <= cal.idx.max()).all()
    sel = pd.concat(
        [
            pd.read_parquet(inp / f"selections_H5_{y}.parquet")
            for y in range(2020, 2027)
        ],
        ignore_index=True,
    )
    assert "outcome" not in sel.columns
    sel = sel.loc[sel.method.isin(["ridge_ordinary", "ridge_concept"])].copy()
    assert not sel.duplicated(["date", "method", "code"]).any()
    assert (
        sel.groupby(["date", "method"])["rank"]
        .agg(["min", "max", "count"])
        .eq([1, 100, 100])
        .all()
        .all()
    )
    # Verify stored ordering among selected names. Global ranking requires full predictions.
    sorted_sel = sel.sort_values(
        ["date", "method", "prediction", "code"], ascending=[True, True, False, True]
    )
    assert (
        sorted_sel["rank"]
        .eq(sorted_sel.groupby(["date", "method"]).cumcount() + 1)
        .all()
    )
    n_immature = int((sel.idx + 6 > cal.idx.max()).sum())
    sel = sel.loc[sel.idx + 6 <= cal.idx.max()].copy()
    sel["entry_date"] = (sel.idx + 1).map(ci)
    sel["exit_date"] = (sel.idx + 6).map(ci)
    prices = pd.read_parquet(
        Path("__CORE_CACHE__/prices_latest.parquet"),
        columns=["date", "code", "open_adj"],
    )
    prices = prices.loc[prices.date <= pd.Timestamp("2026-08-31")]
    assert not prices.duplicated(["date", "code"]).any()
    # Join only selected exact endpoints. Independent implementation uses float64 arithmetic.
    op = prices.set_index(["date", "code"]).open_adj
    sel["entry"] = op.reindex(
        pd.MultiIndex.from_arrays([sel.entry_date, sel.code])
    ).to_numpy()
    sel["exit"] = op.reindex(
        pd.MultiIndex.from_arrays([sel.exit_date, sel.code])
    ).to_numpy()
    ok = (
        np.isfinite(sel.entry)
        & np.isfinite(sel.exit)
        & (sel.entry > 0)
        & (sel.exit > 0)
    )
    sel["return64"] = np.where(ok, sel.exit / sel.entry - 1, np.nan)
    # Original B2 labels are stored as float32 before aggregation, a numerical convention.
    sel["return32"] = sel.return64.astype("float32").astype("float64")
    rebuilt = []
    for n in [20, 100]:
        g = sel.loc[sel["rank"] <= n].groupby(["date", "method"])
        f = g.agg(
            n=("return64", "size"),
            observed=("return64", "count"),
            complete64=("return64", "mean"),
            sum64=("return64", "sum"),
            complete32=("return32", "mean"),
            sum32=("return32", "sum"),
        ).reset_index()
        f["top_n"] = n
        f["zero64"] = f.sum64 / n
        f["zero32"] = f.sum32 / n
        rebuilt.append(f)
    rebuilt = pd.concat(rebuilt, ignore_index=True)
    check = daily.merge(rebuilt, on=["date", "method", "top_n"], validate="one_to_one")
    assert len(check) == len(daily)
    assert check.n_observed.eq(check.observed).all()
    errors = {
        f"{m}_{prec}": float(
            (
                check["portfolio_" + m]
                - check[("complete" if m == "complete" else "zero") + prec]
            )
            .abs()
            .max()
        )
        for m in ["complete", "zero"]
        for prec in ["64", "32"]
    }
    assert max(errors.values()) < 1e-7, errors
    check.to_parquet(out / "daily_reaggregation_comparison.parquet", index=False)
    sel.to_parquet(out / "selections_exact_outcomes.parquet", index=False)
    # Primary diagnostic retains published daily paths; independent outcome replay validates them.
    series = pd.DataFrame(index=sorted(daily.date.unique()))
    for m in ["complete", "zero"]:
        pv = daily.pivot(
            index="date", columns=["method", "top_n"], values="portfolio_" + m
        ).astype("float64")
        for n in [20, 100]:
            series[f"D{n}_{m}"] = 100 * (
                pv[("ridge_concept", n)] - pv[("ridge_ordinary", n)]
            )
        series[f"Q_{m}"] = series[f"D100_{m}"] - series[f"D20_{m}"]
    assert series.notna().all().all()
    assert (
        len(series) == 1609
        and str(series.index.min().date()) == "2020-01-02"
        and str(series.index.max().date()) == "2026-08-21"
    )
    assert abs(series.D100_complete.mean() - 0.139702) < 0.000001
    assert abs(series.D20_complete.mean() - 0.183736) < 0.000001
    series.index.name = "date"
    series.to_csv(out / "paired_daily.csv", encoding="utf-8-sig")
    rows = []
    descriptive = []
    x = series.to_numpy(dtype=np.float64)
    mu = x.mean(axis=0)
    rng = np.random.default_rng(2026100823)
    for L in [20, 40, 60]:
        assert len(x) // L >= 20
        draws = circular_means(x, L, 100000, rng)
        for j, col in enumerate(series.columns):
            lo, hi = np.quantile(draws[:, j], [0.025, 0.975])
            se = draws[:, j].std(ddof=1)
            if col.startswith("Q_"):
                count = int((np.abs(draws[:, j] - mu[j]) >= abs(mu[j])).sum())
                rows.append(
                    dict(
                        estimand=col,
                        block=L,
                        n_dates=len(x),
                        nominal_blocks=len(x) // L,
                        mean_pp=mu[j],
                        ci_low_pp=lo,
                        ci_high_pp=hi,
                        bootstrap_se_pp=se,
                        extreme_count=count,
                        B=100000,
                        p_raw=(count + 1) / 100001,
                    )
                )
            else:
                descriptive.append(
                    dict(
                        metric=col,
                        block=L,
                        mean_pp=mu[j],
                        bootstrap_se_pp=se,
                        pointwise_low_pp=lo,
                        pointwise_high_pp=hi,
                        decision="descriptive_only_no_new_significance_claim",
                    )
                )
    result = pd.DataFrame(rows)
    result["p_holm6"] = holm(result.p_raw)
    result.to_csv(out / "direct_width_inference.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(descriptive).to_csv(
        out / "increment_precision_descriptive.csv", index=False, encoding="utf-8-sig"
    )
    series.groupby(series.index.year).mean().to_csv(
        out / "annual_descriptive.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            dict(excluded_year=y, **series.loc[series.index.year != y].mean().to_dict())
            for y in sorted(set(series.index.year))
        ]
    ).to_csv(out / "leave_year_out_descriptive.csv", index=False, encoding="utf-8-sig")
    # Monte Carlo uncertainty of the reference p value is conditional on its resampling distribution.
    old = pd.read_csv(inp / "increment_inference.csv")
    old = old.loc[(old.H == 5) & (old.model == "ridge") & (old.metric == "return")]
    mc = []
    for row in old.itertuples():
        k = int(round(row.p_raw * 10001 - 1))
        lo = 0 if k == 0 else beta.ppf(0.025, k, 10000 - k + 1)
        hi = 1 if k == 10000 else beta.ppf(0.975, k + 1, 10000 - k)
        mc.append(
            dict(
                top_n=row.top_n,
                missing=row.missing,
                B=10000,
                extreme_count=k,
                p_raw=row.p_raw,
                tail_probability_low=lo,
                tail_probability_high=hi,
                p_holm96=row.p_holm_96,
                note="Monte Carlo tail-probability interval; not a new effect CI or corrected decision",
            )
        )
    pd.DataFrame(mc).to_csv(
        out / "original_p_monte_carlo_precision.csv", index=False, encoding="utf-8-sig"
    )
    receipt = dict(
        status="passed",
        start_epoch=start,
        elapsed_seconds=time.time() - start,
        command=sys.argv,
        python=sys.version,
        numpy=np.__version__,
        pandas=pd.__version__,
        pyarrow=pyarrow.__version__,
        platform=platform.platform(),
        seed=2026100823,
        plan_sha256=sha(plan),
        manifest_sha256=sha(inp / "manifest.json"),
        actual_fits=0,
        bootstrap_replicates_per_length=100000,
        cutoff="2026-08-31",
        last_signal=str(series.index.max().date()),
        last_entry=str(sel.entry_date.max().date()),
        last_exit=str(sel.exit_date.max().date()),
        immature_selection_slots_excluded=n_immature,
        exact_outcome_reaggregation_max_abs_errors=errors,
        return_units="percentage points per complete H5",
        selection_sort_scope="within retained Top100 only; full pool predictive ranking not independently reconstructed",
        correlation_D20_D100=series[["D20_complete", "D100_complete"]]
        .corr()
        .iloc[0, 1],
        source_scope="frozen current daily price cache, predictions-derived selections and daily portfolios; not first-publication source certification",
    )
    (out / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
