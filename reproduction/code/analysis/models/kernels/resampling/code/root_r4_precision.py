"""Whole-family simulation-precision diagnostic; uses retained prediction paths."""

from pathlib import Path
import argparse, json, shutil, sys, time, datetime
import numpy as np
import pandas as pd
from scipy.stats import beta
from root_r2_width import sha, holm, circular_means

HORIZONS = [1, 3, 5, 10, 20, 60, 120, 250]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    inp = root / "work/root_R4_inputs"
    plan = root / "protocols/R4_simulation_precision.md"
    if a.prepare:
        assert not (inp / "manifest.json").exists(), "Frozen inputs already exist"
        inp.mkdir(parents=True, exist_ok=True)
        rows = []
        base = root.parent / "B2_threads4/data"
        for h in HORIZONS:
            for year in range(2020, 2027):
                for kind in ["daily", "ic", "industry_ic"]:
                    s = base / f"{kind}_H{h}_{year}.parquet"
                    d = inp / s.name
                    shutil.copy2(s, d)
                    rows.append(
                        dict(
                            source=str(s),
                            local=str(d.relative_to(root)),
                            sha256=sha(d),
                            bytes=d.stat().st_size,
                        )
                    )
        s = base / "increment_inference.csv"
        d = inp / s.name
        shutil.copy2(s, d)
        rows.append(
            dict(
                source=str(s),
                local=str(d.relative_to(root)),
                sha256=sha(d),
                bytes=d.stat().st_size,
            )
        )
        record = dict(
            frozen_at=datetime.datetime.now(
                datetime.timezone(datetime.timedelta(hours=8))
            ).isoformat(),
            plan_sha256=sha(plan),
            files=rows,
        )
        (inp / "manifest.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (plan.with_suffix(".sha256")).write_text(sha(plan) + "\n", encoding="utf-8")
        print(
            json.dumps(
                dict(
                    files=len(rows),
                    bytes=sum(x["bytes"] for x in rows),
                    plan_sha256=sha(plan),
                )
            )
        )
        return
    out = (a.output or root / "work/root_R4_results").resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    m = json.loads((inp / "manifest.json").read_text(encoding="utf-8"))
    assert sha(plan) == m["plan_sha256"]
    for r in m["files"]:
        assert sha(root / r["local"]) == r["sha256"]
    original = pd.read_csv(inp / "increment_inference.csv")
    assert len(original) == 96
    cal = pd.read_parquet(Path("__CORE_CACHE__/calendar.parquet"))
    cal = cal[cal.date <= pd.Timestamp("2026-08-31")].sort_values("idx")
    di = cal.set_index("date").idx
    ci = cal.set_index("idx").date
    allrows = []
    clocks = []
    for h in HORIZONS:
        frames = {
            k: pd.concat(
                [
                    pd.read_parquet(inp / f"{k}_H{h}_{y}.parquet")
                    for y in range(2020, 2027)
                ],
                ignore_index=True,
            )
            for k in ["daily", "ic", "industry_ic"]
        }
        columns = []
        meta = []
        for row in original.loc[original.H == h].itertuples():
            if row.metric == "return":
                f = frames["daily"].loc[frames["daily"].top_n == row.top_n]
                pv = f.pivot(
                    index="date", columns="method", values="portfolio_" + row.missing
                ).astype("float64")
                scale = 100
            else:
                f = frames[
                    "industry_ic" if row.metric == "industry_conditional_IC" else "ic"
                ]
                pv = f.pivot(index="date", columns="method", values="IC").astype(
                    "float64"
                )
                scale = 1
            s = scale * (pv[row.model + "_concept"] - pv[row.model + "_ordinary"])
            s = s.dropna().sort_index()
            assert len(s) == row.n_dates
            if columns:
                assert columns[0].index.equals(s.index), "Paired date support differs"
            assert abs(s.mean() - row.delta) < (
                1e-5 if row.metric == "return" else 1e-7
            ), (h, row.metric, s.mean(), row.delta)
            columns.append(s)
            meta.append(
                dict(
                    H=h,
                    model=row.model,
                    metric=row.metric,
                    top_n=row.top_n,
                    missing=row.missing,
                    unit=row.unit,
                    original_delta=row.delta,
                    original_p_raw_L20=row.p_raw,
                    original_p_holm96_L20=row.p_holm_96,
                )
            )
        dates = columns[0].index
        idx = dates.map(di)
        assert idx.notna().all() and (idx + h + 1 <= cal.idx.max()).all()
        clocks.append(
            dict(
                H=h,
                n_dates=len(dates),
                signal_first=str(dates.min().date()),
                signal_last=str(dates.max().date()),
                exit_last=str(ci.loc[int(idx.max() + h + 1)].date()),
                all_endpoints_by_cutoff=True,
            )
        )
        x = np.column_stack([s.to_numpy(dtype=np.float64) for s in columns])
        mu = x.mean(axis=0)
        pd.DataFrame(
            x,
            index=dates,
            columns=[
                f'{r["model"]}_{r["metric"]}_{r["top_n"]}_{r["missing"]}' for r in meta
            ],
        ).to_parquet(out / f"paired_deltas_H{h}.parquet")
        for base_L in [20, 40, 60]:
            L = max(base_L, h)
            supported = len(x) // L >= 20
            if supported:
                rng = np.random.default_rng(np.random.SeedSequence([2026100824, h, L]))
                draws = circular_means(x, L, 1000000, rng)
                qs = np.quantile(draws, [0.025, 0.975], axis=0)
                counts = (np.abs(draws - mu) >= np.abs(mu)).sum(axis=0)
            for j, r in enumerate(meta):
                d = dict(
                    r,
                    base_block=base_L,
                    block=L,
                    n_dates=len(x),
                    nominal_blocks=len(x) // L,
                    delta=mu[j],
                    status=(
                        "INFERENCE_ELIGIBLE"
                        if supported
                        else "DEFERRED_INSUFFICIENT_BLOCKS"
                    ),
                )
                if supported:
                    k = int(counts[j])
                    lo = 0.0 if not k else float(beta.ppf(0.025, k, 1000000 - k + 1))
                    hi = (
                        1.0
                        if k == 1000000
                        else float(beta.ppf(0.975, k + 1, 1000000 - k))
                    )
                    d.update(
                        B=1000000,
                        extreme_count=k,
                        ci_low=qs[0, j],
                        ci_high=qs[1, j],
                        p_raw=(k + 1) / 1000001,
                        mc_tail_low=lo,
                        mc_tail_high=hi,
                    )
                else:
                    d.update(
                        B=0,
                        extreme_count=None,
                        ci_low=None,
                        ci_high=None,
                        p_raw=1.0,
                        mc_tail_low=1.0,
                        mc_tail_high=1.0,
                    )
                allrows.append(d)
            print(
                f"Completed H={h} base={base_L} supported={supported} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
        pd.DataFrame(allrows).to_csv(
            out / "checkpoint_uncorrected.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(clocks).to_csv(
            out / "maturity_by_horizon.csv", index=False, encoding="utf-8-sig"
        )
    result = pd.DataFrame(allrows)
    for b, g in result.groupby("base_block"):
        assert len(g) == 96
        result.loc[g.index, "p_holm96"] = holm(g.p_raw)
        result.loc[g.index, "mc_holm_low"] = holm(g.mc_tail_low)
        result.loc[g.index, "mc_holm_high"] = holm(g.mc_tail_high)
    result["significant"] = result.p_holm96 < 0.05
    result["mc_decision_straddles_005"] = (result.mc_holm_low < 0.05) & (
        result.mc_holm_high >= 0.05
    )
    result.to_csv(out / "inference_288.csv", index=False, encoding="utf-8-sig")
    result.loc[result.base_block == 20].assign(
        original_significant=lambda d: d.original_p_holm96_L20 < 0.05,
        decision_changed=lambda d: d.original_significant != d.significant,
    ).to_csv(out / "original_L20_comparison_96.csv", index=False, encoding="utf-8-sig")
    summary = dict(
        status="passed",
        actual_fits=0,
        B_per_H_block=1000000,
        seed_scheme="SeedSequence([2026100824,H,max(base,H)])",
        elapsed_seconds=time.time() - started,
        command=sys.argv,
        python=sys.version,
        numpy=np.__version__,
        pandas=pd.__version__,
        plan_sha256=sha(plan),
        manifest_sha256=sha(inp / "manifest.json"),
        by_base_block=result.groupby("base_block")
        .agg(
            tests=("delta", "size"),
            supported=("status", lambda z: int((z == "INFERENCE_ELIGIBLE").sum())),
            significant=("significant", "sum"),
            mc_boundary_cells=("mc_decision_straddles_005", "sum"),
        )
        .reset_index()
        .to_dict("records"),
    )
    (out / "receipt.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        result.loc[
            result.significant
            | (
                (result.H == 5)
                & (result.model == "ridge")
                & (result.metric == "return")
            ),
            [
                "base_block",
                "H",
                "model",
                "metric",
                "top_n",
                "missing",
                "delta",
                "p_raw",
                "p_holm96",
                "mc_holm_low",
                "mc_holm_high",
            ],
        ].to_string(index=False)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
