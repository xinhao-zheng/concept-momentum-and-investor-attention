"""Post-hoc paired uncertainty audit; frozen daily aggregates, no raw mutation."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "results" / "paired_reranking_daily.csv"
B = 100000


def circular_bootstrap(x, block, seed):
    """Exact n observations, common sampled starts across paired columns."""
    n, k = x.shape
    assert np.isfinite(x).all() and 0 < block <= n
    q, r = divmod(n, block)
    cs = np.vstack([np.zeros((1, k)), np.cumsum(np.vstack([x, x[:block]]), axis=0)])
    starts = np.arange(n)
    full = cs[starts + block] - cs[starts]
    tail = cs[starts + r] - cs[starts]
    rng = np.random.default_rng(seed)
    draws = np.empty((B, k))
    for i in range(0, B, 250):
        z = min(250, B - i)
        ids = rng.integers(n, size=(z, q))
        sums = full[ids].sum(axis=1)
        if r:
            sums += tail[rng.integers(n, size=z)]
        draws[i : i + z] = sums / n
    mean = x.mean(axis=0)
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    p = (1 + (np.abs(draws - mean) >= np.abs(mean)).sum(axis=0)) / (B + 1)
    return mean, low, high, p


def main():
    src = pd.read_csv(SOURCE, parse_dates=["date"])
    assert not src.duplicated(["date", "H", "metric"]).any()
    assert (
        np.max(
            np.abs(
                src.same_pair_universe_reranked
                - src.baseline_float64
                - src.rerank_minus_baseline
            )
        )
        < 1e-12
    )
    columns = [
        "baseline_float64",
        "same_pair_universe_reranked",
        "rerank_minus_baseline",
    ]
    rows = []
    yearly = []
    for (h, metric), group in src.groupby(["H", "metric"]):
        g = group.sort_values("date")
        x = g[columns].to_numpy(float)
        assert np.isfinite(x).all()
        for requested_block in [20, 40]:
            block = max(requested_block, int(h))
            eligible = len(x) // block >= 20
            if eligible:
                mean, lo, hi, p = circular_bootstrap(
                    x, block, 20260923 + int(h) * 100 + requested_block
                )
            else:
                mean = x.mean(axis=0)
                lo = hi = p = np.full(3, np.nan)
            for j, endpoint in enumerate(columns):
                rows.append(
                    dict(
                        H=h,
                        metric=metric,
                        endpoint=endpoint,
                        requested_block=requested_block,
                        actual_block=block,
                        n_dates=len(x),
                        support_blocks=len(x) // block,
                        inference_eligible=eligible,
                        mean=mean[j],
                        pointwise_95ci_low=lo[j],
                        pointwise_95ci_high=hi[j],
                        p_raw=p[j],
                        first_date=g.date.min().date(),
                        last_date=g.date.max().date(),
                    )
                )
        for year, gyear in g.groupby(g.date.dt.year):
            for endpoint in columns:
                yearly.append(
                    dict(
                        H=h,
                        metric=metric,
                        year=year,
                        endpoint=endpoint,
                        n_dates=len(gyear),
                        mean=gyear[endpoint].mean(),
                    )
                )
        print("paired measurement", h, metric, len(x), flush=True)
    out = pd.DataFrame(rows)
    out["p_holm_24_for_shift"] = np.nan
    for block in [20, 40]:
        mask = out.requested_block.eq(block) & out.endpoint.eq("rerank_minus_baseline")
        assert mask.sum() == 24
        out.loc[mask, "p_holm_24_for_shift"] = multipletests(
            out.loc[mask, "p_raw"].fillna(1), method="holm"
        )[1]
    out.to_csv(
        HERE / "paired_rerank_uncertainty.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(yearly).to_csv(
        HERE / "rerank_by_year.csv", index=False, encoding="utf-8-sig"
    )
    qa = {
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "B": B,
        "seed_base": 20260923,
        "status": "post-hoc sensitivity, no causal or held-out claim",
        "scope": "paired changes of a descriptive rank statistic; not conditional return prediction",
        "complete_pairs": True,
        "families": "24 shift tests separately for each block specification",
        "retained_rows": len(src),
        "result_rows": len(out),
        "data_cutoff": "2026-08-31",
        "raw_stock_values_reextracted": False,
    }
    (HERE / "audit_manifest.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out.loc[out.H.eq(3) & out.metric.eq("S")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
