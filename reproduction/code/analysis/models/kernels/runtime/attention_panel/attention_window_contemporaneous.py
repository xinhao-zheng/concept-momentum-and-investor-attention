"""Descriptive H3 selection comparisons on the observed common T/S/K window.

Window start is determined from finite conservative attention measurements,
and each open-to-open H3 outcome must mature by 2026-08-31 inclusive.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ATT = EXP / "dynamics/shared/attention_daily.parquet"
END_EXIT = pd.Timestamp("2026-08-31")
L, B, SEED = 20, 20000, 20260925


def boot(x):
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(SEED)
    k = int(np.ceil(n / L))
    means = []
    for _ in range(0, B, 1000):
        starts = rng.integers(0, n, size=(min(1000, B - len(means)), k))
        ix = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(
            len(starts), -1
        )[:, :n] % n
        means.extend(x[ix].mean(axis=1))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    dates = (
        pl.scan_parquet(ATT)
        .filter(
            pl.all_horizontal(
                [pl.col(c).is_finite() for c in ["T_delta", "S_level", "K_level"]]
            )
        )
        .select("signal_date")
        .unique()
        .sort("signal_date")
        .collect()["signal_date"]
    )
    start = pd.Timestamp(dates.min())
    cal = pd.read_parquet(EXP / "S2/cache/calendar.parquet")
    cal.date = pd.to_datetime(cal.date)
    last_idx = int(cal[cal.date <= END_EXIT].idx.max()) - 4
    last_signal = cal.set_index("idx").date[last_idx]
    s2 = pd.read_parquet(EXP / "S2/data/daily_portfolio_results.parquet")
    s2 = s2[
        (s2.design == "gamma")
        & (s2.clock == "calendar")
        & (s2.H == 3)
        & (s2.top_n == 20)
    ].copy()
    s2.date = pd.to_datetime(s2.date)
    q = s2[s2.date.between(start, last_signal)]
    w = q.pivot(index="date", columns="method", values="portfolio_complete").dropna(
        subset=["FW"]
    )
    pool = q[q.method == "FW"].set_index("date").universe_complete
    series = [
        ("FW - pool", w.FW - pool.reindex(w.index)),
        ("FW - F-only", w.FW - w.F_only),
        ("FW - W-only", w.FW - w.W_only),
    ]
    v = pd.read_csv(EXP / "V/results/daily_reconstructed.csv")
    v.date = pd.to_datetime(v.date)
    j = (
        v[
            (v.N == 20)
            & (v.benchmark == "industry_x_size")
            & v.paired_day
            & v.date.between(start, last_signal)
        ]
        .set_index("date")
        .delta
    )
    series.append(("FW - industry x size", j))
    rows = []
    for name, z in series:
        x = z.dropna() * 100
        lo, hi = boot(x)
        rows.append(
            {
                "comparison": name,
                "n_dates": len(x),
                "first_signal": str(x.index.min().date()),
                "last_signal": str(x.index.max().date()),
                "mean_pp": x.mean(),
                "ci95_low_pp": lo,
                "ci95_high_pp": hi,
            }
        )
    pd.DataFrame(rows).to_csv(
        HERE / "data/attention_window_contemporaneous.csv", index=False
    )
    meta = {
        "window_rule": "signal dates with observed T/S/K common coverage; exact H3 open exit <= 2026-08-31",
        "first_common_attention_date": str(start.date()),
        "last_signal": str(last_signal.date()),
        "bootstrap": {"L": L, "B": B, "seed": SEED},
        "inference": "descriptive pointwise, no additional correction family",
        "rows": rows,
    }
    (HERE / "audit/attention_window_contemporaneous.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
