"""Isolate M2 block-length sensitivity using the original non-overlap algorithm."""

from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / "measurement" / "results"
daily = pd.read_csv(ROOT / "daily_correlations.csv")
names = ["S_minus_T", "S_minus_K", "T_minus_K"]
rows = []
B = 100000
for length in [20, 40]:
    for (transform, h), g in daily.groupby(["transform", "H"]):
        g = g.sort_values("date")
        # Keep the actual full 96-cell family; unsupported endpoints stay p=1.
        L = max(length, int(h))
        k = len(g) // L
        if k < 20:
            for name in names:
                rows.append(
                    dict(
                        L_floor=length,
                        transform=transform,
                        H=h,
                        series=name,
                        n_all=len(g),
                        n_inf=k * L,
                        n_blocks=k,
                        mean=np.nan,
                        low=np.nan,
                        high=np.nan,
                        p=1.0,
                        B=B,
                    )
                )
            continue
        z = g[names].to_numpy()[: k * L].reshape(k, L, 3).mean(axis=1)
        theta = z.mean(axis=0)
        rng = np.random.default_rng(20260922 + int(h))
        boot = np.empty((B, 3))
        for a in range(0, B, 500):
            b = min(500, B - a)
            boot[a : a + b] = z[rng.integers(k, size=(b, k))].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        p = (1 + (abs(boot - theta) >= abs(theta)).sum(axis=0)) / (B + 1)
        for j, name in enumerate(names):
            rows.append(
                dict(
                    L_floor=length,
                    transform=transform,
                    H=h,
                    series=name,
                    n_all=len(g),
                    n_inf=k * L,
                    n_blocks=k,
                    mean=theta[j],
                    low=lo[j],
                    high=hi[j],
                    p=p[j],
                    B=B,
                )
            )
df = pd.DataFrame(rows)
for length in [20, 40]:
    mask = df.L_floor == length
    df.loc[mask, "p_holm_96"] = multipletests(df.loc[mask, "p"], method="holm")[1]
df.to_csv(OUT / "m2_original_algorithm_block_sensitivity.csv", index=False)
print(
    df[
        (df["transform"] == "delta20") & (df.H == 1) & (df.series == "S_minus_K")
    ].to_string(index=False)
)
print(df.assign(reject=df.p_holm_96 < 0.05).groupby("L_floor").reject.sum())
g = daily[(daily["transform"] == "delta20") & (daily.H == 1)].sort_values("date")
PREFIX = 40 * (
    len(g) // 40
)  # the dates used by the 40-day block analysis; 1040 in the 2026-09-18 version
same = g[names].to_numpy()[:PREFIX]
out = []
for L in [20, 40]:
    z = same.reshape(PREFIX // L, L, 3).mean(axis=1)
    theta = z.mean(axis=0)
    rng = np.random.default_rng(20260923)
    boot = np.empty((B, 3))
    for a in range(0, B, 500):
        b = min(500, B - a)
        boot[a : a + b] = z[rng.integers(len(z), size=(b, len(z)))].mean(axis=1)
    p = (1 + (abs(boot - theta) >= abs(theta)).sum(axis=0)) / (B + 1)
    for j, name in enumerate(names):
        out.append(dict(L=L, n=PREFIX, series=name, mean=theta[j], p=p[j], B=B))
pd.DataFrame(out).to_csv(OUT / "m2_same_prefix_length_only.csv", index=False)
print(pd.DataFrame(out).to_string(index=False))
