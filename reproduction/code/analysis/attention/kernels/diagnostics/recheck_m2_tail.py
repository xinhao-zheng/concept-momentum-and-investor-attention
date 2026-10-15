"""Fixed post-audit tail-coverage sensitivity, with unchanged primary M2 rules."""

from pathlib import Path
import os
import json
import hashlib
import datetime
import time

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT
OUT = EXP / "diagnostics/tail_sensitivity"
OUT.mkdir(exist_ok=True)
SOURCE = EXP / "measurement/results/daily_correlations.csv"
B = 100000
NAMES = ["S_minus_T", "S_minus_K", "T_minus_K"]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protocol = dict(
    status="post_audit_sensitivity_not_primary_or_preregistered",
    source=str(SOURCE.relative_to(ROOT)),
    source_sha256=sha(SOURCE),
    cutoff="2026-08-31",
    reason="Determine whether exclusion of the final incomplete nonoverlap block drives the delta20/H1 S-minus-K comparison.",
    primary="Unchanged complete nonoverlap blocks; reproduce the saved 96-cell family separately at base lengths 20 and 40.",
    sensitivity="Use every available daily correlation. Circular moving blocks of L=max(base_length,H), truncate each resample to exact original n; same indices for three paired contrasts. No imputation and no shortened outcome horizon.",
    base_lengths=[20, 40],
    B=B,
    seed_rule="2026100100 + H, separately initialized for each transform and base length",
    family="All four transforms x eight H x three paired differences = 96 per base length. Unsupported cells keep p=1.",
    support_rule="floor(n/L)>=20 as in the existing diagnostic",
    inference="centered two-sided plus-one bootstrap p; pointwise percentile 95% CI; Holm96",
    forbidden="Do not replace primary estimates, change support thresholds, choose block length by significance or interpret the new specification as independent replication.",
)
frozen = OUT / "protocol_before_results.json"
if frozen.exists():
    old = json.loads(frozen.read_text(encoding="utf-8"))
    assert old["source_sha256"] == protocol["source_sha256"]
else:
    protocol["recorded_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    frozen.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
print("PROTOCOL_SAVED", flush=True)
daily = pd.read_csv(SOURCE)
assert daily.date.max() <= "2026-08-31" and daily.future_date.max() <= "2026-08-31"
assert not daily.duplicated(["transform", "H", "date"]).any()


def inference(z, L, kind, seed):
    n = len(z)
    k = n // L
    if k < 20:
        return np.full(3, np.nan), np.full(3, np.nan), np.ones(3)
    rng = np.random.default_rng(seed)
    boot = np.empty((B, 3))
    if kind == "nonoverlap":
        means = np.add.reduceat(z, np.arange(0, n, L), axis=0) / L
        assert n % L == 0
        theta = z.mean(0)
        for a in range(0, B, 500):
            c = min(500, B - a)
            boot[a : a + c] = means[rng.integers(k, size=(c, k))].mean(1)
    else:
        theta = z.mean(0)
        chunks = int(np.ceil(n / L))
        rem = n - (chunks - 1) * L
        cycle = np.concatenate([z, z[:L]], axis=0)
        prefix = np.vstack([np.zeros((1, 3)), np.cumsum(cycle, axis=0)])
        for a in range(0, B, 500):
            c = min(500, B - a)
            starts = rng.integers(n, size=(c, chunks))
            full = prefix[starts[:, :-1] + L] - prefix[starts[:, :-1]]
            last = prefix[starts[:, -1] + rem] - prefix[starts[:, -1]]
            boot[a : a + c] = (full.sum(1) + last) / n
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    p = (1 + (np.abs(boot - theta) >= np.abs(theta)).sum(0)) / (B + 1)
    return lo, hi, p


results = []
start = time.time()
for base in [20, 40]:
    for (trans, h), g in daily.groupby(["transform", "H"]):
        g = g.sort_values("date")
        z = g[NAMES].to_numpy(float)
        assert np.isfinite(z).all()
        L = max(base, int(h))
        k = len(g) // L
        for kind in ["nonoverlap", "circular_full_sample"]:
            used = g.iloc[: k * L] if kind == "nonoverlap" else g
            v = used[NAMES].to_numpy(float)
            seed = (20260922 if kind == "nonoverlap" else 2026100100) + int(h)
            lo, hi, p = inference(v, L, kind, seed)
            for j, name in enumerate(NAMES):
                results.append(
                    dict(
                        method=kind,
                        L_floor=base,
                        L=L,
                        transform=trans,
                        H=int(h),
                        series=name,
                        n_all=len(g),
                        n_used=len(v),
                        tail_excluded=len(g) - len(v),
                        nominal_blocks=k,
                        signal_first=used.date.min(),
                        signal_last=used.date.max(),
                        outcome_last=used.future_date.max(),
                        mean=(
                            float(v[:, j].mean())
                            if k >= 20 or kind != "nonoverlap"
                            else np.nan
                        ),
                        low=lo[j],
                        high=hi[j],
                        p=p[j],
                        B=B,
                        formal_support=k >= 20,
                    )
                )
    print("BASE_LENGTH_COMPLETE", base, flush=True)
df = pd.DataFrame(results)
for _, ix in df.groupby(["method", "L_floor"]).groups.items():
    assert len(ix) == 96
    df.loc[ix, "p_holm_96"] = multipletests(df.loc[ix, "p"], method="holm")[1]
df.to_csv(
    OUT / "all_384_primary_replication_and_tail_sensitivity.csv",
    index=False,
    encoding="utf-8-sig",
)
old = pd.read_csv(EXP / "diagnostics/m2_original_algorithm_block_sensitivity.csv")
new = df[df.method == "nonoverlap"].merge(
    old,
    on=["L_floor", "transform", "H", "series"],
    suffixes=("_new", "_old"),
    validate="one_to_one",
)
errors = {
    c: float(np.nanmax(np.abs(new[c + "_new"] - new[c + "_old"])))
    for c in ["mean", "low", "high", "p", "p_holm_96"]
}
assert max(errors.values()) < 1e-10, errors
assert np.array_equal(new.n_used, new.n_inf) and np.array_equal(
    new.nominal_blocks, new.n_blocks
)

g = daily[(daily["transform"] == "delta20") & (daily.H == 1)].sort_values("date")
prefix = 40 * (len(g) // 40)
v = g[NAMES].to_numpy()[:prefix]
sr = []
for L in [20, 40]:
    lo, hi, p = inference(v, L, "nonoverlap", 20260923)
    for j, name in enumerate(NAMES):
        sr.append(
            dict(
                L=L,
                n=prefix,
                series=name,
                mean=float(v[:, j].mean()),
                p=float(p[j]),
                B=B,
            )
        )
same = pd.DataFrame(sr)
old_same = pd.read_csv(EXP / "diagnostics/m2_same_prefix_length_only.csv")
joined = same.merge(
    old_same,
    on=["L", "n", "series", "B"],
    suffixes=("_new", "_old"),
    validate="one_to_one",
)
same_err = {
    c: float(np.max(np.abs(joined[c + "_new"] - joined[c + "_old"])))
    for c in ["mean", "p"]
}
assert max(same_err.values()) < 1e-10
same.to_csv(
    OUT / "same_prefix_six_cells_reproduced.csv", index=False, encoding="utf-8-sig"
)
focus = df[(df["transform"] == "delta20") & (df.H == 1) & (df.series == "S_minus_K")]
focus.to_csv(OUT / "delta20_H1_S_minus_K_focus.csv", index=False, encoding="utf-8-sig")
report = dict(
    status="PASS",
    source_sha256=sha(SOURCE),
    primary_recomputed_cells=len(new),
    primary_max_errors=errors,
    same_prefix_recomputed_cells=len(same),
    same_prefix_errors=same_err,
    new_sensitivity_cells=int((df.method == "circular_full_sample").sum()),
    focus=focus.to_dict("records"),
    all_actual_outcomes_by_cutoff=bool((df.outcome_last <= "2026-08-31").all()),
    elapsed_seconds=time.time() - start,
    scope="Recomputed from saved daily correlations; initial raw attention ranks were not regenerated in this run. New full-sample circular-block diagnostic is post-audit and does not change original primary rules.",
)
(OUT / "retest_and_sensitivity_summary.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, default=str), flush=True)
