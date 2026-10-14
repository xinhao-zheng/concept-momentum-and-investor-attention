"""Explicitly post-hoc measurement diagnostics. No added hypothesis confirmations."""

import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_m2_measurement import (
    RAW,
    HERE,
    OUTPUT,
    CUTOFF,
    FIELDS,
    NAMES,
    HORIZONS,
    read_source,
    spearman_rows,
)

calendar_df = pd.read_csv(RAW / "stock-main-index-data" / "sh000001.csv")
calendar = pd.DatetimeIndex(pd.to_datetime(calendar_df.candle_end_time))
calendar = calendar[(calendar >= "2016-01-01") & (calendar <= CUTOFF)]
files = sorted((RAW / "stock-popular-concept-detail").glob("*.csv"))
values = np.full((len(calendar), 3, len(files)), np.nan, dtype=np.float64)
original_manifest = pd.read_csv(OUTPUT / "input_checksums.csv").set_index("file")
with ThreadPoolExecutor(max_workers=2) as executor:
    for col, (df, record) in enumerate(executor.map(read_source, files)):
        if record["sha256"] != original_manifest.loc[record["file"], "sha256"]:
            raise RuntimeError("Raw source differs from frozen M2 run")
        positions = calendar.get_indexer(pd.DatetimeIndex(df["交易日期"]))
        valid = positions >= 0
        values[positions[valid], :, col] = df.loc[valid, FIELDS].to_numpy(dtype=float)

complete = np.isfinite(values).all(axis=1)
ranks = np.full_like(values, np.nan)
update_rows = []
for t in range(len(calendar)):
    v = complete[t]
    if v.sum() >= 30:
        ranks[t][:, v] = rankdata(values[t][:, v], axis=1) / v.sum()
    if t == 0:
        continue
    pair = complete[t] & complete[t - 1]
    n = int(pair.sum())
    if not n:
        continue
    a, b = values[t][:, pair], values[t - 1][:, pair]
    for j, name in enumerate(NAMES):
        raw_unchanged = a[j] == b[j]
        f32_unchanged = a[j].astype(np.float32) == b[j].astype(np.float32)
        update_rows.append(
            {
                "date": calendar[t],
                "previous_date": calendar[t - 1],
                "metric": name,
                "n_same_stock_exact_market_pairs": n,
                "n_exact_unchanged": int((a[j] == b[j]).sum()),
                "n_increased": int((a[j] > b[j]).sum()),
                "n_decreased": int((a[j] < b[j]).sum()),
                "fraction_unchanged": float(raw_unchanged.mean()),
                "n_float32_unchanged": int(f32_unchanged.sum()),
                "n_float32_added_zero_changes": int(
                    (f32_unchanged & ~raw_unchanged).sum()
                ),
                "n_current_distinct_values_raw": len(np.unique(a[j])),
                "n_current_distinct_values_float32": len(
                    np.unique(a[j].astype(np.float32))
                ),
            }
        )
updates = pd.DataFrame(update_rows)
updates.to_csv(OUTPUT / "daily_update_frequency.csv", index=False, encoding="utf-8-sig")
summary = []
for name, g in updates.groupby("metric"):
    n = int(g.n_same_stock_exact_market_pairs.sum())
    summary.append(
        {
            "metric": name,
            "n_dates": len(g),
            "n_exact_market_pairs": n,
            "fraction_unchanged_stockday_weighted": g.n_exact_unchanged.sum() / n,
            "fraction_unchanged_date_equal": g.fraction_unchanged.mean(),
            "fraction_increased_stockday_weighted": g.n_increased.sum() / n,
            "fraction_decreased_stockday_weighted": g.n_decreased.sum() / n,
            "n_float32_added_zero_changes": int(g.n_float32_added_zero_changes.sum()),
            "fraction_float32_added_zero_changes": g.n_float32_added_zero_changes.sum()
            / n,
            "float32_lost_distinct_values_sum": int(
                (
                    g.n_current_distinct_values_raw
                    - g.n_current_distinct_values_float32
                ).sum()
            ),
        }
    )
pd.DataFrame(summary).to_csv(
    OUTPUT / "update_frequency_summary.csv", index=False, encoding="utf-8-sig"
)

daily = []
for t in range(5, len(calendar)):
    pair = complete[t] & complete[t - 5]
    n = int(pair.sum())
    if n < 30:
        continue
    # Fixed common universe: equal denominator cancels, half-integer differences are exact.
    reranked_change = rankdata(values[t][:, pair], axis=1) - rankdata(
        values[t - 5][:, pair], axis=1
    )
    n_now, n_past = int(complete[t].sum()), int(complete[t - 5].sum())
    base_change = (
        np.rint(ranks[t][:, pair] * 2 * n_now) * n_past
        - np.rint(ranks[t - 5][:, pair] * 2 * n_past) * n_now
    )
    for h in HORIZONS:
        if t + h >= len(calendar):
            continue
        future = ranks[t + h][:, pair]
        mask = np.isfinite(future).all(axis=0)
        if int(mask.sum()) < 30:
            continue
        baseline = spearman_rows(base_change[:, mask], future[:, mask])
        fixed = spearman_rows(reranked_change[:, mask], future[:, mask])
        if not (np.isfinite(baseline).all() and np.isfinite(fixed).all()):
            continue
        for j, name in enumerate(NAMES):
            daily.append(
                {
                    "date": calendar[t],
                    "H": h,
                    "metric": name,
                    "n_stocks": int(mask.sum()),
                    "baseline_float64": baseline[j],
                    "same_pair_universe_reranked": fixed[j],
                    "rerank_minus_baseline": fixed[j] - baseline[j],
                }
            )
diag = pd.DataFrame(daily)
diag.to_csv(OUTPUT / "paired_reranking_daily.csv", index=False, encoding="utf-8-sig")
agg = (
    diag.groupby(["H", "metric"])
    .agg(
        n_dates=("date", "size"),
        baseline_float64=("baseline_float64", "mean"),
        same_pair_universe_reranked=("same_pair_universe_reranked", "mean"),
        rerank_minus_baseline=("rerank_minus_baseline", "mean"),
    )
    .reset_index()
)
agg.to_csv(OUTPUT / "paired_reranking_summary.csv", index=False, encoding="utf-8-sig")
main = pd.read_csv(OUTPUT / "daily_correlations.csv", parse_dates=["date"])
main = main[main["transform"] == "delta5"].melt(
    id_vars=["date", "H"], value_vars=NAMES, var_name="metric", value_name="main_exact"
)
joined = diag.merge(
    main, on=["date", "H", "metric"], how="outer", validate="one_to_one", indicator=True
)
matched = joined[joined["_merge"] == "both"].copy()
matched["precision_difference"] = matched.baseline_float64 - matched.main_exact
precision = (
    matched.groupby(["H", "metric"])
    .agg(
        n_dates=("date", "size"),
        mean_difference=("precision_difference", "mean"),
        max_abs_difference=("precision_difference", lambda x: x.abs().max()),
    )
    .reset_index()
)
precision.to_csv(OUTPUT / "float64_validation.csv", index=False, encoding="utf-8-sig")
result = {
    "posthoc": True,
    "sources_unchanged": True,
    "row_join_counts": joined["_merge"].value_counts().to_dict(),
    "max_precision_difference": float(matched.precision_difference.abs().max()),
    "precision_gate_0_001_passed": bool(
        matched.precision_difference.abs().max() <= 0.001
    ),
    "no_additional_significance_tests": True,
}
historical_path = (
    HERE / "diagnostics" / "float32_reference" / "results" / "daily_correlations.csv"
)
if historical_path.exists():
    old = pd.read_csv(
        HERE
        / "diagnostics"
        / "float32_reference"
        / "results"
        / "daily_correlations.csv",
        parse_dates=["date"],
    )
    old = old[old["transform"] == "delta5"].melt(
        id_vars=["date", "H"],
        value_vars=NAMES,
        var_name="metric",
        value_name="initial_float32",
    )
    old_comparison = matched.merge(
        old, on=["date", "H", "metric"], validate="one_to_one"
    )
    old_comparison["exact_minus_initial"] = (
        old_comparison.main_exact - old_comparison.initial_float32
    )
    old_comparison.groupby(["H", "metric"]).agg(
        n_dates=("date", "size"),
        mean_difference=("exact_minus_initial", "mean"),
        max_abs_difference=("exact_minus_initial", lambda x: x.abs().max()),
    ).reset_index().to_csv(
        OUTPUT / "rank_precision_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result["max_exact_vs_initial_float32_difference"] = float(
        old_comparison.exact_minus_initial.abs().max()
    )
    result["max_mean_exact_vs_initial_difference"] = float(
        old_comparison.groupby(["H", "metric"]).exact_minus_initial.mean().abs().max()
    )
else:
    result["historical_float32_comparison_status"] = (
        "SKIPPED_ARCHIVED_REFERENCE_UNAVAILABLE"
    )
(OUTPUT / "diagnostic_manifest.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), sharey=True)
colors = {"T": "#C47E40", "S": "#507C9B", "K": "#6B9384"}
names = {"T": "总热度 T", "S": "自选热度 S", "K": "关注黏性 K"}
for ax, name in zip(axes, NAMES):
    g = agg[agg.metric == name].sort_values("H")
    ax.plot(
        g.H,
        g.baseline_float64,
        "--o",
        color="#7E858C",
        label="两期各自股票池排名",
        lw=1.7,
        ms=4,
    )
    ax.plot(
        g.H,
        g.same_pair_universe_reranked,
        "-o",
        color=colors[name],
        label="固定两期共同股票后重排",
        lw=2,
        ms=4,
    )
    ax.axhline(0, color="#90979D", lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS, HORIZONS)
    ax.set_xlabel("未来 H（交易日）")
    ax.set_title(names[name], loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E9ED", lw=0.6)
axes[0].set_ylabel("过去5日位置变化与未来level的秩相关")
axes[0].legend(frameon=False, loc="upper right", fontsize=8)
fig.suptitle(
    "排名所用股票池改变，会改变“热度变化”的含义",
    x=0.07,
    ha="left",
    fontsize=17,
    fontweight="bold",
)
fig.text(
    0.07,
    0.025,
    "事后诊断：相同日期、相同最终配对股票，改变的只是两期排名用的股票池。S 的负号改变，不能解读为反向预测规律。\n本图仅描述，不并入冻结96项家族；未控制当前level，基准位置几何、均值回归和长期规模差异仍未排除。",
    fontsize=9,
    color="#59616A",
)
fig.tight_layout(rect=(0.015, 0.13, 1, 0.9))
fig.savefig(
    HERE / "figures" / "M2C_common_sample_reranking.png", dpi=240, bbox_inches="tight"
)
fig.savefig(HERE / "figures" / "M2C_common_sample_reranking.pdf", bbox_inches="tight")
plt.close(fig)
print(pd.DataFrame(summary).to_string(index=False), flush=True)
print(agg[agg.H.isin([3, 20])].to_string(index=False), flush=True)
print(json.dumps(result, ensure_ascii=False), flush=True)
