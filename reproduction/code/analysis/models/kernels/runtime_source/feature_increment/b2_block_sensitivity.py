"""Post-primary-result requested fixed block-40/60 sensitivity; no new models."""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from b2_model import (
    ROOT,
    DATA,
    AUDIT,
    HORIZONS,
    SEED,
    save_json,
    digest,
    block_bootstrap,
)


def main():
    specification = {
        "primary_result_has_already_been_seen": True,
        "trigger": "Parent/user requested systematic block-length dependence check, not tuning models or replacing primary results",
        "block_floors": [40, 60],
        "length": "max(block_floor,H)",
        "B": 10000,
        "seed": "20260923 + H + 1000*block_floor",
        "sample": "exact same daily contrast series as primary 96 tests",
        "inference_gate": "floor(n_valid_dates / length) >=20, unchanged definition",
        "multiple_testing": "separate Holm96 within each block-floor scenario; 192 sensitivity rows do not expand or change primary confirmation labels",
        "primary_design_sha256": digest(AUDIT / "frozen_design.json"),
        "primary_result_sha256": digest(DATA / "increment_inference.csv"),
    }
    path = AUDIT / "block40_60_sensitivity_design.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf8")) == specification
    else:
        save_json(path, specification)
        save_json(
            AUDIT / "block40_60_sensitivity_time.json",
            {
                "time_utc": pd.Timestamp.now(tz="UTC"),
                "main_results_seen": True,
                "sha256": digest(path),
            },
        )
    daily = pd.read_parquet(DATA / "daily_portfolio_all.parquet")
    ic = pd.read_parquet(DATA / "daily_ic_all.parquet")
    industry = pd.read_parquet(DATA / "daily_industry_conditional_ic_all.parquet")
    primary = pd.read_csv(DATA / "increment_inference.csv")
    all_results = []
    for floor in [40, 60]:
        rows = []
        for h in HORIZONS:
            part = daily[daily.H.eq(h)]
            ipiv = (
                ic[ic.H.eq(h)]
                .pivot(index="date", columns="method", values="IC")
                .sort_index()
            )
            dates = ipiv.index
            spiv = (
                industry[industry.H.eq(h)]
                .pivot(index="date", columns="method", values="IC")
                .reindex(dates)
            )
            arrays, names = [], []
            for model in ["ridge", "hist"]:
                arrays.append(
                    (ipiv[f"{model}_concept"] - ipiv[f"{model}_ordinary"]).to_numpy()
                )
                names.append((model, "IC", 0, "complete"))
                for n in [20, 100]:
                    for missing in ["complete", "zero"]:
                        p = (
                            part[part.top_n.eq(n)]
                            .pivot(
                                index="date",
                                columns="method",
                                values=f"portfolio_{missing}",
                            )
                            .reindex(dates)
                        )
                        arrays.append(
                            (p[f"{model}_concept"] - p[f"{model}_ordinary"]).to_numpy()
                        )
                        names.append((model, "return", n, missing))
                arrays.append(
                    (spiv[f"{model}_concept"] - spiv[f"{model}_ordinary"]).to_numpy()
                )
                names.append((model, "industry_conditional_IC", 0, "complete"))
            x = np.column_stack(arrays)
            length = max(floor, h)
            nvalid = np.isfinite(x).sum(axis=0)
            eligible = nvalid // length >= 20
            mean = np.nanmean(x, axis=0)
            if eligible.any():
                mean, low, high, pval = block_bootstrap(
                    x, length, SEED + h + 1000 * floor, 10000
                )
            else:
                low, high, pval = [np.full(len(names), np.nan) for _ in range(3)]
            for j, (model, metric, n, missing) in enumerate(names):
                scale = 100 if metric == "return" else 1
                rows.append(
                    {
                        "block_floor": floor,
                        "H": h,
                        "model": model,
                        "metric": metric,
                        "top_n": n,
                        "missing": missing,
                        "n_dates": int(nvalid[j]),
                        "block": length,
                        "independent_blocks": int(nvalid[j] // length),
                        "status": (
                            "SENSITIVITY_INFERENCE_ELIGIBLE"
                            if eligible[j]
                            else "DEFERRED_INSUFFICIENT_BLOCKS"
                        ),
                        "delta": mean[j] * scale,
                        "ci_low": low[j] * scale if eligible[j] else np.nan,
                        "ci_high": high[j] * scale if eligible[j] else np.nan,
                        "p_raw": pval[j] if eligible[j] else np.nan,
                        "unit": "percentage_point" if metric == "return" else "IC",
                    }
                )
        out = pd.DataFrame(rows)
        assert len(out) == 96
        raw = out.p_raw.fillna(1).to_numpy()
        order = np.argsort(raw)
        corrected = np.minimum(
            np.maximum.accumulate(raw[order] * (96 - np.arange(96))), 1
        )
        adj = np.empty(96)
        adj[order] = corrected
        out["p_holm_96"] = adj
        out["sensitivity_positive"] = (
            out.status.eq("SENSITIVITY_INFERENCE_ELIGIBLE")
            & out.delta.gt(0)
            & out.p_holm_96.lt(0.05)
        )
        out["sensitivity_negative"] = (
            out.status.eq("SENSITIVITY_INFERENCE_ELIGIBLE")
            & out.delta.lt(0)
            & out.p_holm_96.lt(0.05)
        )
        check = out.merge(
            primary,
            on=["H", "model", "metric", "top_n", "missing"],
            suffixes=("_sensitivity", "_primary"),
        )
        assert len(check) == 96
        assert np.allclose(check.delta_sensitivity, check.delta_primary, atol=1e-12)
        assert (check.n_dates_sensitivity == check.n_dates_primary).all()
        out.to_csv(
            DATA / f"block{floor}_sensitivity.csv", index=False, encoding="utf-8-sig"
        )
        all_results.append(out)
        print(
            "BLOCK_SENSITIVITY",
            floor,
            "positive",
            int(out.sensitivity_positive.sum()),
            "negative",
            int(out.sensitivity_negative.sum()),
            flush=True,
        )
    result = pd.concat(all_results, ignore_index=True)
    result.to_csv(
        DATA / "block40_60_sensitivity_all.csv", index=False, encoding="utf-8-sig"
    )
    keys = ["H", "model", "metric", "top_n", "missing"]
    focus = primary[primary.confirmed_positive].copy()
    main_copy = focus.copy()
    main_copy["block_floor"] = 20
    main_copy["sensitivity_positive"] = main_copy.confirmed_positive
    comparison = pd.concat(
        [main_copy, result.merge(focus[keys], on=keys, how="inner")], ignore_index=True
    )
    comparison.sort_values(keys + ["block_floor"]).to_csv(
        DATA / "primary_positive_block_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# B2 块长40/60敏感性",
        "",
        "在已看过主结果后，按用户统一要求实施；没有改主设计、参数、预测、主96项校正或主确认标签。只增加统计稳健性诊断，不是新增确认检验。",
        "",
        "同一逐日配对差；L=max(40,H)或max(60,H)；各10,000次bootstrap，各Holm96；样本数和点估计与主结果逐项一致断言通过。",
        "",
        "## 原主分析通过项的块依赖",
        "",
        "| H | 模型 | 指标 | TopN | 缺失口径 | 块底长 | 实际块长 | 增量 | 95% CI | Holm p | 敏感性仍通过 |",
        "|---:|---|---|---:|---|---:|---:|---:|---|---:|---|",
    ]
    for r in comparison.sort_values(keys + ["block_floor"]).itertuples():
        lines.append(
            f"| {r.H} | {r.model} | {r.metric} | {r.top_n} | {r.missing} | {r.block_floor} | {r.block} | {r.delta:.6f} | [{r.ci_low:.6f}, {r.ci_high:.6f}] | {r.p_holm_96:.6f} | {bool(r.sensitivity_positive)} |"
        )
    lines += ["", "## 全96项敏感性汇总", ""]
    for floor, p in result.groupby("block_floor"):
        lines.append(
            f'- 块底长{floor}：正向校正通过{int(p.sensitivity_positive.sum())}项，负向{int(p.sensitivity_negative.sum())}项；支持不足{int(p.status.eq("DEFERRED_INSUFFICIENT_BLOCKS").sum())}项。'
        )
    lines += [
        "",
        "没有按敏感性结果重新挑选模型或交易期。原Top100 H5正向结果仍如实保留，块依赖若存在须同时披露。两种缺失口径是同一个经济对照的敏感性，不能算作两项独立发现。",
    ]
    (ROOT / "BLOCK_SENSITIVITY.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    save_json(
        AUDIT / "block_sensitivity_qa.json",
        {
            "same_daily_samples_and_point_estimates_as_primary": "PASS",
            "rows": 192,
            "B_each": 10000,
            "separate_family_size": 96,
            "primary_file_unchanged": digest(DATA / "increment_inference.csv")
            == specification["primary_result_sha256"],
        },
    )
    print(
        comparison[
            keys
            + [
                "block_floor",
                "block",
                "delta",
                "ci_low",
                "ci_high",
                "p_holm_96",
                "sensitivity_positive",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
