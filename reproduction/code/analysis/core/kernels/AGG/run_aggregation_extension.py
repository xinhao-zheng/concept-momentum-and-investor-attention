"""Frozen-cache, post-hoc 192-comparison aggregation extension.

Every output is confined to this new directory. No original scripts are imported.
Selection is persisted and fingerprinted before opening any return endpoints.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np
import polars as pl

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "config"))
import cutoff as CFG  # noqa: E402

SOURCE = CFG.NEW_S2
DESIGN_NOTE = HERE / "design.md"
DESIGNS = ["beta", "gamma"]
TOPNS = [5, 20, 100]
HORIZONS = [1, 3, 5, 10, 20, 60, 120, 250]
BASELINES = ["additive_mean", "product_of_means"]
MISSING = ["complete", "zero"]
B, SEED, CUTOFF = 100000, 20260923, CFG.OBSERVATION_CUTOFF
# Fourth identity date moved from 2026-09-08 (beyond the new cutoff) to the last signal date.
IDENTITY_DATES = ["2017-01-03", "2020-06-30", "2023-01-03", "2026-08-31"]
INPUTS = [
    "cache/frozen_common_scores.parquet",
    "cache/prices_latest.parquet",
    "cache/calendar.parquet",
    "data/daily_portfolio_results.parquet",
    "cache/beta_past_pairs.parquet",
    "cache/gamma_past_pairs.parquet",
    "cache/beta_calendar_scores.parquet",
    "cache/gamma_calendar_scores.parquet",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name, value):
    (HERE / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def freeze():
    if (HERE / "frozen_design.json").exists():
        raise FileExistsError(
            "This run is frozen; use a new output directory for a changed design."
        )
    inventory = [
        {
            "path": str(SOURCE / p),
            "relative_path": p,
            "bytes": (SOURCE / p).stat().st_size,
            "sha256": sha(SOURCE / p),
        }
        for p in INPUTS
    ]
    dump("sourcehash.json", inventory)
    dump(
        "frozen_design.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "POST-HOC SENSITIVITY; not preregistered, not unseen data, not original 960 family",
            "rerun_branch": "2026-08-31 unified observation cutoff; design unchanged except the cutoff and the fourth identity date",
            "design_note": str(DESIGN_NOTE),
            "design_note_sha256": sha(DESIGN_NOTE),
            "script_sha256": sha(__file__),
            "sourcehash_sha256": sha(HERE / "sourcehash.json"),
            "cutoff": CUTOFF,
            "score_windows": "unchanged F5/W10",
            "clock": "calendar",
            "designs": DESIGNS,
            "top_n": TOPNS,
            "horizons": HORIZONS,
            "baselines": BASELINES,
            "missing_policies": MISSING,
            "family_size": 192,
            "bootstrap_B": B,
            "seed": SEED,
            "block_length": "max(20,H)",
            "support": "floor(n_signal_dates / L) >= 20",
            "unsupported": "descriptive mean retained, raw p and CI null, p_for_Holm=1",
            "selection": "same original common pool; descending score, ascending code; frozen before outcomes",
            "endpoint": "exact market-calendar open(t+1) to open(t+H+1); no future-driven reselection",
            "sensitivity": "gamma/calendar/Top20/H3; both baselines and missing rules; L=20,40,60; pointwise only",
            "identity_check_dates_fixed_before_outcomes": IDENTITY_DATES,
            "identity_note": "mean(card product) = product(card means) + population card covariance; not a mechanism",
        },
    )
    print("Input hashes and design frozen", flush=True)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(
        1.0, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    )
    answer = np.empty_like(p)
    answer[order] = adjusted
    return answer


def bootstrap(x, block_length, seed, b=B):
    n, k = x.shape
    valid = np.isfinite(x)
    values = np.where(valid, x, 0.0)
    cv = np.vstack(
        [
            np.zeros((1, k)),
            np.cumsum(np.vstack([values, values[:block_length]]), axis=0),
        ]
    )
    cc = np.vstack(
        [np.zeros((1, k)), np.cumsum(np.vstack([valid, valid[:block_length]]), axis=0)]
    )
    starts = np.arange(n)
    sums = cv[starts + block_length] - cv[starts]
    counts = cc[starts + block_length] - cc[starts]
    q, r = divmod(n, block_length)
    rs, rc = cv[starts + r] - cv[starts], cc[starts + r] - cc[starts]
    rng = np.random.default_rng(seed)
    draws = np.empty((b, k))
    for first in range(0, b, 200):
        size = min(200, b - first)
        pick = rng.integers(0, n, size=(size, q))
        total, den = sums[pick].sum(axis=1), counts[pick].sum(axis=1)
        if r:
            tail = rng.integers(0, n, size=size)
            total += rs[tail]
            den += rc[tail]
        draws[first : first + size] = np.divide(
            total, den, out=np.full_like(total, np.nan), where=den > 0
        )
    mean = np.nanmean(x, axis=0)
    lo, hi = np.nanquantile(draws, [0.025, 0.975], axis=0)
    p = (1 + (np.abs(draws - mean) >= np.abs(mean)).sum(axis=0)) / (b + 1)
    return mean, lo, hi, p


def synthetic_tests():
    a, w = np.array([0.1, 0.6, 0.9]), np.array([0.8, 0.4, 0.7])
    cov = np.mean((a - a.mean()) * (w - w.mean()))
    identity_error = abs(np.mean(a * w) - (a.mean() * w.mean() + cov))
    assert identity_error < 1e-14
    p = np.array([0.01, 0.04, 0.03, 1])
    assert np.allclose(holm(p), [0.04, 0.09, 0.09, 1])
    assert np.allclose(bootstrap(np.zeros((12, 2)), 5, 1, b=100)[3], 1)
    tied = pl.DataFrame(
        {
            "date": [1, 1, 1],
            "code": ["000003", "000001", "000002"],
            "score": [2.0, 2.0, 1.0],
        }
    )
    assert (
        tied.sort(["date", "score", "code"], descending=[False, True, False])["code"][0]
        == "000001"
    )
    sample = np.array([0.02, np.nan, -0.01])
    assert np.isclose(np.nanmean(sample), 0.005) and np.isclose(
        np.nan_to_num(sample).mean(), 0.01 / 3
    )
    return {
        "identity_error": identity_error,
        "holm_known_vector": True,
        "all_zero_null_p1": True,
        "code_ascending_ties": True,
        "complete_vs_fixed_denominator_zero": True,
    }


def select_only():
    from exact_rank_util import add_aggregations, ranks

    common = pl.read_parquet(
        SOURCE / "cache/frozen_common_scores.parquet", columns=["date", "code"]
    )
    selections = []
    for design in DESIGNS:
        scores = add_aggregations(
            pl.read_parquet(SOURCE / f"cache/{design}_calendar_scores.parquet")
        ).join(common, on=["date", "code"])
        scores = scores.rename(
            {
                "A": "additive_mean",
                "A_num": "additive_mean_num",
                "A_den": "additive_mean_den",
                "P": "product_of_means",
                "P_num": "product_of_means_num",
                "P_den": "product_of_means_den",
            }
        )
        selections.append(
            ranks(scores, {m: m for m in ["FW"] + BASELINES}, {"design": design})
        )
    frozen = pl.concat(selections).sort(["design", "method", "date", "rank"])
    frozen.write_parquet(HERE / "frozen_top100_selections.parquet", compression="zstd")
    dump(
        "selection_freeze.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "rows": frozen.height,
            "sha256": sha(HERE / "frozen_top100_selections.parquet"),
            "return_endpoints_not_opened": True,
            "future_missing_not_used": True,
        },
    )
    overlaps = []
    for design in DESIGNS:
        for n in TOPNS:
            fw = frozen.filter(
                (pl.col("design") == design)
                & (pl.col("method") == "FW")
                & (pl.col("rank") <= n)
            ).select("date", "code")
            dates = fw.select("date").unique()
            for base in BASELINES:
                alt = frozen.filter(
                    (pl.col("design") == design)
                    & (pl.col("method") == base)
                    & (pl.col("rank") <= n)
                ).select("date", "code")
                shared = (
                    fw.join(alt, on=["date", "code"])
                    .group_by("date")
                    .len(name="n_intersection")
                )
                overlaps.append(
                    dates.join(shared, on="date", how="left")
                    .with_columns(pl.col("n_intersection").fill_null(0))
                    .with_columns(
                        pl.lit(design).alias("design"),
                        pl.lit(n).alias("top_n"),
                        pl.lit(base).alias("baseline"),
                        (pl.col("n_intersection") / n).alias("intersection_fraction"),
                        (
                            pl.col("n_intersection")
                            / (2 * n - pl.col("n_intersection"))
                        ).alias("jaccard"),
                    )
                )
    overlap = pl.concat(overlaps).sort(["design", "top_n", "baseline", "date"])
    overlap.write_csv(HERE / "overlap_daily.csv")
    overlap.group_by(["design", "top_n", "baseline"]).agg(
        pl.len().alias("n_dates"),
        pl.col("intersection_fraction").mean().alias("mean_intersection_fraction"),
        pl.col("intersection_fraction").min().alias("min_intersection_fraction"),
        pl.col("jaccard").mean().alias("mean_jaccard"),
    ).sort(["design", "top_n", "baseline"]).write_csv(HERE / "overlap_summary.csv")
    print("Past-only rankings frozen before outcome access", flush=True)


def evaluate():
    freeze_record = json.loads(
        (HERE / "selection_freeze.json").read_text(encoding="utf-8")
    )
    assert sha(HERE / "frozen_top100_selections.parquet") == freeze_record["sha256"]
    selected = pl.read_parquet(HERE / "frozen_top100_selections.parquet")
    calendar = pl.read_parquet(SOURCE / "cache/calendar.parquet")
    selected = selected.join(calendar, on="date")
    prices = pl.read_parquet(
        SOURCE / "cache/prices_latest.parquet", columns=["date", "code", "open_adj"]
    ).join(calendar, on="date")
    assert prices.select(pl.struct("date", "code").n_unique()).item() == prices.height
    entry = prices.select(
        "code", (pl.col("idx") - 1).alias("idx"), pl.col("open_adj").alias("entry")
    )
    rows = []
    endpoint_rows = []
    for h in HORIZONS:
        exit_ = prices.select(
            "code",
            (pl.col("idx") - h - 1).alias("idx"),
            pl.col("open_adj").alias("exit"),
        )
        valid = selected.filter(pl.col("idx") + h + 1 <= calendar["idx"].max())
        frame = (
            valid.join(entry, on=["code", "idx"], how="left")
            .join(exit_, on=["code", "idx"], how="left")
            .with_columns(
                (
                    pl.col("entry").is_null()
                    | ~pl.col("entry").is_finite()
                    | (pl.col("entry") <= 0)
                ).alias("entry_missing"),
                (
                    pl.col("exit").is_null()
                    | ~pl.col("exit").is_finite()
                    | (pl.col("exit") <= 0)
                ).alias("exit_missing"),
            )
            .with_columns(
                pl.when(~pl.col("entry_missing") & ~pl.col("exit_missing"))
                .then(pl.col("exit") / pl.col("entry") - 1)
                .otherwise(None)
                .alias("outcome")
            )
        )
        endpoint_rows.append(
            frame.with_columns(pl.lit(h).alias("H")).select(
                "date",
                "idx",
                "code",
                "design",
                "method",
                "rank",
                "H",
                "entry",
                "exit",
                "outcome",
                "entry_missing",
                "exit_missing",
            )
        )
        for n in TOPNS:
            rows.append(
                frame.filter(pl.col("rank") <= n)
                .group_by(["date", "design", "method"])
                .agg(
                    pl.len().alias("n_intended"),
                    pl.col("outcome").count().alias("n_observed"),
                    pl.col("entry_missing").sum().alias("n_entry_missing"),
                    pl.col("exit_missing").sum().alias("n_exit_missing"),
                    pl.col("outcome").mean().alias("portfolio_complete"),
                    pl.col("outcome").fill_null(0).mean().alias("portfolio_zero"),
                )
                .with_columns(pl.lit(n).alias("top_n"), pl.lit(h).alias("H"))
            )
    daily = pl.concat(rows).sort(["design", "H", "top_n", "method", "date"])
    daily.write_parquet(HERE / "daily_portfolio_results.parquet", compression="zstd")
    daily.write_csv(HERE / "daily_portfolio_results.csv")
    pl.concat(endpoint_rows).write_parquet(
        HERE / "selected_endpoints.parquet", compression="zstd"
    )
    original = pl.read_parquet(SOURCE / "data/daily_portfolio_results.parquet").filter(
        (pl.col("clock") == "calendar") & (pl.col("method") == "FW")
    )
    recomputed = daily.filter(pl.col("method") == "FW")
    keys = ["date", "design", "method", "top_n", "H"]
    joined = recomputed.join(original, on=keys, suffix="_original", validate="1:1")
    assert joined.height == recomputed.height == original.height
    comparisons = []
    columns = [
        "n_intended",
        "n_observed",
        "n_entry_missing",
        "n_exit_missing",
        "portfolio_complete",
        "portfolio_zero",
    ]
    for col in columns:
        null_mismatch = joined.select(
            (pl.col(col).is_null() != pl.col(col + "_original").is_null()).sum()
        ).item()
        error = joined.select(
            (pl.col(col).cast(pl.Float64) - pl.col(col + "_original").cast(pl.Float64))
            .abs()
            .max()
        ).item()
        assert null_mismatch == 0 and (error is None or error < 1e-12), (
            col,
            error,
            null_mismatch,
        )
        comparisons.append(
            {
                "column": col,
                "max_abs_difference": error,
                "null_mismatches": null_mismatch,
                "rows": joined.height,
            }
        )
    qa = {
        "FW_reproduction_all_H_all_TopN_both_designs": comparisons,
        "no_future_missing_reselection": daily.select(
            (pl.col("n_intended") == pl.col("top_n")).all()
        ).item(),
        "daily_rows": daily.height,
        "endpoint_rows": sum(x.height for x in endpoint_rows),
    }
    assert qa["no_future_missing_reselection"]
    dump("endpoint_QA.json", qa)
    print("All FW date-level returns and endpoint counts reproduced", flush=True)
    return daily


def statistics(daily):
    paired = []
    for missing in MISSING:
        col = f"portfolio_{missing}"
        fw = daily.filter(pl.col("method") == "FW").select(
            "date", "design", "top_n", "H", pl.col(col).alias("FW_return")
        )
        alt = daily.filter(pl.col("method") != "FW").select(
            "date",
            "design",
            "top_n",
            "H",
            pl.col("method").alias("baseline"),
            pl.col(col).alias("baseline_return"),
        )
        paired.append(
            alt.join(
                fw, on=["date", "design", "top_n", "H"], validate="m:1"
            ).with_columns(
                (pl.col("FW_return") - pl.col("baseline_return")).alias("delta"),
                pl.lit(missing).alias("missing_policy"),
            )
        )
    pairs = pl.concat(paired).sort(
        ["design", "H", "top_n", "baseline", "missing_policy", "date"]
    )
    pairs.write_parquet(HERE / "daily_paired_differences.parquet", compression="zstd")
    pairs.write_csv(HERE / "daily_paired_differences.csv")
    results = []
    columns = [
        (n, base, missing) for n in TOPNS for base in BASELINES for missing in MISSING
    ]
    for di, design in enumerate(DESIGNS):
        for h in HORIZONS:
            matrix = []
            dates = None
            for n, base, missing in columns:
                p = pairs.filter(
                    (pl.col("design") == design)
                    & (pl.col("H") == h)
                    & (pl.col("top_n") == n)
                    & (pl.col("baseline") == base)
                    & (pl.col("missing_policy") == missing)
                ).sort("date")
                if dates is None:
                    dates = p["date"]
                else:
                    assert dates.equals(p["date"])
                matrix.append(p["delta"].to_numpy())
            x = np.column_stack(matrix)
            L = max(20, h)
            supported = len(x) // L >= 20
            mean = np.nanmean(x, axis=0)
            if supported:
                mean, lo, hi, pvalue = bootstrap(x, L, SEED + di * 1000 + h)
            for j, (n, base, missing) in enumerate(columns):
                results.append(
                    {
                        "design": design,
                        "clock": "calendar",
                        "top_n": n,
                        "H": h,
                        "baseline": base,
                        "missing_policy": missing,
                        "n_dates": len(x),
                        "n_finite_paired_dates": int(np.isfinite(x[:, j]).sum()),
                        "block_length": L,
                        "n_full_blocks": len(x) // L,
                        "inference_supported": supported,
                        "mean_delta_pp": float(mean[j] * 100),
                        "ci_low_pp": float(lo[j] * 100) if supported else None,
                        "ci_high_pp": float(hi[j] * 100) if supported else None,
                        "p_raw": float(pvalue[j]) if supported else None,
                        "p_for_holm": float(pvalue[j]) if supported else 1.0,
                        "bootstrap_B": B if supported else 0,
                        "seed": SEED + di * 1000 + h,
                        "status": (
                            "post_hoc_supported" if supported else "descriptive_only"
                        ),
                    }
                )
            print("Inference", design, h, "support", supported, flush=True)
    assert len(results) == 192
    adjusted = holm([r["p_for_holm"] for r in results])
    for r, pvalue in zip(results, adjusted):
        r["p_holm_192"] = float(pvalue)
        r["family_pass_005"] = bool(r["inference_supported"] and pvalue < 0.05)
        r["direction"] = (
            "positive"
            if r["mean_delta_pp"] > 0
            else "negative" if r["mean_delta_pp"] < 0 else "zero"
        )
    summary = pl.DataFrame(results)
    summary.write_csv(HERE / "comparison_summary_192.csv")
    summary.filter(pl.col("mean_delta_pp") <= 0).write_csv(
        HERE / "all_nonpositive_comparisons.csv"
    )
    sensitivity = []
    primary = pairs.filter(
        (pl.col("design") == "gamma") & (pl.col("top_n") == 20) & (pl.col("H") == 3)
    )
    simple_columns = [(base, missing) for base in BASELINES for missing in MISSING]
    x = np.column_stack(
        [
            primary.filter(
                (pl.col("baseline") == base) & (pl.col("missing_policy") == missing)
            )
            .sort("date")["delta"]
            .to_numpy()
            for base, missing in simple_columns
        ]
    )
    for L in [20, 40, 60]:
        mean, lo, hi, pv = bootstrap(x, L, SEED + 2000 + L)
        for j, (base, missing) in enumerate(simple_columns):
            sensitivity.append(
                {
                    "design": "gamma",
                    "top_n": 20,
                    "H": 3,
                    "baseline": base,
                    "missing_policy": missing,
                    "block_length": L,
                    "n_dates": len(x),
                    "n_full_blocks": len(x) // L,
                    "inference_supported": len(x) // L >= 20,
                    "mean_delta_pp": float(mean[j] * 100),
                    "ci_low_pp": float(lo[j] * 100),
                    "ci_high_pp": float(hi[j] * 100),
                    "p_raw": float(pv[j]),
                    "bootstrap_B": B,
                    "seed": SEED + 2000 + L,
                    "status": "pointwise_post_hoc_block_sensitivity_no_new_confirmation",
                }
            )
    pl.DataFrame(sensitivity).write_csv(HERE / "primary_block_sensitivity.csv")
    return summary


def identity_audit():
    calendar = pl.read_parquet(SOURCE / "cache/calendar.parquet")
    records = []
    summaries = []
    for design in DESIGNS:
        frozen = pl.scan_parquet(SOURCE / f"cache/{design}_calendar_scores.parquet")
        for date_string in IDENTITY_DATES:
            target = calendar.filter(
                pl.col("date").dt.strftime("%Y-%m-%d") == date_string
            )["idx"].item()
            p = (
                pl.scan_parquet(SOURCE / f"cache/{design}_past_pairs.parquet")
                .filter((pl.col("idx") >= target - 9) & (pl.col("idx") <= target))
                .collect()
                .sort(["code", "concept", "idx"])
            )

            def rolling(expr, window, minimum):
                return expr.rolling_sum_by(
                    "idx", window_size=f"{window}i", min_samples=minimum
                ).over(["code", "concept"])

            expr = [
                (
                    (
                        rolling(pl.col("loo").clip(lower_bound=-0.999).log1p(), 5, 5)
                    ).exp()
                    - 1
                ).alias("Fraw")
            ]
            for name, value in [
                ("n", pl.col("ret") * 0 + 1),
                ("sx", pl.col("ret")),
                ("sy", pl.col("loo")),
                ("sxx", pl.col("ret") ** 2),
                ("syy", pl.col("loo") ** 2),
                ("sxy", pl.col("ret") * pl.col("loo")),
            ]:
                expr.append(rolling(value, 10, 5).alias(name))
            p = p.with_columns(expr).filter(pl.col("idx") == target)
            vx, vy = pl.col("sxx") - pl.col("sx") ** 2 / pl.col("n"), pl.col(
                "syy"
            ) - pl.col("sy") ** 2 / pl.col("n")
            p = p.with_columns(
                pl.when((vx > 1e-15) & (vy > 1e-15))
                .then(
                    (pl.col("sxy") - pl.col("sx") * pl.col("sy") / pl.col("n"))
                    / (vx * vy).sqrt()
                )
                .otherwise(None)
                .alias("Wraw")
            )

            def rank(name):
                col = (
                    pl.when(pl.col(name).is_finite()).then(pl.col(name)).otherwise(None)
                )
                return col.rank(method="min") / col.count()

            p = p.with_columns(
                rank("Fraw").alias("Fr"), rank("Wraw").alias("Wr")
            ).filter(pl.col("Fr").is_finite() & pl.col("Wr").is_finite())
            # Covariance computed independently from centered member-card products.
            p = p.with_columns(
                pl.col("Fr").mean().over("code").alias("Fmean"),
                pl.col("Wr").mean().over("code").alias("Wmean"),
            )
            q = p.group_by("code").agg(
                (pl.col("Fr") * pl.col("Wr")).mean().alias("mean_product"),
                pl.col("Fr").mean().alias("Fmean"),
                pl.col("Wr").mean().alias("Wmean"),
                ((pl.col("Fr") - pl.col("Fmean")) * (pl.col("Wr") - pl.col("Wmean")))
                .mean()
                .alias("cov_population"),
                pl.len().alias("valid_cards"),
            )
            q = q.with_columns(
                (pl.col("Fmean") * pl.col("Wmean")).alias("product_means")
            ).with_columns(
                (
                    pl.col("mean_product")
                    - pl.col("product_means")
                    - pl.col("cov_population")
                ).alias("identity_residual"),
                pl.lit(design).alias("design"),
                pl.lit(date_string).alias("date"),
            )
            stored = (
                frozen.filter(pl.col("idx") == target)
                .select("code", pl.col("FW").alias("stored_FW"))
                .collect()
            )
            q = q.join(stored, on="code", how="left", validate="1:1").with_columns(
                (pl.col("mean_product") - pl.col("stored_FW")).alias(
                    "reconstructed_minus_frozen_FW"
                )
            )
            max_identity = q["identity_residual"].abs().max()
            assert max_identity < 1e-12
            summaries.append(
                {
                    "design": design,
                    "date": date_string,
                    "stocks": q.height,
                    "cards": p.height,
                    "max_identity_residual": max_identity,
                    "max_reconstructed_minus_frozen_FW": q[
                        "reconstructed_minus_frozen_FW"
                    ]
                    .abs()
                    .max(),
                    "note": "F/W rolling statistics reconstructed on 10-day slice; identity independently checked; no mechanism inference",
                }
            )
            records.append(q)
    pl.concat(records).write_csv(HERE / "member_card_identity_detail.csv")
    pl.DataFrame(summaries).write_csv(HERE / "member_card_identity_summary.csv")
    return summaries
