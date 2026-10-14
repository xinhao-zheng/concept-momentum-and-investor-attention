"""Independent price-ratio checks for sampled outcomes and panel demeaning.

Selection deliberately includes the head/tail and the last mature boundary of
every horizon. This checker reads frozen input prices without calling either
the production return builder or any estimation code.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl

R = Path(__file__).resolve().parent
H = [1, 3, 5, 10, 20, 60, 120, 250]
S = R.parent / "S2/cache"
P = R / "cache/mechanism/mechanism_analysis_panel.parquet"


def main():
    calendar = pd.DatetimeIndex(
        pl.read_parquet(S / "calendar.parquet").sort("idx")["date"].to_numpy()
    )
    assert calendar.max() == pd.Timestamp("2026-08-31")
    keys = pl.read_parquet(
        R / "cache/mechanism/mechanism_sample_keys_before_outcomes.parquet",
        columns=["signal_date", "stock_code"],
    )
    wanted = [keys.sample(n=600, seed=20260928), keys.head(60), keys.tail(60)]
    for h in H:
        for shift in [h, h + 1]:
            boundary = calendar[-1 - shift]
            dates = keys["signal_date"].unique().sort()
            lo = dates.filter(dates <= boundary).tail(2)
            hi = dates.filter(dates > boundary).head(2)
            for day in pl.concat([lo, hi]):
                wanted.append(keys.filter(pl.col("signal_date") == day).head(10))
    sel = pl.concat(wanted).unique().sort(["signal_date", "stock_code"])
    cols = (
        ["lens", "signal_date", "stock_code"]
        + [f"price_return_{h}" for h in H]
        + [f"price_excess_{h}" for h in H]
    )
    q = (
        pl.scan_parquet(P)
        .select(cols)
        .join(sel.lazy(), on=["signal_date", "stock_code"], how="inner")
        .collect()
        .to_pandas()
    )
    prices = (
        pl.scan_parquet(S / "prices_latest.parquet")
        .select("date", "code", "open_adj", "close_adj")
        .filter(pl.col("code").is_in(sel["stock_code"].unique().implode()))
        .collect()
        .to_pandas()
        .set_index(["date", "code"])
        .sort_index()
    )
    lookup = {col: prices[col].to_dict() for col in ["open_adj", "close_adj"]}
    idx = calendar.get_indexer(q.signal_date)
    rows = []
    for lens in ["alpha", "beta", "gamma"]:
        mask = q.lens.eq(lens).to_numpy()
        z = q.loc[mask]
        ii = idx[mask]
        for h in H:
            a, b, col = (
                (0, h, "close_adj") if lens == "alpha" else (1, h + 1, "open_adj")
            )
            expected = []
            for i, code in zip(ii, z.stock_code):
                if i + b >= len(calendar):
                    expected.append(np.nan)
                    continue
                p0 = lookup[col].get((calendar[i + a], code), np.nan)
                p1 = lookup[col].get((calendar[i + b], code), np.nan)
                expected.append(
                    p1 / p0 - 1
                    if np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0
                    else np.nan
                )
            expected = np.asarray(expected).astype(np.float32).astype(float)
            actual = z[f"price_return_{h}"].to_numpy(float)
            finite = np.isfinite(expected) & np.isfinite(actual)
            mismatch = int((np.isfinite(expected) != np.isfinite(actual)).sum())
            err = (
                float(np.max(np.abs(expected[finite] - actual[finite])))
                if finite.any()
                else 0.0
            )
            out = {
                "lens": lens,
                "H": h,
                "checked": len(z),
                "finite_expected": int(np.isfinite(expected).sum()),
                "finite_pattern_mismatches": mismatch,
                "max_abs_error_at_saved_float32_precision": err,
            }
            assert mismatch == 0 and err == 0, out
            rows.append(out)
    # Check sample outcomes use the full same-date panel mean before grouping.
    means = (
        pl.scan_parquet(P)
        .select(["lens", "signal_date"] + [f"price_return_{h}" for h in H])
        .group_by("lens", "signal_date")
        .agg([pl.col(f"price_return_{h}").mean().alias(f"mean_{h}") for h in H])
        .collect()
        .to_pandas()
    )
    joined = q.merge(means, on=["lens", "signal_date"], validate="many_to_one")
    excess = []
    for h in H:
        expected = joined[f"price_return_{h}"] - joined[f"mean_{h}"]
        err = (expected - joined[f"price_excess_{h}"]).abs().max()
        # Independent group reduction can differ in the final float32 bits.
        assert float(err) < 2e-6, {"H": h, "excess_error": float(err)}
        excess.append({"H": h, "max_abs_panel_excess_error": float(err)})
    out = {
        "status": "PASS",
        "cutoff_inclusive": "2026-08-31",
        "selected_stock_days": sel.height,
        "price_ratio_checks": rows,
        "panel_demeaning_checks": excess,
        "selection": "seeded random + sample head/tail + last mature boundary for both clocks at every H",
        "precision": "exact equality after cast to the stored Float32 return precision",
    }
    (R / "audit/exact_endpoint_spot_checks.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": out["status"],
                "selected_stock_days": sel.height,
                "checked_returns": sum(x["checked"] for x in rows),
                "finite_pattern_mismatches": 0,
                "max_return_error": 0,
                "max_excess_error": max(
                    x["max_abs_panel_excess_error"] for x in excess
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
