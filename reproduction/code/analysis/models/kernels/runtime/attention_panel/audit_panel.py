"""Read actual panel dates and verify maturity independently of captions."""

from pathlib import Path
import json
import pandas as pd
import polars as pl

R = Path(__file__).resolve().parent
P = R / "cache/mechanism/mechanism_analysis_panel.parquet"
H = [1, 3, 5, 10, 20, 60, 120, 250]


def main():
    q = pl.read_parquet(P)
    dates = []
    for col, dtype in q.schema.items():
        if isinstance(dtype, pl.Datetime) or dtype == pl.Date:
            dates.append(
                {
                    "column": col,
                    "min": str(q[col].min()),
                    "max": str(q[col].max()),
                    "after_cutoff": q.select(
                        (pl.col(col) > pd.Timestamp("2026-08-31"))
                        .fill_null(False)
                        .sum()
                    ).item(),
                }
            )
    assert all(z["after_cutoff"] == 0 for z in dates), dates
    rows = []
    for lens in ["alpha", "beta", "gamma"]:
        for h in H:
            end = (
                f"attention_target_date_{h}"
                if lens == "alpha"
                else f"open_exit_date_{h}"
            )
            z = q.filter(
                (pl.col("lens") == lens) & pl.col(f"price_return_{h}").is_finite()
            )
            assert z[end].null_count() == 0
            rows.append(
                {
                    "lens": lens,
                    "H": h,
                    "rows": z.height,
                    "dates": z["signal_date"].n_unique(),
                    "first_signal": str(z["signal_date"].min()),
                    "last_signal": str(z["signal_date"].max()),
                    "max_exit": str(z[end].max()),
                }
            )
    own = q.group_by("lens").agg(
        pl.len().alias("rows"),
        pl.col("signal_date").n_unique().alias("dates"),
        pl.col("signal_date").min().alias("min_signal"),
        pl.col("signal_date").max().alias("max_signal"),
    )
    out = {
        "cutoff_inclusive": "2026-08-31",
        "all_actual_date_columns_within_cutoff": True,
        "duplicate_keys": q.height
        - q.select(pl.struct(["lens", "signal_date", "stock_code"]).n_unique()).item(),
        "attention_observation_after_t_minus_2": q.select(
            (pl.col("observation_date") > pl.col("attention_asof_cutoff"))
            .fill_null(False)
            .sum()
        ).item(),
        "membership_gamma_after_previous_state": q.filter(pl.col("lens") == "gamma")
        .select(
            (pl.col("concept_asof_date") > pl.col("concept_state_signal_date"))
            .fill_null(False)
            .sum()
        )
        .item(),
        "by_lens": own.to_dicts(),
        "date_columns": dates,
        "maturity_by_lens_H": rows,
    }
    assert (
        out["duplicate_keys"]
        == out["attention_observation_after_t_minus_2"]
        == out["membership_gamma_after_previous_state"]
        == 0
    )
    (R / "audit/final_panel_cutoff_checks.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(R / "data/maturity_by_lens_H.csv", index=False)
    print(
        json.dumps(
            {
                "rows": q.height,
                "date_columns": len(dates),
                "maturity_rows": len(rows),
                "date_violations": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
