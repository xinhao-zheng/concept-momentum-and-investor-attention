"""Recalculate annual sample descriptions and mature-date counts."""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--core-cache", type=Path)
    parser.add_argument("--b2-cache", type=Path)
    parser.add_argument("--b1-results", type=Path)
    parser.add_argument("--b2-results", type=Path)
    parser.add_argument("--annual-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    import polars as pl
    import pandas as pd

    core = args.core_cache or cfg.output("core/cache")
    features = args.b2_cache or cfg.output("models/B2/cache")
    b1 = args.b1_results or cfg.output("core/replay/B1/comparison_384.csv")
    b2 = args.b2_results or cfg.output("models/B2/data/portfolio_summary.csv")
    output = cfg.output("sample")
    output.mkdir(parents=True, exist_ok=True)
    pool = pl.read_parquet(
        core / "frozen_common_scores.parquet", columns=["date", "code"]
    ).with_columns(pl.col("date").dt.year().alias("year"))
    score = pl.read_parquet(
        core / "gamma_calendar_scores.parquet", columns=["date", "code", "valid_cards"]
    )
    caps = pl.read_parquet(
        features / "feature_panel.parquet", columns=["date", "code", "float_cap"]
    )
    full = pool.join(score, on=["date", "code"], how="left").join(
        caps, on=["date", "code"], how="left"
    )
    membership = pl.scan_parquet(core / "membership_latest.parquet").with_columns(
        pl.col("asof").dt.year().alias("year")
    )
    sizes = (
        membership.group_by("asof", "year", "concept")
        .len()
        .group_by("asof", "year")
        .agg(pl.len().alias("n_concepts"), pl.col("len").median().alias("median_size"))
        .collect(engine="streaming")
    )
    annual = []
    for year in sorted(full["year"].unique().to_list()) + [None]:
        q = full if year is None else full.filter(pl.col("year") == year)
        m = sizes if year is None else sizes.filter(pl.col("year") == year)
        daily = q.group_by("date").len()
        annual.append(
            {
                "year": "All" if year is None else str(year),
                "dates": q["date"].n_unique(),
                "pool_per_date": daily["len"].mean(),
                "stocks": q["code"].n_unique(),
                "concepts": m["n_concepts"].mean(),
                "median_size": m["median_size"].median(),
                "membership_mean": q["valid_cards"].mean(),
                "membership_median": q["valid_cards"].median(),
                "float_cap_bn": q["float_cap"].median() / 1e9,
            }
        )
    pd.DataFrame(annual).to_csv(
        output / "annual_sample.csv", index=False, lineterminator="\n"
    )
    report = {
        "scope": "Recomputed from research keys, membership, score cards and feature inputs",
        "cutoff": cfg.cutoff,
        "common_stock_days": len(pool),
        "unique_stocks": pool["code"].n_unique(),
        "signal_dates": pool["date"].n_unique(),
        "annual_rows": len(annual),
    }
    if not args.annual_only:
        cal = pl.read_parquet(core / "calendar.parquet")
        dates = pool.select("date").unique().join(cal, on="date")
        last = cal["idx"].max()
        daily = pd.read_csv(b1)
        rolling = pd.read_csv(b2)
        rows = []
        for horizon in [1, 3, 5, 10, 20, 60, 120, 250]:
            mature = dates.filter(pl.col("idx") + horizon + 1 <= last)
            d = daily[(daily.design == "gamma") & (daily.H == horizon)]
            r = rolling[rolling.H == horizon]
            if d.n_dates.nunique() != 1 or r.n_dates.nunique() != 1:
                raise ValueError(
                    "The declared sample counts are not unique within a horizon."
                )
            rows.append(
                {
                    "H": horizon,
                    "historical_rules": len(mature),
                    "daily_adaptations": int(d.n_dates.iloc[0]),
                    "rolling_models": int(r.n_dates.iloc[0]),
                    "last_mature_signal": str(mature["date"].max().date()),
                }
            )
        pd.DataFrame(rows).to_csv(
            output / "mature_signal_dates.csv", index=False, lineterminator="\n"
        )
        report["mature_date_rows"] = len(rows)
    (output / "execution.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
