"""Adapt the frozen August price/membership caches to the original mechanism protocol.

Membership-only gamma, observation-row F5/W10 scoring and the original hash/date
sampling are retained. No full-lag signal is substituted for gamma here.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
S2 = EXP / "S2/cache"
SHARED = EXP / "dynamics/shared"
CACHE = ROOT / "cache/abg"
H = (1, 3, 5, 10, 20, 60, 120, 250)


def main():
    for p in [CACHE, ROOT / "data/abg"]:
        p.mkdir(parents=True, exist_ok=True)
    cal = (
        pl.read_parquet(S2 / "calendar.parquet")
        .with_columns(pl.col("date").cast(pl.Datetime("ns")))
        .sort("idx")
    )
    assert cal["date"].max() == pd.Timestamp("2026-08-31")
    dates = pd.DatetimeIndex(cal["date"].to_numpy())
    md = (
        pl.scan_parquet(S2 / "membership_latest.parquet")
        .select(pl.col("asof").cast(pl.Datetime("ns")).alias("concept_asof_date"))
        .unique()
        .sort("concept_asof_date")
        .collect()["concept_asof_date"]
        .to_numpy()
    )
    frames = []
    for design, lag in [("beta", 0), ("gamma", 1)]:
        rows = []
        for i, d in enumerate(dates):
            if i < lag:
                continue
            state = dates[i - lag]
            j = np.searchsorted(md, np.datetime64(state), side="right") - 1
            if j >= 0:
                rows.append(
                    {
                        "signal_date": d,
                        f"{design}_concept_state_signal_date": state,
                        f"{design}_concept_asof_date": pd.Timestamp(md[j]),
                    }
                )
        mapping = pl.from_pandas(pd.DataFrame(rows)).with_columns(
            pl.all().cast(pl.Datetime("ns"))
        )
        scores = (
            pl.scan_parquet(S2 / f"{design}_observations_scores.parquet")
            .select(
                pl.col("date").cast(pl.Datetime("ns")).alias("signal_date"),
                pl.col("code").alias("stock_code"),
                pl.col("FW").alias(f"{design}_score_F5_W10_DEFAULT"),
            )
            .join(mapping.lazy(), on="signal_date", how="inner")
            .sort(["signal_date", "stock_code"])
        )
        dest = CACHE / f"{design}_scores_future_free.parquet"
        scores.sink_parquet(dest, compression="zstd")
        frames.append(pl.scan_parquet(dest))
    frames[0].join(frames[1], on=["signal_date", "stock_code"], how="inner").sort(
        ["signal_date", "stock_code"]
    ).sink_parquet(
        CACHE / "common_beta_gamma_scores_future_free.parquet", compression="zstd"
    )
    pl.scan_parquet(S2 / "prices_latest.parquet").select(
        pl.col("date").cast(pl.Datetime("ns")).alias("signal_date"),
        pl.col("code").alias("stock_code"),
        pl.col("ret").cast(pl.Float32).alias("ret_close_1"),
    ).sink_parquet(CACHE / "open_return_panel.parquet", compression="zstd")
    pl.scan_parquet(S2 / "membership_latest.parquet").select(
        pl.col("asof").cast(pl.Datetime("ns")).alias("concept_asof_date"),
        pl.col("code").alias("stock_code"),
        pl.col("concept"),
    ).sink_parquet(CACHE / "raw_stock_concept_snapshots.parquet", compression="zstd")
    # Per-H target dates are nulled when the exact market-calendar endpoint is unavailable.
    rows = []
    for i, d in enumerate(dates):
        for h in H:
            rows.append(
                {
                    "signal_date": d,
                    "holding_h": h,
                    "close_exit_date": dates[i + h] if i + h < len(dates) else pd.NaT,
                    "open_entry_date": dates[i + 1] if i + 1 < len(dates) else pd.NaT,
                    "open_exit_date": (
                        dates[i + h + 1] if i + h + 1 < len(dates) else pd.NaT
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / "data/abg/abg_horizon_calendar.csv", index=False)
    # Bind source paths and the inclusive cutoff; retain the source algorithms.
    for p in (ROOT / "vendor").glob("*.py"):
        s = p.read_text(encoding="utf-8")
        s = s.replace(
            "ROOT = Path(__file__).resolve().parents[2]",
            "ROOT = Path(__file__).resolve().parents[1]",
        )
        s = s.replace("CQ = ROOT.parent", "CQ = ROOT.parents[2]")
        s = s.replace(
            'CUTOFF = pd.Timestamp("2026-07-01")', 'CUTOFF = pd.Timestamp("2026-09-01")'
        )
        s = s.replace(
            'MEMBERSHIP = next(HARD.rglob("raw_stock_concept_snapshots.parquet"))',
            'MEMBERSHIP = ROOT / "cache/abg/raw_stock_concept_snapshots.parquet"',
        )
        s = s.replace(
            'OPEN_PANEL = next(HARD.rglob("open_return_panel.parquet"))',
            'OPEN_PANEL = ROOT / "cache/abg/open_return_panel.parquet"',
        )
        s = s.replace(
            'ROOT / "runtime" / "shared" / "attention_daily.parquet"',
            'ROOT.parent / "dynamics/shared/attention_daily.parquet"',
        )
        s = s.replace(
            'COVARIATES = CACHE / "stock_pit_covariates_daily.parquet"',
            'COVARIATES = ROOT.parent / "dynamics/shared/stock_pit_covariates_daily.parquet"',
        )
        s = s.replace("all < 2026-07-01", "all <= 2026-08-31")
        p.write_text(s, encoding="utf-8")
    meta = {
        "cutoff_inclusive": "2026-08-31",
        "membership_gamma": "previous market day membership; returns through signal-day close",
        "score_clock": "observation rows",
        "source": str(S2),
        "attention_universe": "original gamma score-eligible full cross-section",
        "common_rows": pl.scan_parquet(
            CACHE / "common_beta_gamma_scores_future_free.parquet"
        )
        .select(pl.len())
        .collect()
        .item(),
    }
    (ROOT / "audit/input_adaptation.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
