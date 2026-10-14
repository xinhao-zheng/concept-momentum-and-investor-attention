"""Rebuild C1 inputs using the original date/hash sampling and network algorithms."""

from pathlib import Path
import argparse, sys, json, importlib.util
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parents[2] / "attention-study"
CACHE = ROOT / "cache/mechanism"
sys.path.insert(0, str(ROOT / "vendor"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "vendor" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sample():
    old_attention = OLD / "runtime/shared/attention_daily.parquet"
    prefix = (
        pl.scan_parquet(old_attention)
        .select("signal_date", "stock_code")
        .filter(pl.col("signal_date") < pd.Timestamp("2026-07-01"))
    )
    suffix = (
        pl.scan_parquet(ROOT / "cache/abg/gamma_scores_future_free.parquet")
        .select("signal_date", "stock_code")
        .filter(pl.col("signal_date") >= pd.Timestamp("2026-07-01"))
    )
    keysource = CACHE / "attention_key_universe_preoutcome.parquet"
    pl.concat([prefix, suffix]).sort(["signal_date", "stock_code"]).sink_parquet(
        keysource, compression="zstd"
    )
    m = load("freeze_mechanism_sample")
    m.ATTENTION = keysource
    sys.argv = [sys.argv[0]]
    m.main()
    new = pl.read_parquet(m.SAMPLE)
    old = pl.read_parquet(
        OLD / "cache/mechanism/mechanism_sample_keys_before_outcomes.parquet"
    )
    keys = ["signal_date", "stock_code"]
    prefix_new = new.filter(pl.col("signal_date") <= old["signal_date"].max())
    a = prefix_new.select(keys).join(old.select(keys), on=keys, how="anti").height
    b = old.select(keys).join(prefix_new.select(keys), on=keys, how="anti").height
    q = {
        "old_rows": old.height,
        "new_rows": new.height,
        "new_dates": new["signal_date"].n_unique(),
        "last_date": str(new["signal_date"].max()),
        "prefix_added": a,
        "prefix_missing": b,
        "date_hash_sample_unchanged_on_historical_prefix": a == b == 0,
    }
    (ROOT / "audit/sample_prefix_comparison.json").write_text(
        json.dumps(q, indent=2), encoding="utf-8"
    )
    print(q, flush=True)
    assert a == b == 0, q
    # Reference columns are diagnostics only. Preserve the saved historical
    # scores on their exact keys, and use the initial S2 reference on new keys.
    # validate_score_kernel.py subsequently replaces both model score covariates
    # with independently verified current-input values before panel attachment.
    for des in ["beta", "gamma"]:
        col = f"{des}_score_F5_W10_DEFAULT"
        new = (
            new.join(
                old.select(*keys, pl.col(col).alias("_archived_reference")),
                on=keys,
                how="left",
                maintain_order="left",
            )
            .with_columns(pl.coalesce("_archived_reference", col).alias(col))
            .drop("_archived_reference")
        )
    new.write_parquet(m.SAMPLE, compression="zstd")


def outcomes():
    s2 = ROOT.parent / "S2/cache"
    cal = pl.read_parquet(s2 / "calendar.parquet").with_columns(
        pl.col("date").cast(pl.Datetime("ns"))
    )
    p = (
        pl.read_parquet(s2 / "prices_latest.parquet")
        .select(pl.col("date").cast(pl.Datetime("ns")), "code", "open_adj", "close_adj")
        .join(cal, on="date")
    )
    keys = (
        pl.read_parquet(CACHE / "mechanism_sample_keys_before_outcomes.parquet")
        .select("signal_date", "stock_code")
        .join(cal.rename({"date": "signal_date"}), on="signal_date")
    )
    for h in (1, 3, 5, 10, 20, 60, 120, 250):
        for clock, a, b, col in [
            ("close", 0, h, "close_adj"),
            ("open", 1, h + 1, "open_adj"),
        ]:
            enter = p.select(
                pl.col("code").alias("stock_code"),
                (pl.col("idx") - a).alias("idx"),
                pl.col(col).alias("a"),
            )
            leave = p.select(
                pl.col("code").alias("stock_code"),
                (pl.col("idx") - b).alias("idx"),
                pl.col(col).alias("b"),
            )
            keys = (
                keys.join(enter, on=["stock_code", "idx"], how="left")
                .join(leave, on=["stock_code", "idx"], how="left")
                .with_columns(
                    pl.when(
                        pl.col("a").is_finite()
                        & (pl.col("a") > 0)
                        & pl.col("b").is_finite()
                        & (pl.col("b") > 0)
                    )
                    .then(pl.col("b") / pl.col("a") - 1)
                    .otherwise(None)
                    .alias(f"future_{clock}_ret_{h}")
                )
                .drop("a", "b")
            )
    keys.drop("idx").write_parquet(
        ROOT / "cache/abg/abg_outcomes_cutoff.parquet", compression="zstd"
    )
    print("outcomes", keys.height, flush=True)


def folds():
    m = load("run_support_diagnostics")
    dates = (
        pl.read_parquet(
            CACHE / "mechanism_analysis_panel.parquet", columns=["signal_date"]
        )["signal_date"]
        .unique()
        .sort()
        .to_list()
    )
    df = m.make_folds([pd.Timestamp(x) for x in dates])
    df.to_csv(ROOT / "data/mechanism/mechanism_expanding_time_folds.csv", index=False)
    print(
        df.groupby("fold_id").signal_date.agg(["min", "max", "count"]).to_string(),
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage", choices=["sample", "winner", "network", "outcomes", "attach", "folds"]
    )
    ap.add_argument("--design", choices=["beta", "gamma"])
    args = ap.parse_args()
    if args.stage == "sample":
        sample()
    elif args.stage == "outcomes":
        outcomes()
    elif args.stage == "folds":
        folds()
    else:
        name = {
            "winner": "build_protocol_winner",
            "network": "build_protocol_network",
            "attach": "attach_exact_outcomes",
        }[args.stage]
        m = load(name)
        sys.argv = [sys.argv[0]] + (["--design", args.design] if args.design else [])
        m.main()
