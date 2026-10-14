"""Frozen, annually rolling ordinary-vs-concept model benchmark.

Only this script's directory is written. Inputs from S2 remain read-only.
Run freeze, prepare, run, infer in that order. No hyperparameter search.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import argparse
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

sys.path.insert(0, str(ROOT.parent / "config"))
import cutoff as CFG  # noqa: E402

OLD = CFG.NEW_S2_CACHE
CUTOFF = CFG.OBSERVATION_CUTOFF
SOURCE = CFG.SOURCE_ROOT / "stock-trading-data"
DATA = ROOT / "data"
AUDIT = ROOT / "audit"
CACHE = ROOT / "cache"
HORIZONS = [1, 3, 5, 10, 20, 60, 120, 250]
ORDINARY = [
    "mom1",
    "mom3",
    "mom5",
    "mom10",
    "mom20",
    "mom60",
    "mom120",
    "mom250",
    "rev5",
    "vol20",
    "vol60",
    "amount",
    "amount20",
    "amount_change5_20",
    "amount_change20_60",
    "amihud20",
    "float_cap",
    "total_cap",
    "turnover20",
]
CONCEPT = ["gamma_calendar_FW", "gamma_calendar_F_only", "gamma_calendar_W_only"]
FEATURES = ORDINARY + ["missing_count"] + CONCEPT
METHODS = ["ridge_ordinary", "ridge_concept", "hist_ordinary", "hist_concept", "FW"]
SEED = 20260923


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_json(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(
        json.dumps(x, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def freeze():
    for p in [DATA, AUDIT, CACHE]:
        p.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": 1,
        "cutoff": CUTOFF,
        "rerun_branch": "2026-08-31 unified observation cutoff; models, features, windows, seeds and families unchanged",
        "H": HORIZONS,
        "prediction_years": list(range(2020, 2027)),
        "models": {
            "ridge": {"alpha": 100.0, "fit_intercept": True},
            "hist": {
                "learning_rate": 0.06,
                "max_iter": 40,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 100,
                "l2_regularization": 10.0,
                "max_bins": 63,
                "early_stopping": False,
                "random_state": SEED,
            },
        },
        "train_window": "previous 3 calendar years; at least 2 distinct signal years; expanding not used",
        "train_dates": "global calendar idx % 20 == 0; ALL ex-ante common-pool stocks on sampled dates",
        "label": "adjusted open[t+H+1] / adjusted open[t+1] - 1, exact global calendar endpoints",
        "training_label_filter": "both endpoints strictly positive finite and endpoint date <= train_end",
        "training_target": "raw return clipped to [-0.8, 2.0], no clipping in evaluation",
        "ordinary_features": ORDINARY,
        "additional_features": CONCEPT,
        "feature_clock": "exact global calendar; rolling min_periods=window; only <= signal t",
        "feature_processing": "daily pool percentile ranks minus 0.5; missing ->0; extra missing_count/19-0.5",
        "features_not_used": [
            "PIT financials unavailable",
            "industry historical release vintage unverified",
        ],
        "universe": "ALL rows in frozen_common_scores, not Top100, no future trading/label screen",
        "selection": "scores descending; code ascending ties; BEFORE projecting evaluation labels",
        "test_metrics": [
            "daily Spearman IC",
            "Top20 raw return main",
            "Top100 raw return sensitivity",
        ],
        "missing_labels": [
            "complete-observed mean",
            "fixed denominator missing-zero sensitivity",
        ],
        "baselines": [
            "pool equal-weight mean",
            "gamma/calendar/FW Top20 and Top100, same pool",
        ],
        "comparisons": "concept minus ordinary for each model; IC and both TopN complete/zero",
        "bootstrap": {
            "B": 10000,
            "seed": SEED,
            "block": "max(20,H)",
            "method": "paired circular moving-block percentile CI and centered two-sided null +1",
        },
        "inference_gate": "floor(n_finite_signal_dates / max(20,H)) >= 20; otherwise descriptive/deferred",
        "multiple_testing": "Holm across all 8H x 2models x (IC + 2TopN x 2missing) = 80 tests; deferred p=1 for correction only",
        "limits": "4 threads; year/H sequential; <8GB intended",
        "interpretation": "historical research reuse, researchers have already seen this sample; not prospective, causal, or net investable returns",
    }
    p = AUDIT / "frozen_design.json"
    if p.exists():
        assert (
            json.loads(p.read_text(encoding="utf-8")) == spec
        ), "Frozen design differs"
    else:
        save_json(p, spec)
        save_json(
            AUDIT / "freeze_time.json",
            {
                "time_utc": pd.Timestamp.now(tz="UTC"),
                "design_sha256": digest(p),
                "real_future_evaluation_not_read_in_this_experiment": True,
                "prior_researcher_sample_exposure": True,
            },
        )
    print("FROZEN", digest(p), flush=True)
    return spec


def industry_amendment():
    """Append a design before labels; the original design is never overwritten."""
    spec = {
        "original_design_sha256": digest(AUDIT / "frozen_design.json"),
        "reason": "User requested available supplier historical industry sensitivity before this experiment read evaluation labels; not motivated by observed model results",
        "industry_source": "stock-trading-data-pro 新版申万一级行业名称 on signal date, supplier historical labels; historical release vintage NOT authenticated PIT",
        "models_unchanged": True,
        "statistic": "For each date, retain known-industry rows with finite outcome and at least 5 observed stocks in their industry. Globally rank predictions and outcome on this identical retained subset, then subtract each industry mean rank; compute Pearson correlation of the residual ranks. Do NOT rerank residuals.",
        "missing_industry": "Exclude from this sensitivity only; never change primary prediction pool or frozen TopN selection",
        "tests_added": "8H x 2models = 16 concept-minus-ordinary conditional rank-correlation tests",
        "multiple_testing": "Unified Holm family size 96 replaces 80 for reporting all tests; original 80-test specification retained verbatim",
        "boundary": "This is an industry-conditional predictive association sensitivity, NOT equivalent to industry-onehot model training and NOT causal adjustment.",
    }
    p = AUDIT / "industry_sensitivity_amendment.json"
    if p.exists():
        assert json.loads(p.read_text(encoding="utf-8")) == spec
    else:
        assert not list(
            DATA.glob("predictions_H*.parquet")
        ), "Amendment must precede model results"
        save_json(p, spec)
        save_json(
            AUDIT / "industry_amendment_time.json",
            {
                "time_utc": pd.Timestamp.now(tz="UTC"),
                "sha256": digest(p),
                "future_labels_not_constructed_in_this_experiment": True,
            },
        )
    print("INDUSTRY_AMENDMENT_FROZEN", digest(p), flush=True)


def prepare():
    freeze()
    start = time.time()
    outfile = CACHE / "feature_panel.parquet"
    if outfile.exists():
        raise FileExistsError(outfile)
    cal = pd.read_parquet(OLD / "calendar.parquet").sort_values("idx")
    dates = pd.DatetimeIndex(cal.date)
    scorefile = OLD / "frozen_common_scores.parquet"
    scores = pl.read_parquet(scorefile, columns=["date", "code"] + CONCEPT)
    # Partition lookup avoids repeatedly scanning the full score file for each source chunk.
    scoreparts = scores.partition_by("code", as_dict=True)
    del scores
    pricefile = pq.ParquetFile(OLD / "prices_latest.parquet")
    manifest, frames, writer = [], [], None
    rows = 0
    source_errors = []
    for rg in range(pricefile.num_row_groups):
        p = pricefile.read_row_group(
            rg, columns=["date", "code", "close_adj", "ret", "amount"]
        ).to_pandas()
        for code, sub in p.groupby("code", sort=False):
            s = scoreparts.pop((code,), None)
            if s is None or not len(s):
                continue
            sub = (
                sub.sort_values("date")
                .drop_duplicates("date", keep="last")
                .set_index("date")
                .reindex(dates)
            )
            f = pd.DataFrame({"date": dates, "code": code, "idx": cal.idx.to_numpy()})
            close = sub.close_adj
            for h in [1, 3, 5, 10, 20, 60, 120, 250]:
                f[f"mom{h}"] = (close / close.shift(h) - 1).to_numpy()
            f["rev5"] = -f.mom5
            for h in [20, 60]:
                f[f"vol{h}"] = sub.ret.rolling(h, min_periods=h).std(ddof=1).to_numpy()
            amount = sub.amount.where(sub.amount > 0)
            a5 = amount.rolling(5, min_periods=5).mean()
            a20 = amount.rolling(20, min_periods=20).mean()
            a60 = amount.rolling(60, min_periods=60).mean()
            f["amount"] = amount.to_numpy()
            f["amount20"] = a20.to_numpy()
            f["amount_change5_20"] = (a5 / a20 - 1).to_numpy()
            f["amount_change20_60"] = (a20 / a60 - 1).to_numpy()
            f["amihud20"] = (
                (sub.ret.abs() / amount).rolling(20, min_periods=20).mean().to_numpy()
            )
            path = SOURCE / f"{code}.csv"
            try:
                payload = path.read_bytes()
                caps = pd.read_csv(
                    io.BytesIO(payload),
                    encoding="gb18030",
                    skiprows=1,
                    usecols=["交易日期", "流通市值", "总市值"],
                )
                caps["交易日期"] = pd.to_datetime(caps["交易日期"], errors="coerce")
                caps = (
                    caps.dropna(subset=["交易日期"])
                    .sort_values("交易日期")
                    .drop_duplicates("交易日期", keep="last")
                )
                caps = caps.set_index("交易日期").reindex(dates)
                f["float_cap"] = caps["流通市值"].where(caps["流通市值"] > 0).to_numpy()
                f["total_cap"] = caps["总市值"].where(caps["总市值"] > 0).to_numpy()
                manifest.append(
                    {
                        "code": code,
                        "path": str(path),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                    }
                )
            except Exception as exc:
                source_errors.append({"code": code, "error": repr(exc)})
                f["float_cap"] = np.nan
                f["total_cap"] = np.nan
            f["turnover20"] = (
                pd.Series(amount.to_numpy() / f.float_cap.to_numpy())
                .rolling(20, min_periods=20)
                .mean()
                .to_numpy()
            )
            f = pl.from_pandas(f).join(s, on=["date", "code"], how="inner")
            f = f.with_columns(pl.col(ORDINARY + CONCEPT).cast(pl.Float32))
            frames.append(f)
        if frames:
            batch = pl.concat(frames)
            table = batch.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(outfile, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(batch)
            frames.clear()
        print(
            "FEATURE_CHUNK",
            rg + 1,
            "/",
            pricefile.num_row_groups,
            "rows",
            rows,
            "sec",
            round(time.time() - start),
            flush=True,
        )
    if writer is not None:
        writer.close()
    assert not scoreparts, "Some common-score stock keys lacked a complete price chunk"
    save_json(AUDIT / "market_cap_source_inventory.json", manifest)
    save_json(
        AUDIT / "feature_build.json",
        {
            "rows": rows,
            "seconds": time.time() - start,
            "source_errors": source_errors,
            "price_sha256": digest(OLD / "prices_latest.parquet"),
            "score_sha256": digest(scorefile),
            "calendar_sha256": digest(OLD / "calendar.parquet"),
            "feature_panel_sha256": digest(outfile),
        },
    )
    if source_errors:
        raise RuntimeError(
            "Market cap source errors; inspect audit, no model fit permitted"
        )
    # Daily processing depends only on the current ex-ante pool. Missing values never drop stocks.
    processed = CACHE / "ranked_feature_panel.parquet"
    writer = None
    for year in range(2016, 2027):
        f = pl.scan_parquet(outfile).filter(pl.col("date").dt.year() == year).collect()
        if not len(f):
            continue
        missing = pl.sum_horizontal(
            [(~pl.col(c).is_finite()).cast(pl.Int8).fill_null(1) for c in ORDINARY]
        )
        f = f.with_columns(
            (missing / len(ORDINARY) - 0.5).cast(pl.Float32).alias("missing_count")
        )
        expressions = []
        for c in ORDINARY + CONCEPT:
            v = pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None)
            rank = v.rank(method="average").over("date") / v.count().over("date") - 0.5
            expressions.append(rank.fill_null(0).fill_nan(0).cast(pl.Float32).alias(c))
        f = (
            f.with_columns(expressions)
            .select(["date", "code", "idx"] + FEATURES)
            .sort(["date", "code"])
        )
        table = f.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(processed, table.schema, compression="zstd")
        writer.write_table(table)
        print("DAILY_FEATURE_RANK", year, len(f), flush=True)
    if writer:
        writer.close()
    save_json(
        AUDIT / "feature_freeze.json",
        {
            "time_utc": pd.Timestamp.now(tz="UTC"),
            "rows": rows,
            "features_sha256": digest(processed),
            "code_sha256": digest(__file__),
            "future_labels_not_constructed_yet": True,
            "seconds": time.time() - start,
        },
    )


def prepare_industry():
    import concurrent.futures

    industry_amendment()
    start = time.time()
    codes = (
        pl.scan_parquet(CACHE / "ranked_feature_panel.parquet")
        .select("code")
        .unique()
        .collect()["code"]
        .to_list()
    )
    files = [
        CFG.SOURCE_ROOT / "stock-trading-data-pro" / f"{code}.csv"
        for code in sorted(codes)
    ]
    out = CACHE / "industry_labels.parquet"
    if out.exists():
        raise FileExistsError(out)

    def read_one(p):
        payload = p.read_bytes()
        f = pd.read_csv(
            io.BytesIO(payload),
            encoding="gb18030",
            skiprows=1,
            usecols=["股票代码", "交易日期", "新版申万一级行业名称"],
        )
        f = f.rename(
            columns={
                "股票代码": "code",
                "交易日期": "date",
                "新版申万一级行业名称": "industry",
            }
        )
        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f = (
            f[f.date.between("2016-11-24", CUTOFF)]
            .sort_values("date")
            .drop_duplicates(["date", "code"], keep="last")
        )
        f["industry"] = f.industry.replace(
            {"": None, "None": None, "nan": None, "未知": None}
        )
        f["industry"] = f.industry.astype("string")
        return pl.from_pandas(f), {
            "path": str(p),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    writer, manifest, rows = None, [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for begin in range(0, len(files), 128):
            results = list(pool.map(read_one, files[begin : begin + 128]))
            frames = [r[0] for r in results]
            manifest.extend(r[1] for r in results)
            f = pl.concat(frames)
            table = f.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(f)
            print(
                "INDUSTRY",
                min(begin + 128, len(files)),
                "/",
                len(files),
                "rows",
                rows,
                "sec",
                round(time.time() - start),
                flush=True,
            )
    writer.close()
    save_json(AUDIT / "industry_source_inventory.json", manifest)
    save_json(
        AUDIT / "industry_build.json",
        {
            "rows": rows,
            "files": len(files),
            "seconds": time.time() - start,
            "sha256": digest(out),
            "supplier_historical_labels_not_authenticated_PIT": True,
        },
    )


def load_labels(h, cal):
    prices = pl.read_parquet(
        OLD / "prices_latest.parquet", columns=["date", "code", "open_adj"]
    ).join(cal, on="date")
    entry = prices.select(
        "code", (pl.col("idx") - 1).alias("idx"), pl.col("open_adj").alias("entry")
    )
    exitp = prices.select(
        "code", (pl.col("idx") - h - 1).alias("idx"), pl.col("open_adj").alias("exit")
    )
    lab = entry.join(exitp, on=["code", "idx"], how="inner")
    return lab.with_columns(
        pl.when(
            (pl.col("entry") > 0)
            & (pl.col("exit") > 0)
            & pl.col("entry").is_finite()
            & pl.col("exit").is_finite()
        )
        .then(pl.col("exit") / pl.col("entry") - 1)
        .otherwise(None)
        .cast(pl.Float32)
        .alias("outcome")
    ).select("code", "idx", "outcome")


def evaluate_predictions(pred, lab, h, year):
    # Freeze actual stock selection before outcome projection; missing future endpoints cannot rerank.
    selections = []
    for method in METHODS:
        s = (
            pred.select("date", "idx", "code", pl.col(method).alias("prediction"))
            .sort(["date", "prediction", "code"], descending=[False, True, False])
            .with_columns(pl.col("code").cum_count().over("date").alias("rank"))
            .filter(pl.col("rank") <= 100)
        )
        selections.append(s.with_columns(pl.lit(method).alias("method")))
    selection = pl.concat(selections)
    sp = DATA / f"selections_H{h}_{year}.parquet"
    if sp.exists():
        assert pl.read_parquet(sp).equals(
            selection
        ), "Previously frozen selection changed"
    else:
        selection.write_parquet(sp, compression="zstd")
        save_json(
            AUDIT / f"selection_freeze_H{h}_{year}.json",
            {
                "time_utc": pd.Timestamp.now(tz="UTC"),
                "sha256": digest(sp),
                "selection_precedes_evaluation_label_join": True,
            },
        )
    labeled = pred.join(lab, on=["code", "idx"], how="left")
    mature = labeled.filter(pl.col("idx") + h + 1 <= int(CALENDAR_MAX))
    pool = mature.group_by("date").agg(
        pl.len().alias("n_pool"),
        pl.col("outcome").count().alias("n_pool_observed"),
        pl.col("outcome").mean().alias("pool_complete"),
        pl.col("outcome").fill_null(0).mean().alias("pool_zero"),
    )
    selected = selection.join(
        mature.select("date", "code", "outcome"), on=["date", "code"], how="inner"
    )
    daily = []
    for n in [20, 100]:
        q = (
            selected.filter(pl.col("rank") <= n)
            .group_by(["date", "method"])
            .agg(
                pl.len().alias("n_intended"),
                pl.col("outcome").count().alias("n_observed"),
                pl.col("outcome").mean().alias("portfolio_complete"),
                pl.col("outcome").fill_null(0).mean().alias("portfolio_zero"),
            )
        )
        q = q.join(pool, on="date").with_columns(
            pl.lit(n).alias("top_n"),
            pl.lit(h).alias("H"),
            pl.lit(year).alias("model_year"),
            (pl.col("portfolio_complete") - pl.col("pool_complete")).alias(
                "excess_complete"
            ),
            (pl.col("portfolio_zero") - pl.col("pool_zero")).alias("excess_zero"),
        )
        daily.append(q)
    ic = []
    for method in METHODS:
        q = (
            mature.filter(pl.col("outcome").is_finite())
            .group_by("date")
            .agg(
                pl.corr(method, "outcome", method="spearman").alias("IC"),
                pl.len().alias("n_ic"),
            )
        )
        ic.append(
            q.with_columns(
                pl.lit(method).alias("method"),
                pl.lit(h).alias("H"),
                pl.lit(year).alias("model_year"),
            )
        )
    pl.concat(daily).sort(["date", "method", "top_n"]).write_parquet(
        DATA / f"daily_H{h}_{year}.parquet", compression="zstd"
    )
    pl.concat(ic).sort(["date", "method"]).write_parquet(
        DATA / f"ic_H{h}_{year}.parquet", compression="zstd"
    )
    industry = (
        pl.scan_parquet(CACHE / "industry_labels.parquet")
        .filter(pl.col("date").dt.year() == year)
        .with_columns(pl.col("date").cast(pl.Datetime("ns")))
        .collect()
    )
    conditional = mature.join(industry, on=["date", "code"], how="left")
    coverage = conditional.group_by("date").agg(
        pl.len().alias("n_pool"), pl.col("industry").count().alias("n_industry_known")
    )
    conditional = conditional.filter(
        pl.col("industry").is_not_null() & pl.col("outcome").is_finite()
    )
    conditional = conditional.with_columns(
        pl.len().over(["date", "industry"]).alias("industry_observed_n")
    ).filter(pl.col("industry_observed_n") >= 5)
    rankcols = METHODS + ["outcome"]
    conditional = conditional.with_columns(
        [
            pl.col(c).rank(method="average").over("date").alias(c + "_rank")
            for c in rankcols
        ]
    )
    conditional = conditional.with_columns(
        [
            (
                pl.col(c + "_rank")
                - pl.col(c + "_rank").mean().over(["date", "industry"])
            ).alias(c + "_residual_rank")
            for c in rankcols
        ]
    )
    sic = []
    for method in METHODS:
        q = conditional.group_by("date").agg(
            pl.corr(
                method + "_residual_rank", "outcome_residual_rank", method="pearson"
            ).alias("IC"),
            pl.len().alias("n_ic"),
            pl.col("industry").n_unique().alias("n_industries"),
        )
        q = q.join(coverage, on="date").with_columns(
            pl.lit(method).alias("method"),
            pl.lit(h).alias("H"),
            pl.lit(year).alias("model_year"),
        )
        sic.append(q)
    pl.concat(sic).sort(["date", "method"]).write_parquet(
        DATA / f"industry_ic_H{h}_{year}.parquet", compression="zstd"
    )


CALENDAR_MAX = None


def run(only_h=None):
    global CALENDAR_MAX
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    spec = freeze()
    industry_amendment()
    assert (AUDIT / "feature_freeze.json").exists()
    assert (CACHE / "industry_labels.parquet").exists()
    assert not json.loads((AUDIT / "feature_build.json").read_text(encoding="utf-8"))[
        "source_errors"
    ]
    cal = pl.read_parquet(OLD / "calendar.parquet")
    CALENDAR_MAX = cal["idx"].max()
    panel = CACHE / "ranked_feature_panel.parquet"
    print("MODEL_RUN_BEGIN", pd.Timestamp.now(tz="UTC"), flush=True)
    for h in ([only_h] if only_h else HORIZONS):
        start = time.time()
        labels = load_labels(h, cal)
        for year in range(2020, 2027):
            if all(
                (DATA / f"{prefix}_H{h}_{year}.parquet").exists()
                for prefix in ["daily", "ic", "industry_ic"]
            ):
                print("RESUME_SKIP", h, year, flush=True)
                continue
            train_start = pd.Timestamp(year=year - 3, month=1, day=1)
            train_end = pd.Timestamp(year=year - 1, month=12, day=31)
            last_train_idx = cal.filter(pl.col("date") <= train_end)["idx"].max()
            train = (
                pl.scan_parquet(panel)
                .filter(
                    (pl.col("date") >= train_start)
                    & (pl.col("date") <= train_end)
                    & (pl.col("idx") % 20 == 0)
                    & (pl.col("idx") + h + 1 <= last_train_idx)
                )
                .collect()
            )
            train = train.join(labels, on=["code", "idx"], how="left").filter(
                pl.col("outcome").is_finite()
            )
            test = (
                pl.scan_parquet(panel)
                .filter(pl.col("date").dt.year() == year)
                .collect()
            )
            if not len(test):
                continue
            years = train["date"].dt.year().n_unique()
            if years < 2:
                save_json(
                    AUDIT / f"fit_H{h}_{year}.json",
                    {
                        "status": "DEFERRED_LESS_THAN_TWO_TRAIN_YEARS",
                        "rows": len(train),
                    },
                )
                continue
            x = np.ascontiguousarray(
                train.select(FEATURES).to_numpy(), dtype=np.float32
            )
            xt = np.ascontiguousarray(
                test.select(FEATURES).to_numpy(), dtype=np.float32
            )
            y = np.clip(train["outcome"].to_numpy(), -0.8, 2.0)
            assert (
                np.isfinite(x).all() and np.isfinite(xt).all() and np.isfinite(y).all()
            )
            info = {
                "H": h,
                "year": year,
                "train_rows": len(train),
                "train_dates": train["date"].n_unique(),
                "train_signal_min": train["date"].min(),
                "train_signal_max": train["date"].max(),
                "max_label_end_idx": int(train["idx"].max() + h + 1),
                "train_end_idx": int(last_train_idx),
                "test_rows": len(test),
                "test_dates": test["date"].n_unique(),
                "fits": [],
            }
            assert info["max_label_end_idx"] <= info["train_end_idx"]
            pred = test.select(
                "date", "idx", "code", pl.col("gamma_calendar_FW").alias("FW")
            )
            for name, cls in [
                ("ridge", Ridge),
                ("hist", HistGradientBoostingRegressor),
            ]:
                for aug, count in [
                    ("ordinary", len(ORDINARY) + 1),
                    ("concept", len(FEATURES)),
                ]:
                    t = time.time()
                    model = cls(**spec["models"][name])
                    with threadpool_limits(limits=4):
                        model.fit(x[:, :count], y)
                        prediction = model.predict(xt[:, :count]).astype(np.float32)
                    pred = pred.with_columns(pl.Series(f"{name}_{aug}", prediction))
                    fit = {
                        "model": name,
                        "features": aug,
                        "seconds": time.time() - t,
                        "training_target_mean": float(y.mean()),
                        "prediction_std": float(prediction.std()),
                    }
                    info["fits"].append(fit)
                    print(
                        "FIT",
                        h,
                        year,
                        name,
                        aug,
                        len(train),
                        len(test),
                        round(fit["seconds"], 1),
                        flush=True,
                    )
            # Persist all-pool predictions for complete independent rank/IC review.
            pred.write_parquet(
                DATA / f"predictions_H{h}_{year}.parquet", compression="zstd"
            )
            save_json(AUDIT / f"fit_H{h}_{year}.json", info)
            del x, xt, y, train, test
            evaluate_predictions(pred, labels, h, year)
            del pred
        print("H_DONE", h, "seconds", round(time.time() - start, 1), flush=True)


def block_bootstrap(x, length, seed, B=10000):
    n, p = x.shape
    finite = np.isfinite(x)
    value = np.where(finite, x, 0.0)
    val = np.vstack([value, value[:length]])
    cnt = np.vstack([finite, finite[:length]])
    cs = np.vstack([np.zeros((1, p)), np.cumsum(val, axis=0)])
    cn = np.vstack([np.zeros((1, p)), np.cumsum(cnt, axis=0)])
    pos = np.arange(n)
    sums, counts = cs[pos + length] - cs[pos], cn[pos + length] - cn[pos]
    q, r = divmod(n, length)
    if r:
        rs, rc = cs[pos + r] - cs[pos], cn[pos + r] - cn[pos]
    rng = np.random.default_rng(seed)
    draws = np.empty((B, p))
    for a in range(0, B, 100):
        k = min(100, B - a)
        idx = rng.integers(0, n, size=(k, q))
        total, den = sums[idx].sum(axis=1), counts[idx].sum(axis=1)
        if r:
            ix = rng.integers(0, n, size=k)
            total += rs[ix]
            den += rc[ix]
        draws[a : a + k] = total / den
    mean = np.nanmean(x, axis=0)
    low, high = np.nanquantile(draws, [0.025, 0.975], axis=0)
    p = (1 + np.sum(np.abs(draws - mean) >= np.abs(mean), axis=0)) / (B + 1)
    return mean, low, high, p


def infer():
    spec = freeze()
    df = pd.concat(
        [pd.read_parquet(p) for p in sorted(DATA.glob("daily_H*_*.parquet"))],
        ignore_index=True,
    )
    ic = pd.concat(
        [pd.read_parquet(p) for p in sorted(DATA.glob("ic_H*_*.parquet"))],
        ignore_index=True,
    )
    industry_ic = pd.concat(
        [pd.read_parquet(p) for p in sorted(DATA.glob("industry_ic_H*_*.parquet"))],
        ignore_index=True,
    )
    df.to_parquet(DATA / "daily_portfolio_all.parquet", index=False)
    ic.to_parquet(DATA / "daily_ic_all.parquet", index=False)
    industry_ic.to_parquet(
        DATA / "daily_industry_conditional_ic_all.parquet", index=False
    )
    rows = []
    for h in HORIZONS:
        part, ip = df[df.H.eq(h)], ic[ic.H.eq(h)]
        arrays, names = [], []
        ipiv = ip.pivot(index="date", columns="method", values="IC").sort_index()
        dates = ipiv.index
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
            sip = (
                industry_ic[industry_ic.H.eq(h)]
                .pivot(index="date", columns="method", values="IC")
                .reindex(dates)
            )
            arrays.append(
                (sip[f"{model}_concept"] - sip[f"{model}_ordinary"]).to_numpy()
            )
            names.append((model, "industry_conditional_IC", 0, "complete"))
        x = np.column_stack(arrays)
        length = max(20, h)
        nvalid = np.isfinite(x).sum(axis=0)
        eligible = nvalid // length >= 20
        means = np.nanmean(x, axis=0)
        if eligible.any():
            means, lo, hi, p = block_bootstrap(
                x, length, SEED + h, spec["bootstrap"]["B"]
            )
        else:
            lo, hi, p = [np.full(len(names), np.nan) for _ in range(3)]
        for j, (model, metric, topn, missing) in enumerate(names):
            scale = 100 if metric == "return" else 1
            rows.append(
                {
                    "H": h,
                    "model": model,
                    "metric": metric,
                    "top_n": topn,
                    "missing": missing,
                    "n_dates": int(nvalid[j]),
                    "block": length,
                    "independent_blocks": int(nvalid[j] // length),
                    "status": (
                        "INFERENCE_ELIGIBLE"
                        if eligible[j]
                        else "DEFERRED_INSUFFICIENT_BLOCKS"
                    ),
                    "delta": float(means[j] * scale),
                    "ci_low": float(lo[j] * scale) if eligible[j] else np.nan,
                    "ci_high": float(hi[j] * scale) if eligible[j] else np.nan,
                    "p_raw": float(p[j]) if eligible[j] else np.nan,
                    "unit": "percentage_point" if metric == "return" else "IC",
                }
            )
    result = pd.DataFrame(rows)
    assert len(result) == 96
    raw = result.p_raw.fillna(1).to_numpy()
    order = np.argsort(raw)
    adj = np.maximum.accumulate(raw[order] * (len(raw) - np.arange(len(raw))))
    adjusted = np.empty_like(adj)
    adjusted[order] = np.minimum(adj, 1)
    result["p_holm_96"] = adjusted
    result["confirmed_positive"] = (
        result.status.eq("INFERENCE_ELIGIBLE")
        & result.delta.gt(0)
        & result.p_holm_96.lt(0.05)
    )
    result["confirmed_negative"] = (
        result.status.eq("INFERENCE_ELIGIBLE")
        & result.delta.lt(0)
        & result.p_holm_96.lt(0.05)
    )
    result.to_csv(DATA / "increment_inference.csv", index=False, encoding="utf-8-sig")
    summary = (
        df.groupby(["H", "method", "top_n"])
        .agg(
            n_dates=("date", "count"),
            gross_return=("portfolio_complete", "mean"),
            excess_pool=("excess_complete", "mean"),
            gross_zero=("portfolio_zero", "mean"),
            n_intended=("n_intended", "sum"),
            n_observed=("n_observed", "sum"),
            mean_pool=("n_pool", "mean"),
        )
        .reset_index()
    )
    summary["missing_fraction"] = 1 - summary.n_observed / summary.n_intended
    summary.to_csv(DATA / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    ics = (
        ic.groupby(["H", "method"])
        .agg(mean_IC=("IC", "mean"), n_dates=("IC", "count"), mean_n=("n_ic", "mean"))
        .reset_index()
    )
    ics.to_csv(DATA / "ic_summary.csv", index=False, encoding="utf-8-sig")
    industry_ic.groupby(["H", "method"]).agg(
        mean_IC=("IC", "mean"),
        n_dates=("IC", "count"),
        mean_n=("n_ic", "mean"),
        mean_industry_known=("n_industry_known", "mean"),
        mean_industries=("n_industries", "mean"),
    ).reset_index().to_csv(
        DATA / "industry_conditional_ic_summary.csv", index=False, encoding="utf-8-sig"
    )
    allfits = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(AUDIT.glob("fit_H*_*.json"))
    ]
    save_json(
        AUDIT / "run_summary.json",
        {
            "fit_year_H_cells": len(allfits),
            "model_fit_count": sum(len(f.get("fits", [])) for f in allfits),
            "positive_confirmations": int(result.confirmed_positive.sum()),
            "negative_confirmations": int(result.confirmed_negative.sum()),
            "deferred_tests": int(result.status.ne("INFERENCE_ELIGIBLE").sum()),
            "family_size": 96,
            "design_sha256": digest(AUDIT / "frozen_design.json"),
            "code_sha256": digest(__file__),
        },
    )
    lines = [
        "# B2 普通特征与概念增量：实际滚动模型实验",
        "",
        "本实验已实际训练并生成全池预测，不是实验计划。历史样本已被研究者看过；冻结只约束本次执行，不能称为真正前瞻样本外。",
        "",
        "## 设计与防泄漏",
        "",
        f"- 信号≤t；标签为全局交易日开盘t+1至t+H+1复权收益；观察截止{CUTOFF}（最后交易日）。",
        "- 2020–2026按自然年预测；每年用此前3自然年中idx%20=0的全池训练行，标签终点严格≤上一年末，至少覆盖2个信号自然年。",
        "- 普通特征：1/3/5/10/20/60/120/250日动量、5日反转、20/60日波动、成交额水平与变化、Amihud、流通/总市值、成交额/流通市值；每日全池rank处理，缺失置中性值并加入缺失数。",
        "- 增量组只追加γ/calendar FW、F_only、W_only；与普通组使用相同模型、训练样本和预测股票池。行业版本不可验证，未加入主模型训练；没有伪造PIT财务。",
        "- 在本轮读取未来标签前追加冻结供应商历史行业标签敏感性：先对同一有效子集的预测与收益做全局秩，再按当日行业去均值，计算残差秩的Pearson相关（不再排名）。行业至少5个有标签股票，缺行业仅排除于敏感性。此非认证PIT，也不等同于行业onehot训练对照。",
        "- 固定ridge和40轮HistGradientBoosting，无调参/模型挑选。全池预测后先固化Top100名单，再投影未来标签；未按未来可成交/标签完整性挑股。",
        "- 收益为毛收益映射，不含费用、涨跌停/停牌成交限制或组合资金约束；缺未来端点同时给完整样本均值和固定分母缺失记零敏感性。",
        "- 配对移动块bootstrap 10,000次；L=max(20,H)；原80项加行业16项，共96项统一Holm校正。少于20个名义完整时间块推迟推断，不解释为零效应；floor(n/L)是支持量门槛，并不保证块间独立。",
        "",
        f'最终模型：{sum(len(f.get("fits", [])) for f in allfits)}个；年×H单元：{len(allfits)}（另有首次时间戳join修复后的4次确定性重复fit）。正向通过校正：{int(result.confirmed_positive.sum())}项；负向通过校正：{int(result.confirmed_negative.sum())}项；支持不足：{int(result.status.ne("INFERENCE_ELIGIBLE").sum())}项。',
        "",
        "## 主要结论",
        "",
        "在本次固定设计下，主Top20收益增量和原始全池IC增量均未通过96项统一Holm校正。局部正向证据仅出现在ridge的H5/Top100敏感性，以及ridge H1/H5、Hist H20的供应商历史行业条件IC敏感性。不能据此宣称概念特征给强普通特征模型带来普遍、稳定或可交易净收益增量。",
        "",
        "H120与H250分别仅12与5个预定义名义完整时间块，因此统计判断推迟；没有将未显著或支持不足解释为零效应。推断n是逐日配对差的日期数（例如H3为1625日），不能把数千万stock-date预测行当作独立样本。",
        "",
        "## 主读数：概念组减普通组（IC与Top20完整标签收益）",
        "",
        "| H | 模型 | 指标 | 日期n | 名义完整时间块 | 增量 | 95% CI | Holm p | 状态 |",
        "|---:|---|---|---:|---:|---:|---|---:|---|",
    ]
    main = result[
        (result.metric.eq("IC")) | (result.top_n.eq(20) & result.missing.eq("complete"))
    ]
    for r in main.itertuples():
        ci = f"[{r.ci_low:.6f}, {r.ci_high:.6f}]" if np.isfinite(r.ci_low) else "推迟"
        lines.append(
            f"| {r.H} | {r.model} | {r.metric} | {r.n_dates} | {r.independent_blocks} | {r.delta:.6f} | {ci} | {r.p_holm_96:.4g} | {r.status} |"
        )
    lines += [
        "",
        "## 8H方向预测水平：平均全池RankIC",
        "",
        "| H | Ridge普通 | Ridge+概念 | Hist普通 | Hist+概念 | 原FW |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    ic_wide = ics.pivot(index="H", columns="method", values="mean_IC")
    for h, r in ic_wide.iterrows():
        lines.append(
            f"| {h} | {r.ridge_ordinary:.6f} | {r.ridge_concept:.6f} | {r.hist_ordinary:.6f} | {r.hist_concept:.6f} | {r.FW:.6f} |"
        )
    lines += [
        "",
        "## 供应商历史行业标签条件增量（非认证PIT敏感性）",
        "",
        "| H | 模型 | 日期n | 行业条件IC增量 | 95% CI | Holm p |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for r in result[result.metric.eq("industry_conditional_IC")].itertuples():
        ci = f"[{r.ci_low:.6f}, {r.ci_high:.6f}]" if np.isfinite(r.ci_low) else "推迟"
        lines.append(
            f"| {r.H} | {r.model} | {r.n_dates} | {r.delta:.6f} | {ci} | {r.p_holm_96:.4g} |"
        )
    lines += [
        "",
        "原始IC衡量全池跨股票的秩相关，包含行业之间和行业内部的共同变动；行业条件IC则先扣除当日行业平均秩，并且只在行业已知、行业有效样本至少5只的子集上计算。这改变了统计对象、样本与噪声结构，因此两者显著性可以不同。它不证明行业是因果混杂变量，也不证明剩余相关源于概念机制；更不等同于行业onehot训练后的模型增量。",
        "",
    ]
    sensitivity_path = DATA / "block40_60_sensitivity_all.csv"
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
        lines += [
            "## 块长40/60事后请求的统一敏感性",
            "",
            "主结果和主确认标签保持不变。此项在已看过主结果后按用户统一要求执行；同一日序列，L=max(40,H)或max(60,H)，各10,000次、分别Holm96，完整192项见BLOCK_SENSITIVITY.md及data/block40_60_sensitivity_all.csv。",
            "",
            "| H | 模型 | 主分析通过的次级指标 | 缺失口径 | 主Holm p | 块40 Holm p | 块60 Holm p |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
        for r in result[result.confirmed_positive].itertuples():
            selected = sensitivity[
                (sensitivity.H.eq(r.H))
                & sensitivity.model.eq(r.model)
                & sensitivity.metric.eq(r.metric)
                & sensitivity.top_n.eq(r.top_n)
                & sensitivity.missing.eq(r.missing)
            ]
            p40 = selected.loc[selected.block_floor.eq(40), "p_holm_96"].iloc[0]
            p60 = selected.loc[selected.block_floor.eq(60), "p_holm_96"].iloc[0]
            label = f"Top{r.top_n}收益" if r.metric == "return" else r.metric
            lines.append(
                f"| {r.H} | {r.model} | {label} | {r.missing} | {r.p_holm_96:.6f} | {p40:.6f} | {p60:.6f} |"
            )
        lines += [
            "",
            "因此不能抹掉ridge H5/Top100的次级正向收益证据：其完整标签与缺失记零口径在块40和60下仍通过校正。但这不是主Top20收益确认，也不是两项独立经济发现。行业条件IC中ridge H1对两种块长保持；ridge H5对更长块不保持，Hist H20在块60不再通过。这些敏感性不触发重新选模型或扩展主确认标签。",
            "",
        ]
    lines += [
        "",
        "收益增量及CI单位为百分点；IC为原始相关系数单位。推迟项的Holm计算占位p=1，不是有效检验p值。",
        "",
        "## 复核文件",
        "",
        "- audit/frozen_design.json 与 freeze_time.json：本轮事前冻结记录。",
        "- data/predictions_H*_*.parquet：所有共同池股票的每日预测（未截成Top100）。",
        "- data/selections_H*_*.parquet：评估前固化的Top100；可重算Top20/100。",
        "- data/daily_portfolio_all.parquet、daily_ic_all.parquet：逐日结果，可由主工程复核配对推断。",
        "- data/increment_inference.csv：全部96项、n、CI、原p和Holm p；portfolio_summary.csv、ic_summary.csv包括FW与pool对照。",
        "- data/daily_industry_conditional_ic_all.parquet：行业条件相关逐日有效样本与行业覆盖；附加协议保留原冻结文件且单独记时。",
        "- audit/fit_H*_*.json：训练日期、最大标签终点、样本量和耗时；全部标签终点≤训练年末断言通过。",
        "",
        "本结果检验“已有量价/市值信息与固定模型条件下概念评分是否有预测和选股增量”，不构成概念机制的因果识别。",
    ]
    (ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["freeze", "amendment", "prepare", "industry", "run", "infer"]
    )
    parser.add_argument("--H", type=int, choices=HORIZONS)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "amendment":
        industry_amendment()
    elif args.stage == "prepare":
        prepare()
    elif args.stage == "industry":
        prepare_industry()
    elif args.stage == "run":
        run(args.H)
    else:
        infer()
