"""One frozen E1 execution. Default is static metadata only; --execute is explicit.

Training targets are created only from each fold's past matured prices. Evaluation
labels are not created/joined until ALL seven years' predictions and Top20 lists
are sealed. No source, protocol, or existing run is overwritten.
"""

from __future__ import annotations
import argparse, datetime, hashlib, json, math, os, sys, time, traceback
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
VERSION = ROOT.parents[1]
PROTOCOL = ROOT / "protocol_frozen.json"
EXPECTED_PROTOCOL_SHA = (
    "4cee8322c274ce026073ce29624f7fc534fe64102dacc87cb5ee53cf2397ad58"
)
for var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[var] = "4"
VENDOR = Path("__MODELS__/runtime_source/feature_increment/vendor")
sys.path.insert(0, str(VENDOR))


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT), f"Forbidden output {path}"
    assert not path.exists(), f"No overwrite {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf8",
    )


def load_protocol():
    assert sha(PROTOCOL) == EXPECTED_PROTOCOL_SHA, "Frozen protocol changed"
    p = json.loads(PROTOCOL.read_text(encoding="utf8"))
    assert p["status"] == "FROZEN_BEFORE_NEW_FITS_OR_EFFECTS"
    assert (
        p["features"]["baseline_matrix_columns"] == len(p["features"]["baseline"]) == 21
    )
    assert (
        p["features"]["enhanced_matrix_columns"] == len(p["features"]["enhanced"]) == 23
    )
    assert p["features"]["enhanced"][:21] == p["features"]["baseline"]
    assert p["model"]["fits"] == 14 and p["model"]["params"]["early_stopping"] is False
    assert len(p["inference"]["cells"]) == 6
    return p


def load_runtime():
    global np, pd, pl, sklearn, HistGradientBoostingRegressor, threadpool_limits, joblib
    import numpy as np
    import pandas as pd
    import polars as pl
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    import joblib

    assert sklearn.__version__ == "1.8.0", sklearn.__version__
    # Dependency location is environment-specific; exact sklearn version is checked above.


def verify_inputs(p):
    inputs, records = {}, []
    for item in p["inputs"]:
        path = Path(item["runtime_path"]).resolve()
        assert path.is_relative_to(Path("__SNAPSHOT__")) or path.is_relative_to(
            Path("__MODELS__").parent
        )
        assert path.stat().st_size == item["bytes"]
        actual = sha(path)
        assert actual == item["sha256"], f'Input hash mismatch: {item["role"]}'
        inputs[item["role"]] = path
        records.append(
            {
                "role": item["role"],
                "resolved_path": str(path),
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return inputs, records


def calendar_load(inputs, cutoff):
    cal = pl.read_parquet(inputs["calendar"]).sort("idx")
    assert cal["date"].n_unique() == len(cal) and cal["idx"].n_unique() == len(cal)
    assert cal["date"].max() <= cutoff
    assert np.array_equal(
        cal["idx"].to_numpy(), np.arange(cal["idx"].min(), cal["idx"].max() + 1)
    )
    return cal


def assert_keys(frame, name, cal):
    assert frame.select(pl.struct(["date", "code"]).n_unique()).item() == len(
        frame
    ), f"{name} duplicate keys"
    assert (
        frame.select(
            pl.any_horizontal(
                pl.col("date").is_null(),
                pl.col("code").is_null(),
                pl.col("idx").is_null(),
            ).any()
        ).item()
        is False
    )
    mapped = (
        frame.select("date", "idx")
        .unique()
        .join(cal, on="date", how="left", suffix="_calendar")
    )
    assert (
        mapped["idx_calendar"].null_count() == 0
        and mapped.select((pl.col("idx") == pl.col("idx_calendar")).all()).item()
    )


def state_prepare(inputs, cal, p, out):
    raw = pd.read_csv(
        inputs["csi300"], usecols=["candle_end_time", "close"], encoding="gb18030"
    )
    dates = pd.to_datetime(raw.candle_end_time, errors="raise")
    assert dates.notna().all() and dates.is_unique
    cutoff = pd.Timestamp(p["cutoff"]["observation"])
    excluded = int((dates > cutoff).sum())
    filtered = raw.loc[dates <= cutoff].copy()
    filtered["date"] = dates[dates <= cutoff]
    filtered = filtered[["date", "close"]].sort_values("date").reset_index(drop=True)
    assert pd.to_numeric(filtered.close, errors="coerce").gt(0).all()
    filtered_path = out / "input_filtered/csi300_through_20260831.csv"
    filtered_path.parent.mkdir()
    filtered.to_csv(filtered_path, index=False)
    filtered["vol20"] = (
        filtered.close.pct_change(fill_method=None)
        .rolling(20, min_periods=20)
        .std(ddof=1)
    )
    calendar = cal.to_pandas()
    maxidx = int(cal["idx"].max())
    states, thresholds = [], []
    for year in p["model"]["years"]:
        history = filtered[
            filtered.date.between(
                pd.Timestamp(year - 3, 1, 1), pd.Timestamp(year - 1, 12, 31)
            )
        ]
        threshold = float(history.vol20.dropna().quantile(0.90, interpolation="linear"))
        evaluation = calendar[
            (calendar.date.dt.year == year) & (calendar.idx + 4 <= maxidx)
        ].copy()
        evaluation = evaluation.merge(
            filtered[["date", "vol20"]], on="date", how="left", validate="one_to_one"
        )
        assert evaluation.vol20.notna().all()
        evaluation["stress"] = evaluation.vol20 > threshold
        evaluation["threshold"] = threshold
        states.append(evaluation)
        thresholds.append(
            {
                "year": year,
                "calibration_min": str(history.date.min().date()),
                "calibration_max": str(history.date.max().date()),
                "calibration_dates": len(history),
                "threshold": threshold,
            }
        )
    state = (
        pd.concat(states, ignore_index=True).sort_values("date").reset_index(drop=True)
    )
    state["fixed_20d_block"] = np.arange(len(state)) // 20
    support = {
        "stress_dates": int(state.stress.sum()),
        "stress_years": int(state.loc[state.stress, "date"].dt.year.nunique()),
        "occupied_fixed20_blocks": int(
            state.loc[state.stress, "fixed_20d_block"].nunique()
        ),
    }
    expected = p["stress"]["observed_support_before_any_new_model_or_effect"]
    assert all(support[k] == expected[k] for k in support), (support, expected)
    assert (
        support["occupied_fixed20_blocks"]
        < p["stress"]["support_gate"]["min_occupied_fixed20_blocks"]
    )
    state.to_parquet(out / "state_calendar.parquet", index=False)
    json_write(
        out / "audit/state_source_receipt.json",
        {
            "created_at": utc(),
            "source_sha256": sha(inputs["csi300"]),
            "filtered_source_sha256": sha(filtered_path),
            "source_rows": len(raw),
            "excluded_after_cutoff_before_transform": excluded,
            "retained_rows": len(filtered),
            "retained_date_min": str(filtered.date.min().date()),
            "retained_date_max": str(filtered.date.max().date()),
            "thresholds": thresholds,
            "support": support,
            "gate": "FAIL_BLOCK_COVERAGE",
            "pressure_inference": "not_estimable",
            "supplier_public_release_capture_revision_time": "unknown; no PIT upgrade",
            "observation_time": "trading-date close; signal_time and available_time are research-clock assumptions",
        },
    )
    return state


def prices_for_labels(inputs, cal, cutoff):
    # Predicate acts before endpoint construction; used with each fold's year-1 cutoff.
    price = (
        pl.scan_parquet(inputs["prices"])
        .filter(pl.col("date") <= cutoff)
        .select("date", "code", "open_adj")
        .collect()
    )
    assert price["date"].max() <= cutoff
    assert price.select(pl.struct(["date", "code"]).n_unique()).item() == len(price)
    price = price.join(cal, on="date", how="left")
    assert price["idx"].null_count() == 0
    return price


def label_table(price):
    entry = price.select(
        "code",
        (pl.col("idx") - 1).alias("idx"),
        pl.col("open_adj").alias("entry"),
        pl.col("date").alias("label_start"),
    )
    exitp = price.select(
        "code",
        (pl.col("idx") - 4).alias("idx"),
        pl.col("open_adj").alias("exit"),
        pl.col("date").alias("label_end"),
    )
    lab = entry.join(exitp, on=["code", "idx"], how="full", coalesce=True)
    valid = (
        (pl.col("entry") > 0)
        & (pl.col("exit") > 0)
        & pl.col("entry").is_finite()
        & pl.col("exit").is_finite()
    )
    return lab.with_columns(
        pl.when(valid)
        .then(pl.col("exit") / pl.col("entry") - 1)
        .otherwise(None)
        .cast(pl.Float32)
        .alias("outcome")
    )


def key_receipt(frame, path):
    frame.select("date", "code", "idx").write_parquet(path, compression="zstd")
    return {
        "rows": len(frame),
        "dates": frame["date"].n_unique(),
        "keys_sha256": sha(path),
    }


def selections_from_predictions(pred, method):
    return (
        pred.select("date", "idx", "code", pl.col(method).alias("prediction"))
        .sort(["date", "prediction", "code"], descending=[False, True, False])
        .with_columns(pl.col("code").cum_count().over("date").alias("rank"))
        .filter(pl.col("rank") <= 20)
        .with_columns(pl.lit(method).alias("method"))
    )


def fit_all(p, inputs, cal, out):
    feature_cols = p["features"]["enhanced"]
    fit_records, sealed_files, clocks = [], [], []
    maxidx = int(cal["idx"].max())
    inventory = json.loads((ROOT / "e1_data_inventory.json").read_text(encoding="utf8"))
    archived_folds = {row["fold_id"]: row for row in inventory["folds"]}
    for year in p["model"]["years"]:
        actual_start = utc()
        train_start = pd.Timestamp(year - 3, 1, 1)
        train_end = pd.Timestamp(year - 1, 12, 31)
        logical_fit = pd.Timestamp(year, 1, 1)
        lastidx = int(cal.filter(pl.col("date") <= train_end)["idx"].max())
        train = (
            pl.scan_parquet(inputs["ranked_features"])
            .filter(
                (pl.col("date") >= train_start)
                & (pl.col("date") <= train_end)
                & (pl.col("idx") + 4 <= lastidx)
            )
            .select(["date", "code", "idx"] + feature_cols)
            .collect()
            .sort(["date", "code"])
        )
        test = (
            pl.scan_parquet(inputs["ranked_features"])
            .filter(pl.col("date").dt.year() == year)
            .select(["date", "code", "idx"] + feature_cols)
            .collect()
            .sort(["date", "code"])
        )
        assert_keys(train, "train", cal)
        assert_keys(test, "test", cal)
        assert test["date"].max() <= pd.Timestamp(p["cutoff"]["observation"])
        train_before = len(train)
        past_prices = prices_for_labels(inputs, cal, train_end)
        training_labels = label_table(past_prices)
        train = train.join(
            training_labels.select(
                "code", "idx", "outcome", "label_start", "label_end"
            ),
            on=["code", "idx"],
            how="left",
        )
        invalid_training = int(
            train.select((~pl.col("outcome").is_finite()).fill_null(True).sum()).item()
        )
        train = train.filter(pl.col("outcome").is_finite())
        assert len(train) == train_before - invalid_training
        assert train["date"].dt.year().n_unique() >= 2
        assert (
            train["label_end"].max()
            <= train_end
            < logical_fit
            < test["date"].min() + datetime.timedelta(hours=15)
        )
        archived = archived_folds[year]
        assert (
            len(train) == archived["archived_b2d_train_rows"]
        ), f"{year}: training rows differ from B2D; stop to investigate"
        assert (
            train["date"].n_unique() == archived["archived_b2d_train_dates"]
        ), f"{year}: training dates differ from B2D"
        assert (
            len(test) == archived["archived_b2d_test_rows_all_calendar_maturity"]
        ), f"{year}: prediction pool differs from B2D"
        assert (
            test["date"].n_unique()
            == archived["archived_b2d_test_dates_all_calendar_maturity"]
        ), f"{year}: prediction dates differ from B2D"
        train_keys = key_receipt(train, out / f"keys/train_{year}.parquet")
        test_keys = key_receipt(test, out / f"keys/predict_{year}.parquet")
        x = np.ascontiguousarray(
            train.select(feature_cols).to_numpy(), dtype=np.float32
        )
        xt = np.ascontiguousarray(
            test.select(feature_cols).to_numpy(), dtype=np.float32
        )
        y = np.clip(train["outcome"].to_numpy(), *p["model"]["target_clip"])
        assert np.isfinite(x).all() and np.isfinite(xt).all() and np.isfinite(y).all()
        pred = test.select("date", "idx", "code")
        fold_fits = []
        for method, count in [("baseline", 21), ("enhanced", 23)]:
            model = HistGradientBoostingRegressor(**p["model"]["params"])
            assert model.get_params(deep=False) == p["model"]["params"]
            t0 = time.monotonic()
            fit_started = utc()
            xfit = np.ascontiguousarray(x[:, :count], dtype=np.float32)
            xpredict = np.ascontiguousarray(xt[:, :count], dtype=np.float32)
            assert xfit.flags.c_contiguous and xpredict.flags.c_contiguous
            with threadpool_limits(limits=4):
                model.fit(xfit, y)
                values = model.predict(xpredict).astype(np.float32)
            assert (
                np.isfinite(values).all()
                and model.n_iter_ == 40
                and model.do_early_stopping_ is False
            )
            pred = pred.with_columns(pl.Series(method, values))
            model_path = out / f"models/{year}_{method}.joblib"
            joblib.dump(model, model_path, compress=3)
            fold_fits.append(
                {
                    "method": method,
                    "n_features": count,
                    "feature_names": feature_cols[:count],
                    "parameters": model.get_params(deep=False),
                    "fit_started_utc": fit_started,
                    "fit_completed_utc": utc(),
                    "fit_predict_seconds": time.monotonic() - t0,
                    "model_sha256": sha(model_path),
                    "same_train_keys_sha256": train_keys["keys_sha256"],
                    "same_predict_keys_sha256": test_keys["keys_sha256"],
                }
            )
            del model, values, xfit, xpredict
        pp = out / f"predictions/predictions_{year}.parquet"
        pred.write_parquet(pp, compression="zstd")
        selected = pl.concat(
            [selections_from_predictions(pred, m) for m in ["baseline", "enhanced"]]
        )
        sizes = selected.group_by("date", "method").len()
        assert sizes["len"].min() == sizes["len"].max() == 20
        sp = out / f"selections/top20_{year}.parquet"
        selected.write_parquet(sp, compression="zstd")
        sealed_files.extend(
            [
                {
                    "role": "predictions",
                    "path": str(pp.relative_to(out)),
                    "sha256": sha(pp),
                },
                {
                    "role": "selections",
                    "path": str(sp.relative_to(out)),
                    "sha256": sha(sp),
                },
            ]
        )
        for role, dates in [
            ("training", train.select("date", "idx").unique()),
            ("prediction", test.select("date", "idx").unique()),
        ]:
            for date, idx in dates.sort("idx").iter_rows():
                maturity = idx + 4 <= maxidx
                start = cal.filter(pl.col("idx") == idx + 1)["date"].to_list()
                end = cal.filter(pl.col("idx") == idx + 4)["date"].to_list()
                clocks.append(
                    {
                        "fold_id": year,
                        "role": role,
                        "signal_time": date + datetime.timedelta(hours=15),
                        "feature_available_time_assumed": date
                        + datetime.timedelta(hours=15),
                        "supplier_actual_available_time": None,
                        "label_start": (
                            start[0] + datetime.timedelta(hours=9, minutes=30)
                            if start
                            else None
                        ),
                        "label_end": (
                            end[0] + datetime.timedelta(hours=9, minutes=30)
                            if end
                            else None
                        ),
                        "fit_cutoff": train_end
                        + datetime.timedelta(hours=23, minutes=59, seconds=59),
                        "logical_fit_time": logical_fit,
                        "predict_time_assumed": (
                            date + datetime.timedelta(hours=15)
                            if role == "prediction"
                            else None
                        ),
                        "label_mature_at_research_cutoff": maturity,
                    }
                )
        record = {
            "fold_id": year,
            "actual_start_utc": actual_start,
            "actual_complete_utc": utc(),
            "logical_fit_time": logical_fit,
            "train_information_cutoff": train_end
            + datetime.timedelta(hours=23, minutes=59, seconds=59),
            "train": train_keys,
            "predict": test_keys,
            "invalid_training_labels_excluded": invalid_training,
            "train_signal_min": train["date"].min(),
            "train_signal_max": train["date"].max(),
            "max_training_label_end": train["label_end"].max(),
            "first_predict_time_assumed": test["date"].min()
            + datetime.timedelta(hours=15),
            "original_ordinary_missing_count_train_mean": float(
                ((train["missing_count"] + 0.5) * 19).mean()
            ),
            "original_ordinary_missing_count_predict_mean": float(
                ((test["missing_count"] + 0.5) * 19).mean()
            ),
            "transformed_nonfinite_values": 0,
            "fits": fold_fits,
        }
        record["archived_B2D_pool_reconciliation"] = {
            "train_rows_match": True,
            "train_dates_match": True,
            "prediction_rows_match": True,
            "prediction_dates_match": True,
            "inventory_sha256": sha(ROOT / "e1_data_inventory.json"),
        }
        json_write(out / f"audit/fold_{year}.json", record)
        fit_records.append(record)
        print("FOLD_FITTED_AND_LISTS_WRITTEN", year, flush=True)
        del train, test, x, xt, y, pred, selected, past_prices, training_labels
    assert sum(len(f["fits"]) for f in fit_records) == 14
    pd.DataFrame(clocks).to_csv(out / "audit/field_clock_boundaries.csv", index=False)
    seal = {
        "sealed_utc": utc(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "code_sha256": sha(__file__),
        "fit_count": 14,
        "all_seven_years_complete": True,
        "evaluation_labels_created_or_joined": False,
        "files": sealed_files,
        "training_overlap_note": "Later folds use matured past labels that overlap earlier evaluation years only as training targets; no prediction-outcome evaluation join has occurred.",
    }
    json_write(out / "audit/all_predictions_and_lists_sealed.json", seal)
    return seal


def evaluation(p, inputs, cal, state, out):
    sealpath = out / "audit/all_predictions_and_lists_sealed.json"
    seal = json.loads(sealpath.read_text(encoding="utf8"))
    assert (
        seal["fit_count"] == 14
        and seal["all_seven_years_complete"]
        and not seal["evaluation_labels_created_or_joined"]
    )
    assert seal["code_sha256"] == sha(__file__) and seal["protocol_sha256"] == sha(
        PROTOCOL
    )
    for item in seal["files"]:
        assert sha(out / item["path"]) == item["sha256"]
    json_write(
        out / "audit/evaluation_label_join_started.json",
        {
            "actual_started_utc": utc(),
            "seal_sha256": sha(sealpath),
            "statement": "All predictions and fixed lists verified before first evaluation outcome construction/join.",
        },
    )
    fullprice = prices_for_labels(inputs, cal, pd.Timestamp(p["cutoff"]["label_end"]))
    lab = label_table(fullprice)
    selection = pl.concat(
        [
            pl.read_parquet(out / f"selections/top20_{y}.parquet")
            for y in p["model"]["years"]
        ]
    )
    intended = selection.join(
        pl.from_pandas(state[["date", "stress", "threshold", "fixed_20d_block"]]),
        on="date",
        how="inner",
    )
    selected = intended.join(lab, on=["code", "idx"], how="left")
    assert len(selected) == len(intended) == 1611 * 20 * 2
    selected = selected.with_columns(
        pl.when(pl.col("entry").is_null())
        .then(pl.lit("missing_entry_row"))
        .when(pl.col("exit").is_null())
        .then(pl.lit("missing_exit_row"))
        .when(pl.col("outcome").is_null() | ~pl.col("outcome").is_finite())
        .then(pl.lit("nonpositive_or_nonfinite_price"))
        .otherwise(pl.lit("observed"))
        .alias("label_status")
    )
    selected.write_parquet(out / "selected_with_labels.parquet", compression="zstd")
    missing = selected.filter(pl.col("label_status") != "observed")
    missing.write_csv(out / "selected_label_exclusions.csv")
    observed = selected.filter(pl.col("outcome").is_finite())
    assert observed["label_end"].max() <= pd.Timestamp(p["cutoff"]["label_end"])
    aggregate = (
        selected.group_by(["date", "method"])
        .agg(
            pl.len().alias("n_intended"),
            pl.col("outcome").count().alias("n_observed"),
            pl.col("outcome").mean().alias("mean_complete"),
            (pl.col("outcome").fill_null(0).sum() / 20).alias("mean_zero"),
        )
        .sort(["date", "method"])
    )
    assert aggregate["n_intended"].min() == aggregate["n_intended"].max() == 20
    aggregate.write_csv(out / "selected_daily_by_model.csv")
    daily = state.copy()
    for method in ["baseline", "enhanced"]:
        f = aggregate.filter(pl.col("method") == method).drop("method").to_pandas()
        f = f.rename(columns={c: f"{method}_{c}" for c in f.columns if c != "date"})
        daily = daily.merge(f, on="date", how="left", validate="one_to_one")
    daily["D_complete"] = daily.enhanced_mean_complete.astype(
        float
    ) - daily.baseline_mean_complete.astype(float)
    daily["D_zero"] = daily.enhanced_mean_zero.astype(
        float
    ) - daily.baseline_mean_zero.astype(float)
    daily.to_csv(out / "daily_paired.csv", index=False)
    missing_summary = {
        "intended_stock_slots": len(selected),
        "observed_stock_slots": len(observed),
        "missing_stock_slots": len(missing),
        "missing_by_reason": missing.group_by("label_status").len().to_dicts(),
        "dates": len(daily),
        "dates_with_invalid_paired_complete": int(
            (~np.isfinite(daily.D_complete)).sum()
        ),
        "calendar_immature_2026_dates_excluded": 4,
        "missing_count_does_not_change_lists": True,
    }
    json_write(out / "audit/selection_missing_receipt.json", missing_summary)
    return daily


def circular_mean_draws(values, length, B, seed):
    """Sum equivalent of fixed circular indices; missing dates stay on calendar."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    finite = np.isfinite(values)
    padded = np.r_[
        np.where(finite, values, 0.0), np.where(finite, values, 0.0)[:length]
    ]
    counts = np.r_[finite.astype(np.int64), finite.astype(np.int64)[:length]]
    cs = np.r_[0.0, np.cumsum(padded)]
    cc = np.r_[0, np.cumsum(counts)]
    pos = np.arange(n)
    blocksum, blockcount = cs[pos + length] - cs[pos], cc[pos + length] - cc[pos]
    q, r = divmod(n, length)
    k = q + bool(r)
    rng = np.random.default_rng(seed)
    result = np.empty(B, dtype=float)
    for begin in range(0, B, 1000):
        size = min(1000, B - begin)
        starts = rng.integers(0, n, size=(size, k))
        total = blocksum[starts[:, :q]].sum(axis=1)
        count = blockcount[starts[:, :q]].sum(axis=1)
        if r:
            last = starts[:, q]
            total += cs[last + r] - cs[last]
            count += cc[last + r] - cc[last]
        assert (count > 0).all(), "Bootstrap draw has no observed paired dates"
        result[begin : begin + size] = total / count
    return result


def infer_and_describe(p, daily, out):
    cells = json.loads(json.dumps(p["inference"]["cells"]))
    values = daily.D_complete.to_numpy(dtype=float)
    n = int(np.isfinite(values).sum())
    years = int(daily.loc[np.isfinite(values), "date"].dt.year.nunique())
    for cell in cells:
        if cell["endpoint"] != "all_dates":
            # Frozen failed state gate: neither bootstrap, p-value, nor CI is computed.
            assert cell["status"] == "not_estimable" and cell["p_for_correction"] == 1
            cell.update(lower95=None, upper95=None, n_dates=174)
            continue
        length = cell["block_length"]
        if years < 4 or n // length < 20:
            cell.update(
                status="not_estimable",
                reason="insufficient valid years or blocks",
                p_for_correction=1,
                lower95=None,
                upper95=None,
                n_dates=n,
            )
            continue
        estimate = float(np.nanmean(values))
        draw = circular_mean_draws(
            values, length, p["inference"]["B"], p["inference"]["seed"]
        )
        lower, upper = np.quantile(draw, [0.025, 0.975], method="linear")
        rawp = float(
            (1 + np.sum(np.abs(draw - estimate) >= abs(estimate))) / (len(draw) + 1)
        )
        cell.update(
            status="estimated",
            reason=None,
            estimate=estimate * 100,
            lower95=float(lower * 100),
            upper95=float(upper * 100),
            p_raw=rawp,
            p_for_correction=rawp,
            n_dates=n,
            effective_block_screen=n // length,
            unit="percentage_points_per_H3",
            ci_type="unadjusted_pointwise95_percentile",
        )
        np.save(out / f"bootstrap_mean_draws_L{length}.npy", draw)
    raw = np.array([c["p_for_correction"] for c in cells], dtype=float)
    order = np.argsort(raw, kind="stable")
    adjusted = np.empty(6)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate((6 - np.arange(6)) * raw[order])
    )
    for i, cell in enumerate(cells):
        cell["p_holm6"] = float(adjusted[i]) if cell["status"] == "estimated" else None
        cell["correction_placeholder_adjusted"] = (
            float(adjusted[i]) if cell["status"] != "estimated" else None
        )
    json_write(
        out / "inference_family.json",
        {"family_size": 6, "cells": cells, "pressure_inference_performed": False},
    )
    pd.DataFrame(cells).to_csv(out / "inference_family.csv", index=False)
    annual = []
    for year, f in daily.groupby(daily.date.dt.year):
        annual.append(
            {
                "year": int(year),
                "n_dates": len(f),
                "n_observed_paired": int(np.isfinite(f.D_complete).sum()),
                "mean_complete_pp": float(f.D_complete.mean() * 100),
                "mean_zero_pp": float(f.D_zero.mean() * 100),
                "leave_one_year_out_complete_pp": float(
                    daily.loc[daily.date.dt.year != year, "D_complete"].mean() * 100
                ),
            }
        )
    pd.DataFrame(annual).to_csv(out / "annual_descriptive.csv", index=False)
    state_desc = []
    for s, f in daily.groupby("stress"):
        state_desc.append(
            {
                "stress": bool(s),
                "n_dates": len(f),
                "mean_complete_pp": float(f.D_complete.mean() * 100),
                "mean_zero_pp": float(f.D_zero.mean() * 100),
                "inference_status": "not_estimable_support_gate",
                "p": None,
                "ci": None,
            }
        )
    json_write(out / "state_point_estimates_descriptive_only.json", state_desc)
    complete = float(daily.D_complete.mean() * 100)
    zero = float(daily.D_zero.mean() * 100)
    both_pass = all(
        c["status"] == "estimated" and c["estimate"] > 0 and c["p_holm6"] < 0.05
        for c in cells[:2]
    )
    json_write(
        out / "result_summary.json",
        {
            "complete_mean_pp": complete,
            "fixed_denominator_zero_mean_pp": zero,
            "missing_rule_change_pp": zero - complete,
            "missing_rule_sign_changed": bool(np.sign(complete) != np.sign(zero)),
            "all_dates_both_lengths_positive_holm6": both_pass,
            "pressure_claim": "not_estimable; no CI or p-value computed",
            "economic_threshold": None,
            "scope": p["interpretation"],
            "material_missing_sensitivity": "Magnitude reported; no post-hoc numeric threshold defines materiality.",
        },
    )


def execute(p):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = ROOT / f"run_{stamp}"
    out.mkdir(exist_ok=False)
    for directory in ["audit", "keys", "models", "predictions", "selections", "tmp"]:
        (out / directory).mkdir()
    for var in ["TEMP", "TMP", "JOBLIB_TEMP_FOLDER"]:
        os.environ[var] = str(out / "tmp")
    json_write(
        out / "audit/execution_started.json",
        {
            "actual_started_utc": utc(),
            "protocol_sha256": sha(PROTOCOL),
            "code_sha256": sha(__file__),
            "frozen_at": p["frozen_at"],
            "requested_fits": 14,
        },
    )
    try:
        load_runtime()
        inputs, input_records = verify_inputs(p)
        json_write(out / "audit/verified_inputs.json", input_records)
        json_write(
            out / "audit/runtime.json",
            {
                "python": sys.version,
                "executable": sys.executable,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "polars": pl.__version__,
                "sklearn": sklearn.__version__,
                "sklearn_path": sklearn.__file__,
                "threads": 4,
            },
        )
        cal = calendar_load(inputs, pd.Timestamp(p["cutoff"]["observation"]))
        state = state_prepare(inputs, cal, p, out)
        fit_all(p, inputs, cal, out)
        daily = evaluation(p, inputs, cal, state, out)
        infer_and_describe(p, daily, out)
        json_write(
            out / "audit/completed.json",
            {
                "completed_utc": utc(),
                "fit_count": 14,
                "status": "COMPLETE",
                "protocol_sha256": sha(PROTOCOL),
                "code_sha256": sha(__file__),
                "source_hashes_after": verify_inputs(p)[1],
            },
        )
        print("E1_COMPLETE", out, flush=True)
    except Exception:
        json_write(
            out / "audit/failure.json",
            {
                "failed_utc": utc(),
                "traceback": traceback.format_exc(),
                "policy": "Preserve run. Diagnose evidence-backed failure before any repair; never tune from effects.",
            },
        )
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly run frozen14 fits and downstream fixed evaluation once.",
    )
    args = ap.parse_args()
    p = load_protocol()
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "STATIC_NO_FIT",
                    "protocol_sha256": sha(PROTOCOL),
                    "code_sha256": sha(__file__),
                    "features": [21, 23],
                    "fits_if_authorized": 14,
                },
                indent=2,
            )
        )
        return
    execute(p)


if __name__ == "__main__":
    main()
