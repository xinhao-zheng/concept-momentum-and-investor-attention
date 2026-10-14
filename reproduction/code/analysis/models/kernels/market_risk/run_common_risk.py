"""Frozen four-model common-risk diagnostic; immutable source, private outputs."""

from pathlib import Path
import os, sys, json, hashlib, datetime, time, importlib.util, traceback, gc

D = Path(__file__).resolve().parent
R = D.parents[1]
S = R / "analysis"
for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[k] = "4"
sys.dont_write_bytecode = True
sys.path.insert(0, "__MODELS__/runtime_source/feature_increment/vendor")
import numpy as np, pandas as pd, polars as pl, sklearn, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

assert sklearn.__version__ == "1.8.0"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def js(p, x):
    p = Path(p)
    assert p.resolve().is_relative_to(D.resolve())
    assert not p.exists(), p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(x, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


P = D / "protocol_frozen.json"
EXPECTED = "f24829dd8958033cbc6d9108604bdc421802648d61b8cfd9015729132d56e081"
assert sha(P) == EXPECTED
protocol = json.loads(P.read_text(encoding="utf-8"))
spec = importlib.util.spec_from_file_location(
    "e1_reference_helpers", Path("__MODELS__/E1/code/run_e1.py")
)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
E.load_runtime()


def risk_features(inputs, cal, out):
    raw = pd.read_csv(
        inputs["csi300"], usecols=["candle_end_time", "close"], encoding="gb18030"
    )
    raw["date"] = pd.to_datetime(raw.candle_end_time)
    excluded = int((raw.date > pd.Timestamp("2026-08-31")).sum())
    raw = raw.loc[raw.date <= pd.Timestamp("2026-08-31")].sort_values("date")
    assert raw.date.is_unique and raw.close.gt(0).all()
    raw["market_ret"] = raw.close.pct_change(fill_method=None)
    raw[["date", "close", "market_ret"]].to_csv(
        out / "csi300_filtered.csv", index=False
    )
    dates = pd.DatetimeIndex(cal["date"].to_numpy())
    m = raw.set_index("date").market_ret.reindex(dates)
    prices = (
        pl.read_parquet(inputs["prices"], columns=["date", "code", "ret"])
        .filter(pl.col("date") <= datetime.datetime(2026, 8, 31))
        .sort(["code", "date"])
    )
    chunks = []
    checks = []
    rng = np.random.default_rng(20261009)
    groups = prices.partition_by("code", as_dict=True, maintain_order=True)
    for ix, (key, g) in enumerate(groups.items()):
        code = key[0]
        s = g.to_pandas().set_index("date").ret.reindex(dates)
        valid = np.isfinite(s) & np.isfinite(m)
        x = m.where(valid, 0.0)
        y = s.where(valid, 0.0)
        z = (
            pd.DataFrame(
                {
                    "n": valid.astype(float),
                    "x": x,
                    "y": y,
                    "xx": x * x,
                    "yy": y * y,
                    "xy": x * y,
                }
            )
            .rolling(60, min_periods=1)
            .sum()
            .shift(1)
        )
        n = z.n
        xx = z.xx - z.x * z.x / n
        yy = z.yy - z.y * z.y / n
        xy = z.xy - z.x * z.y / n
        good = (
            (n >= 45) & (xx > 0) & np.isfinite(xx) & np.isfinite(yy) & np.isfinite(xy)
        )
        beta = (xy / xx).where(good)
        sse = yy - xy * xy / xx
        assert not ((sse < -1e-10) & good).any(), code
        ivol = np.sqrt((sse.clip(lower=0) / (n - 2)).where(good))
        frame = pd.DataFrame(
            {
                "date": dates,
                "code": code,
                "beta60": beta.to_numpy(),
                "ivol60": ivol.to_numpy(),
                "paired_n": n.fillna(0).astype("int16").to_numpy(),
            }
        )
        chunks.append(pl.from_pandas(frame))
        if ix % 97 == 0:
            eligible = np.flatnonzero(good.to_numpy())
            if len(eligible):
                j = int(rng.choice(eligible))
                a = max(0, j - 60)
                v = valid.iloc[a:j].to_numpy()
                xx0 = m.iloc[a:j].to_numpy()[v]
                yy0 = s.iloc[a:j].to_numpy()[v]
                coef = np.linalg.lstsq(
                    np.column_stack([np.ones(len(xx0)), xx0]), yy0, rcond=None
                )[0]
                iv = np.sqrt(
                    np.sum((yy0 - coef[0] - coef[1] * xx0) ** 2) / (len(xx0) - 2)
                )
                assert np.isclose(
                    coef[1], beta.iloc[j], rtol=1e-10, atol=1e-10
                ) and np.isclose(iv, ivol.iloc[j], rtol=1e-10, atol=1e-10)
                checks.append(
                    dict(
                        code=code,
                        signal=dates[j],
                        window_start=dates[a],
                        window_last=dates[j - 1],
                        paired_n=len(xx0),
                        beta_error=float(coef[1] - beta.iloc[j]),
                        ivol_error=float(iv - ivol.iloc[j]),
                    )
                )
        if ix % 500 == 0:
            print("RISK_STOCK", ix, len(groups), flush=True)
    risk = pl.concat(chunks)
    keys = pl.read_parquet(inputs["ranked_features"], columns=["date", "code", "idx"])
    risk = keys.join(risk, on=["date", "code"], how="left", validate="1:1")
    assert len(risk) == 8835000
    risk.write_parquet(out / "risk_raw_on_common_pool.parquet", compression="zstd")
    missing = pl.sum_horizontal(
        [
            (~pl.col(c).is_finite()).fill_null(True).cast(pl.Int32)
            for c in ["beta60", "ivol60"]
        ]
    )
    ranked = risk.with_columns(
        (missing / 2 - 0.5).cast(pl.Float32).alias("risk_missing_count")
    )
    for c in ["beta60", "ivol60"]:
        ranked = ranked.with_columns(
            (
                pl.col(c).rank("average").over("date") / pl.col(c).count().over("date")
                - 0.5
            )
            .fill_null(0)
            .cast(pl.Float32)
            .alias(c)
        )
    ranked = ranked.select(
        "date", "code", "idx", "beta60", "ivol60", "risk_missing_count"
    )
    ranked.write_parquet(out / "risk_ranked_on_common_pool.parquet", compression="zstd")
    js(
        out / "audit/risk_preparation.json",
        dict(
            created=now(),
            source_rows=len(raw) + excluded,
            excluded_after_cutoff_before_any_transform=excluded,
            rows=len(risk),
            first=risk["date"].min(),
            last=risk["date"].max(),
            risk_missing_rows=risk.select(
                pl.any_horizontal(
                    pl.col("beta60").is_null(), pl.col("ivol60").is_null()
                ).sum()
            ).item(),
            window="strictly t-60 to t-1; no signal-day observation",
            spotchecks=checks,
            raw_sha=sha(out / "risk_raw_on_common_pool.parquet"),
            ranked_sha=sha(out / "risk_ranked_on_common_pool.parquet"),
        ),
    )
    return out / "risk_ranked_on_common_pool.parquet"


def fit(inputs, cal, riskpath, out):
    records = []
    seals = []
    features = protocol["features"]
    unions = list(dict.fromkeys(c for cols in features.values() for c in cols))
    risk = pl.scan_parquet(riskpath)
    expected = {
        x["fold_id"]: x
        for x in json.loads(
            (D.parent / "E1/e1_data_inventory.json").read_text(encoding="utf-8")
        )["folds"]
    }
    for year in range(2020, 2027):
        start = pd.Timestamp(year - 3, 1, 1)
        end = pd.Timestamp(year - 1, 12, 31)
        lastidx = cal.filter(pl.col("date") <= end)["idx"].max()

        def make(predicate):
            return (
                pl.scan_parquet(inputs["ranked_features"])
                .filter(predicate)
                .join(risk, on=["date", "code", "idx"], how="left")
                .select(["date", "code", "idx"] + unions)
                .collect()
                .sort(["date", "code"])
            )

        train = make(
            (pl.col("date") >= start)
            & (pl.col("date") <= end)
            & (pl.col("idx") + 4 <= lastidx)
        )
        test = make(pl.col("date").dt.year() == year)
        labs = E.label_table(E.prices_for_labels(inputs, cal, end))
        train = train.join(
            labs.select("code", "idx", "outcome", "label_end"),
            on=["code", "idx"],
            how="left",
        ).filter(pl.col("outcome").is_finite())
        assert (
            train["label_end"].max() <= end and train["date"].dt.year().n_unique() >= 2
        )
        assert len(train) == expected[year]["archived_b2d_train_rows"]
        assert (
            len(test) == expected[year]["archived_b2d_test_rows_all_calendar_maturity"]
        )
        for name, f in [("train", train), ("predict", test)]:
            f.select("date", "code", "idx").write_parquet(
                out / f"keys/{name}_{year}.parquet", compression="zstd"
            )
        pred = test.select("date", "code", "idx")
        rec = []
        y = np.clip(train["outcome"].to_numpy(), -0.8, 2.0)
        for method, cols in features.items():
            x = np.ascontiguousarray(train.select(cols).to_numpy(), dtype=np.float32)
            xt = np.ascontiguousarray(test.select(cols).to_numpy(), dtype=np.float32)
            assert np.isfinite(x).all() and np.isfinite(xt).all()
            model = HistGradientBoostingRegressor(**protocol["model"]["params"])
            tic = time.perf_counter()
            started = now()
            with threadpool_limits(limits=4):
                model.fit(x, y)
                values = model.predict(xt).astype(np.float32)
            assert model.n_iter_ == 40 and not model.do_early_stopping_
            assert np.isfinite(values).all()
            pred = pred.with_columns(pl.Series(method, values))
            mp = out / f"models/{year}_{method}.joblib"
            joblib.dump(model, mp, compress=3)
            rec.append(
                dict(
                    method=method,
                    features=cols,
                    params=model.get_params(),
                    started=started,
                    completed=now(),
                    seconds=time.perf_counter() - tic,
                    sha256=sha(mp),
                )
            )
            del x, xt, model, values
        pp = out / f"predictions/{year}.parquet"
        pred.write_parquet(pp, compression="zstd")
        selected = pl.concat([E.selections_from_predictions(pred, m) for m in features])
        assert selected.group_by("date", "method").len()["len"].min() == 20
        sp = out / f"selections/{year}.parquet"
        selected.write_parquet(sp, compression="zstd")
        seals.extend(
            [dict(path=str(p.relative_to(out)), sha256=sha(p)) for p in [pp, sp]]
        )
        rr = dict(
            year=year,
            train_rows=len(train),
            train_dates=train["date"].n_unique(),
            test_rows=len(test),
            test_dates=test["date"].n_unique(),
            train_key_sha=sha(out / f"keys/train_{year}.parquet"),
            test_key_sha=sha(out / f"keys/predict_{year}.parquet"),
            max_training_label_end=train["label_end"].max(),
            logical_fit_date=datetime.datetime(year, 1, 1),
            fits=rec,
        )
        js(out / f"audit/fold_{year}.json", rr)
        records.append(rr)
        print("FOUR_MODELS_FITTED", year, flush=True)
        del train, test, labs, pred, selected
        gc.collect()
    js(
        out / "audit/predictions_sealed_before_evaluation.json",
        dict(
            created=now(),
            fits=28,
            files=seals,
            protocol_sha=sha(P),
            code_sha=sha(__file__),
            evaluation_labels_joined=False,
        ),
    )
    return records


def evaluate(inputs, cal, out):
    seal = json.loads(
        (out / "audit/predictions_sealed_before_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert seal["fits"] == 28 and seal["protocol_sha"] == sha(P)
    for x in seal["files"]:
        assert sha(out / x["path"]) == x["sha256"]
    js(
        out / "audit/evaluation_start.json",
        dict(
            created=now(),
            prediction_seal_sha=sha(
                out / "audit/predictions_sealed_before_evaluation.json"
            ),
        ),
    )
    selected = pl.concat(
        [
            pl.read_parquet(out / f"selections/{year}.parquet")
            for year in range(2020, 2027)
        ]
    ).filter(pl.col("idx") + 4 <= cal["idx"].max())
    labels = E.label_table(E.prices_for_labels(inputs, cal, pd.Timestamp("2026-08-31")))
    selected = selected.join(labels, on=["code", "idx"], how="left")
    assert len(selected) == 1611 * 20 * 4
    selected.write_parquet(out / "selected_with_labels.parquet", compression="zstd")
    aggregate = (
        selected.group_by("date", "method")
        .agg(
            pl.len().alias("n"),
            pl.col("outcome").count().alias("observed"),
            pl.col("outcome").cast(pl.Float64).mean().alias("complete"),
            (pl.col("outcome").cast(pl.Float64).fill_null(0).sum() / 20).alias("zero"),
        )
        .sort("date", "method")
    )
    aggregate.write_csv(out / "daily_four_models.csv")
    a = aggregate.to_pandas()
    daily = a.pivot(index="date", columns="method", values="complete")
    z = a.pivot(index="date", columns="method", values="zero")
    valid = np.isfinite(daily[["A", "B", "C", "D"]]).all(axis=1)
    for frame in [daily, z]:
        frame["B_minus_A"] = frame.B - frame.A
        frame["D_minus_C"] = frame.D - frame.C
        frame["risk_increment_change"] = frame.D_minus_C - frame.B_minus_A
    contrasts = protocol["inference"]["endpoints"]
    daily.loc[~valid, contrasts] = np.nan
    daily.to_csv(out / "daily_complete_and_contrasts.csv")
    z.to_csv(out / "daily_missing_zero_descriptive.csv")
    rows = []
    for L in [20, 60]:
        # Same seed reset for each contrast produces exactly shared draw paths.
        draws = np.column_stack(
            [
                E.circular_mean_draws(daily[c].to_numpy(), L, 100000, 20261009)
                for c in contrasts
            ]
        )
        np.save(out / f"bootstrap_joint_L{L}.npy", draws)
        assert np.allclose(
            draws[:, 2], draws[:, 1] - draws[:, 0], atol=1e-14, rtol=1e-12
        )
        for j, c in enumerate(contrasts):
            value = float(daily[c].mean())
            bounds = np.quantile(draws[:, j], [0.025, 0.975])
            raw = (1 + np.sum(np.abs(draws[:, j] - value) >= abs(value))) / 100001
            assert (
                int(valid.sum()) // L >= 20
                and len(pd.DatetimeIndex(daily.index[valid]).year.unique()) >= 4
            )
            rows.append(
                dict(
                    contrast=c,
                    block_length=L,
                    estimate_pp=value * 100,
                    lower95_pp=bounds[0] * 100,
                    upper95_pp=bounds[1] * 100,
                    p_raw=raw,
                    n_dates=int(valid.sum()),
                    scope="conditional on fixed trained predictions; pointwise CI; retrospective",
                )
            )
    raw = np.array([x["p_raw"] for x in rows])
    order = np.argsort(raw, kind="stable")
    adjusted = np.empty(6)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate((6 - np.arange(6)) * raw[order])
    )
    for x, pval in zip(rows, adjusted):
        x["p_holm6"] = pval
    pd.DataFrame(rows).to_csv(out / "inference_holm6.csv", index=False)
    annual = daily.groupby(pd.DatetimeIndex(daily.index).year)[contrasts].mean() * 100
    annual["n_dates"] = daily.groupby(pd.DatetimeIndex(daily.index).year).size()
    annual.to_csv(out / "annual_descriptive.csv")
    js(
        out / "result_summary.json",
        dict(
            created=now(),
            four_models="A ordinary+F+sharedriskmissing; B addsW; C addsrisk toA; D addsW toC",
            mean_complete_pp={c: daily[c].mean() * 100 for c in contrasts},
            mean_zero_pp={c: z[c].mean() * 100 for c in contrasts},
            all_four_common_valid_dates=int(valid.sum()),
            intended_slots=len(selected),
            missing_slots=selected["outcome"].null_count(),
            last_signal=selected["date"].max(),
            max_observed_label_end=selected["label_end"].max(),
            inference=rows,
            interpretation=protocol["interpretation"],
        ),
    )


def main():
    if "--execute" not in sys.argv:
        print("STATIC_ONLY", sha(P))
        return
    out = D / (
        "run_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out.mkdir(exist_ok=False)
    for folder in ["audit", "keys", "models", "predictions", "selections", "tmp"]:
        (out / folder).mkdir()
    js(
        out / "audit/started.json",
        dict(
            started=now(),
            protocol_sha=sha(P),
            code_sha=sha(__file__),
            python=sys.version,
            sklearn=sklearn.__version__,
            sklearn_path=sklearn.__file__,
            threads=4,
        ),
    )
    try:
        inputs = {i["role"]: Path(i["runtime_path"]) for i in protocol["inputs"]}
        for i in protocol["inputs"]:
            assert sha(inputs[i["role"]]) == i["sha256"]
        cal = E.calendar_load(inputs, pd.Timestamp("2026-08-31"))
        risk = risk_features(inputs, cal, out)
        fit(inputs, cal, risk, out)
        evaluate(inputs, cal, out)
        for i in protocol["inputs"]:
            assert sha(inputs[i["role"]]) == i["sha256"]
        js(
            out / "audit/completed.json",
            dict(
                completed=now(),
                fits=28,
                status="COMPLETE",
                protocol_sha=sha(P),
                code_sha=sha(__file__),
                sources_unchanged=True,
            ),
        )
        print("COMPLETE", out, flush=True)
    except Exception:
        js(
            out / "audit/failure.json",
            dict(failed=now(), traceback=traceback.format_exc()),
        )
        raise


if __name__ == "__main__":
    main()
