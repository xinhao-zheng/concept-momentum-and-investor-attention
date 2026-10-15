"""Frozen historical diagnostic. No treatment-effect identification claim."""

from pathlib import Path
import sys
import os
import hashlib
import json
import datetime
import shutil
import math
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE / "run"
R6 = HERE.parents[1]
ROOT.mkdir(exist_ok=True)
for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
]:
    os.environ[k] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "environment_vendor"))
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from scipy.stats import norm


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def js(name, obj):
    (ROOT / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf8"
    )


CONTROLS = [
    "ret_1",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "log_float_mv_t",
    "log_total_mv_t",
    "log_amount_t",
    "log_amount_20_mean",
    "amount_to_float_mv_20_mean",
    "amihud_20",
    "beta_hs300_60",
    "idio_vol_hs300_60",
    "drawdown_from_60d_high",
    "market_return_t",
    "suspension_days_20",
    "zero_amount_days_20",
    "institution_net_inflow_to_amount_20",
    "large_net_inflow_to_amount_20",
    "retail_net_inflow_to_amount_20",
    "member_hs300_t",
    "member_zz500_t",
    "member_zz1000_t",
    "score_F5_W10",
    "protocol_concept_degree",
    "network_member_n",
    "peer_count_all",
]
BASE = CONTROLS[:-4]
EX = ["own_T", "own_S", "own_K"]
ALL_EX = EX + ["CategoryShare_T_loo", "CategoryShare_S_loo", "CategoryShare_K_loo"]
START = pd.Timestamp("2022-03-30")
CAL = pd.Timestamp("2022-09-30")
CUT = pd.Timestamp("2026-08-31")

if "--prepare" in sys.argv:
    if (ROOT / "input_manifest.json").exists():
        raise FileExistsError("frozen preparation exists")
    (ROOT / "inputs").mkdir(exist_ok=True)
    manifest = []
    for src, name, role in [
        (
            ROOT / "inputs/panel.parquet",
            "panel.parquet",
            "freshly reconstructed C2 mechanism panel",
        ),
        (
            ROOT / "inputs/folds.csv",
            "folds.csv",
            "freshly reconstructed fixed time folds",
        ),
        (
            ROOT / "inputs/daily_covariates.parquet",
            "daily_covariates.parquet",
            "explicitly selected daily PIT panel; input mode recorded by public launcher",
        ),
    ]:
        dst = ROOT / "inputs" / name
        a, b = sha(src), sha(dst)
        assert a == b
        manifest.append(
            dict(
                source=str(src),
                copy=str(dst),
                sha256=a,
                bytes=dst.stat().st_size,
                role=role,
            )
        )
    cols = [
        "signal_date",
        "stock_code",
        "lens",
        "observation_date",
        "attention_asof_cutoff",
        "feature_asof_date",
        "source_last_trade_date",
        "sw1_industry",
        *CONTROLS,
        *ALL_EX,
    ]
    f = pd.read_parquet(
        ROOT / "inputs/panel.parquet", columns=cols, filters=[("lens", "==", "gamma")]
    )
    f.signal_date = pd.to_datetime(f.signal_date)
    assert f.signal_date.max() <= CUT
    f = (
        f[(f.signal_date >= START) & np.isfinite(f[ALL_EX].to_numpy(float)).all(axis=1)]
        .sort_values(["signal_date", "stock_code"], kind="mergesort")
        .reset_index(drop=True)
    )
    f["row_id"] = np.arange(len(f))
    assert f.duplicated(["stock_code", "signal_date"]).sum() == 0
    ccols = [
        "signal_date",
        "stock_code",
        "feature_asof_date",
        "source_last_trade_date",
        "sw1_industry",
        *BASE,
    ]
    c = (
        pl.scan_parquet(ROOT / "inputs/daily_covariates.parquet")
        .select(ccols)
        .filter(pl.col("signal_date") <= pl.lit(CUT.to_pydatetime()))
        .collect()
    )
    c = c.rename({k: "pre_" + k for k in c.columns if k != "stock_code"}).sort(
        "pre_signal_date"
    )
    left = pl.from_pandas(f[["row_id", "stock_code", "observation_date"]]).sort(
        "observation_date"
    )
    joined = (
        left.join_asof(
            c,
            left_on="observation_date",
            right_on="pre_signal_date",
            by="stock_code",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .sort("row_id")
        .to_pandas()
    )
    assert np.array_equal(joined.row_id, f.row_id)
    for k in joined:
        if k.startswith("pre_"):
            f[k] = joined[k]
    lag = (
        pd.to_datetime(f.observation_date) - pd.to_datetime(f.pre_signal_date)
    ).dt.days
    f["pre_match_ok"] = (
        lag.between(1, 7)
        & (pd.to_datetime(f.pre_feature_asof_date) <= pd.to_datetime(f.pre_signal_date))
        & (
            pd.to_datetime(f.pre_source_last_trade_date)
            <= pd.to_datetime(f.pre_signal_date)
        )
    )
    f["pre_lag_days"] = lag
    cal = []
    for ex in EX:
        v = pd.to_numeric(f[ex])
        use = v.where(f.signal_date < CAL)
        use = use[np.isfinite(use)]
        lo, hi = float(use.quantile(0.01)), float(use.quantile(0.99))
        z = v.where(v.between(lo, hi, inclusive="both"))
        z = z - z.groupby(f.signal_date).transform("mean")
        sd = float(z[f.signal_date < CAL].std(ddof=0))
        f["dose_" + ex] = z / sd
        cal.append(
            dict(
                exposure=ex,
                rows=len(use),
                dates=int(
                    f.loc[
                        (f.signal_date < CAL) & np.isfinite(v), "signal_date"
                    ].nunique()
                ),
                p01=lo,
                p99=hi,
                scale=sd,
                excluded=int((~np.isfinite(z)).sum()),
            )
        )
    f.to_parquet(ROOT / "preoutcome_frame.parquet", index=False)
    clock = dict(
        rows=len(f),
        dates=int(f.signal_date.nunique()),
        observation_after_asof=int(
            (f.observation_date > f.attention_asof_cutoff).sum()
        ),
        current_feature_after_attention=int(
            (f.feature_asof_date > f.observation_date).sum()
        ),
        pre_match_pass=int(f.pre_match_ok.sum()),
        pre_match_fail=int((~f.pre_match_ok).sum()),
        pre_lag_counts=f.pre_lag_days.value_counts(dropna=False).to_dict(),
        all_three_doses_finite=int(
            np.isfinite(f[["dose_" + x for x in EX]]).all(axis=1).sum()
        ),
    )
    js(
        "input_manifest.json",
        dict(
            prepared_utc=now(),
            inputs=manifest,
            protocol_sha256=sha(HERE / "PROTOCOL.md"),
            code_sha256=sha(__file__),
            preoutcome_sha256=sha(ROOT / "preoutcome_frame.parquet"),
            outcome_value_columns_read=False,
            calibration=cal,
            timing=clock,
        ),
    )
    print(json.dumps(clock, ensure_ascii=False, default=str))
    raise SystemExit(0)

if (ROOT / "complete.json").exists():
    raise FileExistsError("Completed diagnostic exists")
spec = json.loads((ROOT / "input_manifest.json").read_text(encoding="utf8"))
assert sha(HERE / "PROTOCOL.md") == spec["protocol_sha256"]
assert sha(ROOT / "preoutcome_frame.parquet") == spec["preoutcome_sha256"]
guard_events = []


def pathguard(event, args):
    if (
        event != "open"
        or not args
        or not isinstance(args[0], (str, bytes, os.PathLike))
    ):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    low = str(p).replace("\\", "/").lower()
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
        isinstance(flags, int)
        and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    )
    inside = p.is_relative_to(ROOT)
    if writing and not inside:
        raise PermissionError("N1 write outside own run: " + str(p))
    allowed = (
        inside
        or p.is_relative_to(HERE / "environment_vendor")
        or p in [Path(__file__).resolve(), HERE / "PROTOCOL.md"]
    )
    if not allowed and p.is_relative_to(Path("__SNAPSHOT__").resolve()):
        guard_events.append(dict(event="DENIED_OUTSIDE_SCIENTIFIC_READ", path=str(p)))
        raise PermissionError("N1 blocked external scientific read: " + str(p))


sys.addaudithook(pathguard)
for bad in [
    Path("__SNAPSHOT__/forbidden_negative_control.parquet"),
    Path("__SNAPSHOT__/forbidden_negative_control.csv"),
]:
    try:
        open(bad, "rb")
    except PermissionError:
        pass
    else:
        raise AssertionError("Read negative control failed")
js(
    "runtime_guard.json",
    dict(
        installed_utc=now(),
        negative_controls=guard_events,
        preparation_code_sha256=spec["code_sha256"],
        estimation_code_sha256=sha(__file__),
        implementation_addendum="Before any outcome effects were fitted, added runtime path guard, prepared-frame integrity assertion, and descriptive industry information concentration. Estimands, sample, models, comparisons and families unchanged.",
    ),
)
for r in spec["inputs"]:
    assert sha(r["copy"]) == r["sha256"]
frame = pd.read_parquet(ROOT / "preoutcome_frame.parquet")
out = pd.read_parquet(
    ROOT / "inputs/panel.parquet",
    columns=["signal_date", "stock_code", "price_excess_3", "open_exit_date_3"],
    filters=[("lens", "==", "gamma")],
)
frame = frame.merge(
    out, on=["signal_date", "stock_code"], how="left", validate="one_to_one", sort=False
)
assert np.array_equal(frame.row_id, np.arange(len(frame)))
y = frame.price_excess_3.to_numpy(float)
dates = pd.to_datetime(frame.signal_date)
end = pd.to_datetime(frame.open_exit_date_3)
assert not (np.isfinite(y) & (end.isna() | (end > CUT))).any()
common = (
    np.isfinite(frame[["dose_" + x for x in EX]]).all(axis=1).to_numpy()
    & np.isfinite(y)
    & frame.pre_match_ok.to_numpy()
)
registry = pd.read_csv(ROOT / "inputs/folds.csv")
folds = []
for fid, g in registry[registry.fold_id >= 0].groupby("fold_id", sort=True):
    s = pd.Timestamp(g.test_start_date_for_fold.iloc[0])
    e = pd.Timestamp(g.test_end_date_for_fold.iloc[0])
    if s < CAL:
        continue
    train = np.flatnonzero(dates < s)
    test = np.flatnonzero((dates >= s) & (dates <= e))
    purge = train[(end.iloc[train] < s).to_numpy() & end.iloc[train].notna().to_numpy()]
    folds.append((int(fid), train, purge, test, s, e))
assert len(folds) == 4
evalmask = common & (dates >= CAL).to_numpy()
assert evalmask.sum() >= 10000 and dates[evalmask].nunique() >= 100
js(
    "before_fit.json",
    dict(
        start_utc=now(),
        code_sha256=sha(__file__),
        common_rows=int(common.sum()),
        evaluation_rows=int(evalmask.sum()),
        evaluation_dates=int(dates[evalmask].nunique()),
        common_exclusions=dict(
            no_pre_match=int((~frame.pre_match_ok).sum()),
            nonfinite_dose=int(
                (~np.isfinite(frame[["dose_" + x for x in EX]]).all(axis=1)).sum()
            ),
            nonfinite_y=int((~np.isfinite(y)).sum()),
        ),
        folds=[
            dict(
                fold_id=a,
                test_start=s,
                test_end=e,
                common_training=int(common[p].sum()),
                common_test=int(common[t].sum()),
            )
            for a, _, p, t, s, e in folds
        ],
    ),
)
fits = []


def residual(target, controls, prefix, role, exidx, aligned):
    res = np.full(len(frame), np.nan, dtype=np.float32)
    cols = [prefix + c for c in controls]
    industry = prefix + "sw1_industry"
    numeric = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    numeric[~np.isfinite(numeric)] = np.nan
    cat = frame[industry].fillna("<MISSING>").astype(str)
    for fid, unpurged, purged, test, s, e in folds:
        train = purged if (role == "outcome" or aligned) else unpurged
        train = train[np.isfinite(target[train])]
        test = test[np.isfinite(target[test])]
        if aligned:
            train = train[common[train]]
            test = test[common[test]]
        assert len(train) >= 1000 and len(test) >= 100
        fit = (
            train[np.linspace(0, len(train) - 1, 30000, dtype=np.int64)]
            if len(train) > 30000
            else train
        )
        vocab = {v: i for i, v in enumerate(sorted(cat.iloc[fit].unique()))}
        a = np.column_stack(
            [numeric[fit], cat.iloc[fit].map(vocab).to_numpy(np.float32)]
        )
        b = np.column_stack(
            [numeric[test], cat.iloc[test].map(vocab).to_numpy(np.float32)]
        )
        seed = 20260714 + (100003 if role == "outcome" else 1000 * exidx) + fid
        m = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=80,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=False,
            categorical_features=[len(controls)],
            random_state=seed,
        )
        m.fit(a, target[fit])
        pred = m.predict(b)
        res[test] = (target[test] - pred).astype(np.float32)
        den = float(np.square(target[test] - np.mean(target[test])).sum())
        r2 = 1 - float(np.square(target[test] - pred).sum()) / den if den else None
        fits.append(
            dict(
                method=METHOD,
                role=role,
                exposure=EX[exidx] if role == "dose" else "Y",
                fold=fid,
                training_rows=len(train),
                fit_rows=len(fit),
                test_rows=len(test),
                r2=r2,
                seed=seed,
                unknown_categories=int(np.isnan(b[:, -1]).sum()),
                last_training_exit=str(end.iloc[train].max()),
                test_start=str(s),
                aligned=aligned,
                control_count=len(controls),
            )
        )
    return res


residuals = {}
with threadpool_limits(limits=2):
    for METHOD, controls, prefix, aligned in [
        ("M0", CONTROLS, "", False),
        ("M1", CONTROLS, "", True),
        ("M2", BASE, "", True),
        ("M3", BASE, "pre_", True),
    ]:
        print(now(), METHOD, "FIT", flush=True)
        yr = residual(y, controls, prefix, "outcome", 0, aligned)
        rr = {"Y": yr}
        for i, ex in enumerate(EX):
            rr[ex] = residual(
                frame["dose_" + ex].to_numpy(float),
                controls,
                prefix,
                "dose",
                i,
                aligned,
            )
        residuals[METHOD] = rr
        pd.DataFrame(
            {
                "signal_date": dates,
                "stock_code": frame.stock_code,
                "common_evaluation": evalmask,
                **rr,
            }
        ).to_parquet(ROOT / (METHOD + "_residuals.parquet"), index=False)
        pd.DataFrame(fits).to_csv(ROOT / "fit_records.csv", index=False)


def meat(z, lag):
    z = np.asarray(z, float)
    z = z - z.mean()
    out = float(z @ z)
    for k in range(1, min(lag, len(z) - 1) + 1):
        out += 2 * (1 - k / (lag + 1)) * float(z[k:] @ z[:-k])
    return out


def pnorm(x):
    return max(float(math.erfc(abs(x) / math.sqrt(2))), np.nextafter(0.0, 1.0))


def holm(values):
    p = np.asarray(values)
    order = np.argsort(p)
    res = np.empty(len(p))
    res[order] = np.minimum(
        1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    )
    return res


rows = []
diff = []
foldrows = []
dailies = {}
theta = {}
diagnostics = []
for method, rr in residuals.items():
    for ex in EX:
        valid = evalmask & np.isfinite(rr[ex]) & np.isfinite(rr["Y"])
        assert np.array_equal(valid, evalmask)
        d = rr[ex][valid].astype(float)
        v = rr["Y"][valid].astype(float)
        den = float(d @ d)
        th = float(d @ v / den)
        theta[method, ex] = th
        score = d * (v - th * d) / den
        daily = pd.Series(score).groupby(dates[valid].to_numpy()).sum()
        dailies[method, ex] = daily
        energy = pd.Series(d * d).groupby(dates[valid].to_numpy()).sum()
        energy = energy / energy.sum()
        industry_energy = (
            pd.Series(d * d)
            .groupby(frame.loc[valid, "sw1_industry"].fillna("<MISSING>").to_numpy())
            .sum()
        )
        industry_energy = industry_energy / industry_energy.sum()
        diagnostics.append(
            dict(
                method=method,
                exposure=ex,
                effective_information_dates=float(1 / (energy @ energy)),
                max_date_information_share=float(energy.max()),
                residual_variance=float(np.var(d)),
                information_top10_dates=float(energy.nlargest(10).sum()),
                effective_information_industries=float(
                    1 / (industry_energy @ industry_energy)
                ),
                max_industry_information_share=float(industry_energy.max()),
                max_information_industry=str(industry_energy.idxmax()),
            )
        )
        for lag in [1, 8, 12]:
            se = math.sqrt(max(0, meat(daily.to_numpy(), lag)))
            rows.append(
                dict(
                    method=method,
                    exposure=ex,
                    hac_lag_sample_dates=lag,
                    coefficient_pp=th * 100,
                    se_pp=se * 100,
                    ci95_low_pp=(th - 1.95996398454 * se) * 100,
                    ci95_high_pp=(th + 1.95996398454 * se) * 100,
                    p=pnorm(th / se),
                    rows=int(valid.sum()),
                    dates=len(daily),
                )
            )
        for fid, _, _, _, s, e in folds:
            m = valid & (dates >= s).to_numpy() & (dates <= e).to_numpy()
            dd = rr[ex][m].astype(float)
            vv = rr["Y"][m].astype(float)
            foldrows.append(
                dict(
                    method=method,
                    exposure=ex,
                    fold=fid,
                    coefficient_pp=float(dd @ vv / (dd @ dd)) * 100,
                    rows=int(m.sum()),
                    dates=int(dates[m].nunique()),
                )
            )
for ex in EX:
    for a, b in [("M1", "M0"), ("M2", "M1"), ("M3", "M2")]:
        assert dailies[a, ex].index.equals(dailies[b, ex].index)
        delta = theta[a, ex] - theta[b, ex]
        ds = dailies[a, ex] - dailies[b, ex]
        for lag in [1, 8, 12]:
            se = math.sqrt(max(0, meat(ds.to_numpy(), lag)))
            diff.append(
                dict(
                    contrast=a + "-" + b,
                    exposure=ex,
                    hac_lag_sample_dates=lag,
                    difference_pp=delta * 100,
                    se_pp=se * 100,
                    ci95_low_pp=(delta - 1.95996398454 * se) * 100,
                    ci95_high_pp=(delta + 1.95996398454 * se) * 100,
                    p=pnorm(delta / se),
                )
            )
summary = pd.DataFrame(rows)
summary["p_holm36"] = holm(summary.p)
contrasts = pd.DataFrame(diff)
contrasts["p_holm27"] = holm(contrasts.p)
summary.to_csv(ROOT / "coefficients_Holm36.csv", index=False)
contrasts.to_csv(ROOT / "paired_differences_Holm27.csv", index=False)
pd.DataFrame(foldrows).to_csv(ROOT / "fold_coefficients_descriptive.csv", index=False)
pd.DataFrame(diagnostics).to_csv(
    ROOT / "residual_information_diagnostics.csv", index=False
)
js(
    "complete.json",
    dict(
        status="COMPUTATION_COMPLETE_CONDITIONAL_DIAGNOSTIC",
        end_utc=now(),
        fits=len(fits),
        coefficient_rows=len(rows),
        contrast_rows=len(diff),
        coefficient_rejections=int((summary.p_holm36 < 0.05).sum()),
        contrast_rejections=int((contrasts.p_holm27 < 0.05).sum()),
        input_hashes_unchanged=all(
            sha(x["copy"]) == x["sha256"] for x in spec["inputs"]
        ),
        code_sha256=sha(__file__),
        outputs={
            p.name: sha(p)
            for p in ROOT.iterdir()
            if p.is_file() and p.name != "complete.json"
        },
    ),
)
print(summary.to_string(index=False))
print(contrasts.to_string(index=False))
