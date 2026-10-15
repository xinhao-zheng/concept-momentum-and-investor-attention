from pathlib import Path
import os
import sys
import json
import hashlib
import datetime
import shutil
import math

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
AUDIT = HERE / "audit"
AUDIT.mkdir(exist_ok=True)
for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[k] = "2"
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "environment_vendor"))
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

O = Path("__SNAPSHOT__/forbidden_old_package")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(p, obj):
    p.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf8"
    )


# Bind copied functions to the configured paths; preserve source files.
PKG = HERE / "verification_package"
PKG.mkdir(exist_ok=True)
edits = []
for src in (HERE / "shared").rglob("*.py"):
    rel = src.relative_to(HERE / "shared")
    dst = PKG / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    s = src.read_text(encoding="utf8")
    old = s
    if rel.as_posix() == "C1/c1_temporal_audit.py":
        for a, b in [
            ("HERE.parents[1] / 'config'", "HERE.parent / 'config'"),
            (
                'HERE.parents[1]/"inputs/mechanism_analysis_panel.parquet"',
                'HERE.parents[1]/"run/inputs/panel.parquet"',
            ),
            (
                'HERE.parents[1]/"inputs/mechanism_expanding_time_folds.csv"',
                'HERE.parents[1]/"run/inputs/folds.csv"',
            ),
        ]:
            assert a in s
            s = s.replace(a, b, 1)
            edits.append({"old": a, "new": b})
    dst.write_text(s, encoding="utf8")
save(AUDIT / "original_function_path_adaptations.json", edits)
# Guard the fitting phase: only this independent N1 tree or installed libraries can be read/written.
denied = []


def guard(event, args):
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
    if writing and not (p.is_relative_to(HERE) or p.is_relative_to(AUDIT)):
        raise PermissionError("WRITE_OUTSIDE_DIAGNOSTIC_ROOT " + str(p))
    if (
        not p.is_relative_to(HERE)
        and not p.is_relative_to(AUDIT)
        and p.is_relative_to(Path("__SNAPSHOT__").resolve())
    ):
        denied.append(str(p))
        raise PermissionError("READ_OUTSIDE_DIAGNOSTIC_ROOT " + str(p))


sys.addaudithook(guard)
for p in [O / "N1/inputs/panel.parquet", Path("__SNAPSHOT__/forbidden.csv")]:
    try:
        open(p, "rb")
    except PermissionError:
        pass
    else:
        raise AssertionError("negative read guard failed")
sys.path[:0] = [
    str(PKG / "config"),
    str(PKG / "C1"),
    str(PKG / "C1S"),
    str(PKG / "vendor"),
]
import c1_temporal_audit as c1
import c1_s_window_branch as c1s

fr = pd.read_parquet(RUN / "preoutcome_frame.parquet")
out = pd.read_parquet(
    RUN / "inputs/panel.parquet",
    columns=["signal_date", "stock_code", "price_excess_3", "open_exit_date_3"],
    filters=[("lens", "==", "gamma")],
)
vf = fr.merge(out, on=["signal_date", "stock_code"], validate="one_to_one", sort=False)
gx, _ = c1.legacy.model_x(vf)
numeric = gx[:, : len(c1.legacy.CONTROLS)].copy()
registry = pd.read_csv(RUN / "inputs/folds.csv")
registry = registry[(registry.fold_id < 0) | registry.fold_id.isin([2, 3, 4, 5])]
base = c1.u.fold_masks(vf.signal_date, registry)
purged, purge_log = c1.legacy.purged_folds(vf, base, 3, "gamma")
if "--reuse_completed_validation" in sys.argv:
    sealed = json.loads(
        (HERE / "verification_completed_fit_stage.json").read_text(encoding="utf8")
    )
    for rel in [
        "run/verification_original_residuals.parquet",
        "run/verification_16_fit_records.csv",
    ]:
        assert sha(HERE / rel) == sealed["completed_artifacts"][rel]
    orig = pd.read_parquet(RUN / "verification_original_residuals.parquet")
    fit_records = pd.read_csv(RUN / "verification_16_fit_records.csv").to_dict(
        "records"
    )
    for ex in ["own_T", "own_S", "own_K"]:
        dose, _ = c1s.dose_window(vf[ex], vf.signal_date, pd.Timestamp("2022-09-30"))
        assert np.array_equal(dose, vf["dose_" + ex].to_numpy(), equal_nan=True)
else:
    orig = pd.DataFrame({"signal_date": vf.signal_date, "stock_code": vf.stock_code})
    fit_records = []
    with threadpool_limits(limits=2):
        for i, ex in enumerate(["own_T", "own_S", "own_K"]):
            dose, cal = c1s.dose_window(
                vf[ex], vf.signal_date, pd.Timestamp("2022-09-30")
            )
            assert np.array_equal(dose, vf["dose_" + ex].to_numpy(), equal_nan=True)
            orig[ex], records = c1.residual(
                vf, numeric, gx, dose, base, c1s.MODE, 1000 * i
            )
            fit_records.extend([{"target": ex, **r} for r in records])
        orig["Y"], records = c1.residual(
            vf, numeric, gx, vf.price_excess_3.to_numpy(float), purged, c1s.MODE, 100003
        )
        fit_records.extend([{"target": "Y", **r} for r in records])
    orig.to_parquet(RUN / "verification_original_residuals.parquet", index=False)
    pd.DataFrame(fit_records).to_csv(
        RUN / "verification_16_fit_records.csv", index=False
    )
assert len(fit_records) == 16 and all(r["status"] == "estimated" for r in fit_records)
m0 = pd.read_parquet(RUN / "M0_residuals.parquet")
assert m0[["signal_date", "stock_code"]].equals(orig[["signal_date", "stock_code"]])
for c in ["own_T", "own_S", "own_K", "Y"]:
    assert np.array_equal(m0[c].to_numpy(), orig[c].to_numpy(), equal_nan=True), c
# Independent numerical reconstruction using explicit Bartlett covariance matrix.
S = pd.read_csv(RUN / "coefficients_Holm36.csv")
C = pd.read_csv(RUN / "paired_differences_Holm27.csv")
fit = pd.read_csv(RUN / "fit_records.csv")
before = json.loads((RUN / "before_fit.json").read_text(encoding="utf8"))
checks = []
daily = {}
theta = {}
valid0 = None
for method in ["M0", "M1", "M2", "M3"]:
    z = pd.read_parquet(RUN / f"{method}_residuals.parquet")
    valid = z.common_evaluation.to_numpy()
    assert int(valid.sum()) == 217831 and z.loc[valid, "signal_date"].nunique() == 185
    if valid0 is None:
        valid0 = valid
    assert np.array_equal(valid, valid0)
    for ex in ["own_T", "own_S", "own_K"]:
        d = z.loc[valid, ex].to_numpy(float)
        y = z.loc[valid, "Y"].to_numpy(float)
        th = np.sum(d * y) / np.sum(d * d)
        sc = d * (y - th * d) / np.sum(d * d)
        dt = (
            pd.DataFrame({"date": z.loc[valid, "signal_date"], "score": sc})
            .groupby("date")
            .score.sum()
            .to_numpy()
        )
        dt = dt - dt.mean()
        theta[method, ex] = th
        daily[method, ex] = dt
        for lag in [1, 8, 12]:
            distance = abs(np.arange(len(dt))[:, None] - np.arange(len(dt))[None, :])
            K = np.where(distance <= lag, 1 - distance / (lag + 1), 0)
            se = np.sqrt(float(dt @ K @ dt))
            r = S[
                (S.method == method)
                & (S.exposure == ex)
                & (S.hac_lag_sample_dates == lag)
            ].iloc[0]
            calc = [
                th * 100,
                se * 100,
                (th - norm.ppf(0.975) * se) * 100,
                (th + norm.ppf(0.975) * se) * 100,
                2 * norm.sf(abs(th / se)),
            ]
            actual = r[
                ["coefficient_pp", "se_pp", "ci95_low_pp", "ci95_high_pp", "p"]
            ].to_numpy(float)
            assert np.allclose(calc, actual, rtol=0, atol=1e-12), (method, ex, lag)
            checks.append(
                {
                    "kind": "coefficient",
                    "method": method,
                    "exposure": ex,
                    "lag": lag,
                    "max_error": float(max(abs(np.array(calc) - actual))),
                }
            )
for ex in ["own_T", "own_S", "own_K"]:
    for a, b in [("M1", "M0"), ("M2", "M1"), ("M3", "M2")]:
        th = theta[a, ex] - theta[b, ex]
        dt = daily[a, ex] - daily[b, ex]
        for lag in [1, 8, 12]:
            distance = abs(np.arange(len(dt))[:, None] - np.arange(len(dt))[None, :])
            K = np.where(distance <= lag, 1 - distance / (lag + 1), 0)
            se = np.sqrt(float(dt @ K @ dt))
            r = C[
                (C.contrast == a + "-" + b)
                & (C.exposure == ex)
                & (C.hac_lag_sample_dates == lag)
            ].iloc[0]
            calc = [
                th * 100,
                se * 100,
                (th - norm.ppf(0.975) * se) * 100,
                (th + norm.ppf(0.975) * se) * 100,
                2 * norm.sf(abs(th / se)),
            ]
            actual = r[
                ["difference_pp", "se_pp", "ci95_low_pp", "ci95_high_pp", "p"]
            ].to_numpy(float)
            assert np.allclose(calc, actual, rtol=0, atol=1e-12), (a, b, ex, lag)
            checks.append(
                {
                    "kind": "contrast",
                    "contrast": a + "-" + b,
                    "exposure": ex,
                    "lag": lag,
                    "max_error": float(max(abs(np.array(calc) - actual))),
                }
            )
# Holm step-down independently with rank permutation and explicit loop.
for table, key in [(S, "p_holm36"), (C, "p_holm27")]:
    order = table.p.sort_values(kind="stable").index
    last = 0
    adjusted = np.zeros(len(table))
    for rank, idx in enumerate(order):
        last = max(last, min(1, (len(table) - rank) * table.loc[idx, "p"]))
        adjusted[idx] = last
    assert np.allclose(adjusted, table[key].to_numpy(), rtol=0, atol=1e-12)
# Reconstruct every role's training/test and thinning indices and support clocks.
dates = pd.to_datetime(vf.signal_date)
ends = pd.to_datetime(vf.open_exit_date_3)
common = (
    np.isfinite(vf[["dose_own_T", "dose_own_S", "dose_own_K"]]).all(axis=1).to_numpy()
    & np.isfinite(vf.price_excess_3)
    & vf.pre_match_ok
)
expected_eval = common & (dates >= pd.Timestamp("2022-09-30"))
assert np.array_equal(expected_eval, valid0)
support = []
for method in ["M0", "M1", "M2", "M3"]:
    for fid, tr, te in base:
        for ex in ["Y", "own_T", "own_S", "own_K"]:
            aligned = method != "M0"
            target = (
                vf.price_excess_3.to_numpy(float)
                if ex == "Y"
                else vf["dose_" + ex].to_numpy(float)
            )
            train = tr.copy()
            test = te.copy()
            if aligned or ex == "Y":
                train = train[
                    (ends.iloc[train] < dates.iloc[test].min()).to_numpy()
                    & ends.iloc[train].notna().to_numpy()
                ]
            train = train[np.isfinite(target[train])]
            test = test[np.isfinite(target[test])]
            if aligned:
                train = train[common.iloc[train].to_numpy()]
                test = test[common.iloc[test].to_numpy()]
            thin = c1.u.deterministic_thin(train, 30000)
            row = fit[
                (fit.method == method) & (fit.fold == fid) & (fit.exposure == ex)
            ].iloc[0]
            assert (
                row.training_rows == len(train)
                and row.fit_rows == len(thin)
                and row.test_rows == len(test)
            )
            if aligned:
                assert (ends.iloc[train] < dates.iloc[test].min()).all()
            support.append(
                {
                    "method": method,
                    "fold": fid,
                    "target": ex,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "fit_rows": len(thin),
                    "train_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
                    "fit_sha256": hashlib.sha256(thin.tobytes()).hexdigest(),
                    "test_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
                    "max_training_exit": str(ends.iloc[train].max()),
                }
            )
for method in ["M1", "M2", "M3"]:
    for fid, _, _ in base:
        group = [x for x in support if x["method"] == method and x["fold"] == fid]
        for key in ["train_sha256", "fit_sha256", "test_sha256"]:
            assert len({x[key] for x in group}) == 1
# Independent nearest predecessor match for all rows, no Polars join_asof.
cols = ["stock_code", "signal_date", "feature_asof_date", "source_last_trade_date"]
daily_source = pd.read_parquet(RUN / "inputs/daily_covariates.parquet", columns=cols)
daily_source = daily_source[daily_source.signal_date <= pd.Timestamp("2026-08-31")]
match_error = 0
matched = 0
source_group_indices = daily_source.groupby("stock_code", sort=False).indices
for stock, g in vf.groupby("stock_code", sort=False):
    cg = daily_source.iloc[source_group_indices[stock]].sort_values("signal_date")
    times = cg.signal_date.to_numpy("datetime64[ns]")
    q = g.observation_date.to_numpy("datetime64[ns]")
    ix = np.searchsorted(times, q, side="left") - 1
    assert (ix >= 0).all()
    ref = cg.iloc[ix]
    assert np.array_equal(ref.signal_date.to_numpy(), g.pre_signal_date.to_numpy())
    assert np.array_equal(
        ref.feature_asof_date.to_numpy(), g.pre_feature_asof_date.to_numpy()
    )
    assert np.array_equal(
        ref.source_last_trade_date.to_numpy(), g.pre_source_last_trade_date.to_numpy()
    )
    matched += len(g)
# Independently evaluate strict timing and cutoff gates on the common evaluation support.
p = vf.loc[valid0]
assert (p.pre_signal_date < p.observation_date).all()
assert (p.pre_feature_asof_date <= p.pre_signal_date).all()
assert (p.pre_source_last_trade_date <= p.pre_signal_date).all()
assert (p.observation_date <= p.attention_asof_cutoff).all()
assert p.open_exit_date_3.max() <= pd.Timestamp("2026-08-31")
assert p.pre_lag_days.between(1, 7).all()
pd.DataFrame(checks).to_csv(RUN / "independent_numeric_checks.csv", index=False)
pd.DataFrame(support).to_csv(RUN / "independent_support_indices.csv", index=False)
receipt = {
    "status": "PASS_ACTUAL_64_PLUS_16_REFITS_AND_INDEPENDENT_NUMERICAL_TIMING_SUPPORT_CHECKS",
    "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "new_diagnostic_refits": 64,
    "diagnostic_origin": "Re-execution of the declared common-support diagnostic; the historical sample is unchanged",
    "validation_fit_stage_reused_after_lookup_optimization": (
        "--reuse_completed_validation" in sys.argv
    ),
    "fit_stage_receipt_sha256": (
        sha(HERE / "verification_completed_fit_stage.json")
        if "--reuse_completed_validation" in sys.argv
        else None
    ),
    "original_function_validation_refits": 16,
    "M0_all_residuals_identical_to_original_functions": True,
    "independent_coefficient_and_contrast_checks": len(checks),
    "maximum_numeric_error": max(x["max_error"] for x in checks),
    "support_role_fold_checks": len(support),
    "aligned_support_checks": 36,
    "nearest_predecessor_rows_verified": matched,
    "common_evaluation_rows": int(valid0.sum()),
    "common_evaluation_dates": int(p.signal_date.nunique()),
    "evaluation_last_signal": str(p.signal_date.max()),
    "evaluation_last_exit": str(p.open_exit_date_3.max()),
    "coefficient_Holm36_rejections": int((S.p_holm36 < 0.05).sum()),
    "contrast_Holm27_rejections": int((C.p_holm27 < 0.05).sum()),
    "coefficient_min_Holm36": float(S.p_holm36.min()),
    "contrast_min_Holm27": float(C.p_holm27.min()),
    "external_read_negative_controls_denied": denied,
    "source_generator_rebuilt": False,
    "causal_identification_certified": False,
    "generic_DML_asymptotic_coverage_certified": False,
    "input_hashes": json.loads(
        (RUN / "input_manifest.json").read_text(encoding="utf8")
    ),
    "verification_code_sha256": sha(__file__),
    "outputs": {
        p.relative_to(HERE).as_posix(): sha(p) for p in RUN.iterdir() if p.is_file()
    },
}
save(HERE / "FINAL_N1_ACCEPTANCE.json", receipt)
print(
    json.dumps(
        {
            k: receipt[k]
            for k in [
                "status",
                "new_diagnostic_refits",
                "original_function_validation_refits",
                "nearest_predecessor_rows_verified",
                "maximum_numeric_error",
            ]
        },
        ensure_ascii=False,
    )
)
