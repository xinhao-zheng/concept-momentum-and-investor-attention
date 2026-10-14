"""S-coverage window branch for the C1 causal-ML audit (plan section C2, steps 3-5 and 7).

The original protocol calibrates every exposure before the first regenerated test fold, where the watchlist
field S has no records. This branch re-sets the historical window to the actual S coverage:
the window starts at the first sampled date with finite S, the past calibration ends at the
first regenerated test-fold start preceded by the protocol's minimum calibration support for S
(>=1,000 rows and >=20 dates), and only the regenerated folds that start on or after that date
are tested. Window, calibration and folds are derived from field coverage and the frozen fold
calendar only, before any outcome is read, and frozen. T and K are re-estimated in the same
window on the identical stock-days (strict intersection of the six exposures). Gates, models,
seeds and family structure follow the original protocol; results form separate families and
never overwrite the original-protocol outputs.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "C1"))
import c1_temporal_audit as c1  # noqa: E402  (sets thread limits, vendor paths and legacy imports)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

legacy, u, CFG = c1.legacy, c1.u, c1.CFG
DATA, AUDIT = HERE / "data", HERE / "audit"
PANEL, FOLDS = c1.PANEL, c1.FOLDS
LENSES, H = c1.LENSES, c1.H
MODE = "past_calibrated_fixed80"
BRANCH = "S_coverage_window"
S_EXPOSURES = ("own_S", "CategoryShare_S_loo")
MIN_CAL_ROWS, MIN_CAL_DATES = 1000, 20


def derive_window():
    """Window from S coverage and the frozen fold calendar only (no outcome columns are read)."""
    registry = pd.read_csv(FOLDS)
    registry["signal_date"] = pd.to_datetime(registry["signal_date"])
    starts = (
        registry[registry.fold_id >= 0]
        .groupby("fold_id")
        .test_start_date_for_fold.first()
        .map(pd.Timestamp)
        .sort_index()
    )
    frame = pd.read_parquet(PANEL, columns=["lens", "signal_date", *legacy.EXPOSURES])
    frame["signal_date"] = pd.to_datetime(frame.signal_date)
    finite_s = np.all(
        [np.isfinite(pd.to_numeric(frame[e], errors="coerce")) for e in S_EXPOSURES],
        axis=0,
    )
    window_start = frame.signal_date[finite_s].min()
    chosen = None
    for fold, start in starts.items():
        ok = True
        for lens in LENSES:
            g = frame[(frame.lens == lens)]
            for e in S_EXPOSURES:
                v = np.isfinite(pd.to_numeric(g[e], errors="coerce")) & (
                    g.signal_date < start
                )
                ok &= (
                    int(v.sum()) >= MIN_CAL_ROWS
                    and g.signal_date[v].nunique() >= MIN_CAL_DATES
                )
        if ok:
            chosen = fold
            break
    assert chosen is not None
    folds = [int(f) for f in starts.index if f >= chosen]
    support = {}
    for lens in LENSES:
        g = frame[frame.lens == lens]
        v = np.isfinite(pd.to_numeric(g["own_S"], errors="coerce")) & (
            g.signal_date < starts[chosen]
        )
        support[lens] = {"rows": int(v.sum()), "dates": int(g.signal_date[v].nunique())}
    return {
        "window_start": str(window_start.date()),
        "calibration_end_exclusive": str(starts[chosen].date()),
        "test_folds": folds,
        "test_period": f"{starts[chosen].date()}..{registry.test_end_date_for_fold.dropna().max()}",
        "calibration_support_own_S_by_lens": support,
    }


def freeze():
    if (AUDIT / "frozen_design.json").exists():
        raise FileExistsError("Design already frozen")
    DATA.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    window = derive_window()
    spec = {
        "frozen_at_utc": c1.now(),
        "protocol_branch": BRANCH,
        "status": "new-window historical analysis set by actual field coverage; not preregistered",
        "known_history": "Prior shorter-window results had been inspected. The expanded S-window calendar is derived from coverage and regenerated folds before expanded-window estimation.",
        "window_rule": "Start at the first sampled signal date with finite S; past calibration ends at the first regenerated test-fold start preceded by >=1,000 finite S rows and >=20 S dates for both S exposures in every lens; test only the regenerated folds starting on or after that date. Derived from coverage and the fold calendar only.",
        **window,
        "sample": "Signal dates >= window_start and the strict intersection of finite own_T/own_S/own_K and CategoryShare_T/S/K_loo, so T, S and K use identical stock-days and endpoints.",
        "mode": MODE,
        "lenses": LENSES,
        "H": H,
        "DML_exposures": legacy.EXPOSURES,
        "AIPW_states": list(legacy.STATE_COLUMNS) + ["TSK_all_high_vs_all_low"],
        "calibration": "Per lens/exposure over window signal dates strictly before calibration_end_exclusive: p01/p99 exclusion, date-centering, fixed past SD; >=1,000 finite rows and >=20 dates.",
        "models": "As original past_calibrated_fixed80: fixed hyperparameters, early_stopping=False, max_iter=80, fit-only industry vocabulary, same seeds and training-row caps.",
        "gates": "DML >=10,000 rows, >=100 test dates, every branch fold estimated; AIPW >=1,000 per train/test group per fold, pooled clip fraction <=0.20, every branch fold estimated.",
        "families": "S_window|DML24 per lens and exposure group; S_window|AIPW32 per lens. Deferred cells enter with p=1 and blank formal inference. Separate from the original-protocol families.",
        "inputs": {str(p): c1.sha(p) for p in [PANEL, FOLDS]},
        "code_sha256": c1.sha(__file__),
        "original_c1_code_sha256": c1.sha(c1.__file__),
    }
    c1.jsave(AUDIT / "frozen_design.json", spec)
    c1.log("BRANCH DESIGN FROZEN " + json.dumps(window))


def dose_window(values, dates, end):
    values = pd.to_numeric(values, errors="coerce")
    calibration = dates < end
    usable = values.where(calibration)
    usable = usable[np.isfinite(usable)]
    cd = int(dates[calibration & values.notna() & np.isfinite(values)].nunique())
    if len(usable) < MIN_CAL_ROWS or cd < MIN_CAL_DATES:
        return np.full(len(values), np.nan), {
            "status": "DEFERRED_NO_PAST_CALIBRATION",
            "calibration_rows": len(usable),
            "calibration_dates": cd,
            "scale": None,
        }
    lo, hi = float(usable.quantile(0.01)), float(usable.quantile(0.99))
    central = values.where(values.between(lo, hi, inclusive="both"))
    centered = central - central.groupby(dates, observed=True).transform("mean")
    scale = float(centered[calibration].std(ddof=0))
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError("Invalid calibration scale")
    return (centered / scale).to_numpy(float), {
        "status": "CALIBRATED",
        "calibration_rows": len(usable),
        "calibration_dates": cd,
        "calibration_end_exclusive": str(end.date()),
        "p01": lo,
        "p99": hi,
        "scale": scale,
    }


def finalize(rows, kind):
    df = pd.DataFrame(rows)
    df["multiplicity_family"] = (
        ("S_window|tree_dml_3features_x_8H|" + df.lens + "|" + df.exposure_group)
        if kind == "dml"
        else ("S_window|tree_aipw_4states_x_8H|" + df.lens)
    )
    df = u.add_multiplicity_fail_closed(df, ["multiplicity_family"])
    df["protocol_branch"] = BRANCH
    return df


def run_lens(lens):
    spec = json.loads((AUDIT / "frozen_design.json").read_text(encoding="utf-8"))
    if (AUDIT / f"complete_{lens}.json").exists():
        raise FileExistsError("Refusing to overwrite complete lens")
    started = time.time()
    start, end = pd.Timestamp(spec["window_start"]), pd.Timestamp(
        spec["calibration_end_exclusive"]
    )
    cols = {
        "lens",
        "signal_date",
        "stock_code",
        "sw1_industry",
        *legacy.CONTROLS,
        *legacy.EXPOSURES,
        *legacy.STATE_COLUMNS.values(),
        *[f"price_excess_{h}" for h in H],
        *[f"open_exit_date_{h}" for h in H],
        *[f"attention_target_date_{h}" for h in H],
    }
    frame = pd.read_parquet(PANEL, columns=sorted(cols), filters=[("lens", "==", lens)])
    frame.signal_date = pd.to_datetime(frame.signal_date)
    rows_all = len(frame)
    inter = np.all(
        [
            np.isfinite(pd.to_numeric(frame[e], errors="coerce"))
            for e in legacy.EXPOSURES
        ],
        axis=0,
    )
    frame = frame[(frame.signal_date >= start) & inter].copy()
    frame.sort_values(["signal_date", "stock_code"], kind="mergesort", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    gx, _ = legacy.model_x(frame)
    numeric = gx[:, : len(legacy.CONTROLS)].copy()
    registry = pd.read_csv(FOLDS)
    registry = registry[
        (registry.fold_id < 0) | registry.fold_id.isin(spec["test_folds"])
    ]
    base = u.fold_masks(frame.signal_date, registry)
    assert [f for f, _, _ in base] == spec["test_folds"]
    assert min(frame.iloc[t].signal_date.min() for _, _, t in base) >= end
    nf = len(base)
    purged, purges = {}, []
    for h in H:
        purged[h], a = legacy.purged_folds(frame, base, h, lens)
        purges += a
    # A window that starts in 2022 leaves no matured long-H training labels before the first test
    # fold; such folds are empty (and fail the fold gate), not leaks.
    for a in purges:
        a["empty_after_purge"] = a["train_rows_after_purge"] == 0
        assert a["empty_after_purge"] or a["target_strictly_before_test"]
    pd.DataFrame(purges).to_csv(AUDIT / f"purge_{lens}.csv", index=False)
    all_folds = lambda ds: len(ds) == nf and all(
        x["status"] == "estimated" for x in ds
    )  # noqa: E731
    states = legacy.build_states(frame)
    fits, calibrations, drows, arows = [], [], [], []
    dr, ds, scales = {}, {}, {}
    for i, ex in enumerate(legacy.EXPOSURES):
        dose, cal = dose_window(frame[ex], frame.signal_date, end)
        scales[ex] = cal["scale"]
        calibrations.append({"lens": lens, "exposure": ex, **cal})
        dr[ex], ds[ex] = c1.residual(frame, numeric, gx, dose, base, MODE, 1000 * i)
        fits += [
            {"lens": lens, "module": "DML", "role": "dose", "variable": ex, "H": 0, **x}
            for x in ds[ex]
        ]
    for h in H:
        c1.log(f"{BRANCH} {lens} H{h}")
        y = pd.to_numeric(frame[f"price_excess_{h}"], errors="coerce").to_numpy(float)
        yr, ys = c1.residual(frame, numeric, gx, y, purged[h], MODE, 100000 + h)
        fits += [
            {
                "lens": lens,
                "module": "DML",
                "role": "outcome",
                "variable": "price_excess",
                "H": h,
                **x,
            }
            for x in ys
        ]
        for ex in legacy.EXPOSURES:
            e = c1.full_est(dr[ex], yr, frame.signal_date, h)
            complete = (
                e["n_rows"] >= 10000
                and e["n_dates"] >= 100
                and all_folds(ds[ex])
                and all_folds(ys)
            )
            drows.append(
                {
                    "lens": lens,
                    "mode": MODE,
                    "exposure": ex,
                    "exposure_group": (
                        "own" if ex.startswith("own_") else "CategoryShare"
                    ),
                    "holding_h": h,
                    **e,
                    "estimability_status": (
                        "COMPLETE"
                        if complete
                        else "DEFERRED_INCOMPLETE_TIME_FOLDS_OR_CALIBRATION"
                    ),
                    "dose_scale": scales[ex],
                }
            )
        for i, (state, w) in enumerate(states.items()):
            _, e, diag = c1.aipw(
                frame, numeric, gx, y, w, purged[h], MODE, h, 1000000 + i * 10000 + h
            )
            fits += [
                {
                    "lens": lens,
                    "module": "AIPW",
                    "role": "joint",
                    "variable": state,
                    "H": h,
                    **x,
                }
                for x in diag
            ]
            support = e["n_rows"] >= 10000 and e["n_dates"] >= 100
            overlap = (
                np.isfinite(e["pooled_propensity_clipped_fraction"])
                and e["pooled_propensity_clipped_fraction"] <= 0.20
            )
            timegate = all_folds(diag)
            arows.append(
                {
                    "lens": lens,
                    "mode": MODE,
                    "state_contrast": state,
                    "holding_h": h,
                    **e,
                    "support_gate_pass": support,
                    "overlap_gate_pass": overlap,
                    "all_time_folds_pass": timegate,
                    "estimability_status": (
                        "COMPLETE"
                        if support and overlap and timegate
                        else "DEFERRED_SUPPORT_OVERLAP_OR_TIME_FOLD_GATE"
                    ),
                }
            )
    for kind, rows in (("dml", drows), ("aipw", arows)):
        finalize(rows, kind).to_csv(DATA / f"{kind}_S_window_{lens}.csv", index=False)
    pd.DataFrame(fits).to_csv(AUDIT / f"nuisance_{lens}.csv", index=False)
    c1.jsave(AUDIT / f"calibration_{lens}.json", calibrations)
    c1.jsave(
        AUDIT / f"complete_{lens}.json",
        {
            "lens": lens,
            "completed_at_utc": c1.now(),
            "elapsed_seconds": time.time() - started,
            "panel_rows_lens": rows_all,
            "branch_rows": len(frame),
            "branch_dates": int(frame.signal_date.nunique()),
            "first_date": str(frame.signal_date.min().date()),
            "last_date": str(frame.signal_date.max().date()),
            "test_folds": spec["test_folds"],
            "purge_violations": 0,
        },
    )
    c1.log(f"COMPLETE {lens} {time.time() - started:.1f}s")


def combine():
    spec = json.loads((AUDIT / "frozen_design.json").read_text(encoding="utf-8"))
    unchanged, resolutions = c1.verify_frozen_inputs(spec["inputs"], CFG.EXP)
    code_ok = c1.sha(__file__) == spec["code_sha256"]
    summary, diag_rows = [], []
    for kind, expected, fam in (("dml", 144, 24), ("aipw", 96, 32)):
        f = pd.concat(
            [pd.read_csv(DATA / f"{kind}_S_window_{lens}.csv") for lens in LENSES],
            ignore_index=True,
        )
        assert len(f) == expected and (f.family_n == fam).all()
        deferred = ~f.estimability_status.eq("COMPLETE")
        assert (
            f.loc[deferred, ["coefficient", "p_holm_family", "se_hac"]]
            .isna()
            .all()
            .all()
        )
        f.to_csv(DATA / f"{kind}_S_window_all.csv", index=False)
        summary.append(
            {
                "branch": BRANCH,
                "module": kind,
                "grid": len(f),
                "complete": int((~deferred).sum()),
                "deferred": int(deferred.sum()),
                "holm_lt_05": int((f.p_holm_family < 0.05).sum()),
                "BH_q_lt_05_secondary": int((f.q_bh_family < 0.05).sum()),
            }
        )
    last = max(spec["test_folds"])
    for lens in LENSES:
        nuis = pd.read_csv(AUDIT / f"nuisance_{lens}.csv")
        cal = pd.DataFrame(
            json.loads((AUDIT / f"calibration_{lens}.json").read_text(encoding="utf-8"))
        )
        for r in pd.read_csv(DATA / f"dml_S_window_{lens}.csv").itertuples():
            c = cal[cal.exposure == r.exposure].iloc[0]
            dose = nuis[
                (nuis.module == "DML")
                & (nuis.role == "dose")
                & (nuis.variable == r.exposure)
            ]
            out = nuis[
                (nuis.module == "DML")
                & (nuis.role == "outcome")
                & (nuis.H == r.holding_h)
            ]
            bad_d, bad_o = (
                dose[dose.status != "estimated"].fold_id.tolist(),
                out[out.status != "estimated"].fold_id.tolist(),
            )
            codes = []
            if c.status != "CALIBRATED":
                codes.append("NO_PAST_CALIBRATION")
            if bad_d or bad_o:
                codes.append("INSUFFICIENT_TIME_FOLDS")
            if last in bad_o and int(out[out.fold_id == last].test_rows.iloc[0]) < 100:
                codes.append("IMMATURE_OUTCOME")
            if r.estimability_status == "COMPLETE":
                codes = ["ESTIMABLE"]
            diag_rows.append(
                {
                    "branch": BRANCH,
                    "lens": lens,
                    "method": "Tree-DML",
                    "cell": r.exposure,
                    "H": r.holding_h,
                    "calibration_rows": c.calibration_rows,
                    "calibration_dates": c.calibration_dates,
                    "dose_folds_failed": ",".join(map(str, bad_d)),
                    "outcome_folds_failed": ",".join(map(str, bad_o)),
                    "n_rows": r.n_rows,
                    "n_dates": r.n_dates,
                    "pooled_clip_fraction": None,
                    "coefficient_pp": (
                        r.coefficient * 100
                        if r.estimability_status == "COMPLETE"
                        else None
                    ),
                    "p_holm_family": r.p_holm_family,
                    "final_status": r.estimability_status,
                    "codes": "|".join(codes),
                }
            )
        for r in pd.read_csv(DATA / f"aipw_S_window_{lens}.csv").itertuples():
            fz = nuis[
                (nuis.module == "AIPW")
                & (nuis.variable == r.state_contrast)
                & (nuis.H == r.holding_h)
            ]
            bad = fz[fz.status != "estimated"].fold_id.tolist()
            codes = []
            if bad:
                codes.append("INSUFFICIENT_TIME_FOLDS")
                if (
                    last in bad
                    and int(
                        fz[fz.fold_id == last][["test0", "test1"]].min(axis=1).iloc[0]
                    )
                    < 1000
                ):
                    codes.append("IMMATURE_OUTCOME")
            if not (r.pooled_propensity_clipped_fraction <= 0.20):
                codes.append("POOR_OVERLAP")
            if not r.support_gate_pass:
                codes.append("INSUFFICIENT_SUPPORT")
            if r.estimability_status == "COMPLETE":
                codes = ["ESTIMABLE"]
            diag_rows.append(
                {
                    "branch": BRANCH,
                    "lens": lens,
                    "method": "AIPW",
                    "cell": r.state_contrast,
                    "H": r.holding_h,
                    "calibration_rows": None,
                    "calibration_dates": None,
                    "dose_folds_failed": "",
                    "outcome_folds_failed": ",".join(map(str, bad)),
                    "n_rows": r.n_rows,
                    "n_dates": r.n_dates,
                    "pooled_clip_fraction": r.pooled_propensity_clipped_fraction,
                    "coefficient_pp": (
                        r.coefficient * 100
                        if r.estimability_status == "COMPLETE"
                        else None
                    ),
                    "p_holm_family": r.p_holm_family,
                    "final_status": r.estimability_status,
                    "codes": "|".join(codes),
                }
            )
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(DATA / "support_diagnostics.csv", index=False, encoding="utf-8-sig")
    diag.assign(code=diag.codes.str.split("|")).explode("code").groupby(
        ["method", "code"]
    ).size().rename("cells").reset_index().to_csv(
        DATA / "support_counts.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary).to_csv(DATA / "summary_counts.csv", index=False)
    c1.jsave(
        AUDIT / "qa_summary.json",
        {
            "source_hashes_unchanged": unchanged,
            "frozen_input_resolution": resolutions,
            "code_hash_unchanged": code_ok,
            "summary": summary,
            "status": "PASS" if code_ok else "CODE_CHANGED_AFTER_FREEZE",
        },
    )
    c1.log("COMBINED " + json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--lens", choices=LENSES)
    p.add_argument("--combine", action="store_true")
    a = p.parse_args()
    with threadpool_limits(limits=4):
        if a.freeze:
            freeze()
        elif a.lens:
            run_lens(a.lens)
        elif a.combine:
            combine()
        else:
            p.error("Select --freeze, --lens or --combine")
