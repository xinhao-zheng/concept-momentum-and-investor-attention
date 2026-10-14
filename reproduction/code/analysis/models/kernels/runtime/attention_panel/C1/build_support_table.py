"""Cell-level support diagnostics for the C1 original-protocol rerun (plan section C3).

One row per branch x lens x exposure/state x H with fold statuses, calibration
support, pooled clipping and every applicable failure code. Reads only this
module's outputs; writes data/support_diagnostics.csv, data/support_counts.csv
and SUMMARY.md (the frozen combine step's markdown table needs `tabulate`,
which the research environment does not provide).
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA, AUDIT = HERE / "data", HERE / "audit"
LENSES = ("alpha", "beta", "gamma")
MODES = ("legacy_replay", "past_calibrated_fixed80")
S_FIRST_FINITE = "2022-03-30"


def folds_text(ids):
    return ",".join(str(int(i)) for i in sorted(ids)) if len(ids) else ""


def main():
    rows = []
    for lens in LENSES:
        nuis = pd.read_csv(AUDIT / f"nuisance_{lens}.csv")
        cal = pd.DataFrame(
            json.loads((AUDIT / f"calibration_{lens}.json").read_text(encoding="utf-8"))
        )
        for mode in MODES:
            dml = pd.read_csv(DATA / f"dml_{lens}_{mode}.csv")
            aipw = pd.read_csv(DATA / f"aipw_{lens}_{mode}.csv")
            n = nuis[nuis["mode"] == mode]
            for r in dml.itertuples():
                c = cal[(cal["mode"] == mode) & (cal.exposure == r.exposure)].iloc[0]
                dose = n[
                    (n.module == "DML")
                    & (n.role == "dose")
                    & (n.variable == r.exposure)
                ]
                out = n[
                    (n.module == "DML") & (n.role == "outcome") & (n.H == r.holding_h)
                ]
                bad_dose = dose[dose.status != "estimated"].fold_id.tolist()
                bad_out = out[out.status != "estimated"].fold_id.tolist()
                codes = []
                if c.status != "CALIBRATED":
                    codes.append("NO_PAST_CALIBRATION")
                if bad_dose or bad_out:
                    codes.append("INSUFFICIENT_TIME_FOLDS")
                if (
                    bad_out
                    and 5 in bad_out
                    and int(out[out.fold_id == 5].test_rows.iloc[0]) < 100
                ):
                    codes.append("IMMATURE_OUTCOME")
                if "_S" in r.exposure and (bad_dose or c.status != "CALIBRATED"):
                    codes.append("SOURCE_UNAVAILABLE_BEFORE_" + S_FIRST_FINITE)
                if r.estimability_status == "COMPLETE":
                    codes = ["ESTIMABLE"]
                rows.append(
                    {
                        "branch": mode,
                        "lens": lens,
                        "method": "Tree-DML",
                        "cell": r.exposure,
                        "H": r.holding_h,
                        "calibration_status": c.status,
                        "calibration_rows": c.calibration_rows,
                        "calibration_dates": c.calibration_dates,
                        "dose_folds_failed": folds_text(bad_dose),
                        "outcome_folds_failed": folds_text(bad_out),
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
            for r in aipw.itertuples():
                f = n[
                    (n.module == "AIPW")
                    & (n.variable == r.state_contrast)
                    & (n.H == r.holding_h)
                ]
                bad = f[f.status != "estimated"].fold_id.tolist()
                codes = []
                if bad:
                    codes.append("INSUFFICIENT_TIME_FOLDS")
                    if (
                        5 in bad
                        and int(
                            f[f.fold_id == 5][["test0", "test1"]].min(axis=1).iloc[0]
                        )
                        < 1000
                    ):
                        codes.append("IMMATURE_OUTCOME")
                    if r.state_contrast in (
                        "S_high_vs_low",
                        "TSK_all_high_vs_all_low",
                    ) and any(i in bad for i in (0, 1)):
                        codes.append("SOURCE_UNAVAILABLE_BEFORE_" + S_FIRST_FINITE)
                if not (r.pooled_propensity_clipped_fraction <= 0.20):
                    codes.append("POOR_OVERLAP")
                if not r.support_gate_pass:
                    codes.append("INSUFFICIENT_SUPPORT")
                if r.estimability_status == "COMPLETE":
                    codes = ["ESTIMABLE"]
                rows.append(
                    {
                        "branch": mode,
                        "lens": lens,
                        "method": "AIPW",
                        "cell": r.state_contrast,
                        "H": r.holding_h,
                        "calibration_status": "not_applicable",
                        "calibration_rows": None,
                        "calibration_dates": None,
                        "dose_folds_failed": "",
                        "outcome_folds_failed": folds_text(bad),
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
    t = pd.DataFrame(rows)
    t.to_csv(DATA / "support_diagnostics.csv", index=False, encoding="utf-8-sig")
    t["primary_code"] = t.codes.str.split("|").str[0]
    counts = (
        t.assign(code=t.codes.str.split("|"))
        .explode("code")
        .groupby(["branch", "method", "code"])
        .size()
        .rename("cells")
        .reset_index()
    )
    counts.to_csv(DATA / "support_counts.csv", index=False, encoding="utf-8-sig")
    summary = pd.read_csv(DATA / "summary_counts.csv")

    def md(df):
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += [
            "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)
        ]
        return "\n".join(lines)

    text = [
        "# Full-window conditional estimation support",
        "",
        "Observation cutoff 2026-08-31. Outcomes are exact mature calendar endpoints; source attention and covariates are extended before fitting. See audit/frozen_design.json for regenerated fold dates and past calibration boundary.",
        "",
        md(summary),
        "",
        "Support diagnostics enumerate calibration, time-fold, group-support and overlap failures; an unavailable coefficient is not a zero effect.",
    ]
    (HERE / "SUMMARY.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
