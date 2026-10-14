"""Check completed support counts, correction families and all purged folds."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent


def main():
    checks = []
    details = []
    items = [
        ("C1", "legacy_replay", "dml", 24),
        ("C1", "legacy_replay", "aipw", 32),
        ("C1", "past_calibrated_fixed80", "dml", 24),
        ("C1", "past_calibrated_fixed80", "aipw", 32),
        ("C1S", "S_window", "dml", 24),
        ("C1S", "S_window", "aipw", 32),
    ]
    for mod, mode, kind, family_n in items:
        path = R / mod / f"data/{kind}_{mode}_all.csv"
        d = pd.read_csv(path)
        complete = d.estimability_status.eq("COMPLETE")
        assert len(d) == (144 if kind == "dml" else 96)
        assert (
            d.loc[~complete, ["coefficient", "se_hac", "p_holm_family"]]
            .isna()
            .all()
            .all()
        )
        maxerr = 0.0
        for fam, g in d.groupby("multiplicity_family"):
            assert len(g) == family_n
            p = g.p_two_sided_hac.fillna(1).to_numpy(float)
            ids = np.argsort(p, kind="stable")
            adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[ids]).clip(
                0, 1
            )
            alladj = np.empty(len(p))
            alladj[ids] = adj
            c = g.estimability_status.eq("COMPLETE").to_numpy()
            if c.any():
                maxerr = max(
                    maxerr,
                    float(np.max(np.abs(g.p_holm_family.to_numpy()[c] - alladj[c]))),
                )
        assert maxerr < 1e-10
        checks.append(
            {
                "branch": mod,
                "mode": mode,
                "method": kind,
                "grid": len(d),
                "estimable": int(complete.sum()),
                "deferred": int((~complete).sum()),
                "Holm05": int((d.p_holm_family < 0.05).sum()),
                "BH05_secondary": int((d.q_bh_family < 0.05).sum()),
                "independent_Holm_max_error": maxerr,
            }
        )
        for lens, g in d.groupby("lens"):
            for h, z in g.groupby("holding_h"):
                details.append(
                    {
                        "branch": mod,
                        "mode": mode,
                        "method": kind,
                        "lens": lens,
                        "H": int(h),
                        "estimable": int(z.estimability_status.eq("COMPLETE").sum()),
                        "cells": len(z),
                    }
                )
    purges = []
    for mod in ["C1", "C1S"]:
        for lens in ["alpha", "beta", "gamma"]:
            d = pd.read_csv(R / mod / f"audit/purge_{lens}.csv")
            empty = d.train_rows_after_purge.eq(0)
            valid = pd.to_datetime(d.loc[~empty, "max_training_exit"]) < pd.to_datetime(
                d.loc[~empty, "test_start"]
            )
            assert valid.all()
            purges.append(
                {
                    "branch": mod,
                    "lens": lens,
                    "checks": len(d),
                    "nonempty_training_folds": int((~empty).sum()),
                    "empty_training_folds": int(empty.sum()),
                    "violations": int((~valid).sum()),
                }
            )
    pd.DataFrame(checks).to_csv(R / "data/S4_support_summary.csv", index=False)
    pd.DataFrame(details).to_csv(R / "data/support_by_lens_H.csv", index=False)
    (R / "audit/final_estimation_checks.json").write_text(
        json.dumps(
            {"status": "PASS", "support_summary": checks, "purge_checks": purges},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(checks).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
