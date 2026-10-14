"""Recompute the paired account diagnostic from released aggregate daily paths."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code/analysis"))
from runtime_config import load_config

SECTION = ROOT / "sections/section-s17-retrospective-account-simulations"


def metrics(frame, rate):
    dates = pd.to_datetime(frame["date"])
    nav = frame["nav"].to_numpy(float)
    if (
        not dates.is_unique
        or not dates.is_monotonic_increasing
        or dates.iloc[-1] != pd.Timestamp("2026-08-31")
        or abs(nav[0] - 1) >= 1e-14
        or not np.all(np.isfinite(nav) & (nav > 0))
    ):
        raise ValueError("Account date or NAV contract failed")
    returns = nav[1:] / nav[:-1] - 1
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    return {
        "final_nav": nav[-1],
        "cagr": nav[-1] ** (1 / years) - 1,
        "ann_vol": returns.std(ddof=1) * np.sqrt(252),
        "sharpe_rf0": returns.mean() / returns.std(ddof=1) * np.sqrt(252),
        "max_drawdown": (nav / np.maximum.accumulate(nav) - 1).min(),
        "mean_exposure": frame["exposure"].iloc[1:].mean(),
        "annual_fee_drag": frame["total_fees"].sum()
        / frame["total_assets"].mean()
        / years,
        "annual_two_way_turnover": frame["commission"].sum()
        / rate
        / frame["total_assets"].mean()
        / years,
        "total_fees": frame["total_fees"].sum(),
        "total_stamp_tax": frame["stamp_tax"].sum(),
        "total_commissions": frame["commission"].sum(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-directory", type=Path, default=SECTION / "data")
    args = parser.parse_args()
    config = load_config(args.config)
    data = args.data_directory.resolve()
    manifest = json.loads((data / "account_paths.json").read_text(encoding="utf-8"))
    rows = []
    for record in manifest["configurations"]:
        frames = {}
        for variant in ["baseline", "single-gate"]:
            item = record["paths"][variant]
            path = (data / item["file"]).resolve()
            if not path.is_relative_to(data):
                raise ValueError("Account path outside declared data directory")
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError("Account file hash differs: " + item["file"])
            frames[variant] = pd.read_csv(path)
        baseline, sensitivity = frames["baseline"], frames["single-gate"]
        if not baseline["date"].equals(sensitivity["date"]):
            raise ValueError("Paired account calendars differ")
        row = {
            k: record[k]
            for k in [
                "configuration_id",
                "holding_period_days",
                "F",
                "W",
                "similarity",
                "top_n",
                "timing",
            ]
        }
        row.update(
            start_date=baseline["date"].iloc[0],
            end_date=baseline["date"].iloc[-1],
            trading_days=len(baseline),
        )
        bm = metrics(baseline, manifest["commission_rate"])
        sm = metrics(sensitivity, manifest["commission_rate"])
        for key in bm:
            row["baseline_" + key] = bm[key]
            row["single_gate_removed_" + key] = sm[key]
            row["delta_" + key] = sm[key] - bm[key]
        delta = sensitivity["nav"].to_numpy() - baseline["nav"].to_numpy()
        row.update(
            nav_changed_days_at_1e_minus12=int((abs(delta) > 1e-12).sum()),
            max_absolute_nav_difference=float(abs(delta).max()),
            cagr_change_class=(
                "positive"
                if row["delta_cagr"] > 1e-12
                else "negative" if row["delta_cagr"] < -1e-12 else "within_tolerance"
            ),
            removed_gate_hits=record["removed_gate_hits"],
        )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("configuration_id").reset_index(drop=True)
    expected = pd.read_csv(data / "paired_single_gate_48.csv")
    if len(result) != 48 or list(result) != list(expected):
        raise ValueError("Expected 48 configurations and the declared metric schema")
    checks = []
    for column in result:
        if pd.api.types.is_numeric_dtype(result[column]):
            equal = bool(
                np.allclose(
                    result[column],
                    expected[column],
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )
            )
        else:
            equal = (
                result[column].astype(str).tolist()
                == expected[column].astype(str).tolist()
            )
        checks.append({"column": column, "equal_at_tolerance": equal})
    output = config.output("accounts/paired-diagnostic")
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        output / "paired_single_gate_48.csv", index=False, float_format="%.17g"
    )
    receipt = {
        "scope": "Independent aggregation of 96 released daily account paths; no account engine run.",
        "status": "PASS" if all(x["equal_at_tolerance"] for x in checks) else "FAIL",
        "accounts": 48,
        "paths": 96,
        "cutoff": config.cutoff,
        "total_removed_gate_hits": int(result.removed_gate_hits.sum()),
        "comparisons": checks,
    }
    (output / "comparison.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
