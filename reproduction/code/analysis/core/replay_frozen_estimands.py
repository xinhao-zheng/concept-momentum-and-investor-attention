"""Load research kernels with configured paths and compare optional references."""

from pathlib import Path
import os
import sys
import ast
import types
import shutil
import json
import time
import hashlib
import traceback
import gc
from runtime_config import load_config

CONFIG = load_config()
ROOT = CONFIG.output("core")
ROOT.mkdir(parents=True, exist_ok=True)
EXP = Path(__file__).resolve().parent / "kernels"
REP = ROOT / "replay"
REP.mkdir(exist_ok=True)
(ROOT / "tmp").mkdir(exist_ok=True)
os.environ.update(
    {
        k: str(CONFIG.threads)
        for k in [
            "POLARS_MAX_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ]
    }
)
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "tmp")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cutoff as CFG
import numpy as np
import pandas as pd
import polars as pl

NAMES = {k: k for k in ["S2", "D1", "D4", "AGG", "B1", "B3", "V", "T2", "T3", "S9"]}
S2 = REP / NAMES["S2"]
(S2 / "cache").mkdir(parents=True, exist_ok=True)
(S2 / "audit").mkdir(exist_ok=True)
(S2 / "data").mkdir(exist_ok=True)


def log(*a):
    print(*a, flush=True)


def dump(name, x):
    (ROOT / name).write_text(
        json.dumps(x, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def sha(p):
    return hashlib.file_digest(open(p, "rb"), "sha256").hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    assert sha(src) == sha(dst), (src, dst)


def cp(k, name):
    copy(EXP / NAMES[k] / name, REP / NAMES[k] / name)


class Bind(ast.NodeTransformer):
    def __init__(self, bindings):
        self.bindings = bindings

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in self.bindings:
                node.value = ast.Subscript(
                    value=ast.Name(id="_AUDIT_BINDINGS", ctx=ast.Load()),
                    slice=ast.Constant(t.id),
                    ctx=ast.Load(),
                )
        return node


def load(k, file, extra=None, name=None, transform=None):
    path = EXP / NAMES[k] / file
    dest = REP / NAMES[k]
    dest.mkdir(exist_ok=True)
    bindings = {"HERE": dest, "ROOT": dest}
    bindings.update(extra or {})
    source = path.read_text(encoding="utf-8-sig")
    if transform:
        source = transform(source)
        (dest / ("adapted_" + file)).write_text(source, encoding="utf-8")
    parsed = ast.parse(source, filename=str(path))
    parsed = Bind(bindings).visit(parsed)
    ast.fix_missing_locations(parsed)
    mod = types.ModuleType(name or "audit_" + k)
    mod.__file__ = str(path)
    mod._AUDIT_BINDINGS = bindings
    sys.modules[mod.__name__] = mod
    copy(path, dest / ("reference_" + file))
    # Fingerprints authenticate the executed portable kernel, not a historical path.
    exec(compile(parsed, str(path), "exec"), mod.__dict__)
    return mod


COMPARISONS = {
    "S2": [
        (
            "data/daily_portfolio_results.parquet",
            ["date", "design", "clock", "method", "top_n", "H"],
        ),
        (
            "data/paired_competition_results.csv",
            ["design", "clock", "H", "top_n", "baseline", "missing", "segment"],
        ),
        (
            "data/portfolio_summary.csv",
            ["design", "clock", "H", "top_n", "method", "segment"],
        ),
        ("data/ic_summary.csv", ["design", "clock", "H", "method"]),
    ],
    "AGG": [
        ("daily_portfolio_results.parquet", ["date", "design", "method", "top_n", "H"]),
        (
            "comparison_summary_192.csv",
            ["design", "top_n", "H", "baseline", "missing_policy"],
        ),
        (
            "primary_block_sensitivity.csv",
            ["design", "top_n", "H", "baseline", "missing_policy", "block_length"],
        ),
    ],
    "D1": [
        ("daily_returns.parquet", ["date", "F", "W", "method", "H", "N"]),
        ("daily_pool_returns.parquet", ["date", "F", "W", "H"]),
        ("comparison_1080.csv", ["F", "W", "H", "N", "baseline", "missing"]),
        ("block_sensitivity_60.csv", ["F", "W", "H", "N", "baseline", "L"]),
    ],
    "B1": [
        ("daily_returns.parquet", ["date", "design", "method", "N", "H"]),
        ("daily_pool_returns.parquet", ["date", "design", "H"]),
        ("comparison_384.csv", ["design", "H", "N", "baseline", "missing"]),
        ("primary_block_sensitivity.csv", ["L", "baseline"]),
    ],
    "B3": [
        ("monthly_cells.csv", ["month", "C", "D"]),
        ("monthly_top_returns.csv", ["month", "method", "N"]),
        ("monthly_contrasts.csv", ["month", "label", "N", "missing"]),
        ("descriptive_cells.csv", ["C", "D"]),
        ("descriptive_contrasts.csv", ["label", "N", "missing"]),
    ],
    "D4": [
        ("daily_matched_controls.csv", ["date", "N", "benchmark"]),
        ("conditional_random_replicates.csv", ["N", "benchmark", "replicate"]),
        ("matched_control_summary.csv", ["N", "H", "benchmark"]),
    ],
    "V": [
        ("results/daily_reconstructed.csv", ["date", "N", "benchmark"]),
        ("results/annual_primary_spreads.csv", ["benchmark", "year"]),
        ("results/leave_one_year_out.csv", ["benchmark", "omitted_year"]),
        ("results/nonoverlap_h3_offsets.csv", ["benchmark", "calendar_offset_mod3"]),
        ("results/fixed_period_bootstrap.csv", ["benchmark", "period", "block_length"]),
        ("results/parameter_annual.csv", ["F", "W", "baseline", "year"]),
        ("results/parameter_fixed_periods.csv", ["F", "W", "baseline", "period"]),
    ],
    "T2": [
        ("results/daily_cells_common_support.csv", ["date", "cell"]),
        (
            "results/paired_inference_Holm20.csv",
            ["contrast", "missing_convention", "block_length"],
        ),
        ("results/cell_levels.csv", ["cell", "missing_convention"]),
    ],
    "T3": [
        (
            "results/abg_daily_portfolios.parquet",
            ["date", "lens", "holding_h", "top_n"],
        ),
        (
            "results/abg_2x2_timing_decomposition.csv",
            ["contrast", "missing_convention", "holding_h", "top_n"],
        ),
        (
            "results/paired_cell_levels.csv",
            ["lens", "missing_convention", "holding_h", "top_n"],
        ),
    ],
}


def compare(k):
    reports = []
    for file, keys in COMPARISONS[k]:
        new = REP / NAMES[k] / file
        old = CONFIG.path("core/reference-results") / NAMES[k] / file
        if not new.exists() or not old.exists():
            continue
        a = pd.read_parquet(new) if new.suffix == ".parquet" else pd.read_csv(new)
        b = pd.read_parquet(old) if old.suffix == ".parquet" else pd.read_csv(old)
        for key in keys:
            if key == "date":
                a[key] = pd.to_datetime(a[key])
                b[key] = pd.to_datetime(b[key])
        j = a.merge(
            b,
            on=keys,
            suffixes=("_new", "_old"),
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        for col in a.columns:
            if col in keys or col not in b:
                continue
            if pd.api.types.is_numeric_dtype(a[col]) and not pd.api.types.is_bool_dtype(
                a[col]
            ):
                av = j[col + "_new"].to_numpy(float)
                bv = j[col + "_old"].to_numpy(float)
                finite = np.isfinite(av) & np.isfinite(bv)
                reports.append(
                    {
                        "module": k,
                        "file": file,
                        "column": col,
                        "fresh_rows": len(a),
                        "saved_rows": len(b),
                        "keys_only_fresh": int((j._merge == "left_only").sum()),
                        "keys_only_saved": int((j._merge == "right_only").sum()),
                        "null_pattern_mismatch": int(
                            (pd.isna(av) != pd.isna(bv)).sum()
                        ),
                        "max_abs_error": (
                            float(np.max(np.abs(av[finite] - bv[finite])))
                            if finite.any()
                            else None
                        ),
                        "over_1e_10": int(
                            (np.abs(av[finite] - bv[finite]) > 1e-10).sum()
                        ),
                    }
                )
            else:
                av = j[col + "_new"]
                bv = j[col + "_old"]
                bad = ~((av == bv) | (av.isna() & bv.isna()))
                reports.append(
                    {
                        "module": k,
                        "file": file,
                        "column": col,
                        "fresh_rows": len(a),
                        "saved_rows": len(b),
                        "keys_only_fresh": int((j._merge == "left_only").sum()),
                        "keys_only_saved": int((j._merge == "right_only").sum()),
                        "null_pattern_mismatch": int((av.isna() != bv.isna()).sum()),
                        "max_abs_error": None,
                        "over_1e_10": int(bad.sum()),
                        "comparison_kind": "exact_boolean_or_text",
                    }
                )
    pd.DataFrame(reports).to_csv(ROOT / f"{k}_numeric_comparison.csv", index=False)
    dump(
        f"{k}_reference_comparison_status.json",
        {
            "stage": k,
            "status": (
                "COMPARED_AVAILABLE_TABLES"
                if reports
                else "REFERENCE_TABLES_NOT_SUPPLIED"
            ),
            "compared_columns": len(reports),
            "scope": "Optional result comparison; absence of reference tables is not a numerical validation pass.",
        },
    )
    return reports
