import os, sys, json, types, shutil, time, importlib.util
from pathlib import Path
from source_bindings import replace_nodes, bind_threads

BASE = Path(__file__).resolve().parents[1]
ORIGINAL_ARGV = list(sys.argv)
CONFIG = (
    json.loads(
        Path(sys.argv[sys.argv.index("--config") + 1]).read_text(encoding="utf8")
    )
    if "--config" in sys.argv
    else {}
)
SRC = BASE / "runtime"
REFERENCE = Path("__SNAPSHOT__/models/runtime/attention_panel")
SOURCE = SRC / "attention_panel"
ROOT = BASE / CONFIG.get("output_name", "C2")
INPUT_S2 = Path(CONFIG.get("input_cache", str(SRC / "S2/cache")))
COMPARE_REFERENCE = bool(CONFIG.get("compare_archived_reference", True))
for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[k] = "2"
os.environ["TEMP"] = os.environ["TMP"] = os.environ["JOBLIB_TEMP_FOLDER"] = str(
    BASE / "tmp"
)
sys.dont_write_bytecode = True
for p in [
    "vendor",
    "cache/abg",
    "cache/mechanism",
    "data/abg",
    "data/mechanism",
    "audit/mechanism",
    "C1/data",
    "C1/audit",
    "C1S/data",
    "C1S/audit",
    "historical_reference/data/mechanism",
]:
    (ROOT / p).mkdir(parents=True, exist_ok=True)
if (SOURCE / "vendor/tabulate").exists():
    shutil.copytree(
        SOURCE / "vendor/tabulate", ROOT / "vendor/tabulate", dirs_exist_ok=True
    )
sys.path[:0] = [
    str(SRC / "feature_increment/vendor"),
    str(SRC / "config"),
    str(ROOT / "vendor"),
]
import numpy as np, pandas as pd, polars as pl


def js(p, x):
    Path(p).write_text(
        json.dumps(x, ensure_ascii=False, indent=2, default=str), encoding="utf8"
    )


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def stage(s):
    print(pd.Timestamp.now(tz="UTC").isoformat(), s, flush=True)


for p in (SOURCE / "vendor").glob("*.py"):
    code = p.read_text(encoding="utf8")
    # Redirect external shared inputs explicitly while all ROOT outputs stay in the audit.
    for name in [
        "attention_daily.parquet",
        "stock_pit_covariates_daily.universe.parquet",
        "stock_pit_covariates_daily.parquet",
    ]:
        replacement = (
            Path(CONFIG["attention_path"])
            if name == "attention_daily.parquet"
            else Path(CONFIG["pit_path"])
        )
        code = replace_nodes(
            code,
            'ROOT.parent / "dynamics/shared/' + name + '"',
            "Path(" + repr(str(replacement)) + ")",
            required=False,
        )
    (ROOT / "vendor" / p.name).write_text(code, encoding="utf8")
if not (ROOT / "cache/abg/common_beta_gamma_scores_future_free.parquet").exists():
    # Source adapter's string rewrite is non-idempotent on its already-vendored CQ path.
    # Disable that code-only migration; all vendor paths were explicitly rebound above.
    prep = replace_nodes(
        (SOURCE / "prepare_inputs.py").read_text(encoding="utf8"),
        "(ROOT / 'vendor').glob('*.py')",
        "[]",
    )
    m = types.ModuleType("prepare_inputs")
    m.__file__ = str(ROOT / "prepare_inputs_adapted.py")
    Path(m.__file__).write_text(prep, encoding="utf8")
    exec(compile(prep, m.__file__, "exec"), m.__dict__)
    m.ROOT = ROOT
    m.EXP = SRC
    m.S2 = INPUT_S2
    m.CACHE = ROOT / "cache/abg"
    m.main()
if CONFIG:
    js(ROOT / "audit/price_correction_input_configuration.json", CONFIG)
stage("fresh sample")
if not (
    ROOT / "cache/mechanism/mechanism_sample_keys_before_outcomes.parquet"
).exists():
    m = load("freeze_mechanism_sample", ROOT / "vendor/freeze_mechanism_sample.py")
    sys.argv = [str(__file__)]
    m.main()
    if (
        COMPARE_REFERENCE
        and (
            REFERENCE / "cache/mechanism/mechanism_sample_keys_before_outcomes.parquet"
        ).exists()
    ):
        key = ["signal_date", "stock_code"]
        a = pl.read_parquet(m.SAMPLE)
        b = pl.read_parquet(
            REFERENCE / "cache/mechanism/mechanism_sample_keys_before_outcomes.parquet"
        )
        q = dict(
            new_rows=a.height,
            source_rows=b.height,
            new_dates=a["signal_date"].n_unique(),
            new_only=a.select(key).join(b.select(key), on=key, how="anti").height,
            source_only=b.select(key).join(a.select(key), on=key, how="anti").height,
        )
        js(ROOT / "audit/sample_key_comparison.json", q)
        assert (
            a.select(key).is_duplicated().sum() == 0
        ), "Duplicate fresh outcome-free sample keys"
    else:
        js(
            ROOT / "audit/sample_key_comparison.json",
            dict(status="SKIPPED_ARCHIVED_REFERENCE_UNAVAILABLE_OR_DISABLED"),
        )
for des in ["beta", "gamma"]:
    stage("fresh winner " + des)
    if not (ROOT / f"cache/mechanism/winner_{des}_F5_W10.parquet").exists():
        m = load("build_protocol_winner", ROOT / "vendor/build_protocol_winner.py")
        sys.argv = [str(__file__), "--design", des]
        m.main()
js(
    ROOT / "audit/full_stock_scores_ready.json",
    dict(
        completed=pd.Timestamp.now(tz="UTC").isoformat(),
        beta=str(ROOT / "cache/mechanism/full_stock_scores_beta.parquet"),
        gamma=str(ROOT / "cache/mechanism/full_stock_scores_gamma.parquet"),
    ),
)
stage("numpy score kernel and current-score controls")
if not (ROOT / "audit/independent_score_kernel.json").exists():
    m = load("validate_score_kernel", SOURCE / "validate_score_kernel.py")
    m.R = ROOT
    m.O = REFERENCE if COMPARE_REFERENCE else ROOT / "unused_archived_reference"
    m.C = ROOT / "cache/mechanism"
    m.main()
for des in ["beta", "gamma"]:
    stage("fresh primary network " + des)
    target = (
        ROOT / f"cache/mechanism/mechanism_features_{des}_primary_preoutcome.parquet"
    )
    if not target.exists():
        m = load("build_protocol_network", ROOT / "vendor/build_protocol_network.py")
        winner = m.CACHE / f"winner_{des}_F5_W10.parquet"
        cards = m.CACHE / f"protocol_membership_cards_{des}_sample_dates.parquet"
        m.build_cards(des, winner, cards)
        m.aggregate_features(des, winner, cards, target)
    js(
        ROOT / f"audit/network_scope_{des}.json",
        dict(
            scope="all primary network cards/exposures fully regenerated; weighted-median diagnostic unused by C1 not regenerated here",
            rows=pl.scan_parquet(target).select(pl.len()).collect().item(),
        ),
    )
stage("fresh exact outcomes and panel")
code = replace_nodes(
    (SOURCE / "build_panel.py").read_text(encoding="utf8"),
    "s2 = ROOT.parent / 'S2/cache'",
    "s2 = Path(" + repr(str(INPUT_S2)) + ")",
)
m = types.ModuleType("build_panel")
m.__file__ = str(ROOT / "build_panel.py")
Path(m.__file__).write_text(code, encoding="utf8")
exec(compile(code, m.__file__, "exec"), m.__dict__)
if not (ROOT / "cache/abg/abg_outcomes_cutoff.parquet").exists():
    m.outcomes()
if not (ROOT / "cache/mechanism/mechanism_analysis_panel.parquet").exists():
    at = load("attach_exact_outcomes", ROOT / "vendor/attach_exact_outcomes.py")
    old_calendar = at.calendar_wide

    def calendar_ns():
        f = old_calendar()
        for c in f.select_dtypes(include=["datetime"]).columns:
            f[c] = f[c].astype("datetime64[ns]")
        return f

    at.calendar_wide = calendar_ns
    sys.argv = [str(__file__)]
    at.main()
m.folds()
stage("panel independent full-column comparison")
if (
    COMPARE_REFERENCE
    and (REFERENCE / "cache/mechanism/mechanism_analysis_panel.parquet").exists()
):
    new = pl.read_parquet(
        ROOT / "cache/mechanism/mechanism_analysis_panel.parquet"
    ).sort(["lens", "signal_date", "stock_code"])
    old = pl.read_parquet(
        REFERENCE / "cache/mechanism/mechanism_analysis_panel.parquet"
    ).sort(["lens", "signal_date", "stock_code"])
    key = ["lens", "signal_date", "stock_code"]
    assert new.select(key).is_duplicated().sum() == 0
    assert old.select(key).is_duplicated().sum() == 0
    new_only = new.select(key).join(old.select(key), on=key, how="anti")
    old_only = old.select(key).join(new.select(key), on=key, how="anti")
    new_only.write_parquet(ROOT / "audit/new_only_panel_keys.parquet")
    old_only.write_parquet(ROOT / "audit/source_only_panel_keys.parquet")
    js(
        ROOT / "audit/panel_key_comparison.json",
        dict(
            new_rows=new.height,
            source_rows=old.height,
            new_only=new_only.height,
            source_only=old_only.height,
            explanation="Comparison with September 28 source, before the documented four-price/PIT/invalid-row corrections; differing keys are retained for inspection, not coerced.",
        ),
    )
    common = new.select(key).join(old.select(key), on=key, how="inner").sort(key)
    new = common.join(new, on=key, how="left").sort(key)
    old = common.join(old, on=key, how="left").sort(key)
    assert new.select(key).equals(old.select(key))
    rows = []
    for c in new.columns:
        if c not in old.columns:
            rows.append(dict(column=c, status="new_only"))
            continue
        a, b = new[c], old[c]
        r = dict(
            column=c,
            dtype=str(a.dtype),
            equal=a.equals(b),
            rows=len(a),
            null_mismatch=int((a.is_null() != b.is_null()).sum()),
        )
        if a.dtype.is_numeric():
            delta = (a.cast(pl.Float64) - b.cast(pl.Float64)).abs()
            r.update(max_abs=delta.max(), different_numeric=int((delta > 0).sum()))
        else:
            r.update(different_nonnull=int((a != b).sum()))
        rows.append(r)
    pd.DataFrame(rows).to_csv(
        ROOT / "audit/full_panel_column_differences.csv", index=False
    )
    del new, old
else:
    js(
        ROOT / "audit/panel_key_comparison.json",
        dict(status="SKIPPED_ARCHIVED_REFERENCE_UNAVAILABLE_OR_DISABLED"),
    )
stage("all C1 and C1S models")
# Historical diagnostics compare current audit with the frozen source, never an unrelated older archive.
for kind in ["dml", "aipw"]:
    for lens in ["alpha", "beta", "gamma"]:
        ref = REFERENCE / f"C1/data/{kind}_{lens}_legacy_replay.csv"
        if COMPARE_REFERENCE and ref.exists():
            shutil.copy2(
                ref,
                ROOT
                / f"historical_reference/data/mechanism/mechanism_tree_{kind}_{lens}.csv",
            )
    ref = REFERENCE / f"C1/data/{kind}_legacy_replay_all.csv"
    if COMPARE_REFERENCE and ref.exists():
        shutil.copy2(
            ref, ROOT / f"historical_reference/data/mechanism/mechanism_tree_{kind}.csv"
        )
import cutoff as CFG

CFG.MECHANISM_PROJECT = ROOT / "historical_reference"
code = bind_threads((SOURCE / "C1/c1_temporal_audit.py").read_text(encoding="utf8"), 2)
(ROOT / "C1/c1_temporal_audit.py").write_text(code, encoding="utf8")
if "--inputs-only" in ORIGINAL_ARGV:
    js(
        ROOT / "audit/input_driver_complete.json",
        dict(completed=pd.Timestamp.now(tz="UTC").isoformat(), configuration=CONFIG),
    )
    stage("C2_INPUTS_COMPLETE")
    sys.exit(0)
c1 = load("c1_temporal_audit", ROOT / "C1/c1_temporal_audit.py")
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=2):
    if not (c1.AUDIT / "frozen_design.json").exists():
        c1.freeze()
    for lens in c1.LENSES:
        if not (c1.AUDIT / f"complete_{lens}.json").exists():
            c1.run_lens(lens)
    c1.combine()
    support = load("build_support_table", SOURCE / "C1/build_support_table.py")
    support.HERE = ROOT / "C1"
    support.DATA = support.HERE / "data"
    support.AUDIT = support.HERE / "audit"
    support.main()
    code = bind_threads(
        (SOURCE / "C1S/c1_s_window_branch.py").read_text(encoding="utf8"), 2
    )
    (ROOT / "C1S/c1_s_window_branch.py").write_text(code, encoding="utf8")
    c1s = load("c1_s_window_branch", ROOT / "C1S/c1_s_window_branch.py")
    if not (c1s.AUDIT / "frozen_design.json").exists():
        c1s.freeze()
    for lens in c1s.LENSES:
        if not (c1s.AUDIT / f"complete_{lens}.json").exists():
            c1s.run_lens(lens)
    c1s.combine()
if not CONFIG:
    at = load(
        "attention_window_contemporaneous",
        SOURCE / "attention_window_contemporaneous.py",
    )
    at.HERE = ROOT
    at.EXP = SRC
    at.main()
js(
    BASE / CONFIG.get("completion_file", "c2_completion.json"),
    dict(
        completed=pd.Timestamp.now(tz="UTC").isoformat(),
        scope="sample, full cards/winners, primary network, exact outcomes, folds, full C1/C1S nuisance refits",
        configuration=CONFIG,
    ),
)
stage("C2_ALL_COMPLETE")
