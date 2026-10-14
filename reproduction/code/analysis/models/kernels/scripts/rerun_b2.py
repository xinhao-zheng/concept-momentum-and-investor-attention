import os, sys, json, types, time, importlib.util, shutil
from pathlib import Path
from source_bindings import replace_nodes, bind_threads

ROOT = Path(__file__).resolve().parents[1]
SRC = Path("__MODELS__/runtime_source")
CONFIG = (
    json.loads(
        Path(sys.argv[sys.argv.index("--config") + 1]).read_text(encoding="utf8")
    )
    if "--config" in sys.argv
    else {}
)
THREADS = int(CONFIG.get("threads", 4 if "--threads4" in sys.argv else 2))
SUFFIX = CONFIG.get("suffix", "_threads4" if THREADS == 4 else "")
for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[k] = str(THREADS)
os.environ["TEMP"] = os.environ["TMP"] = os.environ["JOBLIB_TEMP_FOLDER"] = str(
    ROOT / "tmp"
)
sys.dont_write_bytecode = True
sys.path.insert(0, str(SRC / "feature_increment/vendor"))
sys.path.insert(0, str(SRC / "config"))
import numpy as np, pandas as pd, polars as pl, pyarrow, sklearn, scipy

env = {
    m.__name__: {"version": m.__version__, "path": m.__file__}
    for m in [np, pd, pl, pyarrow, sklearn, scipy]
}
env.update(
    python=sys.version,
    executable=sys.executable,
    threads=THREADS,
    started=pd.Timestamp.now(tz="UTC").isoformat(),
)
(ROOT / f"environment{SUFFIX}.json").write_text(
    json.dumps(env, ensure_ascii=False, indent=2), encoding="utf8"
)


def load_b2(name, out, dense=False):
    src = SRC / "feature_increment/b2_model.py"
    code = bind_threads(src.read_text(encoding="utf8"), THREADS)
    if dense:
        code = replace_nodes(code, "pl.col('idx') % 20", "pl.col('idx') % 1")
    code = replace_nodes(
        code,
        "path = SOURCE / f'{code}.csv'",
        "path = overlay_path(SOURCE / f'{code}.csv')",
    )
    code = replace_nodes(
        code,
        "payload = p.read_bytes()",
        "p = overlay_path(p)\npayload = p.read_bytes()",
    )
    code = replace_nodes(
        code,
        "pricefile = pq.ParquetFile(OLD / 'prices_latest.parquet')",
        "price_full = pl.read_parquet(OLD / 'prices_latest.parquet', columns=['date', 'code', 'close_adj', 'ret', 'amount'])\n"
        "complete_codes = price_full['code'].unique().sort().to_list()\n"
        "batches = [complete_codes[i:i+128] for i in range(0, len(complete_codes), 128)]",
    )
    code = replace_nodes(code, "range(pricefile.num_row_groups)", "range(len(batches))")
    code = replace_nodes(
        code,
        "p = pricefile.read_row_group(rg, columns=['date', 'code', 'close_adj', 'ret', 'amount']).to_pandas()",
        "p = price_full.filter(pl.col('code').is_in(batches[rg])).to_pandas()",
    )
    code = replace_nodes(code, "pricefile.num_row_groups", "len(batches)")
    code += (
        "\n\nOVERLAY = Path("
        + repr(CONFIG["overlay"])
        + ")\ndef overlay_path(p):\n    q=OVERLAY/p.parent.name/p.name\n    return q if q.exists() else p\n"
    )

    adapted = (
        ROOT
        / "scripts"
        / f'{name}_adapted{SUFFIX}{("_t"+str(THREADS)) if CONFIG else ""}.py'
    )
    adapted.write_text(code, encoding="utf8")
    m = types.ModuleType(name)
    m.__file__ = str(adapted)
    sys.modules[name] = m
    exec(compile(code, str(adapted), "exec"), m.__dict__)
    m.ROOT = out
    m.DATA = out / "data"
    m.AUDIT = out / "audit"
    m.CACHE = out / "cache"
    if CONFIG.get("input_cache"):
        m.OLD = Path(CONFIG["input_cache"])
    for d in [m.DATA, m.AUDIT, m.CACHE]:
        d.mkdir(parents=True, exist_ok=True)
    return m


m = load_b2("b2_model", ROOT / ("B2" + SUFFIX))
if CONFIG.get("feature_cache"):
    m.CACHE = Path(CONFIG["feature_cache"])
    for n in ["feature_build.json", "feature_freeze.json"]:
        shutil.copy2(Path(CONFIG["feature_audit"]) / n, m.AUDIT / n)
if SUFFIX and not CONFIG:
    m.CACHE = ROOT / "B2/cache"
    for n in ["feature_build.json", "feature_freeze.json"]:
        shutil.copy2(ROOT / "B2/audit" / n, m.AUDIT / n)
spec = m.freeze()
m.industry_amendment()
if CONFIG:
    m.save_json(
        m.AUDIT / f"audit_branch_input_configuration_threads{THREADS}.json", CONFIG
    )
if not (m.AUDIT / "feature_freeze.json").exists():
    m.prepare()
if not (m.CACHE / "industry_labels.parquet").exists():
    m.prepare_industry()
if "--prepare-only" in sys.argv:
    print("CORRECTED_FEATURE_PREPARATION_COMPLETE", flush=True)
    sys.exit(0)
m.run()
m.infer()
bs = types.ModuleType("block_sensitivity")
bs.__file__ = str(SRC / "feature_increment/b2_block_sensitivity.py")
exec(
    compile(Path(bs.__file__).read_text(encoding="utf8"), bs.__file__, "exec"),
    bs.__dict__,
)
bs.main()
print("B2_FULL_RETRAIN_COMPLETE", flush=True)
d = load_b2("b2_dense", ROOT / ("B2D" + SUFFIX), True)
d.CACHE = m.CACHE
d.HORIZONS = [3]
d.freeze = lambda: spec
d.industry_amendment = lambda: None
for n in ["feature_build.json", "feature_freeze.json"]:
    shutil.copy2(m.AUDIT / n, d.AUDIT / n)
d.run(only_h=3)
daily = pd.concat(
    [pd.read_parquet(p) for p in sorted(d.DATA.glob("daily_H3_*.parquet"))],
    ignore_index=True,
)
ic = pd.concat(
    [pd.read_parquet(p) for p in sorted(d.DATA.glob("ic_H3_*.parquet"))],
    ignore_index=True,
)
cond = pd.concat(
    [pd.read_parquet(p) for p in sorted(d.DATA.glob("industry_ic_H3_*.parquet"))],
    ignore_index=True,
)
arrays = []
names = []
rows = []
for model in ("ridge", "hist"):
    for metric, frame in [("raw_IC", ic), ("industry_IC", cond)]:
        piv = frame.pivot(index="date", columns="method", values="IC").sort_index()
        arrays.append((piv[model + "_concept"] - piv[model + "_ordinary"]).to_numpy())
        names.append((model, metric, 0, "complete", piv.index))
    for n in (20, 100):
        for missing in ("complete", "zero"):
            piv = (
                daily[daily.top_n.eq(n)]
                .pivot(index="date", columns="method", values="portfolio_" + missing)
                .sort_index()
            )
            arrays.append(
                (piv[model + "_concept"] - piv[model + "_ordinary"]).to_numpy()
            )
            names.append((model, "return", n, missing, piv.index))
x = np.column_stack(arrays)
for length in (20, 60):
    means, lower, upper, pvalues = d.block_bootstrap(
        x, length, 20260925 + length, B=10000
    )
    for i, (model, metric, n, missing, dates) in enumerate(names):
        scale = 100 if metric == "return" else 1
        rows.append(
            dict(
                model=model,
                metric=metric,
                top_n=n,
                missing=missing,
                block_length=length,
                n_dates=int(np.isfinite(x[:, i]).sum()),
                estimate=means[i] * scale,
                lower95=lower[i] * scale,
                upper95=upper[i] * scale,
                p_raw=pvalues[i],
                unit="percentage_points" if metric == "return" else "correlation",
                start=str(dates.min().date()),
                end=str(dates.max().date()),
            )
        )
res = pd.DataFrame(rows)
order = np.argsort(res.p_raw.to_numpy())
adjusted = np.maximum.accumulate(
    (len(res) - np.arange(len(res))) * res.p_raw.to_numpy()[order]
)
corrected = np.empty(len(res))
corrected[order] = np.minimum(1, adjusted)
res["p_holm24"] = corrected
res["confirmed_positive"] = (res.estimate > 0) & (res.p_holm24 < 0.05)
res["confirmed_negative"] = (res.estimate < 0) & (res.p_holm24 < 0.05)
res.to_csv(d.DATA / "increment_inference.csv", index=False)
daily.to_parquet(d.DATA / "daily_combined.parquet", index=False)
fits = [json.loads(p.read_text(encoding="utf8")) for p in d.AUDIT.glob("fit_H*.json")]
d.save_json(
    d.AUDIT / "summary.json",
    dict(
        fit_count=sum(len(f["fits"]) for f in fits),
        n_dates=len(x),
        positive_confirmations=int(res.confirmed_positive.sum()),
        negative_confirmations=int(res.confirmed_negative.sum()),
        family_size=24,
    ),
)
(ROOT / f"b2_completion{SUFFIX}.json").write_text(
    json.dumps(
        dict(
            completed=pd.Timestamp.now(tz="UTC").isoformat(),
            B2_fits=224,
            B2D_fits=28,
            threads=THREADS,
            coverage="fresh ordinary and industry features; regenerated labels; every frozen annual model refit; S2 cache dependency to be independently validated",
        ),
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf8",
)
print("B2D_FULL_RETRAIN_COMPLETE", flush=True)
