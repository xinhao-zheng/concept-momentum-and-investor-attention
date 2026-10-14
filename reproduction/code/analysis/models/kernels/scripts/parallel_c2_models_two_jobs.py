"""Lens-disjoint C1/C1S jobs; each estimator retains its original two-thread setting.
Run only after the isolated C2 input driver has completely frozen its panel/folds.
No output files are shared by workers. Combining is strictly after all three finish.
"""

import os, sys, json, subprocess, importlib.util, time
from pathlib import Path
from source_bindings import bind_threads

BASE = Path(__file__).resolve().parents[1]
R = BASE / os.environ.get("CQ_C2_RUN_NAME", "C2")
SRC = BASE / "runtime"
SOURCE = SRC / "attention_panel"
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
sys.path[:0] = [
    str(SRC / "feature_increment/vendor"),
    str(SRC / "config"),
    str(R / "vendor"),
    str(R / "C1"),
]
import pandas as pd
import cutoff as CFG

CFG.MECHANISM_PROJECT = R / "historical_reference"


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


c1 = load("c1_temporal_audit", R / "C1/c1_temporal_audit.py")


def make_s():
    path = R / "C1S/c1_s_window_branch.py"
    if not path.exists():
        path.write_text(
            bind_threads(
                (SOURCE / "C1S/c1_s_window_branch.py").read_text(encoding="utf8"), 2
            ),
            encoding="utf8",
        )
    return load("c1_s_window_branch", path)


if "--worker" in sys.argv:
    stage = sys.argv[-2]
    lens = sys.argv[-1]
    m = c1 if stage == "C1" else make_s()
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=2):
        if not (m.AUDIT / f"complete_{lens}.json").exists():
            m.run_lens(lens)
    sys.exit(0)
assert (R / "audit/input_driver_complete.json").exists(), "Input driver not complete"
manifest = []
for stage in ["C1", "C1S"]:
    m = c1 if stage == "C1" else make_s()
    if not (m.AUDIT / "frozen_design.json").exists():
        m.freeze()
    jobs = []
    for lens in ["alpha", "beta", "gamma"]:
        if (m.AUDIT / f"complete_{lens}.json").exists():
            continue
        log = BASE / f"logs/{R.name}_{stage}_{lens}.log"
        err = BASE / f"logs/{R.name}_{stage}_{lens}_error.log"
        a = log.open("w", encoding="utf8")
        b = err.open("w", encoding="utf8")
        while sum(job[0].poll() is None for job in jobs) >= 2:
            time.sleep(0.5)
        p = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                stage,
                lens,
            ],
            stdout=a,
            stderr=b,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        jobs.append((p, a, b, lens))
        manifest.append(
            dict(
                stage=stage,
                lens=lens,
                pid=p.pid,
                threads=2,
                started=pd.Timestamp.now(tz="UTC").isoformat(),
                stdout=str(log),
                stderr=str(err),
            )
        )
    (R / "audit/parallel_model_execution.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8"
    )
    for p, a, b, lens in jobs:
        rc = p.wait()
        a.close()
        b.close()
        assert rc == 0, (stage, lens, rc)
    m.combine()
    if stage == "C1":
        support = load("support", SOURCE / "C1/build_support_table.py")
        support.HERE = R / "C1"
        support.DATA = support.HERE / "data"
        support.AUDIT = support.HERE / "audit"
        support.main()
    print(pd.Timestamp.now(tz="UTC"), stage, "PARALLEL_ALL_LENSES_COMPLETE", flush=True)
(BASE / (R.name + "_completion.json")).write_text(
    json.dumps(
        dict(
            completed=pd.Timestamp.now(tz="UTC").isoformat(),
            scope="fresh recovered inputs, outcome-free sample, winners, network, exact labels, time folds and full C1/C1S nuisance model refits",
            execution="three lens-disjoint workers per stage; each two threads; C1 then C1S; combined only after workers finish",
            configuration=json.loads(
                (BASE / "scripts" / ("config_" + R.name + ".json")).read_text(
                    encoding="utf8"
                )
            ),
        ),
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf8",
)
print("C2_ALL_MODELS_COMPLETE", flush=True)
