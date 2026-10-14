"""Fresh-card D1/B1/B3 rebuild and new-score economic sensitivity.
This deliberately retains new floating-point results and never substitutes
stored authoritative score doubles to recover legacy rankings.
"""

import os
from pathlib import Path
from runtime_config import load_config

CONFIG = load_config()
ROOT = CONFIG.output("core")
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
import sys, time, json, shutil, gc
import numpy as np, pandas as pd, polars as pl
import score_engine as IS
from exact_rank_util import add_aggregations, ranks as exact_ranks
import replay_frozen_estimands as RP

EXP = RP.EXP
FRESH = ROOT / "cache"
NEW = ROOT / "replay"
NEW.mkdir(exist_ok=True)


def js(p, x):
    p.write_text(
        json.dumps(x, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def ranks(x, cols, extra):
    return exact_ranks(x, cols, extra)


def compare_new(k):
    oldrep = RP.REP
    RP.REP = NEW
    existing = RP.ROOT / f"{k}_numeric_comparison.csv"
    saved_bytes = existing.read_bytes() if existing.exists() else None
    rows = RP.compare(k)
    os.replace(existing, ROOT / f"{k}_new_score_numeric_comparison.csv")
    if saved_bytes is not None:
        existing.write_bytes(saved_bytes)
    RP.REP = oldrep
    return rows


def s2(dest):
    mod = RP.load("S2", "s2_prepare.py", {"ROOT": dest}, name="s2_prepare")
    ev = RP.load("S2", "s2_evaluate.py", {"ROOT": dest})
    ev.evaluate_daily()
    ev.inference()
    compare_new("S2")


def d1():
    dest = NEW / RP.NAMES["D1"]
    (dest / "cache").mkdir(parents=True, exist_ok=True)
    common = pl.read_parquet(
        FRESH / "frozen_common_scores.parquet", columns=["date", "code"]
    )
    allr = []
    checks = []
    for f, w in [(3, 5), (3, 20), (5, 10), (10, 20), (20, 10), (20, 50)]:
        name = f"gamma_calendar_F{f}_W{w}_scores.parquet"
        if (f, w) == (5, 10):
            path = FRESH / "gamma_calendar_scores.parquet"
        else:
            path = FRESH / name
            if not path.exists():
                IS.score("gamma", "calendar", f, w, name)
        x = add_aggregations(pl.read_parquet(path))
        x.write_parquet(dest / f"cache/F{f}_W{w}_scores.parquet")
        if (f, w) == (5, 10):
            x.write_parquet(dest / "cache/F5_W10_authoritative_scores.parquet")
        checks.append(
            dict(
                F=f,
                W=w,
                fresh_rows=len(x),
                comparison="publication CSV compared after evaluation",
            )
        )
        x = x.join(common, on=["date", "code"]).filter(
            pl.all_horizontal(pl.col(["FW", "F_only", "W_only", "A", "P"]).is_finite())
        )
        counts = x.group_by("date").len()
        x = x.join(counts.filter(pl.col("len") >= 100).select("date"), on="date")
        x.select("date", "code").write_parquet(dest / f"cache/F{f}_W{w}_pool.parquet")
        allr.append(
            ranks(
                x,
                {m: m for m in ["FW", "F_only", "W_only", "A", "P"]},
                {"F": f, "W": w},
            )
        )
        del x
        gc.collect()
    pl.concat(allr).sort(["F", "W", "method", "date", "rank"]).write_parquet(
        dest / "frozen_top100_selections.parquet"
    )
    js(
        dest / "selection_freeze.json",
        {
            "ranks_sha256": RP.sha(dest / "frozen_top100_selections.parquet"),
            "independent_fresh_scores": True,
            "no_authoritative_double_substitution": True,
        },
    )
    pd.DataFrame(checks).to_csv(ROOT / "D1_full_score_comparison.csv", index=False)
    m = RP.load("D1", "run_parameter_bridge.py", {"HERE": dest})
    m.evaluate()
    m.stats()
    compare_new("D1")


def b1():
    dest = NEW / RP.NAMES["B1"]
    dest.mkdir(exist_ok=True)
    allr = []
    checks = []
    for d in ["beta", "gamma"]:
        pieces = []
        for bucket in range(8):
            p = (
                pl.scan_parquet(FRESH / f"{d}_past_pairs.parquet")
                .filter(pl.col("code").hash(seed=51) % 8 == bucket)
                .sort(["code", "concept", "idx"])
            )
            outs = []
            for f, w, nf, nw in [(5, 10, 5, 5), (20, 60, 15, 40)]:

                def roll(e, window, minn):
                    return e.rolling_sum_by(
                        "idx", window_size=f"{window}i", min_samples=minn
                    ).over(["code", "concept"])

                terms = {
                    "n": pl.col("ret") * 0 + 1.0,
                    "sx": pl.col("ret"),
                    "sy": pl.col("loo"),
                    "sxx": pl.col("ret") ** 2,
                    "syy": pl.col("loo") ** 2,
                    "sxy": pl.col("ret") * pl.col("loo"),
                }
                q = p.with_columns(
                    (
                        roll(
                            pl.col("loo").clip(lower_bound=-0.999).log1p(), f, nf
                        ).exp()
                        - 1
                    ).alias("Fraw"),
                    *[roll(e, w, nw).alias(n) for n, e in terms.items()],
                )
                vx = pl.col("sxx") - pl.col("sx") ** 2 / pl.col("n")
                vy = pl.col("syy") - pl.col("sy") ** 2 / pl.col("n")
                cov = pl.col("sxy") - pl.col("sx") * pl.col("sy") / pl.col("n")
                q = q.with_columns(
                    pl.when((vx > 1e-15) & (vy > 1e-15)).then(cov / vy).alias("slope")
                ).filter(pl.col("Fraw").is_finite() & pl.col("slope").is_finite())
                outs.append(
                    q.group_by(["date", "code"])
                    .agg(
                        pl.col("Fraw").mean().alias(f"CM{f}"),
                        pl.col("slope").mean().alias(f"beta{w}"),
                    )
                    .collect(engine="streaming")
                )
            pieces.append(outs[0].join(outs[1], on=["date", "code"]))
            print("B1 RAW PART", d, bucket, flush=True)
        raw = pl.concat(pieces)
        base = pl.read_parquet(
            FRESH / "frozen_common_scores.parquet",
            columns=[
                "date",
                "code",
                "mom5",
                "mom20",
                f"{d}_calendar_FW",
                f"{d}_calendar_FW_num",
                f"{d}_calendar_FW_den",
            ],
        ).rename(
            {
                f"{d}_calendar_FW": "FW",
                f"{d}_calendar_FW_num": "FW_num",
                f"{d}_calendar_FW_den": "FW_den",
            }
        )
        x = base.join(raw, on=["date", "code"]).filter(
            pl.all_horizontal(
                pl.col(
                    ["FW", "CM5", "CM20", "beta10", "beta60", "mom5", "mom20"]
                ).is_finite()
            )
        )
        counts = x.group_by("date").len()
        x = x.join(counts.filter(pl.col("len") >= 100).select("date"), on="date")
        x.write_parquet(dest / f"{d}_past_scores.parquet")
        x.select("date", "code").write_parquet(dest / f"{d}_pool.parquet")
        checks.append(
            dict(
                design=d,
                fresh_rows=len(x),
                comparison="publication CSV compared after evaluation",
            )
        )
        allr.append(
            ranks(x, {m: m for m in ["FW", "CM5", "CM20"]}, {"design": d}).drop("score")
        )
        for method, beta, mom in [
            ("BC10", "beta10", "mom5"),
            ("BC60", "beta60", "mom20"),
        ]:
            q = x.sort(
                ["date", beta, "code"], descending=[False, True, False]
            ).with_columns(
                (
                    (
                        (pl.col("code").cum_count().over("date") - 1)
                        * 3
                        / pl.len().over("date")
                    ).floor()
                ).alias("third")
            )
            q = (
                q.sort(
                    ["date", "third", mom, "code"],
                    descending=[False, False, True, False],
                )
                .with_columns(pl.col("code").cum_count().over("date").alias("rank"))
                .filter(pl.col("rank") <= 100)
                .select("date", "code", "rank")
                .with_columns(pl.lit(method).alias("method"), pl.lit(d).alias("design"))
            )
            allr.append(q)
        del x, raw, pieces
        gc.collect()
    pl.concat(allr).write_parquet(dest / "frozen_selections.parquet")
    js(
        dest / "selection_freeze.json",
        {
            "sha256": RP.sha(dest / "frozen_selections.parquet"),
            "independent_fresh_scores": True,
            "no_authoritative_double_substitution": True,
        },
    )
    pd.DataFrame(checks).to_csv(ROOT / "B1_full_score_comparison.csv", index=False)
    m = RP.load("B1", "run_benchmarks.py", {"HERE": dest})
    m.evaluate()
    m.stats()
    compare_new("B1")


def b3():
    dest = NEW / RP.NAMES["B3"]
    dest.mkdir(exist_ok=True)
    from run_core import design_note

    design_note(dest, "B3")
    m = RP.load("B3", "run_monthly_mapping.py", {"HERE": dest, "SRC": FRESH})
    m.freeze()
    m.select()
    m.evaluate()
    compare_new("B3")
