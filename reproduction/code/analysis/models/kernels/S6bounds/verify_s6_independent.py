"""Independent NumPy bound algebra and pandas stock-weight spot reconstruction."""

from pathlib import Path
import os, json, hashlib, datetime

os.environ["POLARS_MAX_THREADS"] = "2"
import numpy as np, pandas as pd, polars as pl

O = Path(__file__).resolve().parent
C = Path("__CORE_CACHE__")
N = O / "inputs"
F = Path("__MODELS__/B2/cache")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8388608), b""):
            h.update(b)
    return h.hexdigest()


d = pd.read_csv(O / "shared_weight_daily_components.csv", parse_dates=["date"])
b = pd.read_csv(O / "missing_bounds_independent.csv")
m = pd.read_csv(N / "daily_matched_independent.csv", parse_dates=["date"])
paired = set(
    m.loc[
        (m.N == 20) & (m.benchmark == "industry_x_size") & np.isfinite(m.delta), "date"
    ]
)
assert len(paired) == 2346
errs = []
for row in b.itertuples():
    q = d[d.benchmark == row.benchmark]
    if row.date_scope == "paired_complete_2346":
        q = q[q.date.isin(paired)]
    assert len(q) == row.dates
    # Construct both endpoint choices explicitly, independently of vectorized source expressions.
    parts = q[
        ["known_sum", "missing_positive_weight", "missing_negative_weight"]
    ].to_numpy()
    extreme = (
        np.array(
            [
                [
                    k + wp * (-1.0) + wn * row.upper_return_support,
                    k + wp * row.upper_return_support + wn * (-1.0),
                ]
                for k, wp, wn in parts
            ]
        ).mean(axis=0)
        * 100
    )
    errs.append(
        float(np.max(np.abs(extreme - [row.lower_spread_pp, row.upper_spread_pp])))
    )
assert max(errs) < 1e-11
dates = pd.to_datetime(
    [
        "2016-12-22",
        "2017-04-07",
        "2017-04-10",
        "2018-01-23",
        "2020-03-16",
        "2023-06-01",
        "2026-08-25",
    ]
)
dt = [x.to_pydatetime() for x in dates]


def load(p, cols, filterdates=True):
    q = pl.scan_parquet(p).select(cols)
    if filterdates:
        q = q.filter(
            pl.col("date")
            .dt.strftime("%Y-%m-%d")
            .is_in([x.date().isoformat() for x in dt])
        )
    return q.collect().to_pandas()


cal = pd.read_parquet(C / "calendar.parquet")
cal = cal[cal.date <= pd.Timestamp("2026-08-31")]
s = load(C / "frozen_common_scores.parquet", ["date", "code"])
f = load(F / "feature_panel.parquet", ["date", "code", "float_cap"])
i = load(F / "industry_labels.parquet", ["date", "code", "industry"])
t = load(N / "all_top20_frozen.parquet", ["date", "code", "method"])
t = t[t.method == "FW"]
required = cal.loc[
    cal.idx.isin(
        [
            int(cal.loc[cal.date == x, "idx"].iloc[0]) + off
            for x in dates
            for off in (1, 4)
        ]
    ),
    "date",
].to_list()
p = (
    pl.scan_parquet(C / "prices_latest.parquet")
    .select("date", "code", "open_adj")
    .filter(
        pl.col("date")
        .dt.strftime("%Y-%m-%d")
        .is_in([x.date().isoformat() for x in required])
    )
    .collect()
    .to_pandas()
    .set_index(["date", "code"])["open_adj"]
)
spot = []
for date in dates:
    q = (
        s[s.date == date]
        .merge(f[f.date == date], on=["date", "code"], validate="one_to_one")
        .merge(i[i.date == date], on=["date", "code"], validate="one_to_one")
    )
    q = q.sort_values(["float_cap", "code"]).reset_index(drop=True)
    q["decile"] = np.arange(len(q)) * 10 // len(q) + 1
    q["selected"] = q.code.isin(t[t.date == date].code).astype(int)
    assert q.selected.sum() == 20, (
        str(date),
        len(q),
        int(q.selected.sum()),
        len(t[t.date == date]),
    )
    idx = int(cal.loc[cal.date == date, "idx"].iloc[0])
    entry = cal.loc[cal.idx == idx + 1, "date"].iloc[0]
    end = cal.loc[cal.idx == idx + 4, "date"].iloc[0]
    en = p.reindex(pd.MultiIndex.from_arrays([[entry] * len(q), q.code])).to_numpy()
    ex = p.reindex(pd.MultiIndex.from_arrays([[end] * len(q), q.code])).to_numpy()
    good = np.isfinite(en) & np.isfinite(ex) & (en > 0) & (ex > 0)
    y = np.full(len(q), np.nan)
    y[good] = ex[good] / en[good] - 1
    for bench in ["pool", "industry_x_size"]:
        if bench == "pool":
            weights = q.selected.to_numpy() / 20.0 - 1.0 / len(q)
        else:
            z = q.groupby(["industry", "decile"])["selected"]
            weights = (
                q.selected.to_numpy() / 20.0
                - (z.transform("sum") / (20 * z.transform("size"))).to_numpy()
            )
        known = np.nansum(weights * y)
        wp = weights[(~good) & (weights > 0)].sum()
        wn = weights[(~good) & (weights < 0)].sum()
        ref = d[(d.date == date) & (d.benchmark == bench)].iloc[0]
        dif = float(
            max(
                abs(
                    np.array([known, wp, wn])
                    - ref[
                        [
                            "known_sum",
                            "missing_positive_weight",
                            "missing_negative_weight",
                        ]
                    ].to_numpy(dtype=float)
                )
            )
        )
        assert abs(weights.sum()) < 1e-12 and dif < 1e-12
        spot.append(
            {
                "date": date.date().isoformat(),
                "benchmark": bench,
                "stocks": len(q),
                "missing": int((~good).sum()),
                "last_outcome": end.date().isoformat(),
                "max_component_error": dif,
            }
        )
removed = sorted(set(d.loc[d.benchmark == "industry_x_size", "date"]) - paired)
assert [x.date().isoformat() for x in removed] == [
    "2017-04-07",
    "2017-04-10",
    "2018-01-23",
]
receipt = {
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Actual full 8,814,334-stock-day reconstruction plus independent NumPy bounds on all dates and pandas weight/price reconstruction on seven predetermined dates. Does not certify missing historical securities or publication vintages.",
    "full_rows": 8814334,
    "pool_dates": 2349,
    "joint_current_dates": 2346,
    "joint_additional_sensitivity_dates": 2349,
    "excluded_joint_dates": [x.date().isoformat() for x in removed],
    "algebra_max_abs_pp": max(errs),
    "spot_checks": spot,
    "last_outcome_cutoff": "2026-08-31",
    "current_joint_bounds": b[
        (b.benchmark == "industry_x_size") & (b.date_scope == "paired_complete_2346")
    ].to_dict("records"),
    "verified_files": {
        p.name: sha(p)
        for p in [
            O / "missing_outcome_bounds.py",
            Path(__file__),
            O / "missing_bounds_calculation_sealed.json",
            O / "missing_bounds_independent.csv",
            O / "shared_weight_daily_components.csv",
            O / "joint_excluded_complete_dates.csv",
        ]
    },
}
(O / "FINAL_S6_SUPPORT_ACCEPTANCE.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        {
            "status": receipt["status"],
            "algebra_max_abs_pp": max(errs),
            "spot_checks": len(spot),
            "joint_dates": 2346,
        }
    )
)
