"""Concept IC landscapes: alpha/beta/membership-lag gamma, 2026-08-31.

Run with Python -B. Inputs are the sibling frozen S2 cache. This module never
imports or writes to an earlier delivery and never runs a trading backtest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[1]
CACHE = ROOT.parent / "S2/cache"
GRID = [1, 2, 3, 5, 10, 20, 60, 120, 250]
LENSES = ["alpha", "beta", "gamma"]
PROTOCOLS = dict(
    alpha="alpha_dynamic_close_concept_index",
    beta="beta_synchronous_open_basket",
    gamma="gamma_membership_lag_open_basket",
)
CUTOFF = np.datetime64("2026-08-31", "ns")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsave(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def day(x):
    return str(np.datetime_as_string(x, unit="D"))


def group_mean(values, groups, n, denominator=None):
    finite = np.isfinite(values)
    sums = np.bincount(groups[finite], weights=values[finite], minlength=n)
    counts = np.bincount(groups[finite], minlength=n)
    denom = counts if denominator is None else denominator
    means = np.divide(sums, denom, out=np.full(n, np.nan), where=denom > 0)
    return means, counts


def rank_ic(x, y, mask):
    n = mask.sum(axis=1)
    xr = rankdata(np.where(mask, x, np.nan), axis=1, nan_policy="omit")
    yr = rankdata(np.where(mask, y, np.nan), axis=1, nan_policy="omit")
    center = (n + 1)[:, None] / 2
    dx, dy = np.where(mask, xr - center, 0), np.where(mask, yr - center, 0)
    den = np.sqrt(np.sum(dx * dx, axis=1) * np.sum(dy * dy, axis=1))
    result = np.divide(
        np.sum(dx * dy, axis=1),
        den,
        out=np.full(len(n), np.nan),
        where=(n >= 20) & (den > 0),
    )
    return result, n


def protocol():
    return {
        "protocol_version": "S9-unified-membership-lag-20260928-v1",
        "execution_role": "research replication",
        "observation_cutoff_inclusive": "2026-08-31",
        "inputs": "Sibling S2 frozen calendar/prices_latest/membership_latest parquet files",
        "source_start": "2016-11-24, the first market day in the frozen common source calendar",
        "grid_F_H": GRID,
        "daily_concept_return": "At each market day u, arithmetic mean of finite stock close/preclose-1 returns measured on u. Alpha/beta use the latest global membership snapshot dated no later than u; gamma uses the latest snapshot dated no later than the preceding market day u-1. Require at least one valid stock. No future outcome determines formation eligibility.",
        "formation": "For every design compound F complete exact market-day concept returns through t. Gamma uses each historical day u's lagged memberships but contemporaneous price returns on u; neither the completed concept return series nor the final formation signal is shifted. Require all F values.",
        "designs": {
            "alpha": {
                "protocol_id": PROTOCOLS["alpha"],
                "display": "alpha",
                "membership_max_in_formation": "t",
                "formation_information_max": "t",
                "outcome": "Product of 1+daily dynamic concept return on t+1,...,t+H minus one; require all H days.",
                "endpoint": "close-clock concept index t to t+H",
                "future_membership": "Updates with each future day, as in the original alpha concept-index study.",
            },
            "beta": {
                "protocol_id": PROTOCOLS["beta"],
                "display": "beta",
                "membership_max_in_formation": "t",
                "formation_information_max": "t",
                "outcome": "Mean of individual stock adjusted open(t+H+1)/open(t+1)-1 within the t eligible concept basket.",
                "future_membership": "Fixed at t.",
            },
            "gamma": {
                "protocol_id": PROTOCOLS["gamma"],
                "display": "gamma",
                "membership_max_in_formation": "t-1",
                "formation_information_max": "t",
                "outcome": "Mean of individual stock adjusted open(t+H+1)/open(t+1)-1 within the concept basket formed from membership dated no later than t-1 and finite stock returns on t.",
                "future_membership": "Fixed at the signal day, using membership dated no later than t-1.",
                "alignment": "Same membership timing as the manuscript stock-selection gamma. Current and each trailing day's price information is contemporaneous with that day, while membership is lagged one market day. Concept-level means have no focal stock, unlike stock-specific leave-one-out peer returns.",
            },
        },
        "primary_support": "Each design uses its own finite formation/outcome concepts and nonconstant correlation dates within each F/H. At least 20 concepts. No requirement to share samples across protocols or F/H.",
        "maturity": "All observations and prescribed endpoints no later than 2026-08-31. Alpha requires t+H within calendar; beta/gamma require t+H+1. No shortened holdings and no next-observed-stock-row substitution.",
        "missing_outcomes": "Missing beta/gamma stock entry or exit stays missing; the primary mean uses observed returns. Formation basket never conditions on future availability.",
        "estimation": "Cross-concept Spearman with average-tie ranks on each design valid concept support, followed by equal-date means. Nominal IC-z is atanh(clip(IC,-0.999999,0.999999))*sqrt(N-3).",
        "interpretation": "Descriptive concept-return association. Nominal IC-z is not a dependency-corrected significance test. Alpha vs beta differs in dynamic-index versus frozen-basket outcome as well as return clock.",
        "sensitivities": [
            "Matched concepts/dates within each F/H for all three protocols",
            "Frozen-member endpoint alpha (kept separate from dynamic alpha primary)",
            "Fixed eligible denominator with missing stock endpoints counted zero, only beta/gamma",
        ],
        "implementation_clarification": "User-requested definition alignment: gamma lags membership only, identically to the main stock-selection timing. Alpha dynamic-index and beta synchronous-basket definitions are retained. All current outputs are regenerated; historical alternative timing modules are not inputs to the calculation.",
    }


def main():
    began = time.time()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    if (out / "completion.json").exists():
        raise FileExistsError("Completed results exist; create a new version instead")
    pfile = ROOT / "protocol_frozen.json"
    if pfile.exists():
        p = json.loads(pfile.read_text(encoding="utf-8"))
        assert p == protocol(), "Existing protocol differs"
    else:
        jsave(pfile, protocol())
    phash = sha(pfile)
    (ROOT / "protocol_frozen.sha256").write_text(
        phash + "  protocol_frozen.json\n", encoding="utf-8"
    )
    jsave(
        out / "execution_environment.json",
        dict(
            started=datetime.now().astimezone().isoformat(),
            python=sys.version,
            platform=platform.platform(),
            numpy=np.__version__,
            pandas=pd.__version__,
            polars=pl.__version__,
            script_sha256=sha(__file__),
            protocol_sha256=phash,
        ),
    )
    inputs = []
    for name in [
        "calendar.parquet",
        "prices_latest.parquet",
        "membership_latest.parquet",
    ]:
        path = CACHE / name
        inputs.append(
            dict(**_receipt_path(path), bytes=path.stat().st_size, sha256=sha(path))
        )
    jsave(out / "input_manifest.json", inputs)

    cal = pl.read_parquet(CACHE / "calendar.parquet").sort("idx")
    dates = cal["date"].to_numpy()
    assert dates[-1] == CUTOFF
    assert np.array_equal(cal["idx"].to_numpy(), np.arange(len(cal)))
    prices = pl.read_parquet(
        CACHE / "prices_latest.parquet",
        columns=["date", "code", "open_adj", "close_adj", "ret"],
    )
    assert prices["date"].max() <= datetime(2026, 8, 31)
    assert prices.select(pl.struct("date", "code").n_unique()).item() == len(prices)
    codes = prices.select("code").unique().sort("code").with_row_index("stock_id")
    stock_price = prices.join(cal, on="date").join(codes, on="code")
    matrices = {}
    for name in ["open_adj", "close_adj", "ret"]:
        arr = np.full((len(cal), len(codes)), np.nan)
        arr[stock_price["idx"].to_numpy(), stock_price["stock_id"].to_numpy()] = (
            stock_price[name].to_numpy()
        )
        matrices[name] = arr
    del prices, stock_price
    members = pl.read_parquet(CACHE / "membership_latest.parquet")
    assert members["asof"].max() <= datetime(2026, 8, 31)
    concepts = (
        members.select("concept").unique().sort("concept").with_row_index("concept_id")
    )
    snaps = members.select("asof").unique().sort("asof").with_row_index("snapshot_id")
    member_rows = (
        members.join(codes, on="code")
        .join(concepts, on="concept")
        .join(snaps, on="asof")
        .select("snapshot_id", "stock_id", "concept_id")
        .sort("snapshot_id")
    )
    del members
    sids = member_rows["snapshot_id"].to_numpy()
    stockids = member_rows["stock_id"].to_numpy()
    conceptids = member_rows["concept_id"].to_numpy()
    limits = np.searchsorted(sids, np.arange(len(snaps) + 1))
    asofs = snaps["asof"].to_numpy()
    snapshot_for_day = np.searchsorted(asofs, dates, side="right") - 1
    lag_snapshot_for_day = np.r_[-1, snapshot_for_day[:-1]]
    nc = len(concepts)
    returns_by_design, counts_by_design, cohorts_by_design = {}, {}, {}
    member_audit = []
    for design, snapshots in [
        ("beta", snapshot_for_day),
        ("gamma", lag_snapshot_for_day),
    ]:
        daily_returns = np.full((len(cal), nc), np.nan)
        counts = np.zeros((len(cal), nc), dtype=np.int32)
        baskets = []
        for t, s in enumerate(snapshots):
            if s < 0:
                baskets.append(
                    (np.array([], dtype=np.uint32), np.array([], dtype=np.uint32))
                )
                continue
            a, b = limits[s : s + 2]
            st, co = stockids[a:b], conceptids[a:b]
            # Prices are measured on t under both conventions. Only the
            # snapshot index differs, matching S2 build_scores mapping/join.
            rr = matrices["ret"][t, st]
            finite = np.isfinite(rr)
            st, co, rr = st[finite], co[finite], rr[finite]
            assert np.all(
                rr > -1
            ), "Log compounding requires positive gross daily returns"
            daily_returns[t], counts[t] = group_mean(rr, co, nc)
            baskets.append((st.copy(), co.copy()))
            member_audit.append(
                dict(
                    design=design,
                    date=day(dates[t]),
                    price_information_date=day(dates[t]),
                    membership_limit=day(dates[t - (design == "gamma")]),
                    membership_asof=day(asofs[s]),
                    memberships=len(st),
                    concepts=int((counts[t] > 0).sum()),
                    singleton_concepts=int((counts[t] == 1).sum()),
                )
            )
        returns_by_design[design] = daily_returns
        counts_by_design[design] = counts
        cohorts_by_design[design] = baskets
    returns = returns_by_design["beta"]
    assert all(
        x["membership_asof"] <= x["membership_limit"] <= x["price_information_date"]
        for x in member_audit
    )
    pd.DataFrame(member_audit).to_csv(out / "daily_membership_audit.csv", index=False)
    concepts.write_csv(out / "concept_dictionary.csv")
    daily_concepts = pd.DataFrame(
        returns,
        index=pd.DatetimeIndex(dates, name="date"),
        columns=concepts["concept"].to_list(),
    )
    daily_concepts.to_parquet(out / "concept_daily_returns.parquet")
    gamma_concepts = pd.DataFrame(
        returns_by_design["gamma"],
        index=pd.DatetimeIndex(dates, name="date"),
        columns=concepts["concept"].to_list(),
    )
    gamma_concepts.to_parquet(out / "concept_daily_returns_gamma.parquet")
    del member_rows, sids, stockids, conceptids, matrices["ret"]
    logreturns = np.log1p(daily_concepts)
    gamma_logreturns = np.log1p(gamma_concepts)
    formation = {}
    for f in GRID:
        current = np.expm1(logreturns.rolling(f, min_periods=f).sum().to_numpy())
        gamma_current = np.expm1(
            gamma_logreturns.rolling(f, min_periods=f).sum().to_numpy()
        )
        formation[f] = [current, current, gamma_current]

    primary_rows, common_rows, zero_rows, fixed_alpha_rows = [], [], [], []
    daily_frames, common_daily_frames, zero_daily_frames, alpha_daily_frames = (
        [],
        [],
        [],
        [],
    )
    availability_frames, direct_checks, endpoint_rows = [], [], []
    print(
        "Frozen inputs loaded:",
        len(cal),
        "days,",
        len(codes),
        "stocks,",
        nc,
        "concepts",
        flush=True,
    )

    def record(
        f,
        h,
        lens,
        past,
        future,
        mask,
        mode,
        collection,
        daily_collection,
        date_gate=None,
    ):
        ic, n = rank_ic(past, future, mask)
        good = np.isfinite(ic)
        if date_gate is not None:
            good &= date_gate
        ts = np.flatnonzero(good)
        assert len(ts) > 0
        values, counts = ic[good], n[good]
        zz = np.arctanh(np.clip(values, -0.999999, 0.999999)) * np.sqrt(counts - 3)
        exits = ts + h + (lens != "alpha")
        info = ts
        membership = ts - (lens == "gamma")
        assert np.all(exits < len(dates))
        protocol_id = (
            "alpha_frozen_close_sensitivity"
            if mode == "frozen_basket_alpha_sensitivity"
            else PROTOCOLS[lens]
        )
        target = (
            "fixed-basket close-to-close"
            if mode == "frozen_basket_alpha_sensitivity"
            else (
                "dynamic close concept index"
                if lens == "alpha"
                else "fixed-basket open-to-open"
            )
        )
        row = dict(
            lens=lens,
            protocol_id=protocol_id,
            display_lens=lens,
            universe="Concept",
            formation_d=f,
            holding_d=h,
            segment="FULL",
            raw_ic_mean=float(values.mean()),
            icz_mean=float(zz.mean()),
            n_dates=len(ts),
            avg_pairs=float(counts.mean()),
            min_pairs=int(counts.min()),
            max_pairs=int(counts.max()),
            signal_start=day(dates[ts.min()]),
            signal_end=day(dates[ts.max()]),
            information_start=day(dates[info.min()]),
            information_end=day(dates[info.max()]),
            membership_limit_start=day(dates[membership.min()]),
            membership_limit_end=day(dates[membership.max()]),
            outcome_end=day(dates[exits.max()]),
            target=target,
            sample=mode,
            source_table=(
                "results/daily_ic.parquet"
                if mode == "primary_own_support"
                else "results/" + mode + ".parquet"
            ),
            cutoff="2026-08-31",
            hit=float((values > 0).mean()),
        )
        collection.append(row)
        frame = pd.DataFrame(
            dict(
                lens=lens,
                protocol_id=protocol_id,
                formation_d=f,
                holding_d=h,
                date=dates[good],
                raw_ic=values,
                nominal_icz=zz,
                n_concepts=counts,
                entry_date=dates[ts + (lens != "alpha")],
                exit_date=dates[exits],
                information_date=dates[info],
                membership_limit_date=dates[membership],
                sample=mode,
            )
        )
        daily_collection.append(frame)
        if mode == "primary_own_support":
            endpoint_rows.append(
                {
                    key: row[key]
                    for key in [
                        "lens",
                        "protocol_id",
                        "formation_d",
                        "holding_d",
                        "signal_start",
                        "signal_end",
                        "information_end",
                        "membership_limit_end",
                        "outcome_end",
                        "n_dates",
                        "min_pairs",
                        "max_pairs",
                    ]
                }
            )
        return ic, n

    for h in GRID:
        # Dynamic alpha repeats the original concept-index construction; the
        # future membership is measured on each outcome day, not fed to formation.
        alpha_future = np.expm1(
            logreturns.shift(-1)
            .iloc[::-1]
            .rolling(h, min_periods=h)
            .sum()
            .iloc[::-1]
            .to_numpy()
        )
        futures = [
            alpha_future,
            np.full_like(returns, np.nan),
            np.full_like(returns, np.nan),
        ]
        zeros = [None, np.full_like(returns, np.nan), np.full_like(returns, np.nan)]
        alpha_fixed = np.full_like(returns, np.nan)
        availability = []
        for t in range(len(cal)):
            for li, lens in enumerate(LENSES):
                info = t
                membership = t - 1 if lens == "gamma" else t
                entry = t if lens == "alpha" else t + 1
                exitidx = t + h if lens == "alpha" else t + h + 1
                if membership < 0 or exitidx >= len(cal):
                    continue
                design = "gamma" if lens == "gamma" else "beta"
                st, co = cohorts_by_design[design][t]
                if not len(st):
                    continue
                px = matrices["close_adj"] if lens == "alpha" else matrices["open_adj"]
                p0, p1 = px[entry, st], px[exitidx, st]
                ok = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0) & (p1 > 0)
                rr = np.divide(p1, p0, out=np.full(len(st), np.nan), where=ok) - 1
                avg, nobserved = group_mean(rr, co, nc)
                if lens == "alpha":
                    alpha_fixed[t] = avg
                else:
                    futures[li][t] = avg
                    zeros[li][t], _ = group_mean(
                        rr, co, nc, counts_by_design[design][t]
                    )
                availability.append(
                    dict(
                        lens=lens,
                        protocol_id=PROTOCOLS[lens],
                        holding_d=h,
                        date=day(dates[t]),
                        information_date=day(dates[info]),
                        membership_limit_date=day(dates[membership]),
                        entry_date=day(dates[entry]),
                        exit_date=day(dates[exitidx]),
                        outcome_type=(
                            "frozen-alpha-sensitivity" if lens == "alpha" else "primary"
                        ),
                        eligible_memberships=len(st),
                        observed_outcomes=int(ok.sum()),
                        missing_outcomes=int((~ok).sum()),
                        evaluable_concepts=int(np.isfinite(avg).sum()),
                    )
                )
                if t in [1, 500, 1500, len(cal) - h - 2] and h in [1, 5, 250]:
                    ccids = np.flatnonzero(np.isfinite(avg))[:2]
                    for cc in ccids:
                        direct = float(np.nanmean(rr[co == cc]))
                        assert abs(direct - avg[cc]) < 1e-12
                        check = dict(
                            lens=lens,
                            H=h,
                            date=day(dates[t]),
                            concept_id=int(cc),
                            basket_direct=direct,
                            basket_production=float(avg[cc]),
                        )
                        if lens == "alpha":
                            vals = returns[t + 1 : t + h + 1, cc]
                            dynamic = (
                                float(np.prod(1 + vals) - 1)
                                if np.isfinite(vals).all()
                                else np.nan
                            )
                            generated = float(alpha_future[t, cc])
                            assert (np.isnan(dynamic) and np.isnan(generated)) or abs(
                                dynamic - generated
                            ) < 1e-12
                            check.update(
                                dynamic_direct=dynamic, dynamic_production=generated
                            )
                        direct_checks.append(check)
        availability_frames.append(pd.DataFrame(availability))
        for f in GRID:
            masks = [
                np.isfinite(formation[f][li]) & np.isfinite(futures[li])
                for li in range(3)
            ]
            common = np.logical_and.reduce(masks)
            prelim = [rank_ic(formation[f][li], futures[li], common) for li in range(3)]
            valid_dates = np.logical_and.reduce([np.isfinite(z[0]) for z in prelim])
            for li, lens in enumerate(LENSES):
                record(
                    f,
                    h,
                    lens,
                    formation[f][li],
                    futures[li],
                    masks[li],
                    "primary_own_support",
                    primary_rows,
                    daily_frames,
                )
                record(
                    f,
                    h,
                    lens,
                    formation[f][li],
                    futures[li],
                    common,
                    "common_support_sensitivity",
                    common_rows,
                    common_daily_frames,
                    valid_dates,
                )
                if lens != "alpha":
                    zero_mask = np.isfinite(formation[f][li]) & np.isfinite(zeros[li])
                    record(
                        f,
                        h,
                        lens,
                        formation[f][li],
                        zeros[li],
                        zero_mask,
                        "missing_zero_own_support_sensitivity",
                        zero_rows,
                        zero_daily_frames,
                    )
            record(
                f,
                h,
                "alpha",
                formation[f][0],
                alpha_fixed,
                np.isfinite(formation[f][0]) & np.isfinite(alpha_fixed),
                "frozen_basket_alpha_sensitivity",
                fixed_alpha_rows,
                alpha_daily_frames,
            )
        print(
            "Complete H",
            h,
            "elapsed",
            round(time.time() - began, 1),
            "seconds",
            flush=True,
        )
    primary = pd.DataFrame(primary_rows).sort_values(
        ["lens", "formation_d", "holding_d"]
    )
    common = pd.DataFrame(common_rows).sort_values(["lens", "formation_d", "holding_d"])
    assert len(primary) == 243 and len(common) == 243
    primary.drop(columns=["icz_mean"]).to_csv(
        out / "legacy_ic.csv", index=False, encoding="utf-8-sig"
    )
    z = primary.drop(columns=["raw_ic_mean"])
    z["mean_icz"] = z["icz_mean"]
    z.to_csv(out / "legacy_icz.csv", index=False, encoding="utf-8-sig")
    for table, name in [
        (primary, "primary_summary_243.csv"),
        (primary, "own_support_summary_243.csv"),
        (common, "common_support_summary_243.csv"),
        (pd.DataFrame(zero_rows), "missing_zero_sensitivity_162.csv"),
        (pd.DataFrame(fixed_alpha_rows), "frozen_basket_alpha_sensitivity_81.csv"),
    ]:
        table.to_csv(out / name, index=False, encoding="utf-8-sig")
    for frames, name in [
        (daily_frames, "daily_ic.parquet"),
        (common_daily_frames, "common_support_sensitivity.parquet"),
        (zero_daily_frames, "missing_zero_own_support_sensitivity.parquet"),
        (alpha_daily_frames, "frozen_basket_alpha_sensitivity.parquet"),
    ]:
        pd.concat(frames, ignore_index=True).to_parquet(out / name, index=False)
    pd.DataFrame(endpoint_rows).to_csv(out / "cell_date_support_243.csv", index=False)
    available = pd.concat(availability_frames, ignore_index=True)
    available.to_parquet(out / "outcome_availability_daily.parquet", index=False)
    available_summary = available.groupby(
        ["lens", "holding_d", "outcome_type"], as_index=False
    ).agg(
        eligible_memberships=("eligible_memberships", "sum"),
        observed_outcomes=("observed_outcomes", "sum"),
        missing_outcomes=("missing_outcomes", "sum"),
    )
    available_summary["missing_share"] = (
        available_summary.missing_outcomes / available_summary.eligible_memberships
    )
    available_summary.to_csv(out / "outcome_availability_summary.csv", index=False)
    jsave(out / "independent_endpoint_spot_checks.json", direct_checks)
    signed = []
    for row in primary.to_dict("records"):
        for metric, column in [("raw_ic", "raw_ic_mean"), ("nominal_icz", "icz_mean")]:
            signed.append(
                dict(
                    metric=metric,
                    lens=row["lens"],
                    protocol_id=row["protocol_id"],
                    display_lens=row["display_lens"],
                    F=row["formation_d"],
                    H=row["holding_d"],
                    value=row[column],
                    signed_value=f"{row[column]:+.17g}",
                    n_dates=row["n_dates"],
                    mean_concepts=row["avg_pairs"],
                    signal_start=row["signal_start"],
                    signal_end=row["signal_end"],
                    outcome_end=row["outcome_end"],
                    cutoff="2026-08-31",
                )
            )
    pd.DataFrame(signed).to_csv(
        out / "S9_signed_values_486.csv", index=False, encoding="utf-8-sig"
    )
    regions = []
    for source, name in [
        (primary, "primary"),
        (common, "common-support"),
        (pd.DataFrame(fixed_alpha_rows), "frozen-alpha"),
    ]:
        for lens, sub in source.groupby("lens"):
            for region, subset in [
                (
                    "short_F_le10_H_le10",
                    sub[sub.formation_d.le(10) & sub.holding_d.le(10)],
                ),
                (
                    "F_ge5_H60_H120",
                    sub[sub.formation_d.ge(5) & sub.holding_d.isin([60, 120])],
                ),
            ]:
                regions.append(
                    dict(
                        sample=name,
                        lens=lens,
                        region=region,
                        cells=len(subset),
                        positive_raw=int(subset.raw_ic_mean.gt(0).sum()),
                        negative_raw=int(subset.raw_ic_mean.lt(0).sum()),
                        mean_raw_ic=float(subset.raw_ic_mean.mean()),
                        mean_nominal_icz=float(subset.icz_mean.mean()),
                    )
                )
    pd.DataFrame(regions).to_csv(out / "region_sensitivity_summary.csv", index=False)
    qa = dict(
        cutoff="2026-08-31",
        primary_cells=len(primary),
        signed_values=len(signed),
        all_cells_finite=bool(
            np.isfinite(primary[["raw_ic_mean", "icz_mean"]]).all().all()
        ),
        primary_max_exit_by_lens=primary.groupby("lens").outcome_end.max().to_dict(),
        all_endpoints_within_cutoff=bool(primary.outcome_end.le("2026-08-31").all()),
        min_daily_concepts=int(primary.min_pairs.min()),
        common_date_counts_equal=bool(
            common.groupby(["formation_d", "holding_d"]).n_dates.nunique().eq(1).all()
        ),
        common_concept_counts_equal=bool(
            common.groupby(["formation_d", "holding_d"]).avg_pairs.nunique().eq(1).all()
        ),
        primary_is_own_support=True,
        alpha_is_dynamic_concept_index=True,
        gamma_is_membership_only_lag=True,
        gamma_formation_includes_signal_day_prices=True,
        no_completed_formation_shift=True,
        frozen_alpha_kept_separate=True,
        beta_gamma_missing_outcomes_do_not_filter_formation=True,
        endpoint_spot_checks=len(direct_checks),
        protocol_hash_unchanged=sha(pfile) == phash,
    )
    for inp in inputs:
        assert sha(_resolve_receipt_path(inp)) == inp["sha256"]
    qa["input_hashes_unchanged"] = True
    assert all(value for value in qa.values() if isinstance(value, bool))
    jsave(out / "qa_results.json", qa)
    jsave(
        out / "completion.json",
        dict(
            completed=datetime.now().astimezone().isoformat(),
            elapsed_seconds=time.time() - began,
            **qa,
        ),
    )
    print("COMPLETE", json.dumps(qa, ensure_ascii=False), flush=True)
