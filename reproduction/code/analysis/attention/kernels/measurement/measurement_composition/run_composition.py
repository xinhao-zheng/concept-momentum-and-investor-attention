"""Post-hoc M2 rank-reference-set decomposition; never writes to old project/source."""

from __future__ import annotations
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "config"))
import cutoff as CFG

RAW = Path("__SNAPSHOT__/raw_snapshot")
OLD = HERE.parent / "results"
CUTOFF = pd.Timestamp(CFG.OBSERVATION_CUTOFF)
FIELDS = ["总热度(万)", "自选热度(万)", "关注粘性(万)"]
METRICS = ["T", "S", "K"]
HORIZONS = [1, 3, 5, 10, 20, 60, 120, 250]
TOL = 1e-12
IC_TOL = 1e-10
RESULTS = HERE / "results"
CACHE = HERE / "cache"


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def csv(frame, name):
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")


def canonical(path):
    return str(path).replace("\\", "/").lower()


def read_source(path):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during read: {path}")
    try:
        text = raw.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = raw.decode("gb18030")
        encoding = "gb18030"
    skip = 0 if text.splitlines()[0].startswith("股票代码,") else 1
    frame = pd.read_csv(
        io.StringIO(text),
        skiprows=skip,
        usecols=["股票代码", "交易日期"] + FIELDS,
        dtype={"股票代码": "string"},
    )
    frame["交易日期"] = pd.to_datetime(frame["交易日期"], errors="raise").dt.normalize()
    codes = frame["股票代码"].dropna().unique()
    if len(codes) != 1 or frame["股票代码"].isna().any():
        raise RuntimeError(f"Code not unique/complete: {path}")
    if frame.duplicated(["股票代码", "交易日期"]).any():
        raise RuntimeError(f"Duplicate stock-date: {path}")
    for field in FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="raise").astype("float64")
    record = dict(
        file=str(path),
        sha256=sha_bytes(raw),
        bytes=len(raw),
        mtime_ns=before.st_mtime_ns,
        encoding=encoding,
        code=str(codes[0]),
        rows_source=len(frame),
        source_first_date=frame["交易日期"].min(),
        source_last_date=frame["交易日期"].max(),
        rows_after_cutoff=int((frame["交易日期"] > CUTOFF).sum()),
    )
    frame = frame[frame["交易日期"] <= CUTOFF].rename(
        columns=dict(zip(["股票代码", "交易日期"] + FIELDS, ["code", "date"] + METRICS))
    )
    frame = frame[["code", "date"] + METRICS].sort_values("date").reset_index(drop=True)
    frame["code"] = frame.code.astype(str)
    record["rows_cutoff"] = len(frame)
    return frame, record


def ingest():
    if (CACHE / "raw_tsk_cutoff.parquet").exists():
        raise FileExistsError(
            "Existing frozen cache; use --from-cache, do not overwrite new batch"
        )
    path = RAW / "stock-main-index-data/sh000001.csv"
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Calendar changed during read")
    cal = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    calendar = pd.DatetimeIndex(pd.to_datetime(cal.candle_end_time).dt.normalize())
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise RuntimeError("Calendar duplicated or unsorted")
    calendar = calendar[(calendar >= "2016-01-01") & (calendar <= CUTOFF)]
    if calendar[-1] != CUTOFF:
        raise RuntimeError("Cutoff is not last retained calendar date")
    pd.DataFrame({"date": calendar}).to_csv(CACHE / "calendar.csv", index=False)
    files = sorted((RAW / "stock-popular-concept-detail").glob("*.csv"))
    if not files:
        raise RuntimeError("No source files")
    values = np.full((len(calendar), 3, len(files)), np.nan, dtype=np.float64)
    codes = []
    manifest = []
    seen = set()
    writer = None
    excluded_nonmarket = 0
    schema = pa.schema(
        [("code", pa.string()), ("date", pa.timestamp("ns"))]
        + [(m, pa.float64()) for m in METRICS]
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for col, (frame, record) in enumerate(pool.map(read_source, files)):
                code = record["code"]
                if code in seen:
                    raise RuntimeError(f"Duplicate code across source files: {code}")
                seen.add(code)
                codes.append(code)
                manifest.append(record)
                pos = calendar.get_indexer(pd.DatetimeIndex(frame.date))
                take = pos >= 0
                excluded_nonmarket += int((~take).sum())
                values[pos[take], :, col] = frame.loc[take, METRICS].to_numpy(float)
                table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        CACHE / "raw_tsk_cutoff.parquet", schema, compression="zstd"
                    )
                writer.write_table(table)
                if (col + 1) % 500 == 0:
                    print(f"Ingested {col+1}/{len(files)} sources", flush=True)
    finally:
        if writer is not None:
            writer.close()
    prior = pd.read_csv(OLD / "input_checksums.csv")
    oldmap = {canonical(r.file): r.sha256 for r in prior.itertuples(index=False)}
    sources = pd.DataFrame(manifest)
    sources["prior_m2_sha256"] = sources.file.map(lambda p: oldmap.get(canonical(p)))
    sources["sha_match_prior_m2"] = sources.sha256 == sources.prior_m2_sha256
    csv(sources, "source_manifest_new_batch.csv")
    common = np.isfinite(values).all(axis=1)
    meta = dict(
        batch_label="observation_cutoff_2026_08_31",
        extraction_finished_utc=datetime.now(timezone.utc).isoformat(),
        source_files=len(files),
        calendar_source=str(path),
        calendar_source_sha256=sha_bytes(payload),
        calendar_days=len(calendar),
        calendar_first=str(calendar[0]),
        calendar_last=str(calendar[-1]),
        source_rows=int(sources.rows_source.sum()),
        cache_rows=int(sources.rows_cutoff.sum()),
        records_after_cutoff_excluded=int(sources.rows_after_cutoff.sum()),
        retained_records_not_in_analysis_calendar=excluded_nonmarket,
        sha_same_count=int(sources.sha_match_prior_m2.sum()),
        sha_changed_count=int((~sources.sha_match_prior_m2).sum()),
        sources_new_to_original=int(sources.prior_m2_sha256.isna().sum()),
        original_sources_absent_now=sum(
            canonical(p) not in set(sources.file.map(canonical)) for p in prior.file
        ),
        common_TSK_stockdays=int(common.sum()),
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        pandas=pd.__version__,
        design_sha256=sha_file(HERE / "DESIGN.md"),
        source_read_errors=0,
        source_duplicate_keys=0,
    )
    save_json(CACHE / "batch_metadata.json", meta)
    save_json(CACHE / "code_order.json", codes)
    return calendar, np.array(codes), values, meta


def load_cache():
    meta = json.loads((CACHE / "batch_metadata.json").read_text(encoding="utf-8"))
    calendar = pd.DatetimeIndex(
        pd.read_csv(CACHE / "calendar.csv", parse_dates=["date"]).date
    )
    codes = np.array(
        json.loads((CACHE / "code_order.json").read_text(encoding="utf-8"))
    )
    values = np.full((len(calendar), 3, len(codes)), np.nan, dtype=np.float64)
    lookup = {c: i for i, c in enumerate(codes)}
    pf = pq.ParquetFile(CACHE / "raw_tsk_cutoff.parquet")
    for g in range(pf.num_row_groups):
        frame = pf.read_row_group(g).to_pandas()
        pos = calendar.get_indexer(pd.DatetimeIndex(frame.date))
        take = pos >= 0
        for code, part in frame.loc[take].groupby("code"):
            values[
                calendar.get_indexer(pd.DatetimeIndex(part.date)), :, lookup[code]
            ] = part[METRICS].to_numpy(float)
    return calendar, codes, values, meta


def half_rank(x):
    h = rankdata(x, axis=1, method="average") * 2
    if not np.equal(h, np.rint(h)).all():
        raise AssertionError("Average rank not half integer")
    return h.astype(np.int64)


def parts(hft, hfp, hct, hcp, nt, npast, nc):
    max_possible = 4 * max(nt, npast, nc) ** 3
    if max_possible > np.iinfo(np.int64).max or max_possible > 2**53:
        raise OverflowError("Scale exceeds exact intended integer/float regime")
    a = hft * npast - hfp * nt
    b = hct - hcp
    en = a * nc - b * nt * npast
    if not np.array_equal(a * nc, b * nt * npast + en):
        raise AssertionError("Exact integer decomposition failed")
    full = a / (2.0 * nt * npast)
    common = b / (2.0 * nc)
    ref = en / (2.0 * nt * npast * nc)
    direct = (hft / (2.0 * nt) - hct / (2.0 * nc)) - (
        hfp / (2.0 * npast) - hcp / (2.0 * nc)
    )
    err = max(float(np.max(abs(full - common - ref))), float(np.max(abs(ref - direct))))
    return a, b, en, full, common, ref, err


def corr_rank(a, b):
    ra = rankdata(a, axis=1, method="average")
    rb = rankdata(b, axis=1, method="average")
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denom = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    return np.divide(
        (ra * rb).sum(axis=1), denom, out=np.full(len(a), np.nan), where=denom > 0
    )


def top_stats(a, b, codes, n):
    if len(a) < n:
        return {
            f"top{n}_{k}": np.nan
            for k in [
                "overlap_fraction",
                "jaccard",
                "full_boundary_ties",
                "common_boundary_ties",
            ]
        }
    ia = np.lexsort((codes, -a))
    ib = np.lexsort((codes, -b))
    overlap = len(set(ia[:n]) & set(ib[:n]))
    return {
        f"top{n}_overlap_fraction": overlap / n,
        f"top{n}_jaccard": overlap / (2 * n - overlap),
        f"top{n}_full_boundary_ties": int((a == a[ia[n - 1]]).sum()),
        f"top{n}_common_boundary_ties": int((b == b[ib[n - 1]]).sum()),
    }


def moments(full, common, ref, a, b, codes):
    matrix = np.vstack([full, common, ref])
    cov = np.cov(matrix, bias=True)
    names = ["full", "common", "reference"]
    row = {}
    for j, name in enumerate(names):
        row[f"mean_{name}"] = float(matrix[j].mean())
        row[f"variance_{name}"] = float(cov[j, j])
        row[f"std_{name}"] = float(np.sqrt(max(0, cov[j, j])))
    row.update(
        cov_full_common=float(cov[0, 1]),
        cov_full_reference=float(cov[0, 2]),
        cov_common_reference=float(cov[1, 2]),
        variance_identity_error=float(
            cov[0, 0] - cov[1, 1] - cov[2, 2] - 2 * cov[1, 2]
        ),
    )
    sf = np.where(abs(full) <= TOL, 0, np.sign(full))
    sc = np.where(abs(common) <= TOL, 0, np.sign(common))
    nz = (sf != 0) & (sc != 0)
    flip = sf * sc < 0
    row.update(
        n_strict_sign_flip=int(flip.sum()),
        n_both_nonzero=int(nz.sum()),
        strict_sign_flip_fraction=float(flip.mean()),
        sign_flip_given_both_nonzero=(
            float(flip.sum() / nz.sum()) if nz.any() else np.nan
        ),
        n_sign_class_mismatch=int((sf != sc).sum()),
        sign_class_mismatch_fraction=float((sf != sc).mean()),
        n_zero_status_changed=int(((sf == 0) ^ (sc == 0)).sum()),
        n_reference_abs_larger=int((abs(ref) > abs(common) + TOL).sum()),
        reference_abs_larger_fraction=float((abs(ref) > abs(common) + TOL).mean()),
    )
    for n in [20, 100]:
        row.update(top_stats(a, b, codes, n))
    return row


def synthetic_tests():
    old = np.arange(1, 101, dtype=float)[None, :]
    new = np.r_[np.arange(-99, 1, dtype=float), old.ravel()][None, :]
    hp = half_rank(old)
    ht = half_rank(new)[:, 100:]
    hc = half_rank(old)
    a, b, en, f, c, e, err = parts(ht, hp, hc, hc, 200, 100, 100)
    expected = (1 - np.arange(1, 101) / 100) / 2
    assert np.allclose(f[0], expected, atol=1e-15) and np.all(c == 0) and err < TOL
    toy1 = np.array([[1, 1, 3, 4], [4, 2, 2, 1]], float)
    toy2 = np.array([[2, 2, 1, 4], [4, 3, 3, 1]], float)
    h1 = half_rank(toy1)
    h2 = half_rank(toy2)
    a, b, en, f, c, e, err2 = parts(h1, h2, h1, h2, 4, 4, 4)
    assert np.all(e == 0) and np.array_equal(a, 4 * b) and err2 < TOL
    return dict(
        insert_low_value_stocks=True,
        same_set_reference_zero=True,
        ties_half_integer=True,
        max_identity_error=max(err, err2),
    )


def analyze(calendar, codes, values):
    valid = np.isfinite(values).all(axis=1)
    counts = valid.sum(axis=1)
    ranks = np.zeros(values.shape, dtype=np.int32)
    for t in range(len(calendar)):
        if counts[t] >= 30:
            ranks[t][:, valid[t]] = half_rank(values[t][:, valid[t]]).astype(np.int32)
    np.savez_compressed(
        CACHE / "full_pool_half_ranks.npz",
        half_ranks=ranks,
        codes=codes,
        dates_ns=calendar.asi8,
    )
    csv(pd.DataFrame({"date": calendar, "n_common_TSK": counts}), "coverage_daily.csv")
    daily = []
    ic = []
    support = []
    writer = None
    maxerr = 0.0
    stockdays = 0
    future_gates = {
        h: dict(
            H=h,
            mature_current_past_common_dates=0,
            future_count_below30=0,
            any_baseline_ic_undefined=0,
            any_fixed_ic_undefined=0,
            either_ic_undefined=0,
            included_dates=0,
        )
        for h in HORIZONS
    }
    component_schema = pa.schema(
        [
            ("date", pa.timestamp("ns")),
            ("code", pa.string()),
            ("n_full", pa.int32()),
            ("n_previous", pa.int32()),
            ("n_common", pa.int32()),
        ]
        + [(f"{m}_{v}", pa.int64()) for m in METRICS for v in ["A", "B", "E_numerator"]]
    )
    try:
        for t in range(5, len(calendar)):
            pair = valid[t] & valid[t - 5]
            nc = int(pair.sum())
            nt = int(counts[t])
            npast = int(counts[t - 5])
            if min(nc, nt, npast) < 30:
                continue
            hct = half_rank(values[t][:, pair])
            hcp = half_rank(values[t - 5][:, pair])
            hft = ranks[t][:, pair].astype(np.int64)
            hfp = ranks[t - 5][:, pair].astype(np.int64)
            a, b, en, f, c, e, err = parts(hft, hfp, hct, hcp, nt, npast, nc)
            maxerr = max(maxerr, err)
            stockdays += nc
            cp = codes[pair]
            frame = {
                "date": np.repeat(calendar[t].to_datetime64(), nc),
                "code": cp,
                "n_full": np.full(nc, nt, dtype=np.int32),
                "n_previous": np.full(nc, npast, dtype=np.int32),
                "n_common": np.full(nc, nc, dtype=np.int32),
            }
            for j, m in enumerate(METRICS):
                frame[f"{m}_A"] = a[j]
                frame[f"{m}_B"] = b[j]
                frame[f"{m}_E_numerator"] = en[j]
            table = pa.Table.from_pydict(frame, schema=component_schema)
            if writer is None:
                writer = pq.ParquetWriter(
                    CACHE / "components_stockday.parquet",
                    component_schema,
                    compression="zstd",
                )
            writer.write_table(table)
            base = dict(
                date=calendar[t],
                previous_date=calendar[t - 5],
                n_full=nt,
                n_previous=npast,
                n_common=nc,
                n_entered=nt - nc,
                n_exited=npast - nc,
                identity_error=err,
            )
            support.append(base)
            for j, m in enumerate(METRICS):
                daily.append(
                    dict(
                        base,
                        scope="current_past_common_no_future_filter",
                        metric=m,
                        n_evaluated=nc,
                        **moments(f[j], c[j], e[j], a[j], b[j], cp),
                    )
                )
            for h in HORIZONS:
                if t + h >= len(calendar):
                    continue
                future_gates[h]["mature_current_past_common_dates"] += 1
                future = ranks[t + h][:, pair]
                mask = (future > 0).all(axis=0)
                n = int(mask.sum())
                if n < 30:
                    future_gates[h]["future_count_below30"] += 1
                    continue
                baseline = corr_rank(a[:, mask], future[:, mask])
                fixed = corr_rank(b[:, mask], future[:, mask])
                if not (np.isfinite(baseline).all() and np.isfinite(fixed).all()):
                    future_gates[h]["any_baseline_ic_undefined"] += int(
                        not np.isfinite(baseline).all()
                    )
                    future_gates[h]["any_fixed_ic_undefined"] += int(
                        not np.isfinite(fixed).all()
                    )
                    future_gates[h]["either_ic_undefined"] += 1
                    continue
                future_gates[h]["included_dates"] += 1
                for j, m in enumerate(METRICS):
                    ic.append(
                        dict(
                            date=calendar[t],
                            H=h,
                            metric=m,
                            n_stocks=n,
                            baseline=baseline[j],
                            fixed_rerank=fixed[j],
                        )
                    )
                    if h == 3:
                        daily.append(
                            dict(
                                base,
                                scope="H3_future_complete_common",
                                metric=m,
                                n_evaluated=n,
                                **moments(
                                    f[j, mask],
                                    c[j, mask],
                                    e[j, mask],
                                    a[j, mask],
                                    b[j, mask],
                                    cp[mask],
                                ),
                            )
                        )
            if len(support) % 200 == 0:
                print(
                    f"Analyzed {len(support)} signal dates through {calendar[t].date()}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
    df = pd.DataFrame(daily)
    ics = pd.DataFrame(ic)
    csv(df, "decomposition_daily.csv")
    csv(pd.DataFrame(support), "pair_support_daily.csv")
    csv(ics, "delta5_ic_all_horizons_new_batch.csv")
    csv(pd.DataFrame(list(future_gates.values())), "future_support_gates.csv")
    summaries = []
    for (scope, metric), g in df.groupby(["scope", "metric"]):
        row = dict(
            scope=scope,
            metric=metric,
            n_dates=len(g),
            sum_evaluated_stockdays=int(g.n_evaluated.sum()),
            first_date=g.date.min(),
            last_date=g.date.max(),
        )
        excluded = {"date", "previous_date", "scope", "metric"}
        for name in g.columns:
            if name not in excluded and pd.api.types.is_numeric_dtype(g[name]):
                row[f"date_mean_{name}"] = g[name].mean()
        total = g.n_evaluated.sum()
        row["stockday_weighted_sign_flip"] = g.n_strict_sign_flip.sum() / total
        row["stockday_weighted_sign_class_mismatch"] = (
            g.n_sign_class_mismatch.sum() / total
        )
        row["stockday_weighted_reference_abs_larger"] = (
            g.n_reference_abs_larger.sum() / total
        )
        row["count_weighted_sign_flip_given_both_nonzero"] = (
            g.n_strict_sign_flip.sum() / g.n_both_nonzero.sum()
        )
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    csv(summary, "decomposition_summary.csv")
    return (
        df,
        ics,
        summary,
        dict(
            identity_max_abs_error=maxerr,
            variance_identity_max_abs_error=float(
                df.variance_identity_error.abs().max()
            ),
            signal_dates=len(support),
            component_stockdays=stockdays,
            component_rows_by_metric=3 * stockdays,
            identity_tolerance=TOL,
            ic_tolerance=IC_TOL,
            future_support_gates=list(future_gates.values()),
        ),
    )


def validate_original(ics):
    olddaily = pd.read_csv(OLD / "daily_correlations.csv", parse_dates=["date"])
    oldbase = olddaily[olddaily["transform"] == "delta5"].melt(
        id_vars=["date", "H", "n_stocks"],
        value_vars=METRICS,
        var_name="metric",
        value_name="baseline_original",
    )
    olddiag = pd.read_csv(OLD / "paired_reranking_daily.csv", parse_dates=["date"])
    olddiag = olddiag.rename(
        columns={
            "baseline_float64": "baseline_diagnostic_original",
            "same_pair_universe_reranked": "fixed_rerank_original",
            "n_stocks": "n_stocks_diagnostic_original",
        }
    )
    prior = oldbase.rename(columns={"n_stocks": "n_stocks_original"}).merge(
        olddiag[
            [
                "date",
                "H",
                "metric",
                "baseline_diagnostic_original",
                "fixed_rerank_original",
                "n_stocks_diagnostic_original",
            ]
        ],
        on=["date", "H", "metric"],
        how="outer",
        validate="one_to_one",
        indicator="old_join",
    )
    joined = ics.merge(
        prior,
        on=["date", "H", "metric"],
        how="outer",
        validate="one_to_one",
        indicator="new_old_join",
    )
    for a, b, name in [
        ("baseline", "baseline_original", "baseline_difference"),
        (
            "baseline",
            "baseline_diagnostic_original",
            "baseline_vs_old_diagnostic_difference",
        ),
        ("fixed_rerank", "fixed_rerank_original", "fixed_rerank_difference"),
    ]:
        joined[name] = joined[a] - joined[b]
    joined["n_stocks_equal"] = (joined.n_stocks == joined.n_stocks_original) & (
        joined.n_stocks == joined.n_stocks_diagnostic_original
    )
    csv(joined, "validation_old_new_daily.csv")
    rows = []
    for (h, m), g in joined.groupby(["H", "metric"]):
        rows.append(
            dict(
                H=h,
                metric=m,
                n_new=int(g.baseline.count()),
                n_original=int(g.baseline_original.count()),
                new_baseline_mean=g.baseline.mean(),
                old_baseline_mean=g.baseline_original.mean(),
                new_fixed_mean=g.fixed_rerank.mean(),
                old_fixed_mean=g.fixed_rerank_original.mean(),
                baseline_max_abs_diff=g.baseline_difference.abs().max(),
                fixed_max_abs_diff=g.fixed_rerank_difference.abs().max(),
                baseline_rows_outside_tol=int(
                    (g.baseline_difference.abs() > IC_TOL).sum()
                ),
                fixed_rows_outside_tol=int(
                    (g.fixed_rerank_difference.abs() > IC_TOL).sum()
                ),
                n_key_mismatches=int((g.new_old_join != "both").sum()),
                n_stock_count_mismatches=int((~g.n_stocks_equal).sum()),
            )
        )
    summary = pd.DataFrame(rows)
    oldsummary = pd.read_csv(OLD / "paired_reranking_summary.csv")
    summary = summary.merge(
        oldsummary[["H", "metric", "baseline_float64", "same_pair_universe_reranked"]],
        on=["H", "metric"],
        validate="one_to_one",
    )
    summary["baseline_mean_vs_saved_summary"] = (
        summary.new_baseline_mean - summary.baseline_float64
    )
    summary["fixed_mean_vs_saved_summary"] = (
        summary.new_fixed_mean - summary.same_pair_universe_reranked
    )
    csv(summary, "validation_old_new_summary.csv")
    stats = dict(
        old_internal_join_counts=prior.old_join.value_counts().to_dict(),
        new_old_join_counts=joined.new_old_join.value_counts().to_dict(),
        all_24_summary_cells=len(summary) == 24,
        max_baseline_abs_difference=float(joined.baseline_difference.abs().max()),
        max_fixed_abs_difference=float(joined.fixed_rerank_difference.abs().max()),
        rows_outside_ic_tolerance=int(
            (
                (joined.baseline_difference.abs() > IC_TOL)
                | (joined.fixed_rerank_difference.abs() > IC_TOL)
            ).sum()
        ),
        key_mismatches=int((joined.new_old_join != "both").sum()),
        stock_count_mismatches=int((~joined.n_stocks_equal).sum()),
    )
    stats["all_delta5_daily_statistics_match_within_tolerance"] = bool(
        stats["rows_outside_ic_tolerance"] == 0
        and stats["key_mismatches"] == 0
        and stats["stock_count_mismatches"] == 0
        and stats["all_24_summary_cells"]
    )
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    started = time.time()
    tests = synthetic_tests()
    save_json(RESULTS / "synthetic_validation.json", tests)
    calendar, codes, values, metadata = load_cache() if args.from_cache else ingest()
    df, ics, summary, checks = analyze(calendar, codes, values)
    checks.update(validate_original(ics))
    checks["synthetic_tests"] = tests
    checks["all_numeric_identity_checks_pass"] = bool(
        checks["identity_max_abs_error"] < TOL
        and checks["variance_identity_max_abs_error"] < TOL
    )
    save_json(RESULTS / "validation.json", checks)
    outputs = []
    for p in sorted(list(RESULTS.glob("*.csv")) + list(CACHE.glob("*"))):
        if p.is_file():
            outputs.append(
                dict(
                    path=str(p.relative_to(HERE)),
                    bytes=p.stat().st_size,
                    sha256=sha_file(p),
                )
            )
    save_json(
        RESULTS / "output_manifest.json",
        dict(
            script_sha256=sha_file(__file__),
            design_sha256=sha_file(HERE / "DESIGN.md"),
            elapsed_seconds=time.time() - started,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            files=outputs,
        ),
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    print(
        summary[
            [
                "scope",
                "metric",
                "n_dates",
                "date_mean_std_full",
                "date_mean_std_common",
                "date_mean_std_reference",
                "date_mean_cov_common_reference",
                "date_mean_strict_sign_flip_fraction",
                "date_mean_reference_abs_larger_fraction",
                "date_mean_top20_overlap_fraction",
                "date_mean_top100_overlap_fraction",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"Finished in {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
