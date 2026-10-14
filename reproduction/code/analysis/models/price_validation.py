"""Verify corrected price values independently of Parquet encoding and row order."""

from pathlib import Path
import hashlib, json, datetime


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_prices(reference, candidate, receipt):
    import polars as pl

    reference, candidate, receipt = map(Path, (reference, candidate, receipt))
    hashes = {"reference_sha256": sha(reference), "candidate_sha256": sha(candidate)}
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf8"))
        if old.get("status") == "PASS" and all(
            old.get(k) == v for k, v in hashes.items()
        ):
            return old
    report = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reference": str(reference),
        "candidate": str(candidate),
        **hashes,
        "keys": ["date", "code"],
        "scope": "Exact keyed values, null/NaN patterns and unique keys; file serialization and physical row order may differ.",
    }
    keys = ["date", "code"]
    columns = ["date", "code", "name", "open_adj", "close_adj", "ret", "amount"]
    numeric = columns[3:]
    a = pl.scan_parquet(reference)
    b = pl.scan_parquet(candidate)
    for frame in (a, b):
        if set(frame.collect_schema().names()) != set(columns):
            raise ValueError(
                "Corrected price schema differs from declared seven-column contract"
            )
    a = a.select(columns).with_columns(
        pl.col("date").cast(pl.Datetime("ns")), pl.col("code").cast(pl.String)
    )
    b = b.select(columns).with_columns(
        pl.col("date").cast(pl.Datetime("ns")), pl.col("code").cast(pl.String)
    )
    report["byte_identical"] = hashes["reference_sha256"] == hashes["candidate_sha256"]
    if report["byte_identical"]:
        report.update(
            status="PASS",
            rows=a.select(pl.len()).collect().item(),
            method="Identical SHA-256 content",
        )
    else:
        years = sorted(
            set(
                a.select(pl.col("date").dt.year().unique()).collect()["date"].to_list()
                + b.select(pl.col("date").dt.year().unique())
                .collect()["date"]
                .to_list()
            )
        )
        if None in years:
            raise ValueError("Null date in corrected prices")
        batches = []
        for year in years:
            frames = [
                f.filter(pl.col("date").dt.year() == year).collect() for f in (a, b)
            ]
            aa, bb = frames
            row = {
                "year": year,
                "reference_rows": len(aa),
                "candidate_rows": len(bb),
                "reference_duplicate_keys": len(aa) - aa.select(keys).unique().height,
                "candidate_duplicate_keys": len(bb) - bb.select(keys).unique().height,
                "reference_null_keys": aa.filter(
                    pl.any_horizontal(pl.col(k).is_null() for k in keys)
                ).height,
                "candidate_null_keys": bb.filter(
                    pl.any_horizontal(pl.col(k).is_null() for k in keys)
                ).height,
            }
            row["reference_only_keys"] = (
                aa.select(keys).join(bb.select(keys), on=keys, how="anti").height
            )
            row["candidate_only_keys"] = (
                bb.select(keys).join(aa.select(keys), on=keys, how="anti").height
            )
            if row["reference_duplicate_keys"] or row["candidate_duplicate_keys"]:
                raise ValueError("Duplicate price keys")
            joined = aa.join(bb, on=keys, suffix="_candidate", validate="1:1")
            row["different_values"] = {
                c: joined.select(
                    (~pl.col(c).eq_missing(pl.col(c + "_candidate"))).sum()
                ).item()
                for c in columns[2:]
            }
            row["null_mismatches"] = {
                c: joined.select(
                    (pl.col(c).is_null() != pl.col(c + "_candidate").is_null()).sum()
                ).item()
                for c in columns[2:]
            }
            row["max_abs_errors"] = {
                c: joined.select(
                    (pl.col(c) - pl.col(c + "_candidate")).abs().max()
                ).item()
                for c in numeric
            }
            batches.append(row)
        report["batches"] = batches
        report["rows"] = sum(z["reference_rows"] for z in batches)
        failed = any(
            any(
                z[k]
                for k in [
                    "reference_duplicate_keys",
                    "candidate_duplicate_keys",
                    "reference_null_keys",
                    "candidate_null_keys",
                    "reference_only_keys",
                    "candidate_only_keys",
                ]
            )
            or any(z["different_values"].values())
            for z in batches
        )
        report["status"] = "FAIL" if failed else "PASS"
        report["method"] = "Full keyed exact-value comparison in calendar-year batches"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    if report["status"] != "PASS":
        raise ValueError(
            "Core prices differ from corrected authority; see " + str(receipt)
        )
    return report
