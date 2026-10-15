"""Select independently rebuilt features using a preserved research row index.

The index has only signal_date and stock_code. It discloses the retained sample
support without asserting recovery of its historical collection/admission rule.
The feature input must be the output of input_build.py, not an archived panel.
"""

from __future__ import annotations
import os

for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ[k] = "1"
from pathlib import Path
from functools import lru_cache
import argparse
import datetime
import hashlib
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--observation-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest_path = a.features / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (
        manifest["status"] == "BUILT_FROM_RAW"
        and manifest["cached_covariate_values_read"] is False
    )
    records = {r["security_id"]: r for r in manifest["records"] if "output_file" in r}
    index = pq.ParquetFile(a.observation_index)
    assert (
        set(index.schema_arrow.names) == {"stock_code", "signal_date"}
        and len(index.schema_arrow) == 2
    )
    if a.output.exists():
        raise FileExistsError("Choose a new output file")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix(".partial.parquet")
    if temporary.exists():
        raise FileExistsError("Unresolved partial output exists")
    timestamp_fields = {"signal_date", "feature_asof_date", "source_last_trade_date"}
    string_fields = {"stock_code", "sw1_industry", "sw2_industry", "sw3_industry"}
    indicator_fields = {
        "source_trade_observed_t",
        "zero_amount_t",
        "limit_up_proxy_t",
        "limit_down_proxy_t",
        "one_price_limit_proxy_t",
        "member_hs300_t",
        "member_sz50_t",
        "member_zz500_t",
        "member_cyb_t",
        "member_zz1000_t",
        "member_zz2000_t",
    }
    writer = None
    schema = None
    previous = None
    rows = 0
    minimum = None
    maximum = None
    securities = set()

    @lru_cache(maxsize=4)
    def load(code):
        path = a.features / "by-stock" / (code + ".parquet")
        assert (
            code in records and sha(path) == records[code]["output_sha256"]
        ), "Raw feature file differs from build receipt"
        d = pd.read_parquet(path).set_index("signal_date", drop=False)
        assert not d.index.duplicated().any() and d.stock_code.eq(code).all()
        return d

    try:
        for batch in index.iter_batches(batch_size=100000):
            keys = batch.to_pandas()
            assert not keys.isna().any().any()
            for code, group in keys.groupby("stock_code", sort=False):
                dates = pd.DatetimeIndex(group.signal_date)
                assert dates.is_monotonic_increasing and not dates.duplicated().any()
                first = (code, dates[0])
                last = (code, dates[-1])
                assert (
                    previous is None or first > previous
                ), "Index must be strictly sorted by stock_code, signal_date"
                previous = last
                source = load(code)
                missing = dates.difference(source.index)
                if len(missing):
                    raise ValueError(
                        f"Raw-built features missing {len(missing)} index dates for {code}"
                    )
                data = source.loc[dates].reset_index(drop=True)
                if schema is None:
                    assert len(data.columns) == 51
                    schema = pa.schema(
                        [
                            (
                                c,
                                (
                                    pa.timestamp("ns")
                                    if c in timestamp_fields
                                    else (
                                        pa.large_string()
                                        if c in string_fields
                                        else (
                                            pa.int8()
                                            if c in indicator_fields
                                            else pa.float64()
                                        )
                                    )
                                ),
                            )
                            for c in data.columns
                        ]
                    )
                    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
                writer.write_table(
                    pa.Table.from_pandas(data, schema=schema, preserve_index=False)
                )
                rows += len(data)
                securities.add(code)
                minimum = (
                    min(minimum, dates.min()) if minimum is not None else dates.min()
                )
                maximum = (
                    max(maximum, dates.max()) if maximum is not None else dates.max()
                )
        if writer is None:
            raise ValueError("Empty observation index")
    finally:
        if writer is not None:
            writer.close()
    assert rows == index.metadata.num_rows
    temporary.replace(a.output)
    receipt = dict(
        schema="preserved-observation-index-selection/1",
        status="SELECTED_FROM_RAW_FEATURES",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rows=rows,
        securities=len(securities),
        date_min=str(minimum.date()),
        date_max=str(maximum.date()),
        index_sha256=sha(a.observation_index),
        feature_build_manifest_sha256=sha(manifest_path),
        raw_constructor_sha256=manifest["constructor_sha256"],
        preprocessing_contract=manifest.get("preprocessing_contract", "full-history"),
        selector_sha256=sha(__file__),
        output_sha256=sha(a.output),
        archived_feature_values_read=False,
        all_selected_feature_file_hashes_verified=True,
        index_role="Preserved research observation support; historical admission rule not claimed to be recovered.",
    )
    a.output.with_suffix(".manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
