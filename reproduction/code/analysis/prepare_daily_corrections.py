"""Build targeted correction overlays from acquired inputs and an event ledger.

The public event ledger contains keys and rules, not replacement price values.
Four verified closing prices must be supplied separately in a local CSV.
Source descriptive headers are skipped and are never copied into outputs.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd

CHECK_FIELDS = ["开盘价", "最高价", "最低价", "收盘价", "前收盘价", "成交量", "成交额"]
FOLDERS = ["stock-trading-data", "stock-trading-data-pro"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument(
        "--replacement-closes",
        type=Path,
        required=True,
        help="Local UTF-8 CSV: stock_code,date,close; independently obtained values.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--csv-encoding", default="gb18030")
    p.add_argument("--skip-metadata-rows", type=int, default=1)
    a = p.parse_args()
    events = pd.read_csv(a.events, dtype=str, keep_default_na=False)
    closes = pd.read_csv(a.replacement_closes, dtype=str, keep_default_na=False)
    assert not closes.duplicated(["stock_code", "date"]).any()
    replacements = events[events.action.eq("replace_field_from_independent_input")]
    assert set(zip(closes.stock_code, closes.date)) == set(
        zip(replacements.stock_code, replacements.date)
    ), "Supply exactly the listed replacement-price keys"
    values = {(r.stock_code, r.date): r.close for r in closes.itertuples()}
    assert not a.output.exists(), "Choose a new output directory"
    a.output.mkdir(parents=True)
    records = []
    for folder in FOLDERS:
        target_dir = a.output / folder
        target_dir.mkdir()
        for code, g in events.groupby("stock_code", sort=True):
            source = a.raw_root / folder / (code + ".csv")
            d = pd.read_csv(
                source,
                encoding=a.csv_encoding,
                skiprows=a.skip_metadata_rows,
                dtype=str,
                keep_default_na=False,
            )
            assert (
                not d["交易日期"].duplicated().any()
            ), "Resolve duplicate input dates explicitly before targeted corrections"
            remove = g[g.action.eq("remove_observation")]
            remove_dates = set(remove.date)
            for event in remove.itertuples():
                row = d[d["交易日期"].eq(event.date)]
                assert len(row) == 1, (
                    code,
                    event.date,
                    "Expected one listed observation",
                )
                if event.validation_rule == "copied_ipo_on_suspended_date":
                    anchor = d[d["交易日期"].eq(event.reference_date)]
                elif event.validation_rule == "terminal_duplicate":
                    anchor = (
                        d[
                            (d["交易日期"] < event.date)
                            & ~d["交易日期"].isin(remove_dates)
                        ]
                        .sort_values("交易日期")
                        .tail(1)
                    )
                else:
                    raise ValueError("Unknown deletion rule")
                assert len(anchor) == 1
                assert np.array_equal(
                    row[CHECK_FIELDS].astype(float).to_numpy(),
                    anchor[CHECK_FIELDS].astype(float).to_numpy(),
                ), (code, event.date, "Duplicate invariant failed")
            d = d[~d["交易日期"].isin(remove_dates)].copy()
            changed = 0
            for event in g[
                g.action.eq("replace_field_from_independent_input")
            ].itertuples():
                assert event.field == "收盘价"
                mask = d["交易日期"].eq(event.date)
                assert mask.sum() == 1
                value = values[(code, event.date)]
                number = float(value)
                assert math.isfinite(number) and number > 0
                row = d.loc[mask].iloc[0]
                assert (
                    float(row["最低价"]) <= number <= float(row["最高价"])
                ), "Supplied close is outside the source daily range"
                next_mask = d["交易日期"].eq(event.reference_date)
                prior = d.loc[next_mask, "前收盘价"].copy()
                d.loc[mask, "收盘价"] = value
                assert d.loc[next_mask, "前收盘价"].equals(prior)
                changed += 1
            continuity = g[g.action.eq("preserve_continuous_canonical_history")]
            assert g.action.isin(
                [
                    "remove_observation",
                    "replace_field_from_independent_input",
                    "preserve_continuous_canonical_history",
                ]
            ).all()
            # Existing canonical history is preserved. No former-code history is appended.
            target = target_dir / source.name
            with target.open("w", encoding=a.csv_encoding, newline="") as f:
                for _ in range(a.skip_metadata_rows):
                    f.write("Locally reconstructed targeted corrections.\n")
                d.to_csv(f, index=False, lineterminator="\n")
            records.append(
                dict(
                    folder=folder,
                    stock_code=code,
                    input_sha256=sha(source),
                    output_sha256=sha(target),
                    removed_rows=len(remove),
                    replaced_close_cells=changed,
                    continuous_history_preserved=bool(len(continuity)),
                    rows=len(d),
                )
            )
    receipt = dict(
        schema="targeted-correction-construction/1",
        status="BUILT_FROM_INPUTS_AND_EVENT_LEDGER",
        events_sha256=sha(a.events),
        local_replacement_input_sha256=sha(a.replacement_closes),
        constructor_sha256=sha(__file__),
        source_inputs_modified=False,
        price_values_in_receipt=False,
        source_descriptive_headers_copied=False,
        csv_encoding=a.csv_encoding,
        skip_metadata_rows=a.skip_metadata_rows,
        records=records,
    )
    (a.output / "manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "files": len(records),
                "removed_rows_across_paired_directories": sum(
                    r["removed_rows"] for r in records
                ),
                "replaced_close_cells_across_paired_directories": sum(
                    r["replaced_close_cells"] for r in records
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
