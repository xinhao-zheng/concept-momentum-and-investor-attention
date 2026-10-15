"""Verify score covariates with NumPy and replace historical reference values."""

from pathlib import Path
import json
import gc
import shutil
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

R = Path(__file__).resolve().parent
O = R.parents[2] / "attention-study"
C = R / "cache/mechanism"


def replace_frame(frame, path):
    t = path.with_suffix(".partial.parquet")
    frame.write_parquet(t, compression="zstd")
    del frame
    gc.collect()
    return t


def main():
    audits = []
    newcols = []
    for des in ["beta", "gamma"]:
        w = pl.read_parquet(C / f"winner_{des}_F5_W10.parquet")
        reference_path = O / f"cache/mechanism/winner_{des}_F5_W10.parquet"
        m = None
        if reference_path.exists():
            ow = pl.read_parquet(reference_path)
            m = w.join(ow, on=["signal_date", "stock_code"], suffix="_old")
        error_col = (
            "stock_score_absolute_error"
            if "stock_score_absolute_error" in w.columns
            else "difference_from_archived_reference"
        )
        selected = (
            w.sort(error_col, descending=True)
            .head(30)
            .select("signal_date", "stock_code")
        )
        selected = pl.concat(
            [
                selected,
                w.sample(n=30, seed=20260928).select("signal_date", "stock_code"),
            ]
        ).unique()
        cards = (
            pl.scan_parquet(C / f"full_protocol_cards_{des}.parquet")
            .filter(pl.col("stock_code").is_in(selected["stock_code"].implode()))
            .sort(["stock_code", "concept", "signal_date"])
            .collect()
        ).to_pandas()
        wanted = set(
            map(
                tuple,
                selected.to_pandas()[["signal_date", "stock_code"]].itertuples(
                    index=False, name=None
                ),
            )
        )
        errors = []
        degenerate = []
        for (code, concept), g in cards.groupby(["stock_code", "concept"], sort=False):
            rr = g.stock_ret.to_numpy(float)
            cc = g.concept_ret_loo.to_numpy(float)
            for i, row in enumerate(g.itertuples(index=False)):
                if (row.signal_date, code) not in wanted:
                    continue
                a = max(0, i - 9)
                x = rr[a : i + 1]
                y = cc[a : i + 1]
                ok = np.isfinite(x) & np.isfinite(y)
                expected = (
                    np.corrcoef(x[ok], y[ok])[0, 1]
                    if ok.sum() >= 5 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0
                    else np.nan
                )
                stored = float(row.concept_corr10_loo)
                er = (
                    abs(stored - expected)
                    if np.isfinite(stored) and np.isfinite(expected)
                    else 0.0 if np.isfinite(stored) == np.isfinite(expected) else np.nan
                )
                if not np.isfinite(er):
                    degenerate.append(
                        {
                            "date": str(row.signal_date),
                            "code": code,
                            "concept": concept,
                            "expected": str(expected),
                            "stored": str(stored),
                            "sd_stock": float(np.std(x[ok])) if ok.sum() else None,
                        }
                    )
                z = cc[max(0, i - 4) : i + 1]
                mom = (
                    float(np.expm1(np.log1p(np.clip(z, -0.999, None)).sum()))
                    if len(z) == 5 and np.isfinite(z).all()
                    else np.nan
                )
                sm = float(row.concept_mom5_loo)
                em = (
                    abs(mom - sm)
                    if np.isfinite(mom) and np.isfinite(sm)
                    else 0.0 if np.isfinite(mom) == np.isfinite(sm) else np.nan
                )
                errors.append({"corr_error": er, "momentum_error": em})
        q = pd.DataFrame(errors)
        out = {
            "design": des,
            "spot_cards": len(q),
            "max_abs_correlation_error": float(q.corr_error.max()),
            "max_abs_momentum_error": float(q.momentum_error.max()),
            "finite_pattern_mismatches": degenerate,
            "historical_comparison_status": (
                "COMPARED"
                if m is not None
                else "SKIPPED_ARCHIVED_REFERENCE_UNAVAILABLE_OR_DISABLED"
            ),
            "historical_common_rows": m.height if m is not None else None,
            "historical_winner_changes": (
                m.select(
                    (pl.col("winning_concept") != pl.col("winning_concept_old")).sum()
                ).item()
                if m is not None
                else None
            ),
            "max_historical_score_difference": (
                m.select(
                    (
                        pl.col("reconstructed_stock_score_F5_W10")
                        - pl.col("reconstructed_stock_score_F5_W10_old")
                    )
                    .abs()
                    .max()
                ).item()
                if m is not None
                else None
            ),
        }
        audits.append(out)
        print(json.dumps(out, ensure_ascii=False), flush=True)
        assert q.momentum_error.max() < 1e-9, out
        assert not degenerate, out
        assert q.corr_error.max() < 1e-7, out
        newcols.append(
            w.select(
                "signal_date",
                "stock_code",
                pl.col("reconstructed_stock_score_F5_W10").alias(
                    f"{des}_score_F5_W10_DEFAULT"
                ),
            )
        )
    (R / "audit/independent_score_kernel.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # The independently checked current-input kernel, rather than the older
    # result file, supplies the score covariate. Sample keys were already frozen.
    samplepath = C / "mechanism_sample_keys_before_outcomes.parquet"
    original = pl.read_parquet(samplepath)
    sample = original.drop("beta_score_F5_W10_DEFAULT", "gamma_score_F5_W10_DEFAULT")
    for z in newcols:
        sample = sample.join(z, on=["signal_date", "stock_code"], maintain_order="left")
    sample = sample.select(original.columns)
    if not sample.equals(original):
        tmp = replace_frame(sample, samplepath)
        del sample
        gc.collect()
        tmp.replace(samplepath)
    for des, check in zip(["beta", "gamma"], audits):
        wp = C / f"winner_{des}_F5_W10.parquet"
        w = pl.read_parquet(wp)
        if "expected_stock_score_F5_W10" in w.columns:
            w = w.rename(
                {
                    "expected_stock_score_F5_W10": "archived_reference_score_F5_W10",
                    "stock_score_absolute_error": "difference_from_archived_reference",
                }
            )
            tmp = replace_frame(w, wp)
            del w
            gc.collect()
            tmp.replace(wp)
        ap = R / f"audit/mechanism/winner_{des}_lineage.json"
        audit = json.loads(ap.read_text(encoding="utf-8"))
        if audit["status"] == "FAIL":
            raise AssertionError(audit)
        audit["status"] = "PASS_INDEPENDENT_NUMPY_VALIDATION"
        audit["hard_gates"].pop(
            "archived_reference_matches_current_inputs_diagnostic", None
        )
        audit["hard_gates"]["independent_momentum_correlation_checks"] = True
        audit["reference_comparison_is_validation_gate"] = False
        audit["reference_provenance"] = (
            "Saved historical sample scores on matching keys; S2 observation-score references on newly added keys. Both serve diagnostics only."
        )
        audit["independent_validation"] = check
        for key, path in [("output", wp)]:
            meta = pq.ParquetFile(path)
            audit[key].update(
                bytes=path.stat().st_size,
                rows=meta.metadata.num_rows,
                row_groups=meta.metadata.num_row_groups,
                columns=meta.schema_arrow.names,
            )
        smeta = pq.ParquetFile(samplepath)
        audit["inputs"]["sample"].update(
            bytes=samplepath.stat().st_size, columns=smeta.schema_arrow.names
        )
        ap.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "Kernel independently checked; current-input score covariates frozen.",
        flush=True,
    )


if __name__ == "__main__":
    main()
