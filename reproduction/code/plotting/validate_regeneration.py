"""Compare regenerated figures and plot tables with publication assets."""

from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "work/figure-validation/summary.json"
    )
    parser.add_argument(
        "--contact-dir",
        type=Path,
        help="Optional contact sheets, kept separate from publication assets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Directory containing regenerated figure subfolders.",
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "figures/index.json").read_text(encoding="utf-8"))
    results = []
    for r in manifest["figures"]:
        folder = ROOT / r["directory"]
        rebuilt = (
            args.output_root / r["id"]
            if args.output_root
            else folder / "analysis/rebuilt"
        )
        receipt = json.loads(
            (rebuilt / "regeneration.json").read_text(encoding="utf-8")
        )
        baseline_png = folder / next(
            x["file"] for x in r["figures"] if x["format"] == "png"
        )
        rebuilt_png = rebuilt / "figures" / (r["output_basename"] + ".png")
        with Image.open(baseline_png) as im:
            a = np.asarray(im.convert("RGB")).astype(np.int16)
        with Image.open(rebuilt_png) as im:
            b = np.asarray(im.convert("RGB")).astype(np.int16)
        sizes_equal = a.shape == b.shape
        delta = np.abs(a - b) if sizes_equal else None
        baseline_pdf = folder / next(
            x["file"] for x in r["figures"] if x["format"] == "pdf"
        )
        rebuilt_pdf = rebuilt / "figures" / (r["output_basename"] + ".pdf")
        with fitz.open(baseline_pdf) as d1, fitz.open(rebuilt_pdf) as d2:
            p1, p2 = d1[0], d2[0]
            dimensions_equal = np.allclose(
                list(p1.rect), list(p2.rect), rtol=0, atol=1e-6
            )
            text_equal = (
                re.sub(r"\s+", " ", p1.get_text()).strip()
                == re.sub(r"\s+", " ", p2.get_text()).strip()
            )
            spans1 = [
                s
                for block in p1.get_text("dict")["blocks"]
                if "lines" in block
                for line in block["lines"]
                for s in line["spans"]
            ]
            spans2 = [
                s
                for block in p2.get_text("dict")["blocks"]
                if "lines" in block
                for line in block["lines"]
                for s in line["spans"]
            ]
            paired_spans = len(spans1) == len(spans2) and all(
                a["text"] == b["text"] for a, b in zip(spans1, spans2)
            )
            text_box_shift = (
                max(
                    float(np.max(np.abs(np.array(a["bbox"]) - np.array(b["bbox"]))))
                    for a, b in zip(spans1, spans2)
                )
                if paired_spans
                else None
            )
            q1 = p1.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
            q2 = p2.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
            pdf_pixels_equal = q1.samples == q2.samples
        tables = []
        for expected in r["reference_plot_data"]:
            reference = folder / expected["file"]
            matches = list(rebuilt.rglob(reference.name))
            if len(matches) != 1:
                tables.append(
                    dict(
                        file=reference.name,
                        status="MISSING_OR_AMBIGUOUS",
                        matches=len(matches),
                    )
                )
                continue
            q = pd.read_csv(reference, float_precision="round_trip")
            z = pd.read_csv(matches[0], float_precision="round_trip")
            try:
                pd.testing.assert_frame_equal(
                    q, z, check_dtype=False, rtol=1e-12, atol=1e-12
                )
                status = "PASS"
                detail = None
            except AssertionError as e:
                status = "DIFFERENT"
                detail = str(e)[:1000]
            tables.append(
                dict(
                    file=reference.name,
                    status=status,
                    rows=len(q),
                    columns=len(q.columns),
                    detail=detail,
                )
            )
        result = dict(
            figure=r["id"],
            source_key=r["source_key"],
            png_shape_equal=bool(sizes_equal),
            png_pixels_equal=bool(sizes_equal and not delta.any()),
            png_changed_pixel_fraction=(
                float(np.any(delta, axis=2).mean()) if sizes_equal else None
            ),
            png_mean_absolute_channel_difference=(
                float(delta.mean()) if sizes_equal else None
            ),
            pdf_page_dimensions_equal=bool(dimensions_equal),
            pdf_visible_text_equal=text_equal,
            pdf_150dpi_pixels_equal=pdf_pixels_equal,
            pdf_text_spans_match=paired_spans,
            pdf_maximum_text_box_shift_pt=text_box_shift,
            plot_tables=tables,
            all_plot_tables_equal=all(t["status"] == "PASS" for t in tables),
            final_captions_retained=receipt["caption_en"] == r["caption_en"]
            and receipt["caption_zh"] == r["caption_zh"],
            frozen_publication_files_unchanged=all(
                file_sha256(folder / x["file"]) == x["sha256"] for x in r["figures"]
            ),
            inputs_unchanged=all(
                file_sha256(folder / x["file"]) == x["sha256"] for x in r["data"]
            ),
        )
        results.append(result)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        (args.report.parent / (r["id"] + ".json")).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            r["id"],
            "PNG",
            result["png_pixels_equal"],
            "PDFtext",
            text_equal,
            "PDFpixels",
            pdf_pixels_equal,
            "TABLES",
            result["all_plot_tables_equal"],
        )
    report = dict(
        scope="Real rendering from aggregate source data; no re-estimation of empirical experiments",
        figure_count=len(results),
        all_inputs_unchanged=all(x["inputs_unchanged"] for x in results),
        all_frozen_files_unchanged=all(
            x["frozen_publication_files_unchanged"] for x in results
        ),
        all_plotted_tables_equal=all(x["all_plot_tables_equal"] for x in results),
        all_final_captions_retained=all(x["final_captions_retained"] for x in results),
        figures=results,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.contact_dir:
        args.contact_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(manifest["figures"]), 6):
            canvas = Image.new("RGB", (1600, 1650), "#EEEEEE")
            draw = ImageDraw.Draw(canvas)
            for j, r in enumerate(manifest["figures"][start : start + 6]):
                rebuilt = (
                    args.output_root / r["id"]
                    if args.output_root
                    else ROOT / r["directory"] / "analysis/rebuilt"
                )
                path = rebuilt / "figures" / (r["output_basename"] + ".png")
                with Image.open(path) as im:
                    thumb = im.convert("RGB")
                    thumb.thumbnail((750, 495))
                    x = 20 + (j % 2) * 800
                    y = 40 + (j // 2) * 550
                    canvas.paste(thumb, (x, y))
                    draw.text((x, y - 25), r["display_label"], fill="black")
            canvas.save(args.contact_dir / f"redrawn_{start+1:02}.png")


if __name__ == "__main__":
    main()
