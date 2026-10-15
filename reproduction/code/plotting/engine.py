"""Render publication figures from released aggregate inputs."""

from pathlib import Path
import argparse
import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
import importlib.metadata as metadata


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def render_figure(repository_root, record, output=None):
    root = Path(repository_root).resolve()
    folder = root / record["directory"]
    output = Path(output).resolve() if output else folder / "analysis/rebuilt"
    frozen = folder / "figures"
    if (
        output == root
        or output == folder
        or output == frozen
        or output in frozen.parents
    ):
        raise ValueError("Choose a separate generated-output directory.")
    for part in ("figures", "data", "qa"):
        (output / part).mkdir(parents=True, exist_ok=True)

    inputs = {}
    for entry in record["data"]:
        path = folder / entry["file"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f'Input hash mismatch: {record["id"]}/{entry["file"]}')
        inputs[entry["key"]] = dict(path=path, sha256=entry["sha256"])
    before = {entry["file"]: file_sha256(folder / entry["file"]) for entry in record["figures"]}
    for entry in record["figures"]:
        if before[entry["file"]] != entry["sha256"]:
            raise ValueError(f'Frozen figure hash mismatch: {entry["file"]}')

    import matplotlib
    from matplotlib.font_manager import FontProperties, findfont

    font = findfont(FontProperties(family="Arial"), fallback_to_default=False)
    from . import figure_core as core

    core.ROOT, core.OUT, core.DATA, core.QA = (
        output,
        output / "figures",
        output / "data",
        output / "qa",
    )
    core.SOURCES, core.MANIFEST, core.CAPTIONS, core.CURRENT_USED = (
        inputs,
        [],
        {},
        set(),
    )
    core.FROZEN_CAPTION = {
        k: record[k] for k in ("display_label", "caption_en", "caption_zh")
    }
    core.style()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if record["renderer"] == "fig_s12":
            from .supp_n1 import render

            render_metadata = render(
                output,
                "en",
                folder / "data",
                core.FROZEN_CAPTION,
                Path(__file__).parent / "figure_palette.json",
            )
        elif record["renderer"] == "fig_s13":
            from .supp_drawdown import render

            render_metadata = render(
                output,
                "en",
                output_root=output,
                data_root=folder / "data",
                caption_record=core.FROZEN_CAPTION,
            )
        else:
            getattr(core, record["renderer"])()
            render_metadata = core.MANIFEST[0]
    glyph_warnings = [str(x.message) for x in caught if "Glyph" in str(x.message)]
    if glyph_warnings:
        raise RuntimeError(glyph_warnings)

    generated = []
    for ext in ("pdf", "svg", "png"):
        original_name = output / "figures" / f'{record["source_key"]}.{ext}'
        final_name = output / "figures" / f'{record["output_basename"]}.{ext}'
        if original_name != final_name:
            original_name.replace(final_name)
        generated.append(
            dict(
                file=final_name.relative_to(output).as_posix(),
                sha256=file_sha256(final_name),
                bytes=final_name.stat().st_size,
            )
        )
    after = {entry["file"]: file_sha256(folder / entry["file"]) for entry in record["figures"]}
    if before != after:
        raise RuntimeError("Frozen publication assets changed during rendering.")
    for entry in record["data"]:
        if file_sha256(folder / entry["file"]) != entry["sha256"]:
            raise RuntimeError("Input file changed during rendering.")

    # Record reproducible inputs and software without machine-specific paths.
    report = dict(
        figure=record["id"],
        display_label=record["display_label"],
        source_key=record["source_key"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        scope="Regeneration from frozen aggregate inputs; no scientific re-estimation",
        status="RENDERED",
        input_hashes_verified=True,
        frozen_figures_unchanged=True,
        caption_source="figures/index.json",
        caption_en=record["caption_en"],
        caption_zh=record["caption_zh"],
        versions={
            x: metadata.version(x)
            for x in ("matplotlib", "numpy", "pandas", "colorspacious", "Pillow")
        },
        python=platform.python_version(),
        font_family="Arial",
        font_file_sha256=file_sha256(font),
        generated_files=generated,
        warnings=[str(x.message) for x in caught],
        render_metadata=render_metadata,
    )
    write_json(output / "regeneration.json", report)
    return report


def run_cli(default_figure=None, repository_root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        default=default_figure,
        help="Displayed identifier, e.g. fig-08 or fig-s13; omitted means all figures.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional separate output directory; one subfolder is created per figure.",
    )
    args = parser.parse_args()
    root = Path(repository_root or Path(__file__).resolve().parents[2])
    manifest = json.loads((root / "figures/index.json").read_text(encoding="utf-8"))
    records = [
        r for r in manifest["figures"] if not args.figure or r["id"] == args.figure
    ]
    if not records:
        parser.error("Figure identifier not found in the frozen manifest.")
    for record in records:
        target = args.output_root / record["id"] if args.output_root else None
        report = render_figure(root, record, target)
        print(f'{record["id"]}: {report["status"]}; frozen files unchanged', flush=True)
