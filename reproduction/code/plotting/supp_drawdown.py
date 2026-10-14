"""Render the complete three-account drawdown comparison without recomputation.

This standalone supplement does not alter the main figure registry. Its output
metadata is supplied separately for publication integration.
"""

from pathlib import Path
import argparse
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .figure_style import PALETTE

STEM = "fig_s13"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(root, language="en", output_root=None, data_root=None, caption_record=None):
    root = Path(root).resolve()
    source = Path(data_root) / "bt_curves.csv"
    summary_source = Path(data_root) / "bt_summary.csv"
    curves = pd.read_csv(source, float_precision="round_trip")
    summary = pd.read_csv(summary_source, float_precision="round_trip")
    curves["date"] = pd.to_datetime(curves.date)
    reference = summary.loc[
        summary.hold.eq("3D")
        & summary.F.eq(5)
        & summary.W.eq(10)
        & summary.similarity.eq("pearson")
        & summary.top_n.eq(10)
    ].set_index("timing")
    output_root = (
        Path(output_root)
        if output_root is not None
        else (root if language == "en" else root / "zh")
    )
    out = output_root / "figures"
    qa = out / "qa"
    data = out / "source_data"
    for path in (out, qa, data):
        path.mkdir(parents=True, exist_ok=True)
    previous_rc = plt.rcParams.copy()
    plt.rcParams.update(
        {
            "font.family": "Arial" if language == "en" else "Microsoft YaHei",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "axes.edgecolor": PALETTE["secondary_text"],
            "xtick.color": PALETTE["secondary_text"],
            "ytick.color": PALETTE["secondary_text"],
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": True,
        }
    )
    fig = plt.figure(figsize=(183 / 25.4, 104 / 25.4))
    axes, selected = [], []
    specs = [
        ("none", "No timing", "无择时", "timing_none", (0, (4.0, 1.8))),
        (
            "breadth",
            "Breadth timing",
            "一级宽度择时",
            "timing_breadth",
            (0, (6.0, 1.5, 1.2, 1.5)),
        ),
        ("breadth2", "Two-level timing", "两级宽度择时", "timing_two_level", "-"),
    ]
    for i, (key, en, zh, color_key, line) in enumerate(specs):
        ax = fig.add_axes([0.105, 0.72 - i * 0.295, 0.87, 0.20])
        axes.append(ax)
        q = curves.loc[curves.series.eq("Top10|" + key)].sort_values("date")
        assert len(q) == 2368 and not q.date.duplicated().any()
        assert q.date.min() == pd.Timestamp("2016-12-01")
        assert q.date.max() == pd.Timestamp("2026-08-31")
        assert np.allclose(q.drawdown, q.nav / q.nav.cummax() - 1, rtol=0, atol=1e-12)
        assert np.isclose(
            q.drawdown.min(), reference.loc[key, "max_drawdown"], rtol=0, atol=1e-12
        )
        assert q.drawdown.min() > -0.5 and q.drawdown.max() <= 0.02
        ax.fill_between(
            q.date,
            100 * q.drawdown,
            0,
            facecolor=PALETTE["range_fill"],
            edgecolor="none",
            zorder=1,
        )
        ax.plot(
            q.date,
            100 * q.drawdown,
            color=PALETTE[color_key],
            ls=line,
            lw=0.85 if key != "breadth2" else 0.95,
            zorder=3,
        )
        ax.axhline(0, color=PALETTE["grid"], lw=0.55, zorder=0)
        ax.set_ylim(-50, 2)
        ax.set_yticks([-40, -20, 0])
        ax.set_xlim(pd.Timestamp("2016-11-01"), pd.Timestamp("2026-10-01"))
        ax.set_ylabel("Drawdown (%)" if language == "en" else "回撤（%）")
        title = en if language == "en" else zh
        ax.set_title(
            "abc"[i] + "  " + title, loc="left", fontsize=8.5, fontweight="bold", pad=6
        )
        value = f'{100*float(reference.loc[key, "max_drawdown"]):.1f}%'
        right = ("Maximum drawdown: " if language == "en" else "最大回撤：") + value
        ax.set_title(
            right, loc="right", fontsize=8, color=PALETTE["secondary_text"], pad=6
        )
        ax.xaxis.set_major_locator(mdates.YearLocator(2, month=1, day=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if i != 2:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.spines["bottom"].set_visible(False)
        selected.append(q)
    assert len({tuple(ax.get_ylim()) for ax in axes}) == 1
    assert len({tuple(ax.get_xlim()) for ax in axes}) == 1
    fig.canvas.draw()
    outputs = {}
    for ext in ("pdf", "svg", "png"):
        target = out / f"{STEM}.{ext}"
        fig.savefig(target, dpi=450)
        outputs[ext] = {
            "path": str(target.relative_to(output_root)),
            "sha256": sha(target),
        }
    fig.savefig(qa / f"{STEM}_print_size.png", dpi=150)
    plotted = pd.concat(selected, ignore_index=True)
    assert len(plotted) == 7104
    plotted.to_csv(data / f"{STEM}_plot_data.csv", index=False, float_format="%.17g")
    geometry = {
        "dimensions_mm": [183, 104],
        "axes": [list(ax.get_position().bounds) for ax in axes],
        "ylim": [-50, 2],
        "dates": ["2016-12-01", "2026-08-31"],
        "observations_per_panel": 2368,
        "daily_observations_retained": True,
        "panels": [x[0] for x in specs],
        "source_sha256": sha(source),
        "summary_source_sha256": sha(summary_source),
        "inference_recomputed": False,
    }
    metadata = {
        "figure": STEM,
        "display_label": "Supplementary Figure S13",
        "caption_en": caption_record["caption_en"],
        "caption_zh": caption_record["caption_zh"],
        "caption": (
            caption_record["caption_en"]
            if language == "en"
            else caption_record["caption_zh"]
        ),
        "sources": ["bt_curves", "bt_summary"],
        "outputs": outputs,
        "geometry": geometry,
        "plot_count": 7104,
        "language": language,
    }
    (qa / f"{STEM}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plt.close(fig)
    plt.rcParams.update(previous_rc)
    return metadata
