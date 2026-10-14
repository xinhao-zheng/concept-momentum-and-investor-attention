"""Shared publication figure style for the CURRENT research package."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = {
    "alpha": "#626A73",  # idealized reference: neutral graphite
    "beta": "#D55E00",  # synchronous membership: warm vermilion
    "gamma": "#0072B2",  # lagged membership: cool blue
    "difference": "#7A5195",
    "a_share": "#D55E00",
    "us": "#0072B2",
    "crypto": "#009E73",
    "positive": "#B4555A",
    "negative": "#4C7899",
    "zero": "#9AA0A6",
    "ink": "#22272E",
    "muted": "#667085",
    "grid": "#DDE3EA",
    "paper": "#FFFFFF",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Noto Sans CJK SC",
                "Microsoft YaHei",
                "DejaVu Sans",
            ],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "axes.titleweight": "semibold",
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.0,
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
            "savefig.facecolor": COLORS["paper"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axes(ax: plt.Axes, *, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(
            axis=grid_axis, color=COLORS["grid"], linewidth=0.55, alpha=0.8, zorder=0
        )
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
    )


def save_figure(
    fig: plt.Figure, output_base: str | Path, *, dpi: int = 400
) -> tuple[Path, Path]:
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    return png, pdf


apply_style()
