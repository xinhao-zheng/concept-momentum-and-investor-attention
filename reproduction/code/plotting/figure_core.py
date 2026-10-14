"""Publication figures from aggregate inputs.

The figure index supplies publication captions. File routing is configured by engine.py.
"""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter
from matplotlib.text import Text
import numpy as np
import pandas as pd

from .figure_style import CONFIG as STYLE_CONFIG, PALETTE, DIV, SEQ

BLUE, ATTENTION, INK = PALETTE["primary"], PALETTE["attention"], PALETTE["text"]
SECONDARY, REF, GRID = PALETTE["secondary_text"], PALETTE["reference"], PALETTE["grid"]
AUX, FILL = PALETTE["secondary_series"], PALETTE["range_fill"]
NEGATIVE, NEUTRAL, POSITIVE = PALETTE["negative"], PALETTE["zero"], PALETTE["positive"]
PALE = FILL
H = [1, 3, 5, 10, 20, 60, 120, 250]
X = np.arange(8)
MANIFEST, CAPTIONS, SOURCES, CURRENT_USED = [], {}, {}, set()
ROOT = OUT = DATA = QA = None
FROZEN_CAPTION = {}


def read(key):
    CURRENT_USED.add(key)
    p = Path(SOURCES[key]["path"])
    assert sha(p) == SOURCES[key]["sha256"], key
    return (
        json.loads(p.read_text(encoding="utf-8"))
        if p.suffix == ".json"
        else pd.read_csv(p, float_precision="round_trip")
    )


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "axes.edgecolor": SECONDARY,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.25,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.unicode_minus": True,
            "mathtext.fontset": "dejavusans",
        }
    )


def canvas(height=110):
    CURRENT_USED.clear()
    return plt.figure(figsize=(183 / 25.4, height / 25.4))


def panel(ax, letter, title):
    ax.set_title(
        f"{letter}  {title}", loc="left", pad=10, fontsize=8.7, fontweight="bold"
    )


def zero(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=SECONDARY, lw=0.75, zorder=0)


def note(fig, text):
    fig.text(0.08, 0.024, text, fontsize=7.2, color=SECONDARY, va="bottom")


def grid(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, lw=0.5)


def heatmap(ax, arr, cmap, vmin, vmax, aspect="auto", grid_width=0):
    rows, cols = arr.shape
    mesh = ax.pcolormesh(
        np.arange(cols + 1) - 0.5,
        np.arange(rows + 1) - 0.5,
        arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        rasterized=False,
        edgecolors="white" if grid_width else "face",
        linewidth=grid_width,
    )
    ax.set(xlim=(-0.5, cols - 0.5), ylim=(rows - 0.5, -0.5))
    ax.set_aspect(aspect)
    return mesh


def heatmap_label_color(value, cmap, vmin, vmax):
    def luminance(rgb):
        rgb = np.asarray(rgb, float)
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return linear @ np.array([0.2126, 0.7152, 0.0722])

    bg = luminance(cmap(colors.Normalize(vmin, vmax)(value))[:3])
    fg = luminance(colors.to_rgb(INK))
    return "white" if 1.05 / (bg + 0.05) > (bg + 0.05) / (fg + 0.05) else INK


def continuous_vector_colorbar(cb):
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    # Opaque same-colour overlap removes PDF-viewer hairlines between vector cells.
    cb.solids.set_edgecolor("face")
    cb.solids.set_linewidth(1.0)
    cb.solids.set_antialiased(False)


def interval(ax, m, lo, hi, y, color=BLUE, marker="o", filled=True):
    ax.errorbar(
        m,
        y,
        xerr=[[m - lo], [hi - m]],
        fmt=marker,
        color=color,
        mfc=color if filled else "white",
        ms=4.3,
        lw=1.15,
        capsize=2.3,
        capthick=0.75,
    )


def legend_handle(color, marker, label, ls="-"):
    return Line2D(
        [], [], color=color, marker=marker, ms=4.2, lw=1.2, label=label, ls=ls
    )


def long_horizons(ax, start=5.5, label_y=0.98):
    ax.axvline(start, color=REF, lw=0.7, ls=":", zorder=0)
    ax.text(
        (start + 7.25) / 2,
        label_y,
        "Descriptive",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7,
        color=SECONDARY,
    )


def horizon(
    ax,
    y,
    color=BLUE,
    marker="o",
    label=None,
    supported=None,
    lo=None,
    hi=None,
    shift=0,
    ls="-",
):
    y = np.asarray(y, float)
    ok = np.ones(8, bool) if supported is None else np.asarray(supported, bool)
    xx = X + shift
    ax.plot(
        xx, np.where(ok, y, np.nan), color=color, ls=ls, lw=1.15, label=label, zorder=3
    )
    for k in range(7):
        if not (ok[k] and ok[k + 1]):
            ax.plot(xx[k : k + 2], y[k : k + 2], color=color, ls="--", lw=0.9, zorder=2)
    if lo is not None:
        low = np.asarray(lo, float)
        high = np.asarray(hi, float)
        ax.vlines(xx[ok], low[ok], high[ok], color=color, lw=0.85, zorder=2)
    ax.plot(xx[ok], y[ok], ls="none", marker=marker, ms=3.8, color=color, zorder=4)
    ax.plot(
        xx[~ok],
        y[~ok],
        ls="none",
        marker=marker,
        ms=4.3,
        mfc="white",
        mec=color,
        mew=1,
        zorder=4,
    )
    ax.set_xticks(X, H)
    ax.set_xlim(-0.35, 7.35)
    ax.set_xlabel("H (trading days)")


def record(stem, frame, suffix="plot_data"):
    frame.to_csv(DATA / f"{stem}_{suffix}.csv", index=False, float_format="%.17g")


def save(fig, stem, plot_count=None):
    # Materialize locator/formatter tick labels before applying the text floor.
    # Empty tick Text objects before the first draw otherwise escape this pass.
    fig.canvas.draw()
    # Publication typography: 8 pt at the manuscript's 183 mm figure width.
    # Fig8 is supplied at 190 mm, so its labels allow for the 183/190 embedding.
    floor = 8.4 if stem == "fig_08" else 8.0
    adjusted = []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text() and text.get_fontsize() < floor:
            adjusted.append(
                {
                    "text": text.get_text(),
                    "before_pt": text.get_fontsize(),
                    "after_pt": floor,
                }
            )
            text.set_fontsize(floor)
    (QA / (stem + "_typography.json")).write_text(
        json.dumps(
            {"font_floor_pt": floor, "adjusted": adjusted}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    outside = []
    skipped_ticks = set()
    # Matplotlib retains Text objects for locator ticks outside the view limits;
    # these are not rendered and must not produce false clipping failures.
    for ax in fig.axes:
        for axis, lim in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lower, upper = sorted(lim)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                if not lower - 1e-12 <= tick.get_loc() <= upper + 1e-12:
                    skipped_ticks.update([id(tick.label1), id(tick.label2)])
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text() or id(t) in skipped_ticks:
            continue
        bb = t.get_window_extent(renderer)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > w + 1 or bb.y1 > h + 1:
            outside.append(t.get_text())
    assert not outside, (stem, outside)
    outputs = {}
    for ext in ["pdf", "svg", "png"]:
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=450)
        outputs[ext] = {
            "path": str(p.relative_to(ROOT)),
            "sha256": sha(p),
            "bytes": p.stat().st_size,
        }
    fig.savefig(QA / f"{stem}_print_size.png", dpi=150)
    row = {
        "figure": stem,
        "width_mm": round(fig.get_figwidth() * 25.4, 3),
        "height_mm": round(fig.get_figheight() * 25.4, 3),
        "sources": sorted(CURRENT_USED),
        "outside_canvas": outside,
        "outputs": outputs,
        "plot_count": plot_count,
    }
    MANIFEST.append(row)
    CAPTIONS[stem] = dict(FROZEN_CAPTION)
    plt.close(fig)
    print(stem, flush=True)


def fig01():
    import sys
    from .fig01_design import render

    render(sys.modules[__name__])


def fig02():
    fig = canvas(127)
    d = read("parameter")
    d = d[d.missing.eq("complete")]
    pairs = [(3, 5), (3, 20), (5, 10), (10, 20), (20, 10), (20, 50)]
    hs = [3, 5, 10]
    ns = [5, 10, 20, 30, 50, 100]
    gs = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.975,
        bottom=0.15,
        top=0.90,
        wspace=0.35,
        hspace=0.60,
        height_ratios=[1.15, 1],
    )
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0])
    e = fig.add_subplot(gs[1, 1])
    pool = d[d.baseline.eq("pool")]
    arr = np.array(
        [
            [
                pool[
                    (pool.F == f) & (pool.W == w) & (pool.N == 20) & (pool.H == h)
                ].delta_pp.iloc[0]
                for h in hs
            ]
            for f, w in pairs
        ]
    )
    im = heatmap(a, arr, SEQ, 0, 0.8, grid_width=0.28)
    for (i, j), v in np.ndenumerate(arr):
        a.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color=heatmap_label_color(v, SEQ, 0, 0.8),
        )
    a.set_xticks(range(3), hs)
    a.set_yticks(range(6), [f"{f}/{w}" for f, w in pairs])
    a.set_xlabel("H (trading days)")
    a.set_ylabel("F / W (days)")
    a.tick_params(length=0)
    panel(a, "a", "Top20 spread over pool (pp)")
    cb = fig.colorbar(im, ax=a, fraction=0.035, pad=0.03)
    continuous_vector_colorbar(cb)
    cb.set_ticks([0, 0.4, 0.8])
    bases = ["pool", "F_only", "W_only", "A", "P"]
    rows = []
    b.set(xlim=(-0.5, 4.5), ylim=(5.5, -1.2))
    b.set_xticks([])
    b.set_yticks(range(6), [f"{f}/{w}" for f, w in pairs])
    b.tick_params(length=0)
    for j, label in enumerate(["Pool", "F only", "W only", "A", "P"]):
        b.text(j, -0.8, label, ha="center", va="center", fontsize=8, color=SECONDARY)
    for i, (f, w) in enumerate(pairs):
        q = d[(d.F == f) & (d.W == w) & (d.N == 20) & (d.H == 3)].set_index("baseline")
        for j, k in enumerate(bases):
            r = q.loc[k]
            v = 0 if abs(r.delta_pp) < 0.0005 else r.delta_pp
            label = f"{v:.3f}" + ("*" if bool(r.pass_Holm540) else "")
            b.text(j, i, label, ha="center", va="center", fontsize=8, color=INK)
            rows.append(
                {
                    "F": f,
                    "W": w,
                    "baseline": k,
                    "delta_pp": r.delta_pp,
                    "Holm540_pass": bool(r.pass_Holm540),
                }
            )
        b.axhline(i + 0.5, color=GRID, lw=0.45)
    for s in b.spines.values():
        s.set_visible(False)
    panel(b, "b", "H3 / Top20: direct differences (pp)")
    for ax, col, fcol, fval, vals, title in [
        (c, "N", "H", 3, ns, "Selection breadth at H3"),
        (e, "H", "N", 20, hs, "Holding window at Top20"),
    ]:
        low = []
        high = []
        ref = []
        for v in vals:
            q = pool[(pool[fcol] == fval) & (pool[col] == v)]
            low.append(q.delta_pp.min())
            high.append(q.delta_pp.max())
            ref.append(q[(q.F == 5) & (q.W == 10)].delta_pp.iloc[0])
        xx = np.arange(len(vals))
        ax.fill_between(xx, low, high, color=FILL, edgecolor=REF, lw=0.5)
        ax.plot(xx, ref, "o-", color=BLUE, ms=4, label="F5/W10")
        ax.set_xticks(xx, vals)
        zero(ax)
        ax.set_ylabel("Gross spread (pp)")
        ax.set_xlabel("Selected stocks N" if col == "N" else "H (trading days)")
        panel(ax, "c" if col == "N" else "d", title)
    c.legend(
        [
            Patch(facecolor=FILL, edgecolor=REF, linewidth=0.5),
            legend_handle(BLUE, "o", "F5/W10 reference"),
        ],
        ["Six-setting range", "F5/W10 reference"],
        loc="lower left",
        fontsize=8,
        labelspacing=0.45,
        handletextpad=0.6,
    )
    for ax in (a, b, c, e, cb.ax):
        ax.tick_params(labelsize=8)
    note(fig, "* Holm540 P < 0.05. Envelopes: minimum–maximum across six settings.")
    record("fig_02", d)
    record("fig_02", pd.DataFrame(rows), "panel_b")
    npos = int((pool.delta_pp > 0).sum())
    npass = int(pool.pass_Holm540.sum())
    assert len(pool) == 108 and arr.min() >= 0 and arr.max() <= 0.8
    save(fig, "fig_02", len(d))


def fig03():
    fig = canvas(103)
    d = read("matched")
    draws = read("draws")
    gs = fig.add_gridspec(
        1, 2, left=0.095, right=0.975, bottom=0.205, top=0.80, wspace=0.29
    )
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    quant = []
    specs = [
        ("industry", REF, "o", "Industry"),
        ("size", AUX, "s", "Size"),
        ("industry_x_size", BLUE, "D", "Industry × size"),
    ]
    for ax in [a, b]:
        ax.set(xlim=(-0.03, 0.46), ylim=(-0.6, 2.6))
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
        ax.set_yticks([2, 1, 0], ["Top5", "Top10", "Top20"])
        zero(ax, "x")
        grid(ax)
    for i, n in enumerate([5, 10, 20]):
        for off, (key, col, mark, label) in zip([0.21, 0, -0.21], specs):
            r = d[(d.N == n) & (d.benchmark == key)].iloc[0]
            interval(
                a,
                r.actual_minus_group_mean * 100,
                r.ci_low * 100,
                r.ci_high * 100,
                2 - i + off,
                col,
                mark,
            )
        r = d[(d.N == n) & (d.benchmark == "industry_x_size")].iloc[0]
        q = draws[(draws.N == n) & (draws.benchmark == "industry_x_size")]
        values = 100 * (r.actual_mean - q.random_mean.to_numpy())
        lo, med, hi = np.quantile(values, [0.05, 0.5, 0.95])
        assert len(values) == 1000
        parts = b.violinplot(
            values,
            positions=[2 - i],
            orientation="horizontal",
            widths=0.53,
            showextrema=False,
            showmedians=False,
            points=200,
            bw_method="scott",
        )
        for body in parts["bodies"]:
            body.set_facecolor(FILL)
            body.set_edgecolor(BLUE)
            body.set_linewidth(0.75)
            body.set_alpha(1)
        interval(b, med, lo, hi, 2 - i, BLUE, "D")
        b.text(0.335, 2 - i + 0.36, f"{med:.3f} pp", ha="right", fontsize=7.2)
        b.text(
            0.45,
            2 - i + 0.36,
            f"n = {int(r.dates):,}",
            ha="right",
            fontsize=7.0,
            color=SECONDARY,
        )
        quant.append(
            {
                "N": n,
                "q05_pp": lo,
                "median_pp": med,
                "q95_pp": hi,
                "paired_dates": r.dates,
                "draws": len(q),
            }
        )
    panel(a, "a", "Matched-control differences")
    panel(b, "b", "Conditional reference distribution")
    a.set_xlabel("Actual − matched-group mean (pp)")
    b.set_xlabel("Actual − random matched portfolio (pp)")
    fig.legend(
        handles=[legend_handle(c, m, l) for _, c, m, l in specs],
        loc="upper left",
        bbox_to_anchor=(0.089, 0.975),
        ncol=3,
        fontsize=7.1,
        columnspacing=1.15,
        handlelength=1.6,
        handletextpad=0.5,
    )
    note(
        fig,
        "a: pointwise 95% time intervals.  b: conditional density and 5th–95th percentiles, not a time CI.",
    )
    record("fig_03", d)
    record("fig_03", pd.DataFrame(quant), "conditional_quantiles")
    joint_counts = "/".join(
        f'{int(d[(d.N==n)&(d.benchmark=="industry_x_size")].iloc[0].dates):,}'
        for n in [5, 10, 20]
    )
    save(fig, "fig_03", 9)


def fig04():
    fig = canvas(123)
    old = read("component")
    new = read("aggregation")
    over = read("overlap")
    lev = read("component_levels")
    old = old[
        (old.design == "gamma")
        & (old.clock == "calendar")
        & (old.H == 3)
        & (old.top_n == 20)
        & (old.missing == "complete")
        & (old.segment == "FULL")
    ]
    new = new[
        (new.design == "gamma")
        & (new.H == 3)
        & (new.top_n == 20)
        & (new.missing_policy == "complete")
    ]
    a = fig.add_axes([0.14, 0.59, 0.32, 0.27])
    b = fig.add_axes([0.65, 0.59, 0.30, 0.27])
    rows = []
    for j, key in enumerate(["F_only", "W_only"]):
        r = old[old.baseline == key].iloc[0]
        interval(a, r.delta_pp, r.ci_low_pp, r.ci_high_pp, 1 - j, BLUE, ["o", "s"][j])
        rows.append(
            {
                "comparison": key,
                "mean": r.delta_pp,
                "lo": r.ci_low_pp,
                "hi": r.ci_high_pp,
            }
        )
    for j, key in enumerate(["additive_mean", "product_of_means"]):
        r = new[new.baseline == key].iloc[0]
        interval(
            b, r.mean_delta_pp, r.ci_low_pp, r.ci_high_pp, 1 - j, BLUE, ["o", "s"][j]
        )
        rows.append(
            {
                "comparison": key,
                "mean": r.mean_delta_pp,
                "lo": r.ci_low_pp,
                "hi": r.ci_high_pp,
            }
        )
    a.set_yticks([1, 0], ["F only", "W only"])
    b.set_yticks([1, 0], ["A: mean", "P: product"])
    a.set_xlim(-0.045, 0.40)
    b.set_xlim(-0.020, 0.025)
    for ax in [a, b]:
        ax.set_ylim(-0.5, 1.5)
        zero(ax, "x")
        grid(ax)
        ax.set_xlabel("FW − comparator, H3 (pp)")
    panel(a, "a", "Single components")
    panel(b, "b", "Same-information alternatives")
    a.text(
        0.02,
        0.95,
        "960-test family",
        transform=a.transAxes,
        fontsize=7.2,
        color=SECONDARY,
        va="top",
    )
    b.text(
        0.02,
        0.95,
        "192-test family",
        transform=b.transAxes,
        fontsize=7.2,
        color=SECONDARY,
        va="top",
    )
    c = fig.add_axes([0.14, 0.19, 0.32, 0.23])
    e = fig.add_axes([0.65, 0.19, 0.30, 0.23])
    panel(c, "c", "Mean Top20 overlap")
    for y, key, label in [
        (1, "additive_mean", "FW vs A"),
        (0, "product_of_means", "FW vs P"),
    ]:
        v = (
            over[(over.design == "gamma") & (over.top_n == 20) & (over.baseline == key)]
            .iloc[0]
            .mean_intersection_fraction
            * 100
        )
        c.barh(y, v, height=0.36, color=BLUE if y == 1 else SECONDARY)
        c.text(
            v - 3, y, f"{v:.2f}%", ha="right", va="center", fontsize=7.4, color="white"
        )
    c.set_yticks([1, 0], ["FW vs A", "FW vs P"])
    c.set(xlim=(0, 100), ylim=(-0.55, 1.55))
    c.set_xticks([0, 50, 100])
    c.set_xlabel("Shared stocks (%)")
    grid(c)
    panel(e, "d", "Pool-relative means")
    q = lev[
        (lev.design == "gamma")
        & (lev.clock == "calendar")
        & (lev.H == 3)
        & (lev.top_n == 20)
        & (lev.segment == "FULL")
    ].set_index("method")
    for y, key, col in [(2, "FW", BLUE), (1, "F_only", REF), (0, "W_only", SECONDARY)]:
        v = q.loc[key, "pool_excess_complete_pp"]
        e.barh(y, v, height=0.40, color=col)
        e.text(v + 0.008, y, f"{v:.4f}", ha="left", va="center", fontsize=7.2)
    e.set_yticks([2, 1, 0], ["FW", "F only", "W only"])
    e.set(xlim=(0, 0.37), ylim=(-0.55, 2.55))
    e.set_xticks([0, 0.1, 0.2, 0.3])
    e.set_xlabel("H3 gross spread (pp)")
    grid(e)
    note(
        fig,
        "Different x-axis ranges in a/b. Pointwise 95% intervals; c/d are descriptive.",
    )
    record("fig_04", pd.DataFrame(rows))
    record("fig_04", q.reset_index(), "levels")
    nm = int(old[old.baseline == "F_only"].n_dates.iloc[0])
    assert (
        nm
        == int(new[new.baseline == "additive_mean"].n_dates.iloc[0])
        == int(q.loc["FW", "n_dates"])
    )
    no = int(over[(over.design == "gamma") & (over.top_n == 20)].n_dates.iloc[0])
    save(fig, "fig_04", 4)


def fig05():
    fig = canvas(127)
    d = read("increment")
    gs = fig.add_gridspec(
        2, 2, left=0.10, right=0.97, bottom=0.14, top=0.86, hspace=0.58, wspace=0.34
    )
    specs = [
        ("IC", 0, "Overall ranking", "IC difference"),
        (
            "industry_conditional_IC",
            0,
            "Within-industry ranking",
            "Conditional IC difference",
        ),
        ("return", 20, "Top20 selection", "Gross return difference (pp)"),
        ("return", 100, "Top100 selection", "Gross return difference (pp)"),
    ]
    used = []
    for k, (metric, n, title, ylab) in enumerate(specs):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        for model, col, mark, off, ls in [
            ("ridge", BLUE, "o", -0.11, "-"),
            ("hist", ATTENTION, "s", 0.11, "-."),
        ]:
            q = (
                d[
                    (d.model == model)
                    & (d.metric == metric)
                    & (d.top_n == n)
                    & (d.missing == "complete")
                ]
                .set_index("H")
                .loc[H]
            )
            horizon(
                ax,
                q.delta,
                col,
                mark,
                supported=q.status == "INFERENCE_ELIGIBLE",
                lo=q.ci_low,
                hi=q.ci_high,
                shift=off,
                ls=ls,
            )
            used.append(q.reset_index())
        long_horizons(ax, label_y=1.10 if k == 2 else 0.98)
        zero(ax)
        ax.set_ylabel(ylab)
        ax.set_ylim((-0.014, 0.020) if k < 2 else (-1.5, 1.7))
        panel(ax, "abcd"[k], title)
        ax.tick_params(labelsize=8)
        for label in ax.texts:
            if label.get_text() == "Descriptive":
                label.set_fontsize(8)
    fig.legend(
        handles=[
            legend_handle(BLUE, "o", "Ridge", "-"),
            legend_handle(ATTENTION, "s", "Boosted trees", "-."),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.975),
        ncol=2,
        fontsize=8,
        columnspacing=2.5,
        handletextpad=0.65,
    )
    note(
        fig,
        "Vertical lines: pointwise 95% intervals. Open points: insufficient time support.",
    )
    record("fig_05", pd.concat(used, ignore_index=True))
    save(fig, "fig_05", 64)


def fig06():
    fig = canvas(126)
    a = read("attention_endpoint")
    b = read("attention_feedback")
    assert (
        len(b) == 72 and not b.duplicated(["attention", "trading", "holding_h"]).any()
    )
    n_joint = int(b.strict_feedback_candidate.sum())
    n_return = int((b.strict_feedback_candidate & b.trading.eq("RET")).sum())
    n_oneway_return = int((b.trading.eq("RET") & (b.A_to_T_max_t_fwer < 0.05)).sum())
    gs = fig.add_gridspec(
        2,
        3,
        left=0.10,
        right=0.98,
        bottom=0.22,
        top=0.89,
        hspace=0.64,
        wspace=0.18,
        height_ratios=[1, 1.6],
    )
    titles = ["T: total popularity", "S: watchlist popularity", "K: stickiness"]
    vmax = max(abs(b.A_to_T_beta).max(), 0.01)
    signmap = []
    for i, p in enumerate("TSK"):
        ax = fig.add_subplot(gs[0, i])
        q = a[a.proxy.eq(p)].copy()
        q = q[q.scale.eq("rank")] if "scale" in q else q
        q = q.set_index("holding_h").loc[H]
        horizon(
            ax,
            q.retained_fraction,
            ATTENTION,
            ["o", "s", "^"][i],
            supported=q.bootstrap_status.eq("COMPLETE"),
        )
        zero(ax)
        ax.axhline(1, color=GRID, lw=0.8, ls=":")
        ax.set_ylim(-0.17, 1.3)
        ax.set_yticks([0, 0.4, 0.8, 1.2])
        panel(ax, "abc"[i], titles[i])
        ax._left_title.set_fontsize(8.3)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_ylabel("Endpoint / current coefficient")
        ax2 = fig.add_subplot(gs[1, i])
        dd = b[b.attention.eq(p)]
        vals = np.array(
            [
                [
                    dd[(dd.trading == t) & (dd.holding_h == h)].A_to_T_beta.iloc[0]
                    for h in H
                ]
                for t in ["RET", "AMT", "TO"]
            ]
        )
        # Transpose the display, not the saved numerical table: three broad
        # outcome columns permit readable labels at the manuscript width.
        im = heatmap(ax2, vals.T, DIV, -vmax, vmax, grid_width=0.22)
        for y, t in enumerate(["RET", "AMT", "TO"]):
            for x, h in enumerate(H):
                r = dd[(dd.trading == t) & (dd.holding_h == h)].iloc[0]
                col = heatmap_label_color(vals[y, x], DIV, -vmax, vmax)
                scaled = 1000 * vals[y, x]
                display = (
                    f"{scaled:.2f}" if abs(scaled) < 0.1 else f"{scaled:.1f}"
                ).replace("-", "−")
                assert float(display.replace("−", "-")) != 0 or vals[y, x] == 0
                ax2.text(y, x, display, ha="center", va="center", fontsize=8, color=col)
                if bool(r.strict_feedback_candidate):
                    ax2.add_patch(
                        Rectangle(
                            (y - 0.46, x - 0.43),
                            0.92,
                            0.86,
                            fill=False,
                            ec=BLUE,
                            lw=0.72,
                        )
                    )
                signmap.append(
                    {
                        "proxy": p,
                        "outcome": t,
                        "H": h,
                        "coefficient": vals[y, x],
                        "coefficient_times_1000": scaled,
                        "display_label": display,
                        "sign": (
                            "negative"
                            if vals[y, x] < 0
                            else "positive" if vals[y, x] > 0 else "zero"
                        ),
                        "joint_candidate": bool(r.strict_feedback_candidate),
                    }
                )
        ax2.set_xticks([0, 1, 2], ["Return", "Value", "Turnover"])
        ax2.set_yticks(X, H)
        ax2.tick_params(length=0, labelsize=8)
        panel(ax2, "def"[i], f"{p} → future outcomes")
        ax2._left_title.set_fontsize(8.3)
        if i == 0:
            ax2.set_ylabel("H (trading days)")
        if i > 0:
            ax2.tick_params(labelleft=False)
        for sp in ax2.spines.values():
            sp.set_visible(False)
    cax = fig.add_axes([0.35, 0.115, 0.42, 0.015])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    continuous_vector_colorbar(cb)
    cb.set_ticks([-0.04, -0.02, 0, 0.02, 0.04], labels=["−40", "−20", "0", "20", "40"])
    cb.ax.tick_params(labelsize=8)
    cb.set_label(
        r"One-way standardized coefficient ($\times10^{-3}$)", fontsize=8, labelpad=3
    )
    note(fig, f"Outlined cells: joint criterion ({n_joint}/72; returns {n_return}/24).")
    record("fig_06", pd.DataFrame(signmap))
    record("fig_06", a, "endpoint_source")
    save(fig, "fig_06", 72)


def fig07():
    fig = canvas(120)
    d = read("bridge")
    ctx = read("context").set_index("metric")
    assert len(d) == 9 and not d.all_date_matching_gate_pass.any()
    gs = fig.add_gridspec(
        2,
        3,
        left=0.10,
        right=0.97,
        bottom=0.15,
        top=0.88,
        hspace=0.70,
        wspace=0.34,
        height_ratios=[0.8, 1],
    )
    a = fig.add_subplot(gs[0, :])
    rows = []
    for y, period in [(1, "past"), (0, "future")]:
        real = ctx.loc[f"{period}_real_corr", "equal_date_mean"]
        ref = ctx.loc[f"{period}_pseudo_corr", "equal_date_mean"]
        a.plot([ref, real], [y, y], color=GRID, lw=2)
        a.plot(ref, y, "s", mfc="white", mec=REF, ms=5, mew=1.1)
        a.plot(real, y, "o", mfc="white", mec=BLUE, ms=5, mew=1.1)
        a.text(ref, y - 0.19, f"{ref:.4f}", ha="center", va="top", fontsize=7.2)
        a.text(real, y - 0.19, f"{real:.4f}", ha="center", va="top", fontsize=7.2)
        rows.append({"period": period, "real": real, "pseudo": ref})
    a.set(xlim=(0, 0.24), ylim=(-0.45, 1.75))
    a.set_yticks([1, 0], ["Past", "Future"])
    a.set_xlabel("Stock–peer return correlation (H20)")
    panel(a, "a", "Concept comovement diagnostics")
    a.legend(
        handles=[
            Line2D(
                [],
                [],
                color=BLUE,
                marker="o",
                mfc="white",
                mec=BLUE,
                ms=5,
                mew=1.1,
                ls="none",
                label="Real concepts",
            ),
            Line2D(
                [],
                [],
                color=REF,
                marker="s",
                mfc="white",
                mec=REF,
                ms=5,
                mew=1.1,
                ls="none",
                label="Pseudo networks",
            ),
        ],
        loc="upper right",
        ncol=2,
        fontsize=8,
        columnspacing=1.6,
        handletextpad=0.45,
        borderaxespad=0.15,
    )
    lim = (
        max(abs(d.diagnostic_ci95_low).max(), abs(d.diagnostic_ci95_high).max()) * 1100
    )
    for i, (out, title) in enumerate(
        [
            ("past_real_minus_pseudo", "Past gap"),
            ("future_real_minus_pseudo", "Future gap"),
            ("future_minus_past", "Future − past"),
        ]
    ):
        ax = fig.add_subplot(gs[1, i])
        q = d[d.outcome.eq(out)].set_index("proxy").loc[list("TSK")]
        for j, (proxy, r) in enumerate(q.iterrows()):
            interval(
                ax,
                r.coefficient * 1000,
                r.diagnostic_ci95_low * 1000,
                r.diagnostic_ci95_high * 1000,
                2 - j,
                ATTENTION,
                ["o", "s", "^"][j],
                False,
            )
        ax.set(xlim=(-lim, lim), ylim=(-0.55, 2.55))
        ax.set_yticks([2, 1, 0], list("TSK"))
        zero(ax, "x")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        panel(ax, "bcd"[i], title)
        ax.set_xlabel(r"Correlation difference ($\times10^{-3}$)")
    nd = int(ctx.n_dates.iloc[0])
    nstock = int(ctx.n_common_stock_dates.iloc[0])
    ndiag = int((d.diagnostic_p_holm_9 < 0.05).sum())
    note(
        fig,
        f"All panels descriptive: matching failed on {nd}/{nd} dates; b–d intervals are diagnostic.",
    )
    record("fig_07", d)
    record("fig_07", pd.DataFrame(rows), "context")
    past_k = d.loc[
        (d.proxy == "K") & (d.outcome == "past_real_minus_pseudo"),
        "diagnostic_p_holm_9",
    ]
    assert len(past_k) == 1
    kp = float(past_k.iloc[0])
    save(fig, "fig_07", 9)


def fig08():
    import sys
    from .fig08_design import render

    render(sys.modules[__name__])


def s01():
    fig = canvas(154)
    d = read("aggregation")
    d = d[d.missing_policy.eq("complete")]
    bound = np.ceil(d.mean_delta_pp.abs().max() * 10) / 10
    gs = fig.add_gridspec(
        2, 2, left=0.105, right=0.975, bottom=0.23, top=0.92, wspace=0.30, hspace=0.42
    )
    for k, (design, base) in enumerate(
        [
            (l, b)
            for l in ["beta", "gamma"]
            for b in ["additive_mean", "product_of_means"]
        ]
    ):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        q = d[(d.design == design) & (d.baseline == base)]
        arr = (
            q.pivot(index="top_n", columns="H", values="mean_delta_pp")
            .loc[[5, 20, 100], H]
            .to_numpy()
            .T
        )
        ok = (
            q.pivot(index="top_n", columns="H", values="inference_supported")
            .loc[[5, 20, 100], H]
            .to_numpy(bool)
            .T
        )
        im = heatmap(ax, arr, DIV, -bound, bound, grid_width=0.22)
        for (y, x), v in np.ndenumerate(arr):
            digits = 4 if 0 < abs(v) < 0.0005 else 3
            ax.text(
                x,
                y,
                f"{v:.{digits}f}".replace("-", "−"),
                ha="center",
                va="center",
                fontsize=8,
                color=heatmap_label_color(v, DIV, -bound, bound),
            )
        # Encode one shared support region instead of repeating 24 cell borders.
        assert ok[:6, :].all() and not ok[6:, :].any()
        ax.add_patch(
            Rectangle(
                (-0.48, 5.52),
                2.96,
                1.96,
                fill=False,
                ec=SECONDARY,
                lw=0.65,
                ls=":",
                zorder=5,
            )
        )
        ax.set_xticks([0, 1, 2], ["Top5", "Top20", "Top100"])
        ax.set_yticks(X, H)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        panel(
            ax,
            "abcd"[k],
            f'{"β" if design=="beta" else "γ"}: FW − {"A" if base=="additive_mean" else "P"}',
        )
        ax.set_xlabel("Selection size")
        if k % 2 == 0:
            ax.set_ylabel("H (trading days)")
    cax = fig.add_axes([0.30, 0.105, 0.48, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    continuous_vector_colorbar(cb)
    cb.set_label("Complete-H gross difference (pp)", fontsize=7.5)
    note(fig, "Dotted region: limited time support. Colour: signed mean difference.")
    record("fig_s03", d)
    full = read("aggregation")
    nsup = int(full.inference_supported.sum())
    npass = int(full.family_pass_005.sum())
    save(fig, "fig_s03", 96)


def s02():
    fig = canvas(106)
    d = read("rank_reference")
    a = fig.add_axes([0.205, 0.24, 0.285, 0.57])
    b = fig.add_axes([0.655, 0.24, 0.30, 0.57])
    rows = []
    for block, off, col, mark in [(20, 0.10, ATTENTION, "o"), (40, -0.10, AUX, "s")]:
        for j, key in enumerate(
            ["baseline_float64", "same_pair_universe_reranked", "rerank_minus_baseline"]
        ):
            r = d[
                (d.metric == "S")
                & (d.H == 3)
                & (d.requested_block == block)
                & (d.endpoint == key)
            ].iloc[0]
            interval(
                a,
                r["mean"],
                r.pointwise_95ci_low,
                r.pointwise_95ci_high,
                2 - j + off,
                col,
                mark,
            )
            rows.append(r)
    a.set_yticks([2, 1, 0], ["Daily full sets", "Fixed pair set", "Paired difference"])
    a.set_ylim(-0.5, 2.5)
    a.set_xlabel("Rank correlation / change")
    zero(a, "x")
    panel(a, "a", "Watchlist change: H3")
    for key, col, mark, label in [
        ("baseline_float64", REF, "o", "Daily full sets"),
        ("same_pair_universe_reranked", ATTENTION, "s", "Fixed pair set"),
    ]:
        q = (
            d[(d.metric == "S") & (d.requested_block == 20) & (d.endpoint == key)]
            .set_index("H")
            .loc[H]
        )
        horizon(b, q["mean"], col, mark, label, supported=q.inference_eligible)
    long_horizons(b, 4.5, label_y=0.75)
    zero(b)
    b.set_ylabel("Mean rank correlation")
    panel(b, "b", "Reference-set sensitivity")
    b.legend(
        handles=[
            legend_handle(REF, "o", "Daily full sets"),
            legend_handle(ATTENTION, "s", "Fixed pair set"),
        ],
        loc="center right",
        fontsize=7,
    )
    fig.legend(
        handles=[
            legend_handle(ATTENTION, "o", "20-day blocks"),
            legend_handle(AUX, "s", "40-day blocks"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.20, 0.10),
        ncol=2,
        fontsize=7.4,
    )
    note(fig, "Same stock–dates and future ranks; attention-rank measurement.")
    record("fig_s08", d)
    n3 = int(
        d[
            (d.metric == "S")
            & (d.H == 3)
            & (d.requested_block == 20)
            & (d.endpoint == "rerank_minus_baseline")
        ].n_dates.iloc[0]
    )
    save(fig, "fig_s08", None)


def s03():
    fig = canvas(106)
    d = read("model_ic")
    r = read("model_returns")
    a = fig.add_axes([0.10, 0.225, 0.405, 0.575])
    b = fig.add_axes([0.705, 0.225, 0.265, 0.575])
    specs = [
        ("FW", BLUE, "D", "Original FW", "-"),
        ("ridge_ordinary", AUX, "o", "Ridge ordinary", "--"),
        ("ridge_concept", AUX, "s", "Ridge +concept", "-"),
        ("hist_ordinary", ATTENTION, "^", "Trees ordinary", "--"),
        ("hist_concept", ATTENTION, "v", "Trees +concept", "-"),
    ]
    shifts = {
        "FW": 0,
        "ridge_ordinary": -0.15,
        "ridge_concept": -0.05,
        "hist_ordinary": 0.05,
        "hist_concept": 0.15,
    }
    q = r[(r.H == 3) & (r.top_n == 20)].set_index("method")
    for j, (method, col, mark, label, ls) in enumerate(specs):
        z = d[d.method == method].set_index("H").loc[H]
        horizon(
            a,
            z.mean_IC,
            col,
            mark,
            supported=np.array(H) < 120,
            ls=ls,
            shift=shifts[method],
        )
        v = q.loc[method, "excess_pool"] * 100
        b.plot(v, 4 - j, marker=mark, color=col, ms=4.8)
        b.text(
            v + (0.012 if v >= 0 else -0.012),
            4 - j + 0.16,
            f"{v:.3f}",
            ha="left" if v >= 0 else "right",
            fontsize=7,
        )
    a.set_ylabel("Mean full-pool RankIC")
    zero(a)
    long_horizons(a, label_y=0.60)
    panel(a, "a", "Whole-pool ranking")
    b.set_yticks(range(4, -1, -1), [x[3] for x in specs])
    b.set(xlim=(-0.10, 0.37), ylim=(-0.6, 4.7))
    zero(b, "x")
    grid(b)
    b.set_xlabel("H3 Top20 − pool (pp)")
    panel(b, "b", "Concentrated selection")
    fig.legend(
        handles=[legend_handle(c, m, l, ls) for _, c, m, l, ls in specs],
        loc="upper left",
        bbox_to_anchor=(0.085, 0.975),
        ncol=3,
        fontsize=7.2,
        columnspacing=1.4,
        handlelength=1.7,
    )
    note(fig, "Descriptive model levels on common evaluation dates.")
    record("fig_s06", d, "ic")
    record("fig_s06", r, "returns")
    nmod = int(q.loc["FW", "n_dates"])
    nrule = int(
        read("component_levels")
        .query(
            "segment=='FULL' and design=='gamma' and clock=='calendar' and H==3 and top_n==20 and method=='FW'"
        )
        .n_dates.iloc[0]
    )
    save(fig, "fig_s06", 40)


def unavailable(ax, detail):
    ax.set_facecolor(PALE)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(
        0.5,
        0.62,
        "No formal estimate",
        transform=ax.transAxes,
        ha="center",
        fontweight="bold",
        fontsize=8,
    )
    ax.text(
        0.5,
        0.39,
        detail,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.5,
        linespacing=1.5,
    )
    ax.set_xticks(X, H)
    ax.set_xlim(-0.4, 7.4)
    ax.set_xlabel("H (trading days)")


def c1_line(ax, q, color, marker, shift=0, ls="-", filled=True, deferred_label=True):
    ok = q.estimability_status.eq("COMPLETE").to_numpy()
    assert q.loc[~ok, ["coefficient", "ci95_low", "ci95_high"]].isna().all().all()
    y = q.coefficient.to_numpy() * 100
    lo = q.ci95_low.to_numpy() * 100
    hi = q.ci95_high.to_numpy() * 100
    xx = X + shift
    ax.plot(
        xx,
        np.where(ok, y, np.nan),
        marker=marker,
        color=color,
        mfc=color if filled else "white",
        mec=color,
        mew=0.9,
        ms=3.6,
        lw=1.15,
        ls=ls,
    )
    ax.vlines(xx[ok], lo[ok], hi[ok], color=color, lw=0.85)
    ax.set_xticks(X, H)
    ax.set_xlim(-0.35, 7.45)
    ax.set_xlabel("H (trading days)")
    zero(ax)
    if deferred_label and not ok[-1] and not getattr(ax, "_has_deferred_label", False):
        ax.axvline(6.5, color=REF, ls=":", lw=0.6)
        ax.text(
            7,
            0.95,
            "H250\ndeferred",
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=6.9,
            bbox=dict(facecolor="white", edgecolor="none", pad=1),
        )
        ax._has_deferred_label = True
    return int(ok.sum())


def causal_windows():
    return read("causal_protocol"), read("causal_s_protocol")


def date_words(value):
    return pd.Timestamp(value).strftime("%d %B %Y").lstrip("0")


def s04():
    fig = canvas(134)
    all_d = read("dml")
    all_w = read("dml_s_window")
    d = all_d[all_d.lens.eq("gamma")]
    w = all_w[all_w.lens.eq("gamma")]
    assert len(d) == len(w) == 48 and w.protocol_branch.eq("S_coverage_window").all()
    original, window = causal_windows()
    ncomplete = int(d.estimability_status.eq("COMPLETE").sum())
    nwin = int(w.estimability_status.eq("COMPLETE").sum())
    s_ok = int(
        w[w.exposure.str.contains("_S")].estimability_status.eq("COMPLETE").sum()
    )
    both = pd.concat([d, w])
    ok = both.estimability_status.eq("COMPLETE")
    ylo = min(-1.4, np.floor(both.loc[ok, "ci95_low"].min() * 100 * 10) / 10)
    yhi = max(1.22, np.ceil(both.loc[ok, "ci95_high"].max() * 100 * 10) / 10 + 0.1)
    gs = fig.add_gridspec(
        2, 3, left=0.105, right=0.975, bottom=0.16, top=0.86, wspace=0.35, hspace=0.56
    )
    specs = [
        ("own_T", "Own total popularity"),
        ("own_S", "Own watchlist"),
        ("own_K", "Own stickiness"),
        ("CategoryShare_T_loo", "Peer total-popularity share"),
        ("CategoryShare_S_loo", "Peer watchlist share"),
        ("CategoryShare_K_loo", "Peer stickiness share"),
    ]
    count = 0
    for k, (key, title) in enumerate(specs):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        panel(ax, "abcdef"[k], title)
        q = d[d.exposure.eq(key)].set_index("holding_h").loc[H]
        r = w[w.exposure.eq(key)].set_index("holding_h").loc[H]
        if q.estimability_status.eq("COMPLETE").any():
            count += c1_line(ax, q, BLUE, "o", -0.13, "-")
        else:
            ax.text(
                0.045,
                0.965,
                "Full-history window:\nno S record before 2022",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                color=SECONDARY,
                linespacing=1.3,
            )
        count += c1_line(
            ax, r, ATTENTION, "s", 0.13, "--", filled=False, deferred_label=False
        )
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        if k % 3 == 0:
            ax.set_ylabel("Coefficient per past SD (pp)")
    fig.legend(
        handles=[
            legend_handle(BLUE, "o", "Full-history window: six test folds"),
            Line2D(
                [],
                [],
                color=ATTENTION,
                marker="s",
                mfc="white",
                mec=ATTENTION,
                ms=4.2,
                lw=1.2,
                ls="--",
                label="S-covered window: four test folds",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=1,
        fontsize=7.2,
    )
    note(
        fig,
        "Unplotted horizons lack temporal support.\nResidual-association diagnostics; pointwise 95% date-HAC intervals.",
    )
    record("fig_s10", pd.concat([d.assign(protocol_branch="original"), w]))
    all_est = pd.concat([all_d, all_w])
    npass = int(
        (
            all_est.estimability_status.eq("COMPLETE") & (all_est.p_holm_family < 0.05)
        ).sum()
    )
    holm_text = (
        "No coefficient in either window passes its diagnostic 24-test Holm family."
        if npass == 0
        else f"{npass} coefficients across both windows and all three designs pass their 24-test Holm families; related specifications are not independent findings."
    )
    save(fig, "fig_s10", count)


def status_table(ax, title, header, rows, footnote):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, 1.0, title, fontsize=8.7, fontweight="bold", va="top")
    xs = [0, 0.36, 0.68]
    y0 = 0.78
    for x, hd in zip(xs, header):
        ax.text(x, y0, hd, fontsize=7.4, color=SECONDARY, va="center")
    ax.plot([0, 1], [y0 - 0.08, y0 - 0.08], color=GRID, lw=0.7)
    for i, row in enumerate(rows):
        y = y0 - 0.22 - i * 0.2
        for x, cell in zip(xs, row):
            ax.text(x, y, cell, fontsize=7.4, va="center", color=INK)
        ax.plot([0, 1], [y - 0.1, y - 0.1], color=GRID, lw=0.5)
    ax.text(
        0,
        y0 - 0.22 - len(rows) * 0.2 + 0.02,
        footnote,
        fontsize=6.9,
        color=SECONDARY,
        va="top",
    )


def clip_range(frame, state):
    v = (
        frame[
            (frame.state_contrast == state)
            & frame.holding_h.isin([1, 3, 5, 10, 20, 60])
        ].pooled_propensity_clipped_fraction
        * 100
    )
    return f"{v.min():.0f}–{v.max():.0f}%"


def s05():
    fig = canvas(110)
    d = read("aipw")
    w = read("aipw_s_window")
    original, window = causal_windows()
    ncomplete = int(d.estimability_status.eq("COMPLETE").sum())
    assert len(d) == 96 and len(w) == 96
    nwin = int(w.estimability_status.eq("COMPLETE").sum())
    for frame in (d, w):
        assert (
            not frame[
                frame.state_contrast.isin(["S_high_vs_low", "TSK_all_high_vs_all_low"])
            ]
            .estimability_status.eq("COMPLETE")
            .any()
        )
    ok = d.estimability_status.eq("COMPLETE")
    ylo = min(-3.8, np.floor(d.loc[ok, "ci95_low"].min() * 100 * 10) / 10)
    yhi = max(1.35, np.ceil(d.loc[ok, "ci95_high"].max() * 100 * 10) / 10 + 0.1)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.975,
        bottom=0.11,
        top=0.84,
        wspace=0.28,
        hspace=0.40,
        height_ratios=[1, 0.63],
    )
    specs = [
        ("T_high_vs_low", "Total popularity: high − low"),
        ("K_high_vs_low", "Stickiness: high − low"),
    ]
    lenses = [
        ("alpha", REF, "o", "α: close-to-close", ":"),
        ("beta", BLUE, "s", "β: synchronous", "-"),
        ("gamma", ATTENTION, "^", "γ: lagged", "--"),
    ]
    count = 0
    for k, (key, title) in enumerate(specs):
        ax = fig.add_subplot(gs[0, k])
        panel(ax, "ab"[k], title)
        for j, (lens, col, mark, label, ls) in enumerate(lenses):
            q = (
                d[(d.state_contrast == key) & (d.lens == lens)]
                .set_index("holding_h")
                .loc[H]
            )
            count += c1_line(ax, q, col, mark, (j - 1) * 0.17, ls)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel("Adjusted difference (pp)")
    fig.legend(
        handles=[legend_handle(c, m, l, ls) for _, c, m, l, ls in lenses],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.975),
        ncol=3,
        fontsize=7.5,
    )
    ax = fig.add_subplot(gs[1, :])
    status_table(
        ax,
        "c  States without a formal estimate",
        ["State contrast", "Full-history window", "S-covered window"],
        [
            [
                "Watchlist S: high − low",
                f'{clip_range(d,"S_high_vs_low")} clipped; early training lacks S',
                f'{clip_range(w,"S_high_vs_low")} clipped',
            ],
            [
                "Joint T/S/K: all high − all low",
                f'{clip_range(d,"TSK_all_high_vs_all_low")} clipped; early training lacks S',
                f'{clip_range(w,"TSK_all_high_vs_all_low")} clipped',
            ],
        ],
        "Share of test propensities clipped to [0.05, 0.95], pooled over H1–H60 and α/β/γ.\nOverlap criterion: at most 20% clipped; these contrasts exceed it.",
    )
    note(
        fig,
        "Pointwise 95% date-HAC intervals. Symbols identify specifications, not corrected significance.",
    )
    record("fig_s11", pd.concat([d.assign(protocol_branch="original"), w]))
    npass = int((pd.concat([d, w]).p_holm_family < 0.05).sum())
    holm_text = (
        "No state contrast in either window passes its 32-test Holm family."
        if npass == 0
        else f"{npass} state contrasts across both windows pass their 32-test Holm families; these related results are not independent findings."
    )
    save(fig, "fig_s11", count)


def s06():
    fig = canvas(107)
    d = read("daily_comparators")
    d = d[(d.design == "gamma") & (d.N == 20) & (d.missing == "complete")]
    a = fig.add_axes([0.12, 0.23, 0.34, 0.59])
    b = fig.add_axes([0.61, 0.23, 0.35, 0.59])
    specs = [
        ("CM5", BLUE, "o", "-"),
        ("CM20", REF, "s", ":"),
        ("BC10", SECONDARY, "^", "-."),
        ("BC60", AUX, "D", (0, (5, 1, 1, 1))),
    ]
    for j, (m, col, mark, ls) in enumerate(specs):
        q = d[d.baseline.eq(m)].set_index("H").loc[H]
        r = q.loc[3]
        interval(a, r.delta_pp, r.lo_pp, r.hi_pp, 3 - j, col, mark)
        horizon(b, q.delta_pp, col, mark, m, q.supported, ls=ls)
    a.set_yticks([3, 2, 1, 0], [m[0] for m in specs])
    a.set(xlim=(-0.10, 1.65), ylim=(-0.6, 3.6))
    zero(a, "x")
    grid(a)
    a.set_xlabel("FW − comparator, H3 (pp)")
    panel(a, "a", "Short-window paired contrasts")
    zero(b)
    long_horizons(b, label_y=0.40)
    b.set_ylabel("FW − comparator (pp)")
    panel(b, "b", "All eight horizons")
    b.legend(
        handles=[legend_handle(c, m, key, ls) for key, c, m, ls in specs],
        loc="upper left",
        ncol=2,
        fontsize=7.2,
    )
    note(
        fig,
        "Literature-inspired adaptations. Open long-horizon estimates are descriptive only.",
    )
    record("fig_s04", d)
    h3 = d[d.H.eq(3)].set_index("baseline")
    nd = int(h3.n_dates.iloc[0])
    pv = (
        ", ".join(f'{h3.loc[m,"p_holm_384"]:.5f}' for m in ["CM5", "CM20", "BC10"])
        + f' and {h3.loc["BC60","p_holm_384"]:.5f}'
    )
    save(fig, "fig_s04", 32)


def s07():
    import sys
    from .monthly_momentum import render

    render(sys.modules[__name__])


def s08():
    fig = canvas(94)
    d = read("decomposition")
    d = (
        d[d.scope.eq("current_past_common_no_future_filter")]
        .set_index("metric")
        .loc[list("TSK")]
    )
    axes = [
        fig.add_axes(r)
        for r in [
            [0.105, 0.33, 0.22, 0.52],
            [0.435, 0.33, 0.21, 0.52],
            [0.77, 0.33, 0.20, 0.52],
        ]
    ]
    a, b, c = axes
    specs = [
        ("full", ATTENTION, -0.17, "o", "Full"),
        ("common", AUX, 0, "s", "Common set"),
        ("reference", REF, 0.17, "^", "Reference"),
    ]
    for key, col, off, mark, label in specs:
        a.plot(
            np.arange(3) + off,
            d[f"date_mean_std_{key}"],
            ls="none",
            marker=mark,
            color=col,
            ms=4.7,
            label=label,
        )
    a.set_yscale("log")
    a.set_ylim(0.00006, 0.8)
    a.set_ylabel("Cross-sectional SD (log scale)")
    panel(a, "a", "Component magnitude")
    v = d.date_mean_strict_sign_flip_fraction * 100
    b.bar(range(3), v, color=ATTENTION, width=0.40)
    for j, value in enumerate(v):
        b.text(j, value + 0.28, f"{value:.2f}%", ha="center", fontsize=7.3)
    b.set_ylim(0, 8.6)
    b.set_ylabel("Sign-flip share (%)")
    panel(b, "b", "Near-zero sign changes")
    for n, off, mark in [(20, -0.08, "s"), (100, 0.08, "^")]:
        c.plot(
            np.arange(3) + off,
            d[f"date_mean_top{n}_overlap_fraction"] * 100,
            ls="none",
            marker=mark,
            color=ATTENTION,
            ms=4.6,
            label=f"Top{n}",
        )
    c.set_ylim(99.15, 100.10)
    c.set_yticks([99.2, 99.6, 100])
    c.set_ylabel("Selection overlap (%)")
    panel(c, "c", "Extreme-list overlap")
    c.legend(
        loc="upper left",
        bbox_to_anchor=(-0.03, -0.22),
        ncol=1,
        fontsize=8,
        handlelength=1.1,
        handletextpad=0.45,
        borderaxespad=0,
        labelspacing=0.3,
    )
    for ax in axes:
        ax.set_xticks(range(3), list("TSK"))
        ax.set_xlim(-0.5, 2.5)
    a.legend(
        handles=[
            Line2D([], [], color=col, marker=mark, ls="none", ms=4.7, label=label)
            for _, col, _, mark, label in specs
        ],
        loc="upper left",
        bbox_to_anchor=(-0.025, -0.22),
        ncol=2,
        fontsize=7.1,
        handlelength=1.1,
        handletextpad=0.45,
        columnspacing=0.9,
        borderaxespad=0,
        labelspacing=0.3,
    )
    ndt = int(d.n_dates.iloc[0])
    note(
        fig,
        f"Descriptive means over {ndt:,} dates. a: log scale. c: axis starts at 99.15%.",
    )
    record("fig_s09", d.reset_index())
    save(fig, "fig_s09", 3)


def s09():
    fig = canvas(138)
    raw = read("legacy_ic")
    z = read("legacy_icz")
    g = [1, 2, 3, 5, 10, 20, 60, 120, 250]
    for q in [raw, z]:
        assert (
            len(q) == 243
            and not q.duplicated(["lens", "formation_d", "holding_d"]).any()
        )
    signed_values = []
    xs = [0.09, 0.36, 0.63]
    width = 0.22
    height = width * 183 / 138
    for i, (y, d, metric, vmax, label) in enumerate(
        [
            (0.55, raw, "raw_ic_mean", 0.05, "Raw IC: time mean"),
            (0.12, z, "icz_mean", 1.0, "Nominal count-scaled IC-z, descriptive"),
        ]
    ):
        fig.text(0.09, y + height + 0.035, label, fontweight="bold", fontsize=8.5)
        for j, (x, lens) in enumerate(zip(xs, ["alpha", "beta", "gamma"])):
            ax = fig.add_axes([x, y, width, height])
            arr = (
                d[d.lens == lens]
                .pivot(index="formation_d", columns="holding_d", values=metric)
                .loc[g, g]
                .to_numpy()
            )
            assert (
                np.isfinite(arr).all() and np.abs(arr).max() <= vmax
            ), "S9 colour scale clips a saved value"
            im = heatmap(ax, arr, DIV, -vmax, vmax, aspect="equal", grid_width=0.35)
            for (r, c), v in np.ndenumerate(arr):
                # An exact-sign cue supplements the diverging hue in grayscale.
                # Short rules remain legible without crowding the 486 cells.
                if v < 0:
                    ax.plot(
                        [c - 0.13, c + 0.13],
                        [r, r],
                        color=heatmap_label_color(v, DIV, -vmax, vmax),
                        lw=0.65,
                        solid_capstyle="butt",
                    )
                signed_values.append(
                    {
                        "panel": "abcdef"[i * 3 + j],
                        "metric": metric,
                        "lens": lens,
                        "formation_d": g[r],
                        "holding_d": g[c],
                        "value": v,
                        "sign": (
                            "negative" if v < 0 else "positive" if v > 0 else "zero"
                        ),
                        "scale_min": -vmax,
                        "scale_max": vmax,
                    }
                )
            ax.set_xticks(range(9), g)
            ax.set_yticks(range(9), g)
            ax.tick_params(length=0, pad=2, labelsize=7)
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )
            ax.set_xlabel("H (trading days)", fontsize=7.3)
            if j == 0:
                ax.set_ylabel("F (trading days)", fontsize=7.3)
            else:
                ax.tick_params(labelleft=False)
            ax.text(
                -0.11,
                1.02,
                "abcdef"[i * 3 + j],
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=9,
            )
            if i == 0:
                fig.text(
                    x + width / 2,
                    0.966,
                    ["α", "β", "γ"][j],
                    ha="center",
                    fontweight="bold",
                    fontsize=8.3,
                )
                fig.text(
                    x + width / 2,
                    0.934,
                    [
                        "Dynamic concept returns",
                        "Synchronous / open returns",
                        "Lagged membership / open returns",
                    ][j],
                    ha="center",
                    fontsize=6.9,
                    color=SECONDARY,
                )
            for sp in ax.spines.values():
                sp.set_visible(False)
        cax = fig.add_axes([0.90, y, 0.015, height])
        cb = fig.colorbar(im, cax=cax)
        continuous_vector_colorbar(cb)
        cb.set_ticks([-vmax, 0, vmax])
        cb.ax.tick_params(labelsize=7)
    note(
        fig,
        "Colour: signed magnitude. Minus marks: negative cells. Each design uses its own valid support.",
    )
    record("fig_s01", raw, "raw_ic")
    record("fig_s01", z, "nominal_icz")
    assert len(signed_values) == 486
    record("fig_s01", pd.DataFrame(signed_values), "signed_value_check")
    save(fig, "fig_s01", 486)


def s11():
    fig = canvas(100)
    d = read("annual_stability")
    d = d[d.year.between(2017, 2026)]
    gs = fig.add_gridspec(
        1, 2, left=0.10, right=0.975, bottom=0.26, top=0.84, wspace=0.22
    )
    for k, (benchmark, title, col, marker) in enumerate(
        [
            ("pool", "FW − research pool", BLUE, "o"),
            ("industry_x_size", "FW − industry × size", SECONDARY, "s"),
        ]
    ):
        ax = fig.add_subplot(gs[0, k])
        q = d[d.benchmark.eq(benchmark)].set_index("year").loc[range(2017, 2027)]
        ax.axvline(8.5, color=REF, ls=":", lw=0.8)
        zero(ax)
        grid(ax, "y")
        ax.plot(
            np.arange(9), q.mean_pp.iloc[:9], ls="none", marker=marker, ms=5, color=col
        )
        ax.plot(
            9,
            q.mean_pp.iloc[9],
            ls="none",
            marker=marker,
            ms=5,
            mfc="white",
            mec=col,
            mew=1.15,
        )
        ax.set_xticks(
            np.arange(10), [str(x)[2:] for x in range(2017, 2026)] + ["26\nYTD"]
        )
        ax.set(xlim=(-0.6, 9.6), ylim=(-0.13, 0.72))
        ax.set_xlabel("Year (20xx)")
        ax.set_yticks([-0.1, 0, 0.2, 0.4, 0.6])
        panel(ax, "ab"[k], title)
        if k == 0:
            ax.set_ylabel("Mean H3 gross difference (pp)")
        else:
            ax.tick_params(labelleft=False)
        for j in [5, 7, 9]:
            val = q.mean_pp.iloc[j]
            ax.text(
                j,
                val + (0.043 if val >= 0 else -0.046),
                f"{val:.3f}",
                ha="center",
                fontsize=7.2,
                color=col,
            )
    y26 = d[(d.year == 2026) & (d.benchmark == "pool")].iloc[0]
    n26 = int(y26.n_dates)
    last26 = pd.Timestamp(y26["last"])
    lastlab = f"{last26.day} {last26:%B}"
    note(
        fig,
        f"Equal-date annual means; no confidence intervals. 2026 YTD: through {lastlab} ({n26} dates).",
    )
    record("fig_s02", d)
    save(fig, "fig_s02", 20)


def s12():
    fig = canvas(121)
    d = read("dense_increment")
    assert len(d) == 24
    t = d[(d.model == "hist") & (d.metric == "industry_IC")].set_index("block_length")
    ptree20, ptree60 = float(t.loc[20, "p_holm24"]), float(t.loc[60, "p_holm24"])
    est = float(t.loc[20, "estimate"])
    nd = int(d.n_dates.iloc[0])
    d0, d1 = pd.Timestamp(d.start.iloc[0]), pd.Timestamp(d.end.iloc[0])
    gs = fig.add_gridspec(
        2, 2, left=0.14, right=0.975, bottom=0.16, top=0.86, hspace=0.60, wspace=0.32
    )
    specs = [
        ("raw_IC", 0, "Overall ranking", "IC difference"),
        ("industry_IC", 0, "Within-industry ranking", "Conditional IC difference"),
        ("return", 20, "Top20 selection", "Gross return difference (pp)"),
        ("return", 100, "Top100 selection", "Gross return difference (pp)"),
    ]
    plotted = []
    for k, (metric, n, title, label) in enumerate(specs):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        zero(ax, "x")
        grid(ax)
        for j, (model, marker) in enumerate([("ridge", "o"), ("hist", "s")]):
            q = (
                d[
                    (d.model == model)
                    & (d.metric == metric)
                    & (d.top_n == n)
                    & (d.missing == "complete")
                ]
                .set_index("block_length")
                .loc[[20, 60]]
            )
            assert q.estimate.nunique() == 1
            y = 1 - j
            value = q.estimate.iloc[0]
            for block, offset, col, ls in [
                (20, 0.095, BLUE, "-"),
                (60, -0.095, REF, "--"),
            ]:
                r = q.loc[block]
                ax.hlines(y + offset, r.lower95, r.upper95, color=col, lw=1.2, ls=ls)
                ax.vlines(
                    [r.lower95, r.upper95],
                    y + offset - 0.035,
                    y + offset + 0.035,
                    color=col,
                    lw=0.7,
                )
            ax.vlines(value, y - 0.095, y + 0.095, color=BLUE, lw=0.7)
            ax.plot(value, y, marker=marker, color=BLUE, ms=4.8)
            ax.text(
                value,
                y + 0.21,
                f"{value:.4f}" if k < 2 else f"{value:.3f}",
                ha="center",
                fontsize=7.1,
                color=SECONDARY,
            )
            plotted.append(q.reset_index())
        ax.set_ylim(-0.42, 1.49)
        ax.set_yticks([1, 0], ["Ridge", "Boosted trees"])
        ax.set_xlabel(label)
        ax.set_xlim((-0.007, 0.0115) if k < 2 else (-0.13, 0.25))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        panel(ax, "abcd"[k], title)
        if k == 1:
            ax.text(
                0.98,
                0.025,
                f"Trees: Holm24 P = {ptree20:.4f} / {ptree60:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.2,
                color=SECONDARY,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
            )
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=1.2, label="L = 20 dates"),
            Line2D([], [], color=REF, lw=1.2, ls="--", label="L = 60 dates"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.55, 0.975),
        ncol=2,
        fontsize=7.6,
    )
    note(
        fig,
        "One estimate per model; two pointwise 95% block-bootstrap intervals. Historical sensitivity.",
    )
    record("fig_s07", pd.concat(plotted, ignore_index=True))
    save(fig, "fig_s07", 16)
