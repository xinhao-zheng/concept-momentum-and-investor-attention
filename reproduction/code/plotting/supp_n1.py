"""Render the nine frozen N1 contrasts; no estimator or fitting code is invoked."""

from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

KEY = "fig_s12"


def file_sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def render(root, language, data_root, caption_record, palette_path):
    out = root / "figures"
    data = Path(data_root)
    qa = root / "qa"
    out.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(data / "conditional_specifications.csv")
    full = pd.read_csv(data / "paired_differences_holm27.csv")
    (root / "data").mkdir(parents=True, exist_ok=True)
    d.to_csv(
        root / "data" / "conditional_specifications.csv",
        index=False,
        float_format="%.17g",
    )
    assert (
        len(d) == 9
        and d.hac_lag_sample_dates.eq(8).all()
        and full.p_holm27.min() > 0.05
    )
    palette = json.loads(Path(palette_path).read_text("utf-8"))["colors"]
    zh = language == "zh"
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "Arial"] if zh else ["Arial"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": palette["secondary_text"],
            "text.color": palette["text"],
            "axes.labelcolor": palette["text"],
            "xtick.color": palette["secondary_text"],
            "ytick.color": palette["secondary_text"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(183 / 25.4, 87 / 25.4))
    gs = fig.add_gridspec(
        1, 3, left=0.225, right=0.978, bottom=0.235, top=0.74, wspace=0.26
    )
    contrasts = ["M1-M0", "M2-M1", "M3-M2"]
    rowlabels = (
        ["M1 − M0\n支持与成熟约束", "M2 − M1\n移除概念控制", "M3 − M2\n控制前移"]
        if zh
        else [
            "M1 − M0\nSupport and maturity",
            "M2 − M1\nRemove concept controls",
            "M3 − M2\nEarlier conventional controls",
        ]
    )
    titles = (
        ["总人气 T", "关注度 S", "粘性 K"]
        if zh
        else ["Total popularity T", "Watchlist S", "Stickiness K"]
    )
    for k, field in enumerate(["own_T", "own_S", "own_K"]):
        ax = fig.add_subplot(gs[0, k])
        q = d[d.exposure.eq(field)].set_index("contrast").loc[contrasts]
        ax.axvline(0, color=palette["reference"], lw=0.8, ls=(0, (2, 2)), zorder=1)
        for j, mark in enumerate(["o", "s", "D"]):
            r = q.iloc[j]
            y = 2 - j
            ax.hlines(
                y,
                r.ci95_low_pp,
                r.ci95_high_pp,
                color=palette["primary"],
                lw=1.25,
                zorder=2,
            )
            ax.vlines(
                [r.ci95_low_pp, r.ci95_high_pp],
                y - 0.05,
                y + 0.05,
                color=palette["primary"],
                lw=0.8,
            )
            ax.plot(
                r.difference_pp,
                y,
                marker=mark,
                ms=4.5,
                color=palette["primary"],
                ls="none",
                zorder=3,
            )
        ax.set_xlim(-0.12, 0.075)
        ax.set_ylim(-0.48, 2.48)
        ax.set_yticks([2, 1, 0], rowlabels if k == 0 else ["", "", ""])
        ax.set_xticks([-0.1, -0.05, 0, 0.05], ["−0.10", "−0.05", "0", "0.05"])
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", length=3)
        for side in ["left", "right", "top"]:
            ax.spines[side].set_visible(False)
        ax.set_title("abc"[k] + "  " + titles[k], loc="left", pad=12, fontweight="bold")
        ax.set_xlabel(
            "配对系数差异（百分点）" if zh else "Paired difference (pp)", labelpad=7
        )
        ax.set_axisbelow(True)
    fig.text(
        0.225,
        0.965,
        "共同 γ/H3 评价样本" if zh else "Common γ/H3 evaluation sample",
        va="top",
        fontsize=9.1,
        fontweight="bold",
    )
    fig.text(
        0.225,
        0.89,
        "217,831 个股票日 · 185 个日期" if zh else "217,831 stock-days · 185 dates",
        va="top",
        fontsize=7.5,
        color=palette["secondary_text"],
    )
    fig.text(
        0.225,
        0.05,
        (
            "逐点 95% 诊断区间 · HAC 8 · Holm27 无通过项"
            if zh
            else "Pointwise 95% diagnostic intervals · HAC 8 · none passes Holm27"
        ),
        va="bottom",
        fontsize=7.1,
        color=palette["secondary_text"],
    )
    # Publication typography; mathematical scripts retain normal scaling.
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text() and text.get_fontsize() < 8:
            text.set_fontsize(8)
        if (
            zh
            and text.get_text()
            and not re.search(r"[\u3400-\u9fff]", text.get_text())
        ):
            text.set_fontfamily(["Arial"])
    outputs = {}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            bb = t.get_window_extent(renderer)
            if (
                bb.x0 < -0.5
                or bb.y0 < -0.5
                or bb.x1 > fig.bbox.width + 0.5
                or bb.y1 > fig.bbox.height + 0.5
            ):
                outside.append(t.get_text())
    assert not outside, outside
    for ext in ["pdf", "svg", "png"]:
        p = out / (KEY + "." + ext)
        fig.savefig(p, dpi=450)
        outputs[ext] = {
            "path": str(p.relative_to(root)),
            "sha256": file_sha256(p),
            "bytes": p.stat().st_size,
        }
    fig.savefig(qa / (KEY + "_print_size.png"), dpi=160)
    plt.close(fig)
    return {
        "figure": KEY,
        "outputs": outputs,
        "caption": caption_record,
        "plot_count": 9,
        "outside_canvas": outside,
    }
