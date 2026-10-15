"""Monthly momentum, concept comovement and return contrasts."""

from pathlib import Path
import hashlib
import json
import numpy as np


def label_map(nmon):
    return {
        "a  Momentum and concept comovement": "a  动量与概念联动",
        "b  Six-month contrasts": "b  六个月收益对比",
        "Past momentum decile": "过去动量十分位",
        "Concept-comovement group": "概念联动分组",
        "Low beta": "低beta组",
        "Middle beta": "中beta组",
        "High beta": "高beta组",
        "Six-month gross return (%)": "六个月毛收益（%）",
        "Six-month difference (pp)": "六个月差值（百分点）",
        "High - low beta": "高－低beta组",
        "FW - BC / Top20": "FW－BC / Top20",
        "FW - BC / Top100": "FW－BC / Top100",
        "FW - pool / Top20": "FW－样本池 / Top20",
        "FW - pool / Top100": "FW－样本池 / Top100",
        f"{nmon} mature formation months; one-month skip and six-month holding.": f"{nmon}个成熟形成月份；跳过一个月后持有六个月。",
    }


def render(b):
    fig = b.canvas(102)
    d = b.read("monthly_cells")
    q = b.read("monthly_contrasts")
    assert len(d) == 30 and not d.duplicated(["C", "D"]).any()
    matrix = (
        d.pivot(index="C", columns="D", values="mean_complete")
        .reindex(index=[1, 2, 3], columns=range(1, 11))
        .to_numpy()
        * 100
    )
    assert matrix.shape == (3, 10) and np.isfinite(matrix).all()
    assert matrix.min() >= 0 and matrix.max() <= 7
    nmon = int(
        q[(q.label == "double_difference") & (q.missing == "complete")].months.iloc[0]
    )
    a = fig.add_axes([0.11, 0.40, 0.48, 0.42])
    cax = fig.add_axes([0.11, 0.225, 0.48, 0.027])
    right = fig.add_axes([0.795, 0.235, 0.18, 0.585])
    b.panel(a, "a", "Momentum and concept comovement")
    mesh = b.heatmap(a, matrix, b.SEQ, 0, 7, grid_width=0.8)
    a.set_xticks(range(10), range(1, 11))
    a.set_yticks(range(3), ["Low beta", "Middle beta", "High beta"])
    a.set_xlabel("Past momentum decile", labelpad=7)
    a.tick_params(axis="both", length=0, pad=4)
    for spine in a.spines.values():
        spine.set_visible(False)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            a.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color=b.heatmap_label_color(value, b.SEQ, 0, 7),
            )
    colorbar = fig.colorbar(
        mesh, cax=cax, orientation="horizontal", ticks=[0, 2, 4, 6, 7]
    )
    b.continuous_vector_colorbar(colorbar)
    colorbar.set_label("Six-month gross return (%)", labelpad=4)
    colorbar.ax.tick_params(length=2, width=0.5)

    specs = [
        ("double_difference", 0, "High - low beta"),
        ("FW_minus_BC", 20, "FW - BC / Top20"),
        ("FW_minus_BC", 100, "FW - BC / Top100"),
        ("FW_minus_pool", 20, "FW - pool / Top20"),
        ("FW_minus_pool", 100, "FW - pool / Top100"),
    ]
    plotted = []
    for j, (key, n, label) in enumerate(specs):
        rows = q[(q.label == key) & (q.N == n) & (q.missing == "complete")]
        assert len(rows) == 1
        value = float(rows.iloc[0].mean_pp)
        right.plot(value, 4 - j, "o", color=b.BLUE if j else b.REF, ms=4.5)
        right.text(value, 4 - j + 0.21, f"{value:.2f}", ha="center", fontsize=7.2)
        plotted.append({"label": key, "N": n, "mean_pp": value})
    right.set_yticks(range(4, -1, -1), [s[2] for s in specs])
    right.set(xlim=(-1.5, 2.9), ylim=(-0.55, 4.7))
    right.set_xticks([-1, 0, 1, 2])
    right.tick_params(axis="y", length=0)
    b.zero(right, "x")
    right.set_xlabel("Six-month difference (pp)")
    b.panel(right, "b", "Six-month contrasts")
    b.note(
        fig, f"{nmon} mature formation months; one-month skip and six-month holding."
    )

    # Retain source values and record the display transformation.
    b.record("fig_s05", d, "cells")
    b.record("fig_s05", q, "contrasts")
    display = d[["C", "D", "months", "mean_complete"]].copy()
    display["display_percent"] = display["mean_complete"] * 100
    display["display_label"] = display["display_percent"].map(lambda x: f"{x:.1f}")
    b.record("fig_s05", display, "heatmap_display")
    b.save(fig, "fig_s05", 35)
    return {
        "cell_count": 30,
        "contrast_count": 5,
        "months": nmon,
        "scale_percent": [0, 7],
        "min_percent": float(matrix.min()),
        "max_percent": float(matrix.max()),
        "labels_zh": label_map(nmon),
        "caption_en": b.FROZEN_CAPTION["caption_en"],
        "caption_zh": b.FROZEN_CAPTION["caption_zh"],
    }
