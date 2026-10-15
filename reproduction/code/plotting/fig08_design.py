"""Account returns, drawdowns and configuration contrasts."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


def render(b):
    """Render account comparisons from the registered figure inputs."""
    fig = b.canvas(170)
    fig.set_size_inches(190 / 25.4, 170 / 25.4)
    summary = b.read("bt_summary")
    curves = b.read("bt_curves")
    pairs = b.read("bt_pairs")
    indices = b.read("bt_indices")
    curves["date"] = pd.to_datetime(curves["date"])
    assert len(summary) == 48 and not summary.execution_anomaly.any()
    assert curves.date.max() == pd.Timestamp("2026-08-31")

    reference = summary.loc[
        summary.hold.eq("3D")
        & summary.F.eq(5)
        & summary.W.eq(10)
        & summary.similarity.eq("pearson")
        & summary.top_n.eq(10)
    ].set_index("timing")
    assert set(reference.index) == {"none", "breadth", "breadth2"}
    assert len(reference) == 3

    palette = b.PALETTE
    timing = {
        "none": dict(
            color=palette.get("timing_none", palette["secondary_series"]),
            marker="o",
            ls=(0, (4.0, 1.8)),
            label="No timing",
        ),
        "breadth": dict(
            color=palette.get("timing_breadth", palette["attention"]),
            marker="s",
            ls=(0, (6.0, 1.5, 1.2, 1.5)),
            label="Breadth",
        ),
        "breadth2": dict(
            color=palette.get("timing_two_level", palette["primary"]),
            marker="^",
            ls="-",
            label="Two-level",
        ),
    }
    # Wealth and reference drawdown share the same date axis.
    # Panel b and Supplementary Fig. S13 compare all account drawdowns.
    left, right = 0.110, 0.975
    nav = fig.add_axes([left, 0.630, right - left, 0.270])
    dd = fig.add_axes([left, 0.470, right - left, 0.115], sharex=nav)
    risk = fig.add_axes([left, 0.080, 0.320, 0.235])
    contrasts = fig.add_axes([0.655, 0.080, 0.320, 0.235])
    assert nav.get_position().x0 == dd.get_position().x0 == risk.get_position().x0
    assert np.allclose(
        [nav.get_position().x1, dd.get_position().x1],
        contrasts.get_position().x1,
        rtol=0,
        atol=1e-12,
    )
    assert risk.get_position().y0 == contrasts.get_position().y0
    assert risk.get_position().y1 == contrasts.get_position().y1
    fig._fig08_geometry = {
        name: list(ax.get_position().bounds)
        for name, ax in [
            ("nav", nav),
            ("drawdown_two_level", dd),
            ("risk", risk),
            ("contrasts", contrasts),
        ]
    }
    end = pd.Timestamp("2026-08-31")
    used_curve_names = []

    for name, style in [("CSI 1000", (0, (1.0, 2.0))), ("CSI 300", (0, (5.0, 2.0)))]:
        q = curves.loc[curves.series.eq(name)].sort_values("date")
        assert len(q) == 2368 and float(q.nav.iloc[0]) == 1.0
        assert q.nav.min() > 0.4 and q.nav.max() < 16
        nav.plot(q.date, q.nav, color=b.REF, lw=0.70, ls=style, zorder=2)
        used_curve_names.append(name)

    for key, spec in timing.items():
        q = curves.loc[curves.series.eq("Top10|" + key)].sort_values("date")
        assert len(q) == 2368 and float(q.nav.iloc[0]) == 1.0
        assert np.isclose(
            q.nav.iloc[-1], reference.loc[key, "final_nav"], rtol=0, atol=1e-12
        )
        assert np.isclose(
            q.drawdown.min(), reference.loc[key, "max_drawdown"], rtol=0, atol=1e-12
        )
        assert q.nav.min() > 0.4 and q.nav.max() < 16
        assert q.drawdown.min() > -0.5 and q.drawdown.max() <= 0.02
        nav.plot(
            q.date,
            q.nav,
            color=spec["color"],
            ls=spec["ls"],
            lw=1.05 if key == "breadth2" else 0.90,
            zorder=4,
        )
        nav.plot(
            q.date.iloc[-1],
            q.nav.iloc[-1],
            marker=spec["marker"],
            ms=3.6,
            color=spec["color"],
            zorder=5,
        )
        if key == "breadth2":
            dd.fill_between(
                q.date,
                100 * q.drawdown,
                0,
                facecolor="#F2F5F6",
                edgecolor="none",
                zorder=1,
            )
            dd.plot(
                q.date, 100 * q.drawdown, color=spec["color"], ls="-", lw=0.65, zorder=3
            )
        used_curve_names.append("Top10|" + key)

    nav.set_yscale("log")
    nav.set_ylim(0.4, 16)
    nav.set_yticks([0.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"])
    nav.minorticks_off()
    nav.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    nav.spines["bottom"].set_visible(False)
    nav.set_ylabel("Net asset value (log scale)")
    nav.axhline(1, color=b.GRID, lw=0.55, zorder=0)
    fig.text(
        left,
        0.966,
        "a  Fixed Top10 configuration",
        fontsize=9,
        fontweight="bold",
        color=b.INK,
        va="baseline",
    )
    timing_handles = [
        Line2D(
            [],
            [],
            color=spec["color"],
            ls=spec["ls"],
            marker=spec["marker"],
            ms=3.8,
            lw=1.15,
            label=spec["label"],
        )
        for key, spec in timing.items()
    ]
    fig.legend(
        handles=timing_handles,
        loc="lower left",
        ncol=3,
        mode="expand",
        bbox_to_anchor=(left, 0.920, right - left, 0.03),
        borderaxespad=0,
        handlelength=2.6,
        handletextpad=0.60,
        columnspacing=1.8,
        fontsize=8,
        labelcolor=b.INK,
    )
    nav.legend(
        handles=[
            Line2D([], [], color=b.REF, lw=0.85, ls=(0, (5.0, 2.0)), label="CSI 300"),
            Line2D([], [], color=b.REF, lw=0.85, ls=(0, (1.0, 2.0)), label="CSI 1000"),
        ],
        loc="upper left",
        ncol=2,
        fontsize=8,
        handlelength=2.4,
        handletextpad=0.5,
        columnspacing=1.3,
        borderaxespad=0.5,
        labelcolor=b.SECONDARY,
    )
    dd.set_ylim(-22, 1)
    dd.set_yticks([-20, -10, 0])
    dd.tick_params(axis="y", labelsize=8, length=2.2)
    dd.axhline(0, color=b.GRID, lw=0.55, zorder=0)
    dd.set_ylabel("Two-level\ndrawdown (%)", fontsize=8)
    dd.xaxis.set_major_locator(mdates.YearLocator(2, month=1, day=1))
    dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    dd.set_xlim(pd.Timestamp("2016-11-01"), pd.Timestamp("2026-10-01"))
    fig._fig08_geometry["drawdown_limits"] = list(dd.get_ylim())
    fig._fig08_geometry["drawdown_date_limits"] = list(dd.get_xlim())
    fig._fig08_geometry["dimensions_mm"] = list(fig.get_size_inches() * 25.4)
    fig._fig08_geometry["drawdown_daily_n"] = 2368
    fig._fig08_geometry["full_drawdown_comparison"] = "Supplementary Figure S13"

    # Timing colors and shapes retain their meaning in the entire design grid.
    for key, spec in timing.items():
        for similarity, filled in [("pearson", True), ("ccc", False)]:
            q = summary.loc[summary.timing.eq(key) & summary.similarity.eq(similarity)]
            risk.plot(
                100 * q.max_drawdown,
                100 * q.cagr,
                ls="none",
                marker=spec["marker"],
                ms=4.0,
                mec=spec["color"],
                mfc=spec["color"] if filled else "white",
                mew=0.75,
                color=spec["color"],
                zorder=3,
            )
    assert summary.max_drawdown.between(-0.65, -0.1).all()
    assert summary.cagr.between(0.1, 0.36).all()
    risk.set(xlim=(-65, -10), ylim=(10, 36))
    risk.set_xticks([-60, -40, -20])
    risk.set_xlabel("Maximum drawdown (%)")
    risk.set_ylabel("Annual return (%)")
    # Avoid background grid behind the dense comparison marks.
    fig.text(
        left,
        0.397,
        "b  All 48 account simulations",
        fontsize=9,
        fontweight="bold",
        color=b.INK,
        va="baseline",
    )
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", ms=4, color=b.INK),
            Line2D([], [], marker="o", ls="none", ms=4, mfc="white", mec=b.INK),
        ],
        labels=["Pearson", "CCC"],
        loc="lower left",
        ncol=2,
        bbox_to_anchor=(left, 0.348, 0.320, 0.025),
        borderaxespad=0,
        fontsize=8,
        handlelength=0.9,
        handletextpad=0.5,
        columnspacing=1.6,
        labelcolor=b.INK,
    )

    order = [
        ("Timing: two-level breadth − none", "Two-level − none", "breadth2"),
        ("Timing: breadth − none", "Breadth − none", "breadth"),
        ("Holding: 3D − 5D", "3-day − 5-day", None),
        ("Similarity: Pearson − CCC", "Pearson − CCC", None),
        ("Selection: Top5 − Top10", "Top5 − Top10", None),
        ("Windows: F5/W10 − F10/W20", "F5/W10 − F10/W20", None),
    ]
    plotted = []
    for i, (key, label, timing_key) in enumerate(order):
        q = pairs.loc[pairs.metric.eq("cagr") & pairs.contrast.eq(key)]
        q = q.sort_values("difference")
        assert len(q) == (16 if timing_key else 24)
        assert (100 * q.difference).between(-11, 18).all()
        y = len(order) - 1 - i
        jitter = ((np.arange(len(q)) % 5) - 2) * 0.075
        color = timing[timing_key]["color"] if timing_key else "#A2AAB0"
        contrasts.plot(
            100 * q.difference,
            y + jitter,
            ls="none",
            marker="o",
            ms=2.35,
            color=color,
            mew=0,
            zorder=3,
        )
        median = 100 * q.difference.median()
        contrasts.vlines(median, y - 0.30, y + 0.30, color=b.INK, lw=1.1, zorder=4)
        plotted.append(q.assign(display=label, median_pp=median))
    b.zero(contrasts, "x")
    # The zero line is the sufficient reference for signed paired contrasts.
    contrasts.set(ylim=(-0.65, 5.65), xlim=(-11, 18))
    contrasts.set_xticks([-10, 0, 10])
    contrasts.set_yticks(range(5, -1, -1), [x[1] for x in order])
    contrasts.set_xlabel("CAGR difference (pp)")
    fig.text(
        0.520,
        0.397,
        "c  Matched design comparisons",
        fontsize=9,
        fontweight="bold",
        color=b.INK,
        va="baseline",
    )
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", ms=3, color=b.AUX),
            Line2D([], [], marker="|", ls="none", ms=7, mew=1.2, color=b.INK),
        ],
        labels=["Matched pair", "Median"],
        loc="lower left",
        ncol=2,
        bbox_to_anchor=(0.520, 0.348, 0.455, 0.025),
        borderaxespad=0,
        handletextpad=0.4,
        columnspacing=1.6,
        fontsize=8,
        labelcolor=b.INK,
    )
    fig._fig08_geometry["pair_count"] = sum(len(q) for q in plotted)
    assert fig._fig08_geometry["pair_count"] == 128
    # The caption states the observation cutoff and transaction costs.
    for ax in (nav, dd, risk, contrasts):
        ax.tick_params(axis="both", labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
    from .alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        fig,
        json_out=Path(b.QA) / "fig08_alignment.json",
        panel_ids=["a_nav", "a_drawdown", "b", "c"],
        row_groups=[["b", "c"]],
        column_groups=[["a_nav", "a_drawdown"]],
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )

    (Path(b.QA) / "fig08_concise_geometry.json").write_text(
        json.dumps(fig._fig08_geometry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    en, zh = b.FROZEN_CAPTION["caption_en"], b.FROZEN_CAPTION["caption_zh"]
    # Retain the complete source table alongside the displayed subset.
    b.record("fig_08", summary)
    b.record("fig_08", pd.concat(plotted, ignore_index=True), "paired_differences")
    b.record("fig_08", curves, "curves")
    b.record(
        "fig_08",
        curves.loc[curves.series.isin(used_curve_names)],
        "displayed_curves",
    )
    b.record("fig_08", reference.reset_index(), "displayed_configurations")
    b.save(fig, "fig_08", 48)
    if "fig_08" in b.CAPTIONS:
        b.CAPTIONS["fig_08"]["caption_zh"] = zh


def chinese_labels(reference):
    """Translate figure labels through the shared localization pass."""
    labels = {
        "a  Fixed Top10 configuration": "a  固定Top10配置下的择时比较",
        "b  All 48 account simulations": "b  全部48组账户模拟",
        "c  Matched design comparisons": "c  设计因素配对比较",
        "Net asset value (log scale)": "净值（对数刻度）",
        "Drawdown (%)": "回撤（%）",
        "Two-level\ndrawdown (%)": "两级择时\n回撤（%）",
        "CSI 300": "沪深300",
        "CSI 1000": "中证1000",
        "Maximum drawdown (%)": "最大回撤（%）",
        "Annual return (%)": "年化复合收益率（%）",
        "Pearson": "Pearson",
        "CCC": "CCC",
        "Configurations in a": "a中的配置",
        "No timing": "无择时",
        "Breadth timing": "一级宽度择时",
        "Two-level timing": "两级宽度择时",
        "Breadth": "宽度择时",
        "Two-level": "两级择时",
        "Ring: panel a": "外圈：a中配置",
        "Matched pair": "单个配对",
        "Median": "中位数",
        "Two-level − none": "两级 − 无择时",
        "Breadth − none": "一级 − 无择时",
        "3-day − 5-day": "3日 − 5日",
        "Pearson − CCC": "Pearson − CCC",
        "Top5 − Top10": "Top5 − Top10",
        "F5/W10 − F10/W20": "F5/W10 − F10/W20",
        "CAGR difference (pp)": "年化复合收益率配对差（百分点）",
        "Daily accounts through 31 August 2026; net of commissions and stamp duty.": "日度账户截至2026年8月31日；已扣除佣金与印花税。",
    }
    for key, en, zh in [
        ("none", "No timing", "无择时"),
        ("breadth", "Breadth timing", "一级宽度择时"),
        ("breadth2", "Two-level timing", "两级宽度择时"),
    ]:
        number = f'{100 * float(reference.loc[key, "cagr"]):.1f}'
        labels[en + " · " + number + "% CAGR"] = zh + " · 年化" + number + "%"
        labels[en + "\n" + number + "% CAGR"] = zh + "\n年化" + number + "%"
    return labels
