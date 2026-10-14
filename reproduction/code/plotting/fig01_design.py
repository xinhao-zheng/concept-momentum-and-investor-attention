"""Membership, score construction and information timing."""

from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np


def chinese_labels():
    return {}


def render(b):
    import matplotlib as mpl

    return _render(b, any("YaHei" in str(f) for f in mpl.rcParams["font.family"]))


def render_cn(b):
    return _render(b, True)


def _render(b, cn=False):
    import matplotlib as mpl

    with mpl.rc_context(
        {
            "font.family": ["Microsoft YaHei", "Arial"] if cn else ["Arial"],
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig = b.canvas(120)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 183), ylim=(120, 0))
        ax.axis("off")
        ink = "#1D2328"
        blue = "#334C5C"
        grey = "#586269"
        grid = "#DCE3E7"
        attention = "#4A626C"
        records = []
        serial = 0

        def txt(x, y, en, zh=None, fs=8, ha="left", weight="normal", color=ink, **kw):
            nonlocal serial
            label = zh if cn and zh is not None else en
            a = ax.text(
                x,
                y,
                label,
                fontsize=fs,
                ha=ha,
                va="center",
                fontweight=weight,
                color=color,
                linespacing=1.25,
                **kw,
            )
            a.set_gid(f"text_{serial:02d}")
            serial += 1
            records.append({"text": label, "x_mm": x, "y_mm": y, "font_pt": fs})
            return a

        def line(x1, y1, x2, y2, color=grid, lw=0.65):
            return ax.plot([x1, x2], [y1, y2], color=color, lw=lw)[0]

        def arrow(x1, y1, x2, y2, dashed=False):
            ax.add_patch(
                FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    arrowstyle="-|>",
                    mutation_scale=8.8,
                    lw=0.85,
                    color=grey,
                    shrinkA=0,
                    shrinkB=0,
                    linestyle=(0, (3, 2)) if dashed else "-",
                )
            )

        def title(y, en, zh):
            txt(4, y, en, zh, fs=9.2, weight="bold")

        title(
            5.5,
            "a  From overlapping concepts to a stock ranking",
            "a  从交叠概念到股票排序",
        )
        txt(19, 14, "Membership", "成员关系", weight="bold", ha="center")
        member = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0]])
        for j in range(3):
            txt(12 + j * 7.2, 19, f"C{j+1}", ha="center", color=grey)
        for i in range(5):
            y = 24 + i * 4
            txt(6.8, y, f"S{i+1}", ha="right", color=grey)
            for j in range(3):
                ax.add_patch(
                    Rectangle(
                        (9.7 + j * 7.2, y - 1.35),
                        4.6,
                        2.7,
                        facecolor=blue if member[i, j] else "white",
                        edgecolor=grid,
                        lw=0.55,
                    )
                )
        txt(43.5, 15, "F  Peer strength", "F  同伴收益强度", weight="bold", color=blue)
        txt(43.5, 21, r"Peer return over $L_F$", r"$L_F$ 日同伴收益")
        txt(
            43.5,
            30,
            "W  Stock–peer alignment",
            "W  股票与同伴同步性",
            weight="bold",
            color=blue,
        )
        txt(43.5, 36, r"Return correlation over $L_W$", r"$L_W$ 日收益相关性")
        txt(
            43.5,
            43,
            "Peer returns exclude the focal stock",
            "同伴收益剔除目标股票",
            fs=7.6,
            color=grey,
        )
        txt(109, 15, "Rank and combine", "排序与组合", weight="bold")
        txt(109, 24, r"$f=\mathrm{rank}(F),\ w=\mathrm{rank}(W)$", fs=8.2)
        txt(
            109,
            36,
            r"$FW_i=\mathrm{mean}_{c\in C_i}(f_{ic}w_{ic})$",
            fs=8.2,
            color=blue,
        )
        txt(171, 22, "Top N", "前 N 只", weight="bold", ha="center")
        txt(171, 33, "Freeze\nstock list", "固定\n选股名单", ha="center")
        arrow(32, 30, 39, 30)
        arrow(99, 30, 105, 30)
        arrow(157, 30, 163, 30)
        line(4, 49, 179, 49)
        title(55, "b  Main research clock (γ)", "b  主研究时序（γ）")
        xs = [22, 72, 119, 168]
        labels = [
            ("Membership record", "成员记录"),
            ("Score and fixed list", "评分并固定名单"),
            ("Entry open", "起始开盘"),
            ("Outcome open", "结束开盘"),
        ]
        times = [
            (r"$\leq t-1$", r"$\leq t-1$"),
            (r"$t$ close", r"$t$ 收盘"),
            (r"$t+1$", r"$t+1$"),
            (r"$t+H+1$", r"$t+H+1$"),
        ]
        for x, (en, zh), (te, tz) in zip(xs, labels, times):
            txt(x, 64, en, zh, ha="center")
            line(x, 69, x, 71.5, blue, 0.9)
            txt(x, 77, te, tz, ha="center", color=blue)
        for x1, x2 in zip(xs[:-1], xs[1:]):
            arrow(x1 + 1, 70, x2 - 3, 70)
        line(xs[2], 83, xs[3], 83, blue, 1.05)
        line(xs[2], 81.7, xs[2], 84.3, blue, 0.8)
        line(xs[3], 81.7, xs[3], 84.3, blue, 0.8)
        txt(
            (xs[2] + xs[3]) / 2,
            89,
            "Complete H-day return",
            "完整 H 日收益",
            ha="center",
            color=blue,
        )
        line(4, 94.5, 179, 94.5)
        title(
            100.5,
            "c  Attention interpretation and separate measurements",
            "c  注意力解释与独立测量",
        )
        for x, en, zh in [
            (28, "Shared thematic attention", "共同主题注意力"),
            (91, "Coordinated trading", "协同交易"),
            (155, "Return continuation", "收益延续"),
        ]:
            txt(x, 108, en, zh, ha="center", color=blue)
        arrow(56, 108, 70, 108, True)
        arrow(115, 108, 133, 108, True)
        txt(
            4,
            116,
            "Separate stock-level T / S / K: participation and return analyses",
            "独立的个股 T / S / K 测量：交易参与与收益分析",
            fs=7.8,
            color=attention,
        )
        txt(
            179,
            116,
            "Dashed arrows: hypothesis",
            "虚线：解释假说",
            ha="right",
            fs=7.8,
            color=grey,
        )
        fig._fig01_geometry = {
            "dimensions_mm": [183, 120],
            "panels": ["a", "b", "c"],
            "archetype": "membership-score-clock schematic",
            "membership_matrix_shape": [5, 3],
            "empirical_plotted_values": 0,
            "hypothesis_arrows": "dashed",
            "solid_arrows": "calculation or chronology",
            "editable_text": True,
            "language": "zh" if cn else "en",
            "labels": records,
        }
        b.save(fig, "fig_01")
        b.CAPTIONS["fig_01"]["caption_zh"] = b.FROZEN_CAPTION["caption_zh"]
