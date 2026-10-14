"""Explicit checks for the article's fixed Figure 8 axes geometry.

This project-local checker uses Matplotlib rectangles directly. It is not a
copy of the more general historical figure_qa module and imports no skill code.
"""

from pathlib import Path
import json


def require_matplotlib_panel_alignment(
    fig,
    *,
    json_out,
    panel_ids,
    row_groups,
    column_groups,
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    strict=True
):
    fig.canvas.draw()
    if len(fig.axes) != len(panel_ids):
        raise AssertionError("The fixed figure must supply one id per axes.")
    width, height = [float(x) for x in fig.get_size_inches() * 72]
    boxes = {}
    for name, ax in zip(panel_ids, fig.axes):
        box = ax.get_position()
        boxes[name] = [
            float(x)
            for x in (box.x0 * width, box.y0 * height, box.x1 * width, box.y1 * height)
        ]
    checks = []
    for axis, groups, edges in [
        ("row", row_groups, (1, 3)),
        ("column", column_groups, (0, 2)),
    ]:
        for names in groups:
            for edge in edges:
                values = [boxes[name][edge] for name in names]
                delta = max(values) - min(values)
                checks.append(
                    dict(
                        group=names,
                        axis=axis,
                        edge=edge,
                        max_difference_pt=delta,
                        passed=delta <= tolerance_pt,
                    )
                )
    for name, box in boxes.items():
        checks.append(
            dict(
                panel=name,
                check="inside_canvas",
                passed=box[0] >= 0
                and box[1] >= 0
                and box[2] <= width
                and box[3] <= height,
            )
        )
    report = dict(
        schema="article_fixed_axes_alignment_v1",
        verdict="PASS" if all(x["passed"] for x in checks) else "FAIL",
        panel_boxes_pt=boxes,
        tolerance_pt=tolerance_pt,
        checks=checks,
        scope="Explicit row/column edges and canvas bounds; no inference about scientific equivalence",
    )
    target = Path(json_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["verdict"] != "PASS":
        raise AssertionError(report)
    return report
