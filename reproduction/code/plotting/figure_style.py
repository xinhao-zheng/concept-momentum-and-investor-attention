"""Shared publication palette and perceptually ordered continuous ramps."""

from pathlib import Path
import json
import sys
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

HERE = Path(__file__).resolve().parent
from colorspacious import cspace_convert

CONFIG = json.loads((HERE / "figure_palette.json").read_text(encoding="utf-8"))
PALETTE = CONFIG["colors"]


def lab_ramp(start, end, count):
    ends = cspace_convert([to_rgb(start), to_rgb(end)], "sRGB1", "CIELab")
    rgb = cspace_convert(np.linspace(*ends, count), "CIELab", "sRGB1")
    # The white endpoint has tiny round-trip errors, not meaningful gamut loss.
    if rgb.min() < -1e-5 or rgb.max() > 1.00001:
        raise ValueError("Palette interpolation leaves the sRGB gamut")
    return np.clip(rgb, 0, 1)


_seq = lab_ramp(PALETTE["zero"], PALETTE["sequential_end"], 256)
_neg = lab_ramp(PALETTE["negative"], PALETTE["zero"], 257)
_pos = lab_ramp(PALETTE["zero"], PALETTE["positive"], 257)
SEQ = ListedColormap(_seq, name="white_to_steel_blue")
DIV = ListedColormap(np.vstack([_neg, _pos[1:]]), name="steel_blue_white_bronze")

assert np.all(np.diff(cspace_convert(_seq, "sRGB1", "CIELab")[:, 0]) < 0)
assert np.all(np.diff(cspace_convert(_neg, "sRGB1", "CIELab")[:, 0]) > 0)
assert np.all(np.diff(cspace_convert(_pos, "sRGB1", "CIELab")[:, 0]) < 0)
assert np.allclose(DIV(0.5)[:3], to_rgb(PALETTE["zero"]), atol=1e-5)
