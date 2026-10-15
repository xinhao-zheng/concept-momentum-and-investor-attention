"""Observation boundary and configured output paths for the retained design."""

from pathlib import Path
from runtime_config import load_config
import hashlib
import json

RUNTIME = load_config()
OBSERVATION_CUTOFF = LAST_ALLOWED_MARKET_DATE = RUNTIME.cutoff
START = "2016-11-24"
EXP = RUNTIME.output("core/replay")
SOURCE_ROOT = RUNTIME.path("raw_snapshot")
CALENDAR_SOURCE = SOURCE_ROOT / "stock-main-index-data/sh000001.csv"
NEW_S2 = EXP / "S2"
NEW_S2_CACHE = NEW_S2 / "cache"
NEW_S2_AUDIT = NEW_S2 / "audit"
NEW_S2_DATA = NEW_S2 / "data"
HORIZONS = [1, 3, 5, 10, 20, 60, 120, 250]


def digest(path, chunk=4 * 1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def assert_not_frozen_source(path):
    p = Path(path).resolve()
    if not p.is_relative_to(RUNTIME.output_root):
        raise PermissionError("Writes must stay within output_root")
    return p
