"""Shared constants and hashing helpers for evaluation of frozen core selections. Score construction is implemented by score_engine.py."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "config"))
import cutoff as CFG  # noqa: E402

SOURCE = CFG.SOURCE_ROOT
CACHE = ROOT / "cache"
AUDIT = ROOT / "audit"
CFG.assert_not_frozen_source(CACHE)
CUTOFF = CFG.OBSERVATION_CUTOFF
START = CFG.START
METHODS = ["FW", "F_only", "W_only", "mom5", "rev5", "mom20"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
