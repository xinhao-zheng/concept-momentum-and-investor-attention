"""Validate external inputs required by the attention measurement branch."""

from pathlib import Path


def validate_measurement_calendar(snapshot_root: Path, cutoff: str) -> Path:
    """Check the calendar contract used by M2 before any estimation starts."""
    import pandas as pd

    path = snapshot_root / "raw_snapshot/stock-main-index-data/sh000001.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "M2 requires the historical index calendar: " + str(path)
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "candle_end_time" not in frame.columns:
        raise ValueError(
            "M2 calendar requires the candle_end_time column: " + str(path)
        )
    calendar = pd.DatetimeIndex(pd.to_datetime(frame["candle_end_time"]).dt.normalize())
    if (
        calendar.hasnans
        or calendar.has_duplicates
        or not calendar.is_monotonic_increasing
    ):
        raise ValueError(
            "M2 calendar dates must be valid, unique and increasing: " + str(path)
        )
    retained = calendar[(calendar >= "2016-01-01") & (calendar <= cutoff)]
    if retained.empty or retained[-1] != pd.Timestamp(cutoff):
        raise ValueError(
            "M2 calendar must include the observation cutoff "
            + cutoff
            + ": "
            + str(path)
        )
    return path
