"""Shared paths and resource limits for independent research calculations."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import os

REPRODUCTION_ROOT = Path(__file__).resolve().parents[2]
OBSERVATION_CUTOFF = "2026-08-31"


@dataclass(frozen=True)
class RuntimeConfig:
    data_root: Path
    snapshot_root: Path
    output_root: Path
    cutoff: str
    threads: int
    config_file: Path | None = None

    def path(self, relative_path: str) -> Path:
        candidate = (self.snapshot_root / relative_path).resolve()
        if not candidate.is_relative_to(self.snapshot_root):
            raise ValueError("Input path must stay within snapshot_root")
        return candidate

    def output(self, relative_path: str) -> Path:
        candidate = (self.output_root / relative_path).resolve()
        if not candidate.is_relative_to(self.output_root):
            raise ValueError("Output path must stay within output_root")
        return candidate

    def apply_threads(self) -> None:
        for key in (
            "POLARS_MAX_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[key] = str(self.threads)


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    explicit = path or os.environ.get("CONCEPT_QUANT_CONFIG")
    source = Path(explicit).expanduser().resolve() if explicit else None
    values = json.loads(source.read_text(encoding="utf-8-sig")) if source else {}
    base = source.parent if source else REPRODUCTION_ROOT
    data = Path(values.get("data_root", "../data")).expanduser()
    data = data.resolve() if data.is_absolute() else (base / data).resolve()
    snapshot = Path(values.get("snapshot_root", "research-snapshot"))
    snapshot = (
        snapshot.resolve() if snapshot.is_absolute() else (data / snapshot).resolve()
    )
    output_value = values.get("output_root")
    output = (
        Path(output_value).expanduser() if output_value else REPRODUCTION_ROOT / "work"
    )
    output = output.resolve() if output.is_absolute() else (base / output).resolve()
    cutoff = values.get("cutoff", OBSERVATION_CUTOFF)
    if cutoff != OBSERVATION_CUTOFF:
        raise ValueError("This replication fixes the observation cutoff at 2026-08-31")
    threads = int(values.get("threads", 2))
    if threads < 1:
        raise ValueError("threads must be positive")
    if output == snapshot or output.is_relative_to(snapshot):
        raise ValueError("output_root must not be inside the input snapshot")
    config = RuntimeConfig(data, snapshot, output, cutoff, threads, source)
    config.apply_threads()
    if source:
        os.environ["CONCEPT_QUANT_CONFIG"] = str(source)
    return config
