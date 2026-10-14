"""Verify explicitly configured frozen input hashes; no historical machine-path aliases."""

from pathlib import Path
import hashlib


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_frozen_inputs(inputs, experiment_root):
    unchanged, resolutions = {}, []
    for recorded, expected in inputs.items():
        resolved = Path(recorded)
        if not resolved.is_absolute():
            resolved = Path(experiment_root) / resolved
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        actual = digest(resolved)
        if actual != expected:
            raise ValueError("Frozen SHA-256 mismatch: " + str(resolved))
        unchanged[recorded] = True
        resolutions.append(
            dict(
                recorded_path=recorded,
                resolved_path=str(resolved),
                resolution_rule="explicit_configured_path",
                expected_sha256=expected,
                actual_sha256=actual,
                sha256_matches=True,
            )
        )
    return unchanged, resolutions
