"""Verify released files against the SHA256 checksum list."""

from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify(root):
    root = root.resolve()
    records = []
    for line in (
        (root / "reproduction/checksums.sha256")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if line.strip():
            digest, name = line.split("  ", 1)
            records.append((digest, name))
    errors = []
    total_bytes = 0
    for expected, name in records:
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            errors.append({"file": name, "error": "path outside archive"})
            continue
        if not path.is_file():
            errors.append({"file": name, "error": "missing file"})
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        total_bytes += path.stat().st_size
        if digest != expected:
            errors.append({"file": name, "error": "content differs"})
    return {
        "status": "PASS" if not errors else "FAIL",
        "scope": "Released file integrity; no scientific calculation.",
        "files": len(records),
        "bytes": total_bytes,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
