"""Render published table values without changing their stored precision."""

from pathlib import Path
import argparse, csv, json

ROOT = Path(__file__).resolve().parents[2]


def render(directory):
    out = directory / "tables"
    out.mkdir(exist_ok=True)
    emitted = []
    for src in sorted((directory / "data").glob("published*.csv")):
        with src.open(encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert rows and all(len(r) == len(rows[0]) for r in rows)
        rows = [[x.replace("|", r"\|") for x in row] for row in rows]
        lines = [
            "| " + " | ".join(rows[0]) + " |",
            "| " + " | ".join(["---"] * len(rows[0])) + " |",
        ]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        target = out / (src.stem + ".md")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        emitted.append(str(target.relative_to(ROOT)))
    return emitted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table")
    a = p.parse_args()
    entries = json.loads((ROOT / "tables/index.json").read_text(encoding="utf-8"))
    if a.table:
        entries = [e for e in entries if e["id"] == a.table]
        if not entries:
            p.error("Unknown table identifier")
    for entry in entries:
        for path in render(ROOT / entry["path"]):
            print(path)


if __name__ == "__main__":
    main()
