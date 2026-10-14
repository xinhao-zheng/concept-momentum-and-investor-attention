from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from plotting.engine import run_cli

if __name__ == "__main__":
    run_cli(repository_root=ROOT)
