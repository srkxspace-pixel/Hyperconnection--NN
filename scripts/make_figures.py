"""Run folders -> figures F1-F7 (PDF + PNG) in results/figures.

  python scripts/make_figures.py                        # results/raw -> results/figures
  python scripts/make_figures.py --raw path/to/raw --out path/to/figures --cost results/tables/cost_<gpu>.csv

No GPU. Figures that the available runs cannot support are skipped with a message (F2 needs two data fractions, F5 a cost
CSV, F6 runs with connections.pt and torch).
"""
import argparse
from pathlib import Path

from hcsmall.analysis.figures import make_all

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "results/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "results/figures")
    ap.add_argument("--cost", type=Path, help="cost CSV from scripts/profile_cost.py (default: newest results/tables/cost_*.csv)")
    a = ap.parse_args()
    cost = a.cost
    if cost is None:
        found = sorted((ROOT / "results/tables").glob("cost_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        cost = found[0] if found else None
    written = make_all(a.raw, a.out, cost)
    print(f"wrote {len(written)} files to {a.out}: " + ", ".join(sorted({p.stem for p in written})))


if __name__ == "__main__":
    main()
