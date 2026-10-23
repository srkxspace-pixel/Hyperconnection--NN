"""Run folders -> tables T1-T4 (CSV + LaTeX) plus runs.csv, coverage.csv, T0_setup.tex and macros.tex.

  python scripts/aggregate.py                          # results/raw -> results/tables
  python scripts/aggregate.py --raw path/to/raw --out path/to/tables --cost results/tables/cost_<gpu>.csv

Needs only pandas, scipy and pyyaml (no torch, no GPU). Refuses a results folder whose runs mix training budgets: aggregate
one plan per folder. Statistics follow D10: paired by seed, paired t-test and a bootstrap 95% CI.
"""
import argparse
from pathlib import Path

from hcsmall.analysis.tables import build_all, coverage
from hcsmall.analysis.runs import load_runs

ROOT = Path(__file__).resolve().parents[1]


def newest_cost(folder: Path):
    found = sorted(folder.glob("cost_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "results/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "results/tables")
    ap.add_argument("--cost", type=Path, help="cost CSV from scripts/profile_cost.py (default: newest cost_*.csv in --out)")
    a = ap.parse_args()
    cost = a.cost or newest_cost(a.out)
    written = build_all(a.raw, a.out, cost, ROOT / "configs")
    df = load_runs(a.raw)
    print(f"{len(df)} runs from {a.raw}; cost table: {cost.name if cost else 'none (T3 and F5 need scripts/profile_cost.py)'}")
    print("runs per data setting x model x connection:")
    print(coverage(df).pivot_table(index=["data", "model"], columns="conn", values="runs", fill_value=0).astype(int).to_string())
    print(f"wrote {len(written)} files to {a.out}")


if __name__ == "__main__":
    main()
