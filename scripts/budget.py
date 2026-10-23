"""Turn a timing CSV from scripts/time_epoch.py or scripts/profile_cost.py into GPU-hour budgets.

  python scripts/profile_cost.py                                  # on the machine that will run the study
  python scripts/budget.py results/tables/cost_<gpu>.csv                          # full Tier 1
  python scripts/budget.py results/tables/cost_<gpu>.csv --epochs 30 --seeds 3
  python scripts/budget.py cost.csv --models resnet20 resnet56 --connections residual dhc_n4 --data p100 p10 --seeds 1
  python scripts/budget.py cost.csv --epochs 30 --gpus 2 --weekly-hours 30        # two GPUs, a 30 h/week quota

Assumptions, all printed: one epoch = 50,000 images; per-epoch cost = timed training step x steps; `--overhead`
(default 1.10) covers the test evaluation after every epoch and the analysis passes. D6(b) "same steps" (`--steps
same_steps`, the default) makes a 10%- or 25%-data run cost as much as a 100% run; D6(a) "same epochs" makes it cost its
fraction. Runs that did not fit in memory are excluded, so those totals are lower bounds.
"""
import argparse
import csv
import math
from pathlib import Path

MODELS = ["resnet20", "resnet56", "resnet110", "vit_tiny_d6", "vit_tiny_d12", "vit_tiny_d24"]
CONNS = ["residual", "dhc_n2", "dhc_n4", "mhc_n4"]
FRACTION = {"p100": 1.0, "p25": 0.25, "p10": 0.10}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--connections", nargs="+", default=CONNS)
    ap.add_argument("--data", nargs="+", default=["p100", "p10"], choices=list(FRACTION))
    ap.add_argument("--steps", choices=["same_steps", "same_epochs"], default="same_steps")
    ap.add_argument("--overhead", type=float, default=1.10)
    ap.add_argument("--gpus", type=int, default=1, help="GPUs used in parallel (e.g. 2 on Kaggle T4 x2)")
    ap.add_argument("--weekly-hours", type=float, default=0, help="session hours per week you may use (0 = do not print weeks)")
    a = ap.parse_args()
    rows = {(r["model"], r["connection"]): r for r in csv.DictReader(a.csv.open(encoding="utf-8"))}
    data_factor = sum(1.0 if a.steps == "same_steps" else FRACTION[d] for d in a.data)

    def hours(m, c, epochs):
        r = rows.get((m, c))
        return float(r["epoch_s"]) * epochs * a.overhead / 3600 if r and r["epoch_s"] else math.nan

    def total(epochs, seeds):
        per_cell = sum(h for m in a.models for c in a.connections if not math.isnan(h := hours(m, c, epochs)))
        return per_cell * data_factor * seeds

    n_runs = len(a.models) * len(a.connections) * len(a.data) * a.seeds
    skipped = [f"{m}/{c}" for m in a.models for c in a.connections if math.isnan(hours(m, c, 1))]
    print(f"timing file: {a.csv.name}  (batch {next(iter(rows.values()))['batch_size']}, overhead x{a.overhead}, "
          f"data {'/'.join(a.data)} with {a.steps})")
    print(f"\nHours for ONE run at {a.epochs} epochs:")
    print(f"{'model':14s}" + "".join(f"{c:>10s}" for c in a.connections))
    for m in a.models:
        print(f"{m:14s}" + "".join(f"{(f'{h:.2f}' if not math.isnan(h := hours(m, c, a.epochs)) else 'n.a.'):>10s}" for c in a.connections))
    t = total(a.epochs, a.seeds)
    print(f"\nYour plan ({len(a.models)} models x {len(a.connections)} connections x {len(a.data)} data settings x "
          f"{a.seeds} seeds = {n_runs} runs, {a.epochs} epochs): {t:,.1f} GPU-hours = {t / 24:.1f} days on one GPU")
    if a.gpus > 1 or a.weekly_hours:
        wall = t / a.gpus
        weeks = f" = {wall / a.weekly_hours:.1f} weeks at {a.weekly_hours:g} session-hours per week" if a.weekly_hours else ""
        print(f"  on {a.gpus} GPU{'s' if a.gpus > 1 else ''} in parallel: about {wall:,.1f} hours of session time{weeks}"
              "  (assumes the GPUs are kept busy)")
    if skipped:
        print(f"  lower bound: no timing for {', '.join(skipped)} (did not fit in memory?), excluded")
    print("\nSensitivity of this plan, GPU-hours:")
    print(f"{'epochs':>8s}" + "".join(f"{f'{s} seed' + ('s' if s > 1 else ''):>12s}" for s in (3, 2, 1)))
    for e in (200, 100, 50, 30, 20, 10):
        print(f"{e:>8d}" + "".join(f"{total(e, s):>12,.1f}" for s in (3, 2, 1)))


if __name__ == "__main__":
    main()
