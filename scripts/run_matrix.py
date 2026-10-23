"""Launch a tier of runs. Resumable: a run whose <results_dir>/<run_id>/final.json exists is skipped, and an
interrupted run continues from its last finished epoch.

  python scripts/run_matrix.py --tier smoke                 # every model x connection, 2 tiny epochs
  python scripts/run_matrix.py --tier 1 --dry-run           # list the 144 Tier 1 runs and which are done
  python scripts/run_matrix.py --tier 1 --set train.epochs=30
  python scripts/run_matrix.py --tier 1 --only resnet56     # only run ids containing this text
  python scripts/run_matrix.py --tier 1 --models resnet20 vit_tiny_d6 --connections residual dhc_n4 --seeds 0 1

`--models/--connections/--data/--seeds` replace the corresponding lists of the tier file (a sub-grid).
Order: seed 0 for every cell first, then seed 1, ... so a partial campaign still covers every cell.
Each run is a separate process; its log is <results_dir>/<run_id>/train.log and failures go to <results_dir>/_failures.txt.
Redirect output with e.g. --set paths.results_dir=/content/drive/MyDrive/nn09/raw --set paths.runs_dir=/content/drive/MyDrive/nn09/runs
A results folder that already holds a run with *different* settings is refused (use another results_dir per plan).

Free GPU sessions end (Colab and Kaggle after at most 12 h), and some machines have two GPUs:
  --stop-after-hours 11   no new run is started in the last minutes, and the running one pauses cleanly after the epoch that
                          would not end in time (progress saved; exit 75): run the same command again to continue.
  --claim-dir DIR         several launchers (one per GPU, each with its own CUDA_VISIBLE_DEVICES) share the work: a run is
                          taken by creating a file DIR/<run_id>, so no run is started twice. Use a fresh DIR per session.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from hcsmall.utils.io import load_config, load_grid
from hcsmall.utils.runid import run_id_of

ROOT = Path(__file__).resolve().parents[1]
PAUSED = 75          # exit code of scripts/train.py when train.stop_at was reached (progress saved)


def claim(claim_dir, run_id: str) -> bool:
    """True if this launcher may run `run_id`: the claim file is created atomically, so exactly one launcher gets it."""
    if claim_dir is None:
        return True
    claim_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(claim_dir / run_id, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", required=True, help="1, 2, smoke, ... (configs/experiments/tier<N>.yaml or <name>.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--models", nargs="+"), ap.add_argument("--connections", nargs="+"), ap.add_argument("--data", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--only", default="", help="run only run ids containing this text")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-after-hours", type=float, help="session budget: pause cleanly before this many hours have passed")
    ap.add_argument("--min-start-minutes", type=float, default=10, help="do not start a new run with less time than this left")
    ap.add_argument("--claim-dir", type=Path, help="shared folder for several launchers (see above)")
    a = ap.parse_args()
    t_end = time.time() + a.stop_after_hours * 3600 if a.stop_after_hours else None

    exp = ROOT / "configs/experiments" / (f"tier{a.tier}.yaml" if a.tier.isdigit() else f"{a.tier}.yaml")
    g = load_grid(str(exp))["grid"]
    for key, chosen in (("models", a.models), ("connections", a.connections), ("data", a.data), ("seeds", a.seeds)):
        if chosen:
            g[key] = chosen
    jobs = []
    for seed in g["seeds"]:
        for d in g["data"]:
            for m in g["models"]:
                for c in g["connections"]:
                    cfg = load_config(str(exp), m, c, d, [f"seed={seed}", *a.set])
                    rid = run_id_of(cfg)
                    if a.only in rid:
                        results_dir = ROOT / cfg["paths"]["results_dir"]      # absolute paths (e.g. on Drive) replace ROOT
                        jobs.append((rid, m, c, d, seed, results_dir / rid))
    done = sum((out / "final.json").exists() for *_, out in jobs)
    print(f"{len(jobs)} runs, {done} already complete", flush=True)
    for i, (rid, m, c, d, seed, out) in enumerate(jobs, 1):
        if (out / "final.json").exists():
            continue
        if a.dry_run:
            print(f"  todo  {rid}")
            continue
        if t_end and t_end - time.time() < a.min_start_minutes * 60:
            print(f"time budget reached: {rid} and the runs after it are not started; run the same command again to continue",
                  flush=True)
            break
        if not claim(a.claim_dir, rid):
            continue                                  # another launcher (another GPU) has taken this run
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(exp), "--model", m, "--connection", c,
               "--data", d, "--seed", str(seed)] + [x for s in a.set for x in ("--set", s)]
        if t_end:
            cmd += ["--set", f"train.stop_at={int(t_end)}"]
        with (out / "train.log").open("a", encoding="utf-8") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        if rc == 0:
            print(f"[{i}/{len(jobs)}] {rid}: ok", flush=True)
        elif rc == PAUSED:
            print(f"[{i}/{len(jobs)}] {rid}: paused at the time limit, progress saved; run the same command again to continue",
                  flush=True)
            break
        else:
            tail = "".join((out / "train.log").read_text(encoding="utf-8", errors="replace").splitlines(True)[-8:])
            print(f"[{i}/{len(jobs)}] {rid}: FAILED (exit {rc}); see {out / 'train.log'}", flush=True)
            with (out.parent / "_failures.txt").open("a", encoding="utf-8") as f:
                f.write(f"{rid} exit={rc}\n{tail}\n")


if __name__ == "__main__":
    main()
