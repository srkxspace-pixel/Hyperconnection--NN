"""One config -> one run directory in results/raw/.

  python scripts/train.py --config configs/experiments/smoke.yaml
  python scripts/train.py --config configs/experiments/tier1.yaml --model resnet56 --connection dhc_n4 \
         --data cifar100_p10 --seed 1 --set train.epochs=30
"""
import argparse
import sys

from hcsmall.training.trainer import Paused, train
from hcsmall.utils.io import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/experiments/<name>.yaml")
    ap.add_argument("--model"), ap.add_argument("--connection"), ap.add_argument("--data")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. train.epochs=30")
    a = ap.parse_args()
    overrides = ([f"seed={a.seed}"] if a.seed is not None else []) + a.set
    try:
        train(load_config(a.config, a.model, a.connection, a.data, overrides))
    except Paused as e:                  # train.stop_at reached: progress is saved, run the same command again to continue
        print(e, flush=True)
        sys.exit(75)                     # EX_TEMPFAIL; scripts/run_matrix.py stops launching further runs on it


if __name__ == "__main__":      # required on Windows: DataLoader workers re-import this module
    main()
