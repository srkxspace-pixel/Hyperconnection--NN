"""Cost-table inputs (T3, F5): parameters, forward MACs, peak training memory, training and inference throughput for every
model x connection, measured on the GPU you run it on (one batch size, fp32).

  python scripts/profile_cost.py                                  # all 6 models x 4 Tier-1 connections
  python scripts/profile_cost.py --models resnet20 resnet56 --connections residual dhc_n4 --steps 30

Writes results/tables/cost_<gpu>.csv. The columns of scripts/time_epoch.py are included, so scripts/budget.py reads it too.
MACs (1 MAC = 2 FLOPs) come from torch.utils.flop_counter and cover matmul / conv / attention only. The elementwise work
that dominates the overhead of HC and mHC is not counted, which is why memory and throughput are measured, not derived.
(Named profile_cost.py, not profile.py: a script called profile.py shadows the stdlib `profile` module that cProfile imports.)
"""
import argparse
import csv
import re
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from time_epoch import CONNECTIONS, MODELS, time_one      # same folder: scripts/ is on sys.path when run as a script
from hcsmall.models import build_model
from hcsmall.utils.io import load_config

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def inference_and_macs(model: str, connection: str, bs: int, steps: int, device: torch.device) -> dict:
    """Inference images per second (eval mode, one fixed batch) and forward GMACs per image."""
    cfg = load_config(str(ROOT / "configs/experiments/smoke.yaml"), model, connection, "cifar100_p100")
    net = build_model(cfg).to(device).eval()
    size = cfg["data"]["img_size"]
    x = torch.randn(bs, 3, size, size, device=device)
    try:
        for _ in range(5):
            net(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            net(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ips = bs * steps / (time.time() - t0)
        with FlopCounterMode(display=False) as fc:
            net(x[:1])
        return {"infer_img_per_s": round(ips), "macs_g": round(fc.get_total_flops() / 2 / 1e9, 4)}
    except torch.cuda.OutOfMemoryError:
        return {"infer_img_per_s": float("nan"), "macs_g": float("nan")}
    finally:
        del net, x
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--connections", nargs="+", default=CONNECTIONS)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    rows = []
    print(f"device: {gpu}  torch {torch.__version__}  batch {a.batch_size}")
    for m in a.models:
        for c in a.connections:
            r = time_one(m, c, a.batch_size, a.steps, device)
            r["train_img_per_s"] = r.pop("img_per_s")
            r.update(inference_and_macs(m, c, a.batch_size, 30, device) if not r["note"] else
                     {"infer_img_per_s": float("nan"), "macs_g": float("nan")})
            rows.append(r)
            print(f"{m:13s} {c:9s} params {r['params']:>9,d}  GMACs {r['macs_g']}  train {r['train_img_per_s']} img/s  "
                  f"infer {r['infer_img_per_s']} img/s  peak {r['peak_mem_mb']} MiB {r['note']}", flush=True)
    out = ROOT / "results/tables" / f"cost_{re.sub(r'[^A-Za-z0-9]+', '_', gpu).strip('_')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
