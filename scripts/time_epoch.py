"""Time a training step of every model x connection pair on this machine, for budget planning.

  python scripts/time_epoch.py                                  # all 6 models x 4 Tier-1 connections
  python scripts/time_epoch.py --models resnet56 vit_tiny_d12 --connections residual dhc_n4 --steps 30

Uses one fixed random batch, so it measures GPU compute and memory only (not the data loader). The step is the real
one: same model, optimizer, fp32, cudnn.benchmark. Writes results/tables/timing_<device>.csv and prints a table.
Run it on the machine that will run Tier 1: epoch/run estimates assume 50,000 images per epoch (one CIFAR-100 pass).
"""
import argparse
import csv
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from hcsmall.models import build_model
from hcsmall.training.optim import build_optimizer
from hcsmall.utils.io import load_config

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["resnet20", "resnet56", "resnet110", "vit_tiny_d6", "vit_tiny_d12", "vit_tiny_d24"]
CONNECTIONS = ["residual", "dhc_n2", "dhc_n4", "mhc_n4"]


def time_one(model: str, connection: str, bs: int, steps: int, device: torch.device) -> dict:
    cfg = load_config(str(ROOT / "configs/experiments/smoke.yaml"), model, connection, "cifar100_p100")
    torch.manual_seed(0)
    net = build_model(cfg).to(device).train()
    opt = build_optimizer(net, cfg["train"])
    x = torch.randn(bs, 3, cfg["data"]["img_size"], cfg["data"]["img_size"], device=device)
    y = torch.randint(0, cfg["data"]["num_classes"], (bs,), device=device)
    row = {"model": model, "connection": connection, "params": sum(p.numel() for p in net.parameters()),
           "batch_size": bs, "step_ms": None, "img_per_s": None, "peak_mem_mb": None, "epoch_s": None, "run200_h": None, "note": ""}
    try:
        for i in range(5 + steps):
            if i == 5:                       # warm-up done: cudnn autotuning (large temporary workspaces) is behind us
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
            loss = F.cross_entropy(net(x), y, label_smoothing=cfg["train"]["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) / steps * 1000
        epoch = 50_000 / bs * ms / 1000
        row.update(step_ms=round(ms, 1), img_per_s=round(bs / ms * 1000), epoch_s=round(epoch, 1), run200_h=round(200 * epoch / 3600, 2),
                   peak_mem_mb=round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0)
    except torch.cuda.OutOfMemoryError:
        row["note"] = "OOM"
    del net, opt, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--connections", nargs="+", default=CONNECTIONS)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    rows = []
    print(f"device: {name}  torch {torch.__version__}  batch {a.batch_size}")
    print(f"{'model':14s} {'connection':9s} {'params':>9s} {'step ms':>8s} {'img/s':>7s} {'peak MiB':>8s} {'epoch s':>8s} {'200ep h':>8s}")
    for m in a.models:
        for c in a.connections:
            r = time_one(m, c, a.batch_size, a.steps, device)
            rows.append(r)
            print(f"{m:14s} {c:9s} {r['params']:9d} {r['step_ms'] or '-':>8} {r['img_per_s'] or '-':>7} "
                  f"{r['peak_mem_mb'] if r['peak_mem_mb'] is not None else '-':>8} {r['epoch_s'] or '-':>8} {r['run200_h'] or r['note'] or '-':>8}", flush=True)
    out = ROOT / "results/tables" / f"timing_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
