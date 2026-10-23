"""A small synthetic results folder with known answers, in the exact layout the trainer writes."""
import json
from pathlib import Path

import yaml

MODELS = {"resnet20": ("resnet", 20, 60.0), "vit_tiny_d6": ("vit", 6, 50.0)}      # name -> (family, depth, base accuracy)
OFFSET = {0: -0.2, 1: 0.0, 2: 0.2}                                                 # per-seed noise shared by all arms
# accuracy of a connection relative to the residual model on the same seed
GAIN = {"residual": [0.0, 0.0, 0.0], "dhc_n2": [1.0, 1.0, 1.0], "mhc_n4": [0.5, -0.5, 0.1]}
LAYERS = 4


def make_run(raw: Path, model: str, conn: str, percent: int, seed: int, epochs: int = 3, diverged: bool = False,
             acc=None, with_pt: bool = True) -> Path:
    family, depth, base = MODELS[model]
    kind = "residual" if conn == "residual" else conn.split("_n")[0]
    n = 1 if conn == "residual" else int(conn.split("_n")[1])
    acc = acc if acc is not None else base - (20.0 if percent == 10 else 0.0) + OFFSET[seed] + GAIN[conn][seed]
    run_id = f"{model}_{conn}_cifar100p{percent}_s{seed}"
    d = raw / run_id
    d.mkdir(parents=True)
    cfg = {"model": {"family": family, "depth": depth}, "connection": {"kind": kind, "n": n},
           "data": {"name": "cifar100", "percent": percent}, "seed": seed,
           "train": {"epochs": epochs, "batch_size": 128}}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    final = {"run_id": run_id, "seed": seed, "epochs_done": epochs, "epochs": epochs, "diverged": diverged,
             "final_acc": None if diverged else acc / 100, "best_acc": None if diverged else (acc + 0.5) / 100,
             "train_seconds": 10.0, "peak_mem_mb": 500.0, "params": 1000, "total_steps": 10 * epochs,
             "device": "Test GPU", "torch": "0.0"}
    (d / "final.json").write_text(json.dumps(final), encoding="utf-8")
    (d / "metrics.csv").write_text("epoch,step,train_loss,train_acc,test_loss,test_acc,lr\n" + "".join(
        f"{e},{10 * e},1.0,0.1,1.0,{(acc - 5 + 5 * e / epochs) / 100},0.1\n" for e in range(1, epochs + 1)), encoding="utf-8")
    ck = sorted({1, epochs})
    (d / "grad_norms.csv").write_text("epoch,layer,grad_norm\n" + "".join(
        f"{e},{k},{1.0 + 0.1 * k}\n" for e in ck for k in range(LAYERS)), encoding="utf-8")
    (d / "similarity.csv").write_text("epoch,layer,cosine,cka\n" + "".join(
        f"{e},{k},{'nan' if k == 2 else 0.5 + 0.1 * k},{0.9 - 0.05 * k}\n" for e in ck for k in range(LAYERS)), encoding="utf-8")
    if with_pt and conn != "residual":
        import torch
        eff = {k: {"w_read": torch.eye(n)[k % n], "w_write": torch.ones(n), "R": torch.eye(n) * 0.9 + 0.1 / n}
               for k in range(LAYERS)}
        torch.save({"state": {}, "effective": eff}, d / "connections.pt")
    return d


def make_raw(raw: Path, seeds=(0, 1, 2), percents=(100, 10), conns=tuple(GAIN)) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for percent in percents:
            for conn in conns:
                for seed in seeds:
                    make_run(raw, model, conn, percent, seed)
    return raw


def make_cost(path: Path) -> Path:
    rows = ["model,connection,params,macs_g,batch_size,step_ms,train_img_per_s,infer_img_per_s,peak_mem_mb,note"]
    mem = {"residual": 1.0, "dhc_n2": 1.5, "mhc_n4": 2.0}
    for m, size in (("resnet20", 100.0), ("vit_tiny_d6", 400.0)):
        for c in GAIN:
            rows.append(f"{m},{c},{int(1e6)},0.1,128,50,{2000 / mem[c]},{6000 / mem[c]},{size * mem[c]},")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path
