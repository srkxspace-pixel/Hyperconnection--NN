"""One run: a resolved config in, results/raw/<run_id>/ out."""
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ..analysis.connection_viz import effective_weights
from ..analysis.grad_norms import layer_grad_norms
from ..analysis.similarity import consecutive_similarity
from ..data import make_data
from ..models import build_model
from ..utils.io import git_commit, write_json
from ..utils.runid import run_id_of
from ..utils.seed import set_seed
from .optim import build_optimizer
from .schedule import lr_at

# Settings that may differ between a run and its resume / rerun without changing the experiment.
_IGNORED = ("root", "paths", "train.num_workers", "train.checkpoint_every", "train.stop_at", "train.save_weights",
            "data.mean", "data.std")


class Paused(Exception):
    """Raised after a finished epoch when the next one would not end before `train.stop_at` (a session time limit).
    The resume checkpoint is written first, so running the same command again continues the run."""


def _config_changes(old: dict, new: dict, prefix: str = "") -> list:
    """Dotted keys whose values differ between two resolved configs, ignoring where / how the run executes."""
    out = []
    for k in sorted(set(old) | set(new)):
        key = f"{prefix}{k}"
        a, b = old.get(k), new.get(k)
        if key in _IGNORED:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            out += _config_changes(a, b, key + ".")
        elif a != b:
            out.append(f"{key}: {a!r} -> {b!r}")
    return out


def _batches(loader):
    while True:                      # re-iterating reshuffles; epoch_steps need not equal one pass (D6)
        yield from loader


@torch.no_grad()
def evaluate(model, test, device, batch_size: int = 256) -> tuple[float, float]:
    """Loss and accuracy on `test` = (images, labels) tensors, in eval mode."""
    x_all, y_all = test
    model.eval()
    loss = correct = 0
    for i in range(0, len(y_all), batch_size):
        x, y = x_all[i:i + batch_size].to(device), y_all[i:i + batch_size].to(device)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
    model.train()
    return loss / len(y_all), correct / len(y_all)


def _write_csv(path: Path, header: list, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def checkpoint_epochs(epochs: int, fractions: list) -> set:
    """End of epoch 1 and of 25% / 50% / 100% of training."""
    return {1} | {max(1, round(f * epochs)) for f in fractions}


def train(cfg: dict) -> Path:
    t, an = cfg["train"], cfg["analysis"]
    root = Path(cfg["root"])
    rid = run_id_of(cfg)
    out, scratch = root / cfg["paths"]["results_dir"] / rid, root / cfg["paths"]["runs_dir"] / rid
    stored = out / "config.yaml"
    if stored.exists():                           # same run id, different settings: refuse instead of skipping / mixing
        changes = _config_changes(yaml.safe_load(stored.read_text(encoding="utf-8")), cfg)
        if changes:
            raise RuntimeError(f"{out} already holds this run with different settings ({'; '.join(changes)}). "
                               "Use another paths.results_dir for a different plan.")
    if (out / "final.json").exists():
        print(f"[skip] {rid}: final.json exists")
        return out
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    if not stored.exists():
        stored.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)           # built before any data RNG use: equal seeds give equal backbones
    data = make_data(cfg)
    opt = build_optimizer(model, t)
    epoch_steps = t["epoch_steps"] or len(data["train_loader"])
    total, warmup = t["epochs"] * epoch_steps, t["warmup_epochs"] * epoch_steps
    ckpts = checkpoint_epochs(t["epochs"], an["checkpoints"])
    n_params = sum(p.numel() for p in model.parameters())

    rows, grad_rows, sim_rows, start, best, elapsed, peak = [], [], [], 0, 0.0, 0.0, 0.0
    if (scratch / "last.pt").exists():            # resume an interrupted run from its last finished epoch
        ck = torch.load(scratch / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"])
        rows, grad_rows, sim_rows = ck["rows"], ck["grad_rows"], ck["sim_rows"]
        start, best, elapsed, peak = ck["epoch"], ck["best"], ck["elapsed"], ck["peak"]
        print(f"[resume] {rid} from epoch {start}")

    stream, diverged = _batches(data["train_loader"]), False
    model.train()
    for epoch in range(start + 1, t["epochs"] + 1):
        t0, loss_sum, correct, seen = time.time(), torch.zeros((), device=device), torch.zeros((), device=device), 0
        for i in range(epoch_steps):
            step = (epoch - 1) * epoch_steps + i
            if device.type == "cuda" and i == 3 and epoch == start + 1:
                torch.cuda.reset_peak_memory_stats()      # steady state: drop cudnn autotuning workspaces from the peak
            lr = lr_at(step, total, warmup, t["lr"])
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = next(stream)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=t["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(y)
            correct += (logits.argmax(1) == y).sum()
            seen += len(y)
            if i % 50 == 0 and not torch.isfinite(loss_sum):
                diverged = True
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.time() - t0
        elapsed += train_time
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
            torch.cuda.reset_peak_memory_stats()
        if diverged or not math.isfinite(float(loss_sum)):
            diverged = True
            print(f"[diverged] {rid} at epoch {epoch}")
            break
        test_loss, test_acc = evaluate(model, data["test"], device)
        best = max(best, test_acc)
        rows.append([epoch, epoch * epoch_steps, float(loss_sum) / seen, float(correct) / seen, test_loss, test_acc, lr])
        print(f"{rid} epoch {epoch}/{t['epochs']} train {rows[-1][2]:.3f}/{rows[-1][3]:.3f} test {test_loss:.3f}/{test_acc:.3f} "
              f"lr {lr:.4f} {train_time:.0f}s", flush=True)
        if epoch in ckpts:
            gx, gy = data["grad"]
            for layer, v in enumerate(layer_grad_norms(model, gx, gy, t["batch_size"], t["label_smoothing"], device)):
                grad_rows.append([epoch, layer, v])
            for layer, (cos, cka) in enumerate(consecutive_similarity(model, data["probe"][0], device)):
                sim_rows.append([epoch, layer, cos, cka])
            _write_csv(out / "grad_norms.csv", ["epoch", "layer", "grad_norm"], grad_rows)
            _write_csv(out / "similarity.csv", ["epoch", "layer", "cosine", "cka"], sim_rows)
        _write_csv(out / "metrics.csv", ["epoch", "step", "train_loss", "train_acc", "test_loss", "test_acc", "lr"], rows)
        # time limit: stop here if another epoch (as long as this one, plus 30%) would not end before train.stop_at
        pause = bool(t.get("stop_at")) and epoch < t["epochs"] and time.time() + 1.3 * (time.time() - t0) > t["stop_at"]
        if epoch < t["epochs"] and (pause or epoch % t["checkpoint_every"] == 0):      # the last epoch needs no resume point
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "rows": rows, "grad_rows": grad_rows,
                        "sim_rows": sim_rows, "epoch": epoch, "best": best, "elapsed": elapsed, "peak": peak},
                       scratch / "last.pt")
        if pause:
            raise Paused(f"[paused] {rid} after epoch {epoch}/{t['epochs']}: time limit reached, progress saved")

    if rows and not diverged and model.conns[0].n > 1:
        torch.save({"state": {k: v.cpu() for k, v in model.conns.state_dict().items()},
                    "effective": effective_weights(model, data["probe"][0], device)}, out / "connections.pt")
    if rows and not diverged and t.get("save_weights"):      # the demo page (app/server.py) rebuilds the network from cfg and loads these
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "cfg": cfg}, out / "model.pt")
    done = len(rows)
    write_json(out / "final.json", {
        "run_id": rid, "seed": cfg["seed"], "epochs_done": done, "epochs": t["epochs"], "diverged": diverged,
        "final_acc": rows[-1][5] if rows and not diverged else None, "best_acc": best if rows else None,
        "final_test_loss": rows[-1][4] if rows and not diverged else None,
        "train_seconds": elapsed, "train_img_per_s": done * epoch_steps * t["batch_size"] / elapsed if elapsed else None,
        "peak_mem_mb": peak, "params": n_params, "n_train": data["n_train"], "total_steps": total,
        "git_commit": git_commit(root), "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__})
    (scratch / "last.pt").unlink(missing_ok=True)
    try:
        scratch.rmdir()              # the run is complete: leave no empty scratch directory behind
    except OSError:
        pass
    return out
