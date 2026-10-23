"""Load finished runs from a results folder and compute the paired statistics of D10. No torch needed."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

CONNECTIONS = ["residual", "dhc_n2", "dhc_n4", "mhc_n4", "shc_n2"]
MODELS = ["resnet20", "resnet56", "resnet110", "vit_tiny_d6", "vit_tiny_d12", "vit_tiny_d24"]
DATA = ["cifar100_p100", "cifar100_p25", "cifar100_p10", "tinyimagenet_p100"]
LABEL = {"residual": "Residual", "dhc_n2": "DHC×2", "dhc_n4": "DHC×4", "mhc_n4": "mHC×4", "shc_n2": "SHC×2"}
TEX = {"residual": "Residual", "dhc_n2": r"DHC$\times$2", "dhc_n4": r"DHC$\times$4", "mhc_n4": r"mHC$\times$4",
       "shc_n2": r"SHC$\times$2"}
MODEL_LABEL = {"resnet20": "ResNet-20", "resnet56": "ResNet-56", "resnet110": "ResNet-110",
               "vit_tiny_d6": "ViT-Tiny-6", "vit_tiny_d12": "ViT-Tiny-12", "vit_tiny_d24": "ViT-Tiny-24"}
DATA_LABEL = {"cifar100_p100": "CIFAR-100, 100% of the training set", "cifar100_p25": "CIFAR-100, 25%",
              "cifar100_p10": "CIFAR-100, 10%", "tinyimagenet_p100": "Tiny-ImageNet, 100%"}
DATA_SHORT = {"cifar100_p100": "100%", "cifar100_p25": "25%", "cifar100_p10": "10%", "tinyimagenet_p100": "Tiny-ImageNet"}


def order(values, canonical) -> list:
    """The values present, in canonical order (unknown ones last, sorted)."""
    present = set(values)
    return [v for v in canonical if v in present] + sorted(present - set(canonical))


def load_runs(raw: Path) -> pd.DataFrame:
    """One row per finished run (a folder with config.yaml and final.json). Accuracies are in percent."""
    rows = []
    for fj in sorted(Path(raw).glob("*/final.json")):
        run = fj.parent
        if not (run / "config.yaml").exists():
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        f = json.loads(fj.read_text(encoding="utf-8"))
        m, c, d, t = cfg["model"], cfg["connection"], cfg["data"], cfg["train"]
        pct = lambda v: 100.0 * v if v is not None else math.nan
        rows.append(dict(
            run_id=f["run_id"], family=m["family"], depth=m["depth"],
            model=f"resnet{m['depth']}" if m["family"] == "resnet" else f"vit_tiny_d{m['depth']}",
            conn="residual" if c["kind"] == "residual" else f"{c['kind']}_n{c['n']}",
            data=f"{d['name']}_p{d['percent']}", seed=cfg["seed"], epochs=t["epochs"],
            epoch_steps=f["total_steps"] // t["epochs"], batch_size=t["batch_size"],
            final_acc=pct(f["final_acc"]), best_acc=pct(f["best_acc"]), diverged=bool(f["diverged"]),
            epochs_done=f["epochs_done"], train_seconds=f["train_seconds"], peak_mem_mb=f["peak_mem_mb"],
            params=f["params"], device=f["device"], torch=f["torch"], path=str(run)))
    return pd.DataFrame(rows)


def budget_problems(df: pd.DataFrame) -> list:
    """Runs of one family on one data setting must share their training budget (one variable at a time)."""
    out = []
    for (data, family), g in df.groupby(["data", "family"]):
        budgets = g[["epochs", "epoch_steps", "batch_size"]].drop_duplicates().to_dict("records")
        if len(budgets) > 1:
            out.append(f"{family} on {data} mixes training budgets: {budgets}")
    return out


def accs(df: pd.DataFrame, model: str, data: str, conn: str) -> pd.Series:
    """Final test accuracy per seed of the non-diverged runs of one table cell."""
    g = df[(df.model == model) & (df.data == data) & (df.conn == conn) & ~df.diverged]
    return g.set_index("seed").final_acc.dropna()


def paired(df: pd.DataFrame, model: str, data: str, conn: str, base: str = "residual", n_boot: int = 10_000) -> dict:
    """D10: difference to the baseline paired by seed, with a paired t-test and a percentile bootstrap 95% CI.

    `sig` needs both: p < 0.05 and a CI that excludes 0. With few seeds the bootstrap CI is coarse (with three pairs it
    excludes 0 exactly when all three differences have the same sign), which is why D11 asks for more seeds when borderline.
    """
    a, b = accs(df, model, data, conn), accs(df, model, data, base)
    common = a.index.intersection(b.index)
    d = (a[common] - b[common]).to_numpy(float)
    out = dict(n=len(d), delta=math.nan, sd=math.nan, p=math.nan, lo=math.nan, hi=math.nan, sig=False)
    if len(d) == 0:
        return out
    out["delta"] = float(d.mean())
    if len(d) < 2:
        return out
    out["sd"] = float(d.std(ddof=1))
    out["p"] = (0.0 if d.mean() != 0 else 1.0) if out["sd"] == 0 else float(stats.ttest_rel(a[common], b[common]).pvalue)
    means = d[np.random.default_rng(0).integers(0, len(d), (n_boot, len(d)))].mean(1)
    out["lo"], out["hi"] = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    out["sig"] = bool(out["p"] < 0.05 and (out["lo"] > 0 or out["hi"] < 0))
    return out
