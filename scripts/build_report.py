"""Fill docs/templates/*.md with the numbers in results/ and write docs/PROJECT_REPORT.md and docs/STUDENT_GUIDE.md.

  python scripts/build_report.py            # write both files
  python scripts/build_report.py --check    # exit 1 if the written files differ from what the data gives

Rule: no result is typed by hand. The templates contain prose and [[placeholders]]; every placeholder is filled from
results/tables/*.csv and results/raw/*/ (accuracy, cost, similarity, gradient norms). An unknown placeholder is an error.
Needs only pandas and pyyaml (no torch, no GPU).
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TABLES, RAW, TEMPLATES, DOCS = ROOT / "results/tables", ROOT / "results/raw", ROOT / "docs/templates", ROOT / "docs"
OUTPUTS = {"project_report.md": "PROJECT_REPORT.md", "student_guide.md": "STUDENT_GUIDE.md"}

MODELS = ["resnet20", "resnet56", "vit_tiny_d6"]
MODEL_NAME = {"resnet20": "ResNet-20", "resnet56": "ResNet-56", "vit_tiny_d6": "ViT-Tiny (6 blocks)"}
CONNS = ["residual", "dhc_n2", "dhc_n4", "mhc_n4"]
CONN_NAME = {"residual": "Normal shortcut", "dhc_n2": "HC, 2 lanes", "dhc_n4": "HC, 4 lanes", "mhc_n4": "mHC, 4 lanes"}
DATAS = {"p100": "cifar100_p100", "p10": "cifar100_p10"}
DATA_NAME = {"p100": "all training data", "p10": "10% of the training data"}
MINUS = "−"


def sgn(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}f}".replace("-", MINUS)


def f(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace("-", MINUS)


def rng(xs, fmt=f) -> tuple:
    return fmt(min(xs)), fmt(max(xs))


def md_table(header: list, rows: list) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def mechanism(runs: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-layer similarity at the end of training and the gradient-norm profile after epoch 1, per run."""
    recs = []
    for r in runs.itertuples():
        sim, grad = pd.read_csv(RAW / r.run_id / "similarity.csv"), pd.read_csv(RAW / r.run_id / "grad_norms.csv")
        s, g = sim[sim.epoch == sim.epoch.max()], grad[grad.epoch == grad.epoch.min()].grad_norm.to_numpy()
        recs.append(dict(model=r.model, conn=r.conn, data=r.data, cka=s.cka.mean(), cos=s.cosine.mean(), gratio=g[-1] / g[0]))
    return pd.DataFrame(recs).set_index(["model", "conn", "data"])


def build_values() -> dict:
    runs = pd.read_csv(TABLES / "runs.csv")
    cost = pd.read_csv(TABLES / "T3_cost.csv").set_index(["model", "connection"])
    mech = mechanism(runs)
    acc = runs.set_index(["model", "conn", "data"])
    v: dict = {}

    def put(key, val):
        v[key] = val if isinstance(val, str) else str(val)

    put("n_runs", len(runs))
    put("n_seeds", int(runs.groupby(["model", "conn", "data"]).size().max()))
    put("epochs", int(runs.epochs.iloc[0]))
    put("steps_per_epoch", int(runs.epoch_steps.iloc[0]))
    put("batch_size", int(runs.batch_size.iloc[0]))
    put("gpu_hours", f(runs.train_seconds.sum() / 3600, 1))
    put("diverged", int(runs.diverged.sum()))
    put("device", runs.device.iloc[0])
    n = int(runs.groupby(["model", "conn", "data"]).size().max())
    put("once", "once" if n == 1 else f"{n} times")
    put("runword", "one run" if n == 1 else f"{n} runs")

    for m in MODELS:
        for d, dn in DATAS.items():
            base = acc.loc[(m, "residual", dn), "final_acc"]
            for c in CONNS:
                a = acc.loc[(m, c, dn), "final_acc"]
                put(f"acc.{m}.{c}.{d}", f(a))
                put(f"best.{m}.{c}.{d}", f(acc.loc[(m, c, dn), "best_acc"]))
                put(f"gain.{m}.{c}.{d}", sgn(a - base))
                put(f"absgain.{m}.{c}.{d}", f(abs(a - base)))

    def gains(models, conns, d):
        return [acc.loc[(m, c, DATAS[d]), "final_acc"] - acc.loc[(m, "residual", DATAS[d]), "final_acc"] for m in models for c in conns]

    resnets, vit = ["resnet20", "resnet56"], ["vit_tiny_d6"]
    for name, models, conns in (("resnet_hc", resnets, ["dhc_n2", "dhc_n4"]), ("resnet_all", resnets, CONNS[1:]), ("vit_all", vit, CONNS[1:])):
        for d in DATAS:
            lo, hi = rng(gains(models, conns, d), sgn)
            put(f"agg.{name}.{d}.lo", lo)
            put(f"agg.{name}.{d}.hi", hi)
            lo, hi = rng([abs(x) for x in gains(models, conns, d)])
            put(f"agg.{name}.{d}.abslo", lo)
            put(f"agg.{name}.{d}.abshi", hi)

    # cost: parameters, profiled step time and memory, real training time (100% data)
    hours = runs.assign(h=runs.train_seconds / 3600).set_index(["model", "conn", "data"]).h
    for m in MODELS:
        for c in CONNS:
            r = cost.loc[(m, c)]
            put(f"params.{m}.{c}", f"{int(r.params):,}")
            put(f"paramsup.{m}.{c}", f(100 * (r.params / cost.loc[(m, "residual"), "params"] - 1), 1))
            put(f"gmacs.{m}.{c}", f(r.macs_g, 3))
            put(f"slow.{m}.{c}", f(r.train_slowdown_x, 1))
            put(f"mem.{m}.{c}", f(r.mem_x, 1))
            put(f"infer.{m}.{c}", f(r.infer_slowdown_x, 1))
            put(f"memmb.{m}.{c}", f(r.peak_mem_mb, 0))
            put(f"hours.{m}.{c}", f(hours[(m, c, DATAS["p100"])], 2))
            put(f"hoursx.{m}.{c}", f(hours[(m, c, DATAS["p100"])] / hours[(m, "residual", DATAS["p100"])], 1))
    for c in CONNS[1:]:
        for key, col in (("slow", "train_slowdown_x"), ("mem", "mem_x")):
            lo, hi = rng([cost.loc[(m, c), col] for m in MODELS], lambda x: f(x, 1))
            put(f"range.{key}.{c}.lo", lo)
            put(f"range.{key}.{c}.hi", hi)
        lo, hi = rng([hours[(m, c, DATAS["p100"])] / hours[(m, "residual", DATAS["p100"])] for m in MODELS], lambda x: f(x, 1))
        put(f"range.hoursx.{c}.lo", lo)
        put(f"range.hoursx.{c}.hi", hi)
        lo, hi = rng([100 * (cost.loc[(m, c), "params"] / cost.loc[(m, "residual"), "params"] - 1) for m in MODELS], lambda x: f(x, 1))
        put(f"range.paramsup.{c}.lo", lo)
        put(f"range.paramsup.{c}.hi", hi)
    for key, conns in (("hc", ["dhc_n2", "dhc_n4"]), ("all", CONNS[1:])):           # ranges over several connection types
        for name, vals in (("hoursx", [hours[(m, c, DATAS["p100"])] / hours[(m, "residual", DATAS["p100"])] for m in MODELS for c in conns]),
                           ("slow", [cost.loc[(m, c), "train_slowdown_x"] for m in MODELS for c in conns]),
                           ("mem", [cost.loc[(m, c), "mem_x"] for m in MODELS for c in conns])):
            lo, hi = rng(vals, lambda x: f(x, 1))
            put(f"range.{name}.{key}.lo", lo)
            put(f"range.{name}.{key}.hi", hi)
    put("step_ms.resnet20.residual", f(cost.loc[("resnet20", "residual"), "step_ms"], 1))
    put("imgs.resnet20.residual", f"{int(cost.loc[('resnet20', 'residual'), 'train_img_per_s']):,}")

    # mechanism
    for m in MODELS:
        for c in CONNS:
            for d in DATAS:
                r = mech.loc[(m, c, DATAS[d])]
                put(f"cka.{m}.{c}.{d}", f(r.cka, 3))
                put(f"cos.{m}.{c}.{d}", f(r["cos"], 3))
                put(f"gratio.{m}.{c}.{d}", f(r.gratio, 2))
        for d in DATAS:
            lo, hi = rng([mech.loc[(m, c, DATAS[d]), "cka"] for c in CONNS], lambda x: f(x, 3))
            put(f"cka.{m}.range.{d}.lo", lo)
            put(f"cka.{m}.range.{d}.hi", hi)

    # tables
    def acc_cell(m, c, d):
        a = acc.loc[(m, c, DATAS[d]), "final_acc"]
        return f(a) if c == "residual" else f"{f(a)} ({sgn(a - acc.loc[(m, 'residual', DATAS[d]), 'final_acc'])})"

    rows = [[MODEL_NAME[m], DATA_NAME[d]] + [acc_cell(m, c, d) for c in CONNS] for d in DATAS for m in MODELS]
    put("table.accuracy", md_table(["Model", "Training data"] + [CONN_NAME[c] for c in CONNS], rows))

    rows = []
    for m in MODELS:
        for c in CONNS:
            r = cost.loc[(m, c)]
            rows.append([MODEL_NAME[m], CONN_NAME[c], f"{int(r.params):,}", f(r.macs_g, 3), f(r.peak_mem_mb, 0), f(r.mem_x, 1) + "x",
                         f(r.train_slowdown_x, 1) + "x", f(hours[(m, c, DATAS['p100'])], 2), f(hours[(m, c, DATAS['p100'])] / hours[(m, 'residual', DATAS['p100'])], 1) + "x"])
    put("table.cost", md_table(["Model", "Connection", "Parameters", "GMACs", "Peak memory (MiB)", "Memory vs normal",
                                "Step time vs normal", "Real training time (h)", "Time vs normal"], rows))

    rows = [[MODEL_NAME[m], CONN_NAME[c], f(mech.loc[(m, c, DATAS["p100"]), "cka"], 3), f(mech.loc[(m, c, DATAS["p100"]), "cos"], 3),
             f(mech.loc[(m, c, DATAS["p100"]), "gratio"], 2)] for m in MODELS for c in CONNS]
    put("table.mechanism", md_table(["Model", "Connection", "Mean CKA of consecutive layers", "Mean cosine of consecutive layers",
                                     "Gradient norm, last / first layer"], rows))

    rows = [[r.run_id, MODEL_NAME[r.model], CONN_NAME[r.conn], DATA_NAME[[k for k, x in DATAS.items() if x == r.data][0]], f(r.final_acc), f(r.best_acc),
             f(r.train_seconds / 3600, 2), f(r.peak_mem_mb, 0)] for r in runs.sort_values(["data", "model", "conn"], ascending=[False, True, True]).itertuples()]
    put("table.runs", md_table(["Run", "Model", "Connection", "Data", "Final top-1 (%)", "Best top-1 (%)", "Training time (h)", "Peak memory (MiB)"], rows))

    # training recipes, read from the configuration files that were stored with the runs
    cfg = {k: yaml.safe_load((RAW / f"{k}_residual_n1_cifar100p100_s0" / "config.yaml").read_text(encoding="utf-8"))["train"]
           for k in ("resnet20", "vittiny6")}

    def aug(t):
        a = t["augment"]
        return f"random crop (padding {a['crop_padding']}), flip" + (f", RandAugment ({a['randaugment']['num_ops']} operations, strength {a['randaugment']['magnitude']})" if a["randaugment"] else "")

    def opt(t):
        return f"SGD, momentum {t['momentum']}, Nesterov" if t["optimizer"] == "sgd" else "AdamW"

    r, vt = cfg["resnet20"], cfg["vittiny6"]
    rows = [["Optimizer", opt(r), opt(vt)], ["Learning rate", r["lr"], vt["lr"]], ["Weight decay", r["weight_decay"], vt["weight_decay"]],
            ["Schedule", f"{r['schedule']} to zero, warm-up {r['warmup_epochs']} epochs", f"{vt['schedule']} to zero, warm-up {vt['warmup_epochs']} epochs"],
            ["Label smoothing", r["label_smoothing"], vt["label_smoothing"]], ["Augmentation", aug(r), aug(vt)],
            ["Batch size", r["batch_size"], vt["batch_size"]], ["Epochs", r["epochs"], vt["epochs"]],
            ["Steps per epoch", r["epoch_steps"], vt["epoch_steps"]]]
    put("table.setup", md_table(["Setting", "ResNets", "ViT-Tiny"], rows))
    put("vit_warmup", int(vt["warmup_epochs"]))
    return v


def fill(text: str, values: dict, name: str) -> str:
    def sub(m):
        key = m.group(1)
        if key not in values:
            raise KeyError(f"{name}: unknown placeholder [[{key}]]")
        return values[key]

    return re.sub(r"\[\[([A-Za-z0-9_.]+)\]\]", sub, text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if the written files are not what the templates and data give")
    a = ap.parse_args()
    values = build_values()
    stale = []
    for tpl, out in OUTPUTS.items():
        banner = f"<!-- Generated by scripts/build_report.py from docs/templates/{tpl} and results/. Edit the template, not this file. -->\n\n"
        text = banner + fill((TEMPLATES / tpl).read_text(encoding="utf-8"), values, tpl)
        path = DOCS / out
        if a.check:
            if not path.exists() or path.read_text(encoding="utf-8").replace("\r\n", "\n") != text:
                stale.append(out)
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
            print("wrote", path.relative_to(ROOT), f"({len(text.split())} words)")
    if a.check:
        print("reports are up to date" if not stale else "STALE: " + ", ".join(stale))
    return int(bool(stale))


if __name__ == "__main__":
    sys.exit(main())
