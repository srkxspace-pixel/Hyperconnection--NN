"""Tables T1-T4 as CSV + LaTeX, plus a coverage table, the setup table and LaTeX macros. No torch needed.

Every number comes from the run folders (and the cost CSV); nothing is typed by hand. Accuracies are final-epoch top-1 in
percent (final, not best, is the headline); std is the sample std over seeds.
"""
import math
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from .runs import (CONNECTIONS, DATA, DATA_LABEL, DATA_SHORT, LABEL, MODEL_LABEL, MODELS, TEX, accs, budget_problems,
                   load_runs, order, paired)

MISSING = "--"
STAR = "$^{*}$"


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def f2(x, nd: int = 2) -> str:
    return MISSING if _nan(x) else f"{x:.{nd}f}"


def signed(x, nd: int = 2) -> str:
    """+0.45 / $-$0.45 for LaTeX text."""
    return MISSING if _nan(x) else f"{x:+.{nd}f}".replace("-", "$-$")


def tabular(header: list, rows: list, align: str, groups: Optional[list] = None) -> str:
    """booktabs tabular. `groups` = optional (label, span) list for a first header row."""
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule"]
    if groups:
        lines.append(" & ".join(f"\\multicolumn{{{k}}}{{c}}{{{g}}}" if k > 1 else g for g, k in groups) + r" \\")
        start, rules = 1, []
        for g, k in groups:
            if k > 1:
                rules.append(f"\\cmidrule(lr){{{start}-{start + k - 1}}}")
            start += k
        lines.append("".join(rules))
    lines += [" & ".join(header) + r" \\", r"\midrule"] + [" & ".join(r) + r" \\" for r in rows]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def t1(df: pd.DataFrame, data: str) -> tuple:
    """T1 for one data setting: mean +- std of final accuracy per model x connection; best per row in bold; * = significant
    difference from the residual baseline (paired t-test p < 0.05 and bootstrap 95% CI excluding 0, D10)."""
    sub = df[df.data == data]
    conns = order(sub.conn, CONNECTIONS)
    recs, rows = [], []
    for m in order(sub.model, MODELS):
        rec, mean, cell = {"model": m}, {}, {}
        for c in conns:
            a = accs(df, m, data, c)
            pr = paired(df, m, data, c) if c != "residual" else dict(p=math.nan, sig=False)
            mean[c] = a.mean() if len(a) else math.nan
            rec.update({f"{c}_mean": mean[c], f"{c}_std": a.std(ddof=1) if len(a) > 1 else math.nan, f"{c}_n": len(a),
                        f"{c}_p": pr["p"], f"{c}_sig": pr["sig"]})
        best = max((c for c in conns if not _nan(mean[c])), key=lambda c: mean[c], default=None)
        for c in conns:
            if _nan(mean[c]):
                cell[c] = MISSING
                continue
            num = f2(mean[c])
            cell[c] = (f"\\textbf{{{num}}}" if c == best else num) + (f"$\\pm${f2(rec[f'{c}_std'])}" if rec[f"{c}_n"] > 1 else "")
            cell[c] += STAR if rec[f"{c}_sig"] else ""
        recs.append(rec)
        rows.append([MODEL_LABEL.get(m, m)] + [cell[c] for c in conns])
    return pd.DataFrame(recs), tabular(["Model"] + [TEX[c] for c in conns], rows, "l" + "c" * len(conns))


def t2(df: pd.DataFrame) -> tuple:
    """T2: gain over residual (connection minus residual, paired by seed) with bootstrap 95% CI, per model / data setting."""
    recs = []
    for data in order(df.data, DATA):
        sub = df[df.data == data]
        for m in order(sub.model, MODELS):
            for c in order(sub.conn, CONNECTIONS):
                if c != "residual":
                    p = paired(df, m, data, c)
                    recs.append(dict(data=data, model=m, conn=c, n_pairs=p["n"], delta=p["delta"], delta_sd=p["sd"],
                                     ci_lo=p["lo"], ci_hi=p["hi"], p_ttest=p["p"], significant=p["sig"]))
    long = pd.DataFrame(recs)
    if long.empty:
        return long, ""
    cols = [(d, c) for d in order(long.data, DATA) for c in order(long[long.data == d].conn, CONNECTIONS)]
    rows = []
    for m in order(long.model, MODELS):
        cells = []
        for d, c in cols:
            r = long[(long.model == m) & (long.data == d) & (long.conn == c)]
            if r.empty:
                cells.append(MISSING)
                continue
            r = r.iloc[0]
            txt = signed(r.delta)
            if r.n_pairs > 1:
                txt += f" [{signed(r.ci_lo)}, {signed(r.ci_hi)}]"
            cells.append(txt + (STAR if r.significant else ""))
        rows.append([MODEL_LABEL.get(m, m)] + cells)
    groups = [("Model", 1)] + [(DATA_SHORT.get(d, d), sum(1 for x, _ in cols if x == d)) for d in order(long.data, DATA)]
    return long, tabular([""] + [TEX[c] for _, c in cols], rows, "l" + "c" * len(cols), groups)


def t3(cost: pd.DataFrame) -> tuple:
    """T3: cost per model x connection from scripts/profile_cost.py, with the overhead relative to the residual model."""
    cost = cost.copy()
    base = cost[cost.connection == "residual"].set_index("model")
    cost["mem_x"] = [r.peak_mem_mb / base.peak_mem_mb.get(r.model, math.nan) for r in cost.itertuples()]
    cost["train_slowdown_x"] = [base.train_img_per_s.get(r.model, math.nan) / r.train_img_per_s for r in cost.itertuples()]
    cost["infer_slowdown_x"] = [base.infer_img_per_s.get(r.model, math.nan) / r.infer_img_per_s for r in cost.itertuples()]
    rows = []
    for m in order(cost.model, MODELS):
        for c in order(cost[cost.model == m].connection, CONNECTIONS):
            r = cost[(cost.model == m) & (cost.connection == c)].iloc[0]
            rows.append([MODEL_LABEL.get(m, m), TEX[c], f2(r.params / 1e6, 3), f2(r.macs_g, 3), f2(r.peak_mem_mb, 0),
                         f2(r.mem_x), f2(r.train_img_per_s, 0), f2(r.train_slowdown_x), f2(r.infer_img_per_s, 0),
                         f2(r.infer_slowdown_x)])
    header = ["Model", "Connection", "Params (M)", "GMACs", "Mem (MiB)", r"$\times$ mem", "Train img/s", r"$\times$ time",
              "Infer img/s", r"$\times$ time"]
    return cost, tabular(header, rows, "ll" + "r" * 8)


def t4(df: pd.DataFrame) -> tuple:
    """T4: stability per cell: seed std, number of diverged runs, mean gap between best and final accuracy."""
    recs = []
    for (data, m, c), g in df.groupby(["data", "model", "conn"]):
        ok = g[~g.diverged]
        recs.append(dict(data=data, model=m, conn=c, n_runs=len(g), n_diverged=int(g.diverged.sum()),
                         seed_std=ok.final_acc.std(ddof=1) if len(ok) > 1 else math.nan,
                         best_minus_final=(ok.best_acc - ok.final_acc).mean() if len(ok) else math.nan))
    long = pd.DataFrame(recs)
    tex = {}
    for data in order(long.data, DATA):
        sub = long[long.data == data]
        conns = order(sub.conn, CONNECTIONS)
        rows = []
        for m in order(sub.model, MODELS):
            cells = []
            for c in conns:
                r = sub[(sub.model == m) & (sub.conn == c)]
                cells.append(MISSING if r.empty else f"{f2(r.iloc[0].seed_std)} / {int(r.iloc[0].n_diverged)} / {f2(r.iloc[0].best_minus_final)}")
            rows.append([MODEL_LABEL.get(m, m)] + cells)
        tex[data] = tabular(["Model"] + [TEX[c] for c in conns], rows, "l" + "c" * len(conns))
    return long, tex


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Runs found per data setting x model x connection (to see at a glance what is still missing)."""
    return df.groupby(["data", "model", "conn"]).size().rename("runs").reset_index()


def setup_table(df: pd.DataFrame, configs: Path) -> str:
    """The training recipes, read from the YAML configs (so the paper cannot drift from what was run)."""
    load = lambda name: yaml.safe_load((configs / name).read_text(encoding="utf-8"))["train"]
    r, v = {**load("base.yaml"), **load("models/_resnet.yaml")}, {**load("base.yaml"), **load("models/_vit.yaml")}   # base <- model

    def aug(t):
        a = t["augment"]
        s = f"crop (pad {a['crop_padding']})" + (", flip" if a["flip"] else "")
        return s + (f", RandAugment ({a['randaugment']['num_ops']} ops, mag. {a['randaugment']['magnitude']})" if a["randaugment"] else "")

    def epochs(family):
        e = sorted(df[df.family == family].epochs.unique())
        return ", ".join(str(int(x)) for x in e) if len(e) else MISSING

    bs = lambda t: str(t["batch_size"])
    opt = {"sgd": lambda t: f"SGD, momentum {t['momentum']}" + (", Nesterov" if t["nesterov"] else ""), "adamw": lambda t: "AdamW"}
    rows = [["Optimizer", opt[r["optimizer"]](r), opt[v["optimizer"]](v)], ["Learning rate", str(r["lr"]), str(v["lr"])],
            ["Weight decay", str(r["weight_decay"]), str(v["weight_decay"])],
            ["Schedule", f"{r['schedule']}, warmup {r['warmup_epochs']} ep.", f"{v['schedule']}, warmup {v['warmup_epochs']} ep."],
            ["Label smoothing", str(r["label_smoothing"]), str(v["label_smoothing"])], ["Augmentation", aug(r), aug(v)],
            ["Batch size", bs(r), bs(v)], ["Epochs (this study)", epochs("resnet"), epochs("vit")]]
    return tabular(["Setting", "ResNet", "ViT-Tiny"], rows, "lll")


def macros(df: pd.DataFrame) -> str:
    """LaTeX macros so the paper text never hand-types a count."""
    seeds = int(df.groupby(["data", "model", "conn"]).size().max())
    ep = lambda fam: "/".join(str(int(x)) for x in sorted(df[df.family == fam].epochs.unique())) or MISSING
    return "\n".join([f"\\newcommand{{\\NRuns}}{{{len(df)}}}", f"\\newcommand{{\\NDiverged}}{{{int(df.diverged.sum())}}}",
                      f"\\newcommand{{\\NSeeds}}{{{seeds}}}", f"\\newcommand{{\\EpochsResNet}}{{{ep('resnet')}}}",
                      f"\\newcommand{{\\EpochsViT}}{{{ep('vit')}}}",
                      f"\\newcommand{{\\GpuName}}{{{'; '.join(sorted(df.device.unique()))}}}", ""])


def build_all(raw: Path, out: Path, cost_csv: Optional[Path] = None, configs: Optional[Path] = None) -> list:
    """Write every table under `out`; returns the written paths. Raises if the runs mix training budgets."""
    df = load_runs(raw)
    if df.empty:
        raise ValueError(f"no finished runs (folders with config.yaml and final.json) under {raw}")
    problems = budget_problems(df)
    if problems:
        raise ValueError("; ".join(problems) + ". Aggregate one plan per results folder.")
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def save(name, frame=None, tex=None):
        for ext, payload in (("csv", frame), ("tex", tex)):
            if payload is not None:
                p = out / f"{name}.{ext}"
                p.write_text(payload, encoding="utf-8") if ext == "tex" else payload.to_csv(p, index=False)
                written.append(p)

    save("runs", df.drop(columns=["path"]))
    save("coverage", coverage(df))
    for data in order(df.data, DATA):
        save(f"T1_{data}", *t1(df, data))
    long2, tex2 = t2(df)
    save("T2_gain", long2, tex2 or None)
    long4, tex4 = t4(df)
    save("T4_stability", long4)
    for data, tex in tex4.items():
        save(f"T4_{data}", None, tex)
    if cost_csv is not None and Path(cost_csv).exists():
        save("T3_cost", *t3(pd.read_csv(cost_csv)))
    if configs is not None and (Path(configs) / "models/_resnet.yaml").exists():
        save("T0_setup", None, setup_table(df, Path(configs)))
    save("macros", None, macros(df))
    return written
