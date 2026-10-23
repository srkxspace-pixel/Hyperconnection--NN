"""Figures F1-F7 as PDF + PNG from a results folder.

No GPU. torch is imported only by F6 (it reads connections.pt). Colours follow the validated reference palette in fixed
slot order, so a connection keeps its colour in every figure; marker shapes are the second encoding for print and colour-
blind readers (aqua, yellow and magenta are below 3:1 on white), and every multi-series panel has a legend plus direct labels.
"""
import math
import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter  # noqa: E402

from .runs import (CONNECTIONS, DATA, DATA_SHORT, LABEL, MODEL_LABEL, MODELS, accs, budget_problems,  # noqa: E402
                   load_runs, order, paired)

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"residual": "#2a78d6", "dhc_n2": "#eb6834", "dhc_n4": "#1baf7a", "mhc_n4": "#eda100", "shc_n2": "#e87ba4"}
MARK = {"residual": "o", "dhc_n2": "s", "dhc_n4": "^", "mhc_n4": "D", "shc_n2": "v"}
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#e34948", "#f0efec", "#256abf"])    # red - neutral - blue
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
FAMILY = {"resnet": "ResNet", "vit": "ViT-Tiny"}


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})


def model_key(family: str, depth: int) -> str:
    return f"resnet{depth}" if family == "resnet" else f"vit_tiny_d{depth}"


def save(fig, out: Path, name: str) -> list:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        paths.append(out / f"{name}.{ext}")
        fig.savefig(paths[-1])
    plt.close(fig)
    return paths


def end_labels(ax, ends: list, gap: float = 0.07) -> None:
    """Direct labels right of each line end, nudged apart. Text is ink; a coloured marker beside it carries identity."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    span = (hi - lo) or 1.0
    ends = sorted(ends, key=lambda e: e[1])
    ys = []
    for e in ends:
        ys.append(e[1] if not ys else max(e[1], ys[-1] + gap * span))
    if max(abs(yy - e[1]) for yy, e in zip(ys, ends)) > 2 * gap * span:
        return                                    # lines end too close together: labels would sit far from them, the legend carries identity
    for (x, _, text, col, mk), yy in zip(ends, ys):
        ax.annotate(mk_char(mk), xy=(x, yy), xytext=(5, 0), textcoords="offset points", va="center", ha="center",
                    color=col, fontsize=6, annotation_clip=False)
        ax.annotate(text, xy=(x, yy), xytext=(11, 0), textcoords="offset points", va="center", ha="left",
                    color=INK2, fontsize=6.5, annotation_clip=False)


def mk_char(marker: str) -> str:
    return {"o": "●", "s": "■", "^": "▲", "D": "◆", "v": "▼"}.get(marker, "●")


def _legend(ax, **kw) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        ax.legend(handles, labels, **kw)


def f1_accuracy_vs_depth(df: pd.DataFrame, out: Path) -> list:
    """F1 (RQ1): final accuracy against depth, one panel per data setting x model family, one line per connection."""
    datas, fams = order(df.data, DATA), [f for f in FAMILY if f in set(df.family)]
    fig, axes = plt.subplots(len(datas), len(fams), figsize=(3.7 * len(fams), 2.4 * len(datas)), squeeze=False)
    for i, data in enumerate(datas):
        for j, fam in enumerate(fams):
            ax, sub = axes[i][j], df[(df.data == data) & (df.family == fam)]
            depths = sorted(sub.depth.unique())
            ends = []
            for c in order(sub.conn, CONNECTIONS):
                pts = [(k, accs(df, model_key(fam, d), data, c)) for k, d in enumerate(depths)]
                pts = [(k, a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0) for k, a in pts if len(a)]
                if pts:
                    x, y, e = (np.array(v, float) for v in zip(*pts))
                    ax.errorbar(x, y, yerr=e, color=COLOR[c], marker=MARK[c], ms=4.5, lw=1.4, capsize=2, label=LABEL[c], zorder=3)
                    ends.append((x[-1], y[-1], LABEL[c], COLOR[c], MARK[c]))
            ax.set_xticks(range(len(depths)), [str(d) for d in depths])
            ax.set_xlim(-0.2, len(depths) - 1 + 0.2)
            ax.set_title(f"{FAMILY[fam]}, {DATA_SHORT.get(data, data)}")
            ax.set_xlabel("depth (ResNet layers / ViT blocks)")
            ax.set_ylabel("top-1 test accuracy (%)")
            end_labels(ax, ends)
            _legend(ax, loc="best")
    fig.tight_layout(w_pad=4.5)
    return save(fig, out, "F1_accuracy_vs_depth")


def f2_gain_vs_data(df: pd.DataFrame, out: Path) -> list:
    """F2 (RQ2): accuracy gain over residual against the data fraction, one line per depth."""
    frac = {"cifar100_p10": 10, "cifar100_p25": 25, "cifar100_p100": 100}
    datas = [d for d in frac if d in set(df.data)]
    conns = [c for c in order(df.conn, CONNECTIONS) if c != "residual"]
    fams = [f for f in FAMILY if f in set(df.family)]
    if len(datas) < 2 or not conns:
        print("F2 skipped: needs at least two data fractions and one non-residual connection")
        return []
    fig, axes = plt.subplots(len(fams), len(conns), figsize=(2.6 * len(conns) + 0.6, 2.4 * len(fams)), squeeze=False, sharey="row")
    shapes = ["o", "s", "^", "D"]
    for i, fam in enumerate(fams):
        for j, c in enumerate(conns):
            ax = axes[i][j]
            ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
            for k, depth in enumerate(sorted(df[df.family == fam].depth.unique())):
                pts = []
                for d in datas:
                    p = paired(df, model_key(fam, depth), d, c)
                    if p["n"]:
                        err = (p["delta"] - p["lo"], p["hi"] - p["delta"]) if p["n"] > 1 else (0.0, 0.0)
                        pts.append((frac[d], p["delta"], *err))
                if pts:
                    x, y, lo, hi = (np.array(v, float) for v in zip(*pts))
                    ax.errorbar(x, y, yerr=[lo, hi], color=COLOR[c], marker=shapes[k % 4], ms=4.5, lw=1.2,
                                ls=["-", "--", ":", "-."][k % 4], capsize=2, label=f"depth {depth}", zorder=3)
            ax.set_xscale("log")
            ax.set_xticks([frac[d] for d in datas], [f"{frac[d]}%" for d in datas])
            ax.minorticks_off()
            ax.set_title(f"{FAMILY[fam]}: {LABEL[c]}")
            ax.set_xlabel("training data")
            if j == 0:
                ax.set_ylabel("accuracy gain over residual (pp)")
            _legend(ax, loc="best")
    fig.tight_layout()
    return save(fig, out, "F2_gain_vs_data_fraction")


def _profiles(df, model, data, conn, fname, col, which) -> Optional[np.ndarray]:
    rows = []
    for path in df[(df.model == model) & (df.data == data) & (df.conn == conn)].path:
        t = pd.read_csv(Path(path) / fname)
        t = t[t.epoch == (t.epoch.min() if which == "first" else t.epoch.max())].sort_values("layer")
        rows.append(t[col].to_numpy(float))
    return np.vstack(rows) if rows else None


def _deepest(df, data):
    return [(f, int(df[(df.family == f) & (df.data == data)].depth.max())) for f in FAMILY
            if len(df[(df.family == f) & (df.data == data)])]


def _layer_figure(df, out, name, specs, ylabel_of, which, logy=False) -> list:
    data = "cifar100_p100" if "cifar100_p100" in set(df.data) else order(df.data, DATA)[0]
    targets = _deepest(df, data)
    fig, axes = plt.subplots(len(specs), len(targets), figsize=(3.7 * len(targets), 2.3 * len(specs)), squeeze=False)
    for j, (fam, depth) in enumerate(targets):
        model = model_key(fam, depth)
        for i, (fname, col) in enumerate(specs):
            ax, ends = axes[i][j], []
            for c in order(df[(df.model == model) & (df.data == data)].conn, CONNECTIONS):
                prof = _profiles(df, model, data, c, fname, col, which)
                if prof is None:
                    continue
                with warnings.catch_warnings():       # layers that are nan for every seed (ResNet stage transitions)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    x, y = np.arange(prof.shape[1]), np.nanmean(prof, axis=0)
                    lo, hi = np.nanmin(prof, 0), np.nanmax(prof, 0)
                ax.plot(x, y, color=COLOR[c], lw=1.4, marker=MARK[c], ms=3.0, markevery=max(1, len(x) // 10), label=LABEL[c], zorder=3)
                if len(prof) > 1:
                    ax.fill_between(x, lo, hi, color=COLOR[c], alpha=0.15, lw=0, zorder=2)
                ends.append((x[-1], y[-1], LABEL[c], COLOR[c], MARK[c]))
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"{MODEL_LABEL[model]}, {DATA_SHORT.get(data, data)}" if i == 0 else "")
            ax.set_xlabel("layer index")
            ax.set_ylabel(ylabel_of(col))
            if not logy:                              # the nudging in end_labels is linear
                end_labels(ax, ends)
            _legend(ax, loc="best")
    fig.tight_layout(w_pad=4.5)
    return save(fig, out, name)


def f3_collapse(df: pd.DataFrame, out: Path) -> list:
    """F3 (RQ3): similarity of consecutive layers' hidden states (end of training), deepest ResNet and ViT."""
    return _layer_figure(df, out, "F3_representation_collapse", [("similarity.csv", "cka"), ("similarity.csv", "cosine")],
                         lambda col: "linear CKA of consecutive layers" if col == "cka" else "cosine of consecutive layers", "last")


def f4_gradients(df: pd.DataFrame, out: Path) -> list:
    """F4 (RQ3): per-layer gradient norm at the first checkpoint (end of epoch 1), deepest ResNet and ViT."""
    return _layer_figure(df, out, "F4_gradient_norms", [("grad_norms.csv", "grad_norm")],
                         lambda col: "gradient norm (L2), end of epoch 1", "first", logy=True)


def _log_axis(ax, values) -> None:
    """Log x axis with a few plain-number ticks (the default minor labels collide when the range is narrow)."""
    ax.set_xscale("log")
    lo, hi = min(values), max(values)
    ticks = sorted({int(round(v, -1)) or int(v) for v in np.geomspace(lo, hi, 4)}) if hi > lo else [int(lo)]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo / 1.15, hi * 1.15)


def _pareto(points: dict) -> set:
    """Keys of the points no other point beats on both axes (lower memory, higher accuracy)."""
    return {k for k, (m, a) in points.items() if not any((m2 <= m and a2 >= a) and (m2 < m or a2 > a)
                                                          for k2, (m2, a2) in points.items() if k2 != k)}


def f5_accuracy_vs_cost(df: pd.DataFrame, cost: pd.DataFrame, out: Path) -> list:
    """F5 (RQ4): accuracy against peak training memory. One panel per connection: grey = residual model, coloured = the
    same model with that connection, arrow = the move; ring = Pareto-optimal (memory down, accuracy up) within the family."""
    data = "cifar100_p100" if "cifar100_p100" in set(df.data) else order(df.data, DATA)[0]
    conns = [c for c in order(df.conn, CONNECTIONS) if c != "residual"]
    fams = [f for f in FAMILY if f in set(df.family)]
    fig, axes = plt.subplots(len(fams), len(conns), figsize=(2.5 * len(conns) + 0.6, 2.4 * len(fams)), squeeze=False)
    mem = cost.set_index(["model", "connection"]).peak_mem_mb
    for i, fam in enumerate(fams):
        models = order(df[df.family == fam].model, MODELS)
        pts = {}
        for m in models:
            for c in ["residual"] + conns:
                a = accs(df, m, data, c)
                if len(a) and (m, c) in mem.index and not math.isnan(mem[(m, c)]):
                    pts[(m, c)] = (float(mem[(m, c)]), float(a.mean()))
        front = _pareto(pts)
        for j, c in enumerate(conns):
            ax = axes[i][j]
            for m in models:
                if (m, "residual") in pts and (m, c) in pts:
                    (x0, y0), (x1, y1) = pts[(m, "residual")], pts[(m, c)]
                    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8), zorder=2)
                    ax.plot([x0], [y0], marker="o", ms=5, mfc="white", mec=MUTED, ls="", zorder=3, label="residual" if m == models[0] else None)
                    ax.plot([x1], [y1], marker=MARK[c], ms=5, color=COLOR[c], ls="", zorder=4, label=LABEL[c] if m == models[0] else None)
                    ax.annotate(MODEL_LABEL[m].split("-")[-1], (x0, y0), xytext=(-4, 4), textcoords="offset points", fontsize=6, color=INK2, ha="right")
                    for key, (x, y) in (((m, "residual"), (x0, y0)), ((m, c), (x1, y1))):
                        if key in front:
                            ax.plot([x], [y], marker="o", ms=9, mfc="none", mec=INK, mew=0.7, ls="", zorder=5)
            shown = [pts[k][0] for k in pts if k[1] in ("residual", c)]
            if shown:
                _log_axis(ax, shown)
            ax.set_title(f"{FAMILY[fam]}: {LABEL[c]}")
            ax.set_xlabel("peak training memory (MiB)")
            if j == 0:
                ax.set_ylabel("top-1 test accuracy (%)")
            _legend(ax, loc="best")
    fig.tight_layout()
    return save(fig, out, "F5_accuracy_vs_cost")


def f6_connection_patterns(df: pd.DataFrame, out: Path) -> list:
    """F6 (RQ3/RQ5): learned mixing matrices R (source -> target stream) at six layers, and the read / write weights of
    every layer, for one ResNet and one ViT (seed with the lowest number)."""
    import torch
    data = "cifar100_p100" if "cifar100_p100" in set(df.data) else order(df.data, DATA)[0]
    picks = []
    for fam, prefer in (("resnet", ("resnet56", "resnet110", "resnet20")), ("vit", ("vit_tiny_d12", "vit_tiny_d24", "vit_tiny_d6"))):
        for m in prefer:
            for c in ("dhc_n4", "dhc_n2", "mhc_n4", "shc_n2"):
                g = df[(df.model == m) & (df.data == data) & (df.conn == c) & ~df.diverged].sort_values("seed")
                if len(g) and (Path(g.iloc[0].path) / "connections.pt").exists():
                    picks.append((m, c, Path(g.iloc[0].path)))
                    break
            else:
                continue
            break
    if not picks:
        print("F6 skipped: no run with connections.pt")
        return []
    K = 6
    fig, axes = plt.subplots(len(picks), K + 2, figsize=(1.35 * (K + 2) + 1.2, 2.0 * len(picks) + 0.6), squeeze=False,
                             gridspec_kw={"width_ratios": [1] * K + [1.5, 1.5], "hspace": 0.85, "wspace": 0.45})
    for i, (m, c, path) in enumerate(picks):
        eff = torch.load(path / "connections.pt", map_location="cpu")["effective"]
        layers = sorted(eff)
        R = np.stack([eff[k]["R"].numpy() for k in layers])
        rd, wr = np.stack([eff[k]["w_read"].numpy() for k in layers]), np.stack([eff[k]["w_write"].numpy() for k in layers])
        lim = max(1e-6, float(np.abs(R).max()))
        shown = np.unique(np.linspace(0, len(layers) - 1, K).round().astype(int))
        for k, L in enumerate(shown):
            ax = axes[i][k]
            im = ax.imshow(R[L], cmap=DIVERGING, vmin=-lim, vmax=lim)
            ax.set_xticks(range(R.shape[1]), [str(t + 1) for t in range(R.shape[1])])
            ax.set_yticks(range(R.shape[1]), [str(t + 1) for t in range(R.shape[1])])
            ax.grid(False)
            ax.set_title(f"layer {layers[L]}", fontsize=7)
            if k == 0:
                ax.set_ylabel(f"{MODEL_LABEL[m]}\n{LABEL[c]}\nsource stream", fontsize=7)
            ax.set_xlabel("target stream", fontsize=6)
        for k in range(len(shown), K):
            axes[i][k].axis("off")
        fig.colorbar(im, ax=list(axes[i][:K]), shrink=0.8, pad=0.01, label="R")
        top = max(1.0, float(max(rd.max(), wr.max())))                # one honest scale for both weight panels
        for ax, w, title in ((axes[i][K], rd, "read weight"), (axes[i][K + 1], wr, "write weight")):
            im2 = ax.imshow(w.T, cmap=SEQUENTIAL, aspect="auto", vmin=0.0, vmax=top)
            ax.grid(False)
            ax.set_title(title, fontsize=7)
            ax.set_xlabel("layer", fontsize=6)
            ax.set_yticks(range(w.shape[1]), [str(t + 1) for t in range(w.shape[1])])
        fig.colorbar(im2, ax=[axes[i][K], axes[i][K + 1]], shrink=0.8, pad=0.02, label="weight")
    return save(fig, out, "F6_connection_patterns")


def f7_training_curves(df: pd.DataFrame, out: Path) -> list:
    """F7: test accuracy against epoch for one mid-depth ResNet and one ViT (band = min/max over seeds)."""
    datas = order(df.data, DATA)[:2]
    prefer = (("resnet56", "resnet20", "resnet110"), ("vit_tiny_d12", "vit_tiny_d6", "vit_tiny_d24"))      # first one present per family
    models = [m for m in (next((m for m in fam if m in set(df.model)), None) for fam in prefer) if m] or [order(df.model, [])[0]]
    fig, axes = plt.subplots(len(datas), len(models), figsize=(3.7 * len(models), 2.3 * len(datas)), squeeze=False)
    for i, data in enumerate(datas):
        for j, m in enumerate(models):
            ax, ends = axes[i][j], []
            for c in order(df[(df.model == m) & (df.data == data)].conn, CONNECTIONS):
                curves = []
                for path in df[(df.model == m) & (df.data == data) & (df.conn == c)].path:
                    t = pd.read_csv(Path(path) / "metrics.csv")
                    curves.append(t.set_index("epoch").test_acc * 100)
                if not curves:
                    continue
                cur = pd.concat(curves, axis=1)
                mean = cur.mean(axis=1)
                ax.plot(cur.index, mean, color=COLOR[c], lw=1.4, label=LABEL[c], zorder=3)
                if cur.shape[1] > 1:
                    ax.fill_between(cur.index, cur.min(axis=1), cur.max(axis=1), color=COLOR[c], alpha=0.15, lw=0, zorder=2)
                ends.append((cur.index[-1], float(mean.iloc[-1]), LABEL[c], COLOR[c], MARK[c]))
            ax.set_title(f"{MODEL_LABEL[m]}, {DATA_SHORT.get(data, data)}")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("epoch")
            ax.set_ylabel("top-1 test accuracy (%)")
            end_labels(ax, ends)
            _legend(ax, loc="lower right")
    fig.tight_layout(w_pad=4.5)
    return save(fig, out, "F7_training_curves")


def make_all(raw: Path, out: Path, cost_csv: Optional[Path] = None) -> list:
    """Write every figure that the available runs support; says which ones it skipped and why."""
    style()
    df = load_runs(raw)
    if df.empty:
        raise ValueError(f"no finished runs under {raw}")
    problems = budget_problems(df)
    if problems:
        raise ValueError("; ".join(problems) + ". Plot one plan per results folder.")
    written = f1_accuracy_vs_depth(df, out) + f2_gain_vs_data(df, out) + f3_collapse(df, out) + f4_gradients(df, out)
    if cost_csv is not None and Path(cost_csv).exists():
        written += f5_accuracy_vs_cost(df, pd.read_csv(cost_csv), out)
    else:
        print("F5 skipped: no cost CSV (run scripts/profile_cost.py on the GPU)")
    try:
        written += f6_connection_patterns(df, out)
    except ImportError:
        print("F6 skipped: torch is needed to read connections.pt")
    return written + f7_training_curves(df, out)
