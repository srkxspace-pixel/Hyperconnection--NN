"""Config loading and small run-provenance helpers."""
import copy
import json
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import yaml


def deep_merge(a: dict, b: dict) -> dict:
    """Return a copy of `a` updated recursively with `b`."""
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _load(path: Path, root: Path) -> dict:
    """One YAML file; its `defaults: [a/b, ...]` (paths under `root`, no extension) are merged first, in order."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg: dict = {}
    for d in raw.pop("defaults", []):
        cfg = deep_merge(cfg, _load(root / f"{d}.yaml", root))
    return deep_merge(cfg, raw)


def set_path(cfg: dict, assignment: str) -> None:
    """Apply `a.b.c=value` (value parsed as YAML) in place."""
    key, value = assignment.split("=", 1)
    *parents, leaf = key.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(value)


def load_config(experiment: str, model: Optional[str] = None, connection: Optional[str] = None,
                data: Optional[str] = None, overrides: Iterable[str] = ()) -> dict:
    """Resolve base <- model <- connection <- data <- experiment <- overrides.

    `experiment` is configs/experiments/<name>.yaml; its `defaults` name the model / connection / data files,
    which the arguments replace. `grid` is not part of a single run and is dropped.
    """
    exp_path = Path(experiment).resolve()
    root = exp_path.parent.parent                                   # the configs/ directory
    exp = yaml.safe_load(exp_path.read_text(encoding="utf-8")) or {}
    chosen = {Path(d).parts[0]: d for d in exp.pop("defaults", [])}
    for category, name in (("models", model), ("connections", connection), ("data", data)):
        if name:
            chosen[category] = f"{category}/{name}"
    exp.pop("grid", None)
    cfg = _load(root / "base.yaml", root)
    for category in ("models", "connections", "data"):
        if category in chosen:
            cfg = deep_merge(cfg, _load(root / f"{chosen[category]}.yaml", root))
    cfg = deep_merge(cfg, exp)
    for item in overrides:
        set_path(cfg, item)
    cfg["root"] = str(root.parent)               # repo root: every cfg["paths"] entry is relative to it
    return cfg


def load_grid(experiment: str) -> dict:
    """The `grid` and `train` blocks of an experiment file (used by scripts/run_matrix.py)."""
    exp = yaml.safe_load(Path(experiment).read_text(encoding="utf-8")) or {}
    return {"grid": exp["grid"], "train": exp.get("train", {})}


def git_commit(cwd: Optional[Path] = None) -> str:
    """HEAD hash, suffixed with `-dirty` if the working tree has uncommitted changes. Without a git repository (a
    code snapshot, e.g. the Colab notebook) it reads `.snapshot_commit` if present, else returns 'unknown'."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        return run("rev-parse", "HEAD") + ("-dirty" if run("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        snap = Path(cwd or ".") / ".snapshot_commit"
        return snap.read_text(encoding="utf-8").strip() if snap.exists() else "unknown"


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
