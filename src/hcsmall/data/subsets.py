"""Class-balanced, nested, deterministic training subsets whose indices are saved once and reused."""
import json
from pathlib import Path

import numpy as np


def class_balanced_indices(labels, percent: float, seed: int) -> np.ndarray:
    """Per class, the first round(percent/100 * class size) entries (at least 1, at most all) of a seeded
    permutation of that class's indices. The permutation does not depend on `percent`, so subsets are
    nested: the 10% subset is contained in the 25% subset, which is contained in the full set."""
    labels, out = np.asarray(labels), []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        perm = np.random.default_rng([seed, int(c)]).permutation(idx)
        out.append(perm[: min(len(idx), max(1, round(len(idx) * percent / 100)))])
    return np.sort(np.concatenate(out))


def load_or_make_subset(path: Path, labels, percent: float, seed: int) -> np.ndarray:
    """Read the stored indices if the file exists (and check they are class-balanced for these labels),
    else create and store them. The stored file, not a regeneration, is the source of truth."""
    labels = np.asarray(labels)
    sizes = np.bincount(labels)
    expected = np.minimum(sizes, np.maximum(1, np.round(sizes * percent / 100))).astype(int)
    if path.exists():
        idx = np.array(json.loads(path.read_text(encoding="utf-8"))["indices"])
        assert len(np.unique(idx)) == len(idx) and (np.bincount(labels[idx], minlength=len(sizes)) == expected).all(), \
            f"{path} is not a class-balanced {percent}% subset of these labels"
        return idx
    idx = class_balanced_indices(labels, percent, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"percent": percent, "seed": seed, "n": len(idx), "indices": idx.tolist()}), encoding="utf-8")
    return idx
