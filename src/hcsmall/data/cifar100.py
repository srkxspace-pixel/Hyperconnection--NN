"""CIFAR-100 loading: augmentation in DataLoader workers, class-balanced saved subsets, fixed probe / gradient batches."""
import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms as T

from .subsets import class_balanced_indices, load_or_make_subset


def _transforms(cfg: dict, train: bool):
    d, a = cfg["data"], cfg["train"]["augment"]
    norm = T.Normalize(d["mean"], d["std"])
    if not train:
        return T.Compose([T.ToTensor(), norm])
    ops = [T.RandomCrop(d["img_size"], padding=a["crop_padding"])]
    if a["flip"]:
        ops.append(T.RandomHorizontalFlip())
    if a["randaugment"]:
        ops.append(T.RandAugment(**a["randaugment"]))          # num_ops, magnitude
    return T.Compose(ops + [T.ToTensor(), norm])


def _channel_stats(data: np.ndarray, chunk: int = 5000) -> tuple[list, list]:
    """Per-channel mean and (population) std of uint8 images (N, H, W, 3) scaled to [0, 1], accumulated in float64
    chunks (a one-shot float64 copy of CIFAR-100 is 1.2 GB)."""
    s = s2 = 0.0
    for i in range(0, len(data), chunk):
        a = data[i:i + chunk].astype(np.float64) / 255.0
        s, s2 = s + a.sum((0, 1, 2)), s2 + (a * a).sum((0, 1, 2))
    n = data.shape[0] * data.shape[1] * data.shape[2]
    mean = s / n
    return mean.tolist(), np.sqrt(s2 / n - mean ** 2).tolist()


def _num_workers(setting) -> int:
    """'auto': 0 on Windows (spawned workers re-import torch and re-load the dataset: ~1 min each time, while the
    PIL pipeline costs ~0.1 ms per image), else up to 4."""
    if setting == "auto":
        return 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    return int(setting)


def _dataset(cfg: dict, train: bool):
    """The raw dataset (no transform). Give it a transform with `_with`, which shares the image array."""
    d = cfg["data"]
    if d["name"] == "fake":                                    # offline stand-in for tests: random images and labels
        size = d.get("fake_size", 256) if train else d.get("fake_size", 256) // 2
        ds = datasets.FakeData(size, (3, d["img_size"], d["img_size"]), d["num_classes"], None, random_offset=0 if train else 10**6)
        ds.targets = [int(ds[i][1]) for i in range(len(ds))]   # FakeData has no stored labels; this reads them once
        return ds
    return datasets.CIFAR100(Path(cfg["root"]) / cfg["paths"]["data_dir"], train=train, download=True)


def _with(ds, tf):
    out = copy.copy(ds)                                        # shallow: the 150 MB image array is shared
    out.transform = tf
    return out


def make_data(cfg: dict) -> dict:
    """Returns train_loader (augmented, shuffled), test (x, y) tensors, n_train, and fixed unaugmented tensors:
    probe (class-balanced test images for the similarity / CKA analysis) and grad (training images for the
    gradient-norm analysis)."""
    d, t, an = cfg["data"], cfg["train"], cfg["analysis"]
    train_base, test_base = _dataset(cfg, True), _dataset(cfg, False)
    if d["mean"] is None:                                      # measured on the full training set, so identical for every subset
        d["mean"], d["std"] = ([0.5] * 3, [0.25] * 3) if d["name"] == "fake" else _channel_stats(train_base.data)
    train_set = _with(train_base, _transforms(cfg, True))
    clean_train, clean_test = _with(train_base, _transforms(cfg, False)), _with(test_base, _transforms(cfg, False))
    labels = np.asarray(train_base.targets)
    if d["percent"] == 100:
        idx = np.arange(len(labels))
    else:
        path = Path(cfg["root"]) / cfg["paths"]["subsets_dir"] / f"{d['name']}_p{d['percent']}_s{d['subset_seed']}.json"
        idx = load_or_make_subset(path, labels, d["percent"], d["subset_seed"])
    cuda, nw = torch.cuda.is_available(), _num_workers(t["num_workers"])
    g = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(Subset(train_set, idx), batch_size=t["batch_size"], shuffle=True, drop_last=True, generator=g,
                        num_workers=nw, pin_memory=cuda, persistent_workers=nw > 0)
    probe_idx = class_balanced_indices(np.asarray(test_base.targets), 100 * an["probe_images"] / len(test_base), an["seed"])
    grad_idx = np.random.default_rng(an["seed"]).permutation(idx)[: an["grad_batches"] * t["batch_size"]]
    stack = lambda ds, ix: (torch.stack([ds[i][0] for i in ix]), torch.tensor([int(ds[i][1]) for i in ix]))
    return {"train_loader": loader, "test": stack(clean_test, range(len(clean_test))), "n_train": len(idx),
            "probe": stack(clean_test, probe_idx), "grad": stack(clean_train, grad_idx)}
