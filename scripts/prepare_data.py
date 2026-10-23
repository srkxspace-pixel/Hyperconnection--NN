"""Download CIFAR-100, measure the normalisation statistics, and write the saved subset / probe index files.

Run once from the repo root:  python scripts/prepare_data.py
The indices land in configs/data/subsets/ and are committed, so every run (and every machine) uses the same subsets.
"""
from pathlib import Path

import numpy as np
from torchvision import datasets

from hcsmall.data.cifar100 import _channel_stats
from hcsmall.data.subsets import class_balanced_indices, load_or_make_subset
from hcsmall.utils.io import load_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = load_config(str(ROOT / "configs/experiments/smoke.yaml"))
    train = datasets.CIFAR100(ROOT / cfg["paths"]["data_dir"], train=True, download=True)
    test = datasets.CIFAR100(ROOT / cfg["paths"]["data_dir"], train=False, download=True)
    mean, std = _channel_stats(train.data)                               # what training uses when mean/std are null
    print("mean:", [round(v, 4) for v in mean])
    print("std: ", [round(v, 4) for v in std])
    labels, out = np.asarray(train.targets), ROOT / cfg["paths"]["subsets_dir"]
    seed = cfg["data"]["subset_seed"]
    for percent in (10, 25):
        idx = load_or_make_subset(out / f"cifar100_p{percent}_s{seed}.json", labels, percent, seed)
        print(f"p{percent}: {len(idx)} training images, {len(idx) // 100} per class")
    probe = class_balanced_indices(np.asarray(test.targets), 100 * cfg["analysis"]["probe_images"] / len(test), cfg["analysis"]["seed"])
    print(f"probe: {len(probe)} test images (class-balanced) -> {out / 'cifar100_probe.json'}")
    (out / "cifar100_probe.json").write_text(f'{{"seed": {cfg["analysis"]["seed"]}, "indices": {probe.tolist()}}}', encoding="utf-8")


if __name__ == "__main__":
    main()
