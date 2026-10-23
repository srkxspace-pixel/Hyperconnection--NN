"""Write app/classes.json and app/classes.png from the local CIFAR-100 files, so the demo page needs no dataset (D20).

  python scripts/make_app_assets.py        # needs data/cifar-100-python; run once, the two outputs are committed

classes.json: {"names": the 100 class names, "superclass": the superclass name of each class}.
classes.png:  the first test image of every class, 32 x 32 each, on a 10 x 10 grid (class c at column c % 10, row c // 10).
Needs numpy and PIL only (no PyTorch).
"""
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "cifar-100-python"


def load(name: str) -> dict:
    with (BASE / name).open("rb") as f:
        return pickle.load(f, encoding="latin1")       # the official, checksum-verified dataset files, as torchvision does


def main() -> None:
    meta, test = load("meta"), load("test")
    fine, coarse = np.asarray(test["fine_labels"]), np.asarray(test["coarse_labels"])
    first = [int(np.argmax(fine == c)) for c in range(len(meta["fine_label_names"]))]
    sprite = Image.new("RGB", (320, 320))
    for c, i in enumerate(first):
        sprite.paste(Image.fromarray(test["data"][i].reshape(3, 32, 32).transpose(1, 2, 0)), (32 * (c % 10), 32 * (c // 10)))
    sprite.save(ROOT / "app" / "classes.png", optimize=True)
    names = {"names": meta["fine_label_names"], "superclass": [meta["coarse_label_names"][int(coarse[i])] for i in first]}
    (ROOT / "app" / "classes.json").write_text(json.dumps(names, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote app/classes.json ({len(first)} classes) and app/classes.png")


if __name__ == "__main__":
    main()
