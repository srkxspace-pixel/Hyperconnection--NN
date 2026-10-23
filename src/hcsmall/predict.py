"""Classify one image with a saved run: <run folder>/model.pt, written when a run trains with train.save_weights=true (D20).

Used by the demo page (app/server.py). Nothing here trains anything.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms as T

from .models import build_model


def load_run(run_dir, device: str = "cpu") -> dict:
    """The network of a finished run in eval mode: {'model', 'cfg', 'device'}. The config that rebuilds it is in the file."""
    ck = torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=True)
    model = build_model(ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    return {"model": model.eval(), "cfg": ck["cfg"], "device": device}


def prepare(image: Image.Image, size: int) -> Image.Image:
    """Any photo -> the square the network was trained on: honour the EXIF rotation, centre-crop, resize (antialiased).
    A CIFAR image (already size x size) comes out unchanged."""
    return ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (size, size), Image.Resampling.BICUBIC)


def to_tensor(image: Image.Image, cfg: dict) -> torch.Tensor:
    """(1, 3, size, size), normalised with the training-set mean / std exactly like the training evaluation."""
    d = cfg["data"]
    return T.Compose([T.ToTensor(), T.Normalize(d["mean"], d["std"])])(prepare(image, d["img_size"])).unsqueeze(0)


@torch.no_grad()
def classify(run: dict, image: Image.Image) -> tuple:
    """(probabilities over all classes as a list, the picture the network actually saw)."""
    x = to_tensor(image, run["cfg"]).to(run["device"])
    return F.softmax(run["model"](x), dim=1)[0].tolist(), prepare(image, run["cfg"]["data"]["img_size"])
