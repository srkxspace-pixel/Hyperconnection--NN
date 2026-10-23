"""Model factory: one place that turns a resolved config into a network."""
from torch import nn

from .resnet_cifar import PreActResNet
from .vit_tiny import ViTTiny


def build_model(cfg: dict) -> nn.Module:
    """cfg keys used: model.{family, depth, dim, heads, mlp_ratio}, connection.{kind, n},
    data.{num_classes, img_size, patch_size}."""
    m, c, d = cfg["model"], cfg["connection"], cfg["data"]
    if m["family"] == "resnet":
        return PreActResNet(m["depth"], d["num_classes"], c["kind"], c["n"])
    if m["family"] == "vit":
        return ViTTiny(m["depth"], d["num_classes"], c["kind"], c["n"], d["img_size"], d["patch_size"],
                       m["dim"], m["heads"], m["mlp_ratio"])
    raise ValueError(f"unknown model family: {m['family']!r}")
