"""Optimizer with the weight-decay parameter groups of D7 / D8 / D9 / D17 (identical rules for every connection)."""
import torch
from torch import nn


def split_decay(model: nn.Module, decay_norm_and_bias: bool) -> tuple[list, list]:
    """(decayed, not decayed) parameters.

    - static HC / mHC parameters (Connection.no_decay): never decayed (D9, D17);
    - dynamic projections (the other connection parameters): decayed like weight matrices (D17, [HC] section 4);
    - everything else: decayed, except, when `decay_norm_and_bias` is false (ViT recipe), biases, norm weights,
      the position embedding and the class token.
    """
    conn_static, conn_dynamic = set(), set()
    for conn in getattr(model, "conns", []):
        for name, p in conn.named_parameters(recurse=False):
            (conn_static if name in conn.no_decay else conn_dynamic).add(id(p))
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if id(p) in conn_static:
            no_decay.append(p)
        elif id(p) in conn_dynamic or decay_norm_and_bias:
            decay.append(p)
        elif p.ndim < 2 or name in ("pos_embed", "cls_token"):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    t = train_cfg
    decay, no_decay = split_decay(model, t["decay_norm_and_bias"])
    groups = [{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(groups, lr=t["lr"], momentum=t["momentum"], nesterov=t["nesterov"])
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=t["lr"])
    raise ValueError(f"unknown optimizer: {t['optimizer']!r}")
