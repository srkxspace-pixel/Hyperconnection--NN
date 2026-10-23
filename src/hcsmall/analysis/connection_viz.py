"""Learned connection weights, averaged over a probe set (data for figure F6)."""
import torch
from torch import Tensor, nn


@torch.no_grad()
def effective_weights(model: nn.Module, x: Tensor, device: torch.device, chunk: int = 250) -> dict:
    """{layer: {"w_read": (n,), "w_write": (n,), "R": (n, n)}}: the weights each layer actually used on the probe
    images, averaged over images (and tokens); R[i, j] is the weight from source stream i to target stream j.
    Empty for the residual baseline. Static connections give their parameters; dynamic ones, their mean.
    Images go through `chunk` at a time; chunk means are combined weighted by chunk size (exact)."""
    if model.conns[0].n == 1:
        return {}
    sums: dict[int, list] = {}
    for k, conn in enumerate(model.conns):
        def wrapped(H, k=k, orig=conn.weights):
            w = orig(H)
            dims = (0, 2) if w[0].dim() == 3 else (0,)           # tokens: (B, n, T[, n]); conv: (B, n[, n])
            acc = [t.mean(dims).cpu() * len(H) for t in w]
            sums[k] = acc if k not in sums else [a + b for a, b in zip(sums[k], acc)]
            return w

        conn.weights = wrapped
    was_training = model.training
    model.eval()
    try:
        for i in range(0, len(x), chunk):
            model(x[i:i + chunk].to(device))
    finally:
        for conn in model.conns:
            conn.__dict__.pop("weights", None)                   # drop the instance attribute: the class method is back
        model.train(was_training)
    return {k: {"w_read": a / len(x), "w_write": b / len(x), "R": c / len(x)} for k, (a, b, c) in sums.items()}
