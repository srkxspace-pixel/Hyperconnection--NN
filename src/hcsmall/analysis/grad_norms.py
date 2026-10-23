"""Per-layer gradient norms."""
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def layer_grad_norms(model: nn.Module, x: Tensor, y: Tensor, batch_size: int, label_smoothing: float,
                     device: torch.device) -> list[float]:
    """L2 norm of the gradient of the training loss w.r.t. each layer's branch (and shortcut) parameters,
    averaged over the full batches of (x, y). Connection parameters are excluded so the profile is comparable
    across connection types. The model is used in train mode (batch statistics); BatchNorm running statistics
    and the grads are restored / cleared afterwards, so the run is not disturbed."""
    shortcuts = getattr(model, "shortcuts", [None] * len(model.branches))
    layers = [list(b.parameters()) + (list(s.parameters()) if s is not None else [])
              for b, s in zip(model.branches, shortcuts)]
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    was_training = model.training
    model.train()
    total, batches = torch.zeros(len(layers), device=device), 0
    for i in range(0, len(x) - batch_size + 1, batch_size):
        xb, yb = x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb, label_smoothing=label_smoothing).backward()
        for k, ps in enumerate(layers):
            total[k] += torch.sqrt(sum(p.grad.pow(2).sum() for p in ps))
        batches += 1
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for k, v in model.named_buffers():
            v.copy_(buffers[k])
    model.train(was_training)
    return (total / max(1, batches)).tolist()
