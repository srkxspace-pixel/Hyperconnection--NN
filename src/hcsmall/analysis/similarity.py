"""Consecutive-layer similarity of the (reduced) hidden state: per-sample cosine and linear CKA.

Linear CKA is Kornblith et al., ICML 2019."""
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gram(x: Tensor) -> Tensor:
    z = x.float()                                 # float32 matmul (float64 is ~32x slower on consumer GPUs)
    z = z - z.mean(0, keepdim=True)
    return (z @ z.T).double()                     # reductions in float64


def linear_cka(gram_x: Tensor, gram_y: Tensor) -> float:
    """Linear CKA from centered Gram matrices: <Kx, Ky>_F / (|Kx|_F |Ky|_F)."""
    return float((gram_x * gram_y).sum() / (gram_x.norm() * gram_y.norm()))


@torch.no_grad()
def consecutive_similarity(model: nn.Module, x: Tensor, device: torch.device, chunk: int = 250) -> list[tuple[float, float]]:
    """For every layer k, (cosine, CKA) between the reduced hidden state before and after the layer, on the probe
    set `x`. Eval mode, so the result does not depend on `chunk`: the images go through the network `chunk` at a
    time (small allocations, robust on a shared 4 GiB GPU), the per-layer states are kept on the CPU and the
    Gram matrices are formed one layer at a time. Cosine is nan where the two states differ in shape (ResNet stage
    transitions)."""
    was_training = model.training
    model.eval()
    states: list[list[Tensor]] = []               # states[k] = CPU chunks of the state before/after layer k-1/k

    def keep(h: Tensor, k: int) -> None:
        (states[k].append if k < len(states) else lambda s: states.append([s]))(h.flatten(1).cpu())

    for i in range(0, len(x), chunk):
        count = [0]

        def trace(h: Tensor) -> None:
            keep(h, count[0])
            count[0] += 1

        model(x[i:i + chunk].to(device), trace=trace)
    out: list[tuple[float, float]] = []
    prev_x = prev_g = None
    for parts in states:
        cur = torch.cat(parts).to(device)
        g = _gram(cur)
        if prev_x is not None:
            cos = float(F.cosine_similarity(prev_x, cur, dim=1).mean()) if prev_x.shape == cur.shape else float("nan")
            out.append((cos, linear_cka(prev_g, g)))
        prev_x, prev_g = cur, g
    model.train(was_training)
    return out
