"""Learning-rate schedule, evaluated per optimizer step."""
import math


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    """Linear warmup to `base` over `warmup` steps, then cosine decay to 0 at step `total` (D7: no warmup; D8: 10 epochs)."""
    if step < warmup:
        return base * (step + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
