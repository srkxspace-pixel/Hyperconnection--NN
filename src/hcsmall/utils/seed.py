"""Seeding: every run sets and logs its seeds."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and all CUDA devices).

    cudnn.benchmark stays on for speed, so GPU runs are seeded
    but not bit-exact; the paper must state this.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
