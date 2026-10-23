import random

import numpy as np
import torch

from hcsmall.utils.runid import make_run_id
from hcsmall.utils.seed import set_seed


def test_run_id_format():
    # {model}{depth}_{connection}_n{n}_{dataset}p{fraction}_s{seed}
    assert make_run_id("resnet", 56, "dhc", 4, "cifar100", 10, 1) == "resnet56_dhc_n4_cifar100p10_s1"


def test_set_seed_reproducible():
    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item()

    set_seed(3)
    a = draw()
    set_seed(3)
    assert a == draw()
    set_seed(4)
    assert a != draw()
