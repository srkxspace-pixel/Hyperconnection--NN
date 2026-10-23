"""Forward/backward shapes for n in {1, 2, 4}, both layouts, plus the connection parameter-count formulas."""
import pytest
import torch
from helpers import D, build, run

from hcsmall.connections import make_connection

CASES = [("residual", 1), ("shc", 2), ("dhc", 2), ("dhc", 4), ("mhc", 4)]


@pytest.mark.parametrize("layout", ["tokens", "conv"])
@pytest.mark.parametrize("kind,n", CASES)
def test_forward_backward(layout, kind, n):
    parts = build(layout)
    ref, _ = run("residual", 1, layout, parts)
    out, conns = run(kind, n, layout, parts)
    assert out.shape == ref.shape            # the conv stack halves the resolution and doubles channels at layer 3
    out.pow(2).sum().backward()
    for c in conns:
        for name, p in c.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("layout", ["tokens", "conv"])
def test_connection_param_counts(layout, n):
    """[HC] App. B: static n(n+2); dynamic adds d(n+2)+2. mHC (docs/method_notes.md §8): nd(n^2+2n) + n^2+2n + 3."""
    count = lambda kind: sum(p.numel() for p in make_connection(kind, n, D, 0, layout).parameters())
    assert count("shc") == n * (n + 2)
    assert count("dhc") == n * (n + 2) + D * (n + 2) + 2
    assert count("mhc") == n * D * (n * n + 2 * n) + n * n + 2 * n + 3
    assert count("residual") == 0
