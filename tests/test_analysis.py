"""Analysis passes: CKA properties, chunk independence, and no side effects on the training state."""
import math

import pytest
import torch

from hcsmall.analysis.connection_viz import effective_weights
from hcsmall.analysis.grad_norms import layer_grad_norms
from hcsmall.analysis.similarity import _gram, consecutive_similarity, linear_cka
from hcsmall.models import build_model

DEV = torch.device("cpu")


def model(family, depth, kind, n):
    torch.manual_seed(0)
    return build_model({"model": {"family": family, "depth": depth, "dim": 48, "heads": 3, "mlp_ratio": 4},
                        "connection": {"kind": kind, "n": n}, "data": {"num_classes": 10, "img_size": 32, "patch_size": 4}})


def test_cka_properties():
    torch.manual_seed(0)
    X = torch.randn(200, 30)
    Q, _ = torch.linalg.qr(torch.randn(30, 30))
    assert linear_cka(_gram(X), _gram(X)) == pytest.approx(1.0)
    assert linear_cka(_gram(X), _gram(3.0 * X @ Q)) == pytest.approx(1.0)       # invariant to rotation and scale
    assert linear_cka(_gram(X), _gram(torch.randn(200, 30))) < 0.5               # independent features: far below 1
    assert linear_cka(_gram(X), _gram(X + 5.0)) == pytest.approx(1.0)            # centering: a constant shift is irrelevant


@pytest.mark.parametrize("family,depth,kind,n", [("resnet", 20, "dhc", 2), ("vit", 6, "mhc", 4)])
def test_similarity_does_not_depend_on_chunking(family, depth, kind, n):
    net = model(family, depth, kind, n)
    torch.manual_seed(1)
    x = torch.randn(30, 3, 32, 32)
    a, b = consecutive_similarity(net, x, DEV, chunk=7), consecutive_similarity(net, x, DEV, chunk=1000)
    assert len(a) == len(b) == (9 if family == "resnet" else 12)
    for (c1, k1), (c2, k2) in zip(a, b):
        assert k1 == pytest.approx(k2, abs=1e-9) and (c1 == pytest.approx(c2, abs=1e-9) or (math.isnan(c1) and math.isnan(c2)))
    if family == "resnet":                                    # cosine is undefined where the shape changes: layers 3 and 6
        assert [math.isnan(c) for c, _ in a] == [i in (3, 6) for i in range(9)]


def test_effective_weights_static_equals_parameters_and_ignores_chunking():
    net = model("resnet", 20, "shc", 4)
    with torch.no_grad():
        for c in net.conns:
            c.a_r.add_(0.2 * torch.randn_like(c.a_r))
    x = torch.randn(10, 3, 32, 32)
    w = effective_weights(net, x, DEV, chunk=3)
    assert len(w) == 9 and torch.allclose(w[2]["R"], net.conns[2].a_r) and torch.allclose(w[2]["w_read"], net.conns[2].a_m)
    d = model("vit", 6, "dhc", 2)
    with torch.no_grad():
        for p in d.conns[1].parameters():
            p.add_(0.2 * torch.randn_like(p))
    a, b = effective_weights(d, x, DEV, chunk=3), effective_weights(d, x, DEV, chunk=100)
    assert all(torch.allclose(a[k][f], b[k][f], atol=1e-9) for k in a for f in a[k])
    assert effective_weights(model("resnet", 20, "residual", 1), x, DEV) == {}
    assert "weights" not in d.conns[1].__dict__                # the temporary wrapper is removed again


def test_grad_norms_leave_the_training_state_alone():
    net = model("resnet", 20, "dhc", 2)
    net.train()
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
    before = {k: v.clone() for k, v in net.named_buffers()}
    norms = layer_grad_norms(net, x, y, 8, 0.0, DEV)
    assert len(norms) == 9 and all(v > 0 and math.isfinite(v) for v in norms)
    assert all(torch.equal(before[k], v) for k, v in net.named_buffers())      # BatchNorm running stats restored
    assert all(p.grad is None for p in net.parameters()) and net.training
