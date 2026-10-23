"""The central guarantee: every HC / mHC stack and model computes the residual function at init (float64)."""
import pytest
import torch
from helpers import build, run

from hcsmall.models import build_model

KINDS = [("shc", 2), ("shc", 4), ("dhc", 2), ("dhc", 4), ("mhc", 2), ("mhc", 4)]
MODELS = [("resnet", 20), ("resnet", 56), ("resnet", 110), ("vit", 6), ("vit", 12), ("vit", 24)]
MODEL_KINDS = [("shc", 2), ("dhc", 2), ("dhc", 4), ("mhc", 4)]       # the Tier-1 arms plus the Tier-2 SHC ablation


def make_model(family, depth, kind, n):
    torch.manual_seed(0)         # equal seeds must give equal backbones whatever the connection
    cfg = {"model": {"family": family, "depth": depth, "dim": 192, "heads": 3, "mlp_ratio": 4},
           "connection": {"kind": kind, "n": n}, "data": {"num_classes": 100, "img_size": 32, "patch_size": 4}}
    return build_model(cfg)


@pytest.mark.parametrize("family,depth", MODELS)
@pytest.mark.parametrize("kind,n", MODEL_KINDS)
@pytest.mark.parametrize("train", [True, False])
def test_model_equals_residual_at_init(family, depth, kind, n, train):
    ref, hc = make_model(family, depth, "residual", 1), make_model(family, depth, kind, n)
    backbone = lambda m: {k: v for k, v in m.state_dict().items() if not k.startswith("conns.")}
    for (k1, v1), (k2, v2) in zip(backbone(ref).items(), backbone(hc).items()):
        assert k1 == k2 and torch.equal(v1, v2)         # same seed -> identical backbone
    x = torch.randn(2, 3, 32, 32)
    ref.train(train), hc.train(train)
    # relative: an untrained ResNet-110 in eval mode (default BN statistics) has logits ~1e4, and mHC's
    # exp/log/sigmoid path adds float64 round-off of ~1e-13 relative. Real wiring errors are O(1) relative.
    torch.testing.assert_close(hc(x), ref(x), atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("layout", ["tokens", "conv"])
@pytest.mark.parametrize("kind,n", KINDS)
@pytest.mark.parametrize("train", [True, False])
def test_equals_residual_at_init(layout, kind, n, train):
    parts = build(layout)
    for m in parts[0] + [s for s in parts[1] if s is not None]:
        m.train(train)           # BatchNorm: batch statistics (train) and running statistics (eval)
    ref, _ = run("residual", 1, layout, parts)
    out, _ = run(kind, n, layout, parts)
    torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("layout", ["tokens", "conv"])
@pytest.mark.parametrize("kind,n", [("shc", 4), ("dhc", 4), ("mhc", 4)])
def test_equivalence_is_not_vacuous(layout, kind, n):
    """Moving the connection parameters must change the output, i.e. the connection is really used."""
    parts = build(layout)
    for m in parts[0]:
        m.eval()
    ref, _ = run("residual", 1, layout, parts)
    torch.manual_seed(1)
    out, _ = run(kind, n, layout, parts, perturb=0.3)
    assert (out - ref).abs().max() > 1e-3
