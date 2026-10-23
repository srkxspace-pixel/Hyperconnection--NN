"""Residual models have the standard ResNet / ViT-Tiny parameter counts (closed forms, derived from the architecture)."""
import pytest

from hcsmall.models import build_model


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def resnet_params(depth, K):
    """Pre-act ResNet-6k+2: stem, 3 stages (16/32/64), 2 BN+conv per block, a 1x1 projection at the 2 transitions."""
    k, total, cin = (depth - 2) // 6, 3 * 16 * 9, 16
    for stage, c in enumerate((16, 32, 64)):
        for i in range(k):
            ci = cin if i == 0 else c
            total += ci * c * 9 + c * c * 9 + 2 * ci + 2 * c
            if i == 0 and stage > 0:
                total += ci * c
        cin = c
    return total + 2 * 64 + 64 * K + K


def vit_params(depth, K, img=32, patch=4, dim=192, ratio=4):
    tokens = (img // patch) ** 2 + 1
    block = (2 * dim + dim * 3 * dim + 3 * dim + dim * dim + dim          # LN1, qkv, proj
             + 2 * dim + dim * ratio * dim + ratio * dim + ratio * dim * dim + dim)   # LN2, fc1, fc2
    return (3 * patch * patch * dim + dim) + dim + tokens * dim + depth * block + 2 * dim + dim * K + K


def resnet(depth, K):
    return build_model({"model": {"family": "resnet", "depth": depth}, "connection": {"kind": "residual", "n": 1},
                        "data": {"num_classes": K}})


def vit(depth, K, img=32, patch=4):
    return build_model({"model": {"family": "vit", "depth": depth, "dim": 192, "heads": 3, "mlp_ratio": 4},
                        "connection": {"kind": "residual", "n": 1},
                        "data": {"num_classes": K, "img_size": img, "patch_size": patch}})


@pytest.mark.parametrize("depth", [20, 56, 110])
@pytest.mark.parametrize("K", [10, 100])
def test_resnet_matches_closed_form(depth, K):
    assert n_params(resnet(depth, K)) == resnet_params(depth, K)


@pytest.mark.parametrize("depth,published", [(20, 0.27e6), (56, 0.85e6), (110, 1.7e6)])
def test_resnet_close_to_published_counts(depth, published):
    # He et al. CVPR 2016, Table 6 (CIFAR-10, K=10). They use identity "option A" shortcuts, i.e. no projection
    # parameters; ours adds the two 1x1 projections of D4a/D15a, so we sit slightly above (< 3%).
    assert n_params(resnet(depth, 10)) == pytest.approx(published, rel=0.03)


@pytest.mark.parametrize("depth", [6, 12, 24])
def test_vit_matches_closed_form(depth):
    assert n_params(vit(depth, 100)) == vit_params(depth, 100)


def test_vit_tiny_is_the_standard_vit_tiny():
    """Patch 16 at 224 px with a 1000-class head is the standard ViT-Tiny/16: 5,717,416 parameters."""
    assert n_params(vit(12, 1000, img=224, patch=16)) == 5_717_416
