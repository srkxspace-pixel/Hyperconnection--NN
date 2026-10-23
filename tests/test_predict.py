"""Saved weights for the demo page (D20): the trainer writes model.pt only when asked, and hcsmall.predict rebuilds the same
network from it and prepares an image exactly like the training evaluation does."""
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageChops
from torchvision import transforms as T

from hcsmall.models import build_model
from hcsmall.predict import classify, load_run, prepare, to_tensor
from hcsmall.training.trainer import train
from hcsmall.utils.io import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def float64():              # overrides tests/conftest.py: weights are saved and reloaded in float32 like a real run
    yield


def fake_cfg(tmp_path, model="resnet20", connection="dhc_n2", save=False):
    return load_config(str(ROOT / "configs/experiments/smoke.yaml"), model, connection, "cifar100_p100", [
        "data.name=fake", "data.fake_size=64", "train.num_workers=0", "train.batch_size=8", "train.epochs=2",
        "train.epoch_steps=2", "train.warmup_epochs=1", "analysis.grad_batches=2", "analysis.probe_images=20",
        f"train.save_weights={'true' if save else 'false'}",
        f"paths.results_dir={tmp_path / 'raw'}", f"paths.runs_dir={tmp_path / 'runs'}"])


def test_weights_are_saved_only_when_asked(tmp_path):
    assert not (train(fake_cfg(tmp_path / "off")) / "model.pt").exists()
    out = train(fake_cfg(tmp_path / "on", save=True))
    ck = torch.load(out / "model.pt", weights_only=True)                             # plain data only: safe to load
    assert set(ck) == {"model", "cfg"} and ck["cfg"]["data"]["mean"] is not None     # the measured mean / std travel along


@pytest.mark.parametrize("model,connection", [("resnet20", "dhc_n2"), ("vit_tiny_d6", "mhc_n4")])
def test_reloaded_run_is_the_same_network(tmp_path, model, connection):
    cfg = fake_cfg(tmp_path, model, connection)
    cfg["data"]["mean"], cfg["data"]["std"] = [0.5] * 3, [0.25] * 3
    net = build_model(cfg).eval()                                                    # random weights are enough for a round trip
    torch.save({"model": net.state_dict(), "cfg": cfg}, tmp_path / "model.pt")
    run = load_run(tmp_path)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(net(x), run["model"](x), atol=1e-6) and not run["model"].training
        probs, seen = classify(run, Image.new("RGB", (50, 80), "red"))               # not square: centre-cropped
    assert len(probs) == 100 and abs(sum(probs) - 1) < 1e-4 and seen.size == (32, 32)


def test_preparation_matches_the_training_evaluation():
    cfg = {"data": {"img_size": 32, "mean": [0.5071, 0.4865, 0.4409], "std": [0.2673, 0.2564, 0.2762]}}
    cifar = Image.effect_noise((32, 32), 60).convert("RGB")                          # stands in for a CIFAR image
    assert ImageChops.difference(prepare(cifar, 32), cifar).getbbox() is None        # a 32 x 32 image is left alone
    d = cfg["data"]
    expected = T.Compose([T.ToTensor(), T.Normalize(d["mean"], d["std"])])(cifar).unsqueeze(0)
    assert torch.equal(to_tensor(cifar, cfg), expected)
    assert prepare(Image.new("RGB", (200, 80)), 32).size == (32, 32)
