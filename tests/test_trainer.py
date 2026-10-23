"""End-to-end plumbing on random offline data: one run writes every per-run artifact (config, metrics, final summary, gradient and similarity logs, connection weights), and is resumable/skippable."""
import json
from pathlib import Path

import pytest
import torch

import hcsmall.training.trainer as trainer
from hcsmall.models import build_model
from hcsmall.training.optim import split_decay
from hcsmall.training.schedule import lr_at
from hcsmall.training.trainer import checkpoint_epochs, run_id_of, train
from hcsmall.utils.io import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def float64():              # overrides tests/conftest.py: the trainer runs in float32 like a real run
    yield


def fake_cfg(tmp_path, model="resnet20", connection="dhc_n2"):
    return load_config(str(ROOT / "configs/experiments/smoke.yaml"), model, connection, "cifar100_p100", [
        "data.name=fake", "data.fake_size=64", "train.num_workers=0", "train.batch_size=8", "train.epochs=2",
        "train.epoch_steps=2", "train.warmup_epochs=1", "analysis.grad_batches=2", "analysis.probe_images=20",
        f"paths.results_dir={tmp_path / 'raw'}", f"paths.runs_dir={tmp_path / 'runs'}"])


@pytest.mark.parametrize("model,connection", [("resnet20", "dhc_n2"), ("vit_tiny_d6", "mhc_n4"), ("resnet20", "residual")])
def test_run_writes_all_artifacts(tmp_path, model, connection):
    cfg = fake_cfg(tmp_path, model, connection)
    out = train(cfg)
    final = json.loads((out / "final.json").read_text())
    assert final["epochs_done"] == 2 and not final["diverged"] and 0 <= final["final_acc"] <= 1
    assert len((out / "metrics.csv").read_text().splitlines()) == 3                   # header + 2 epochs
    layers = 9 if model == "resnet20" else 12                                         # blocks / sublayers
    assert len((out / "grad_norms.csv").read_text().splitlines()) == 1 + 2 * layers   # checkpoints at epoch 1 and 2
    assert len((out / "similarity.csv").read_text().splitlines()) == 1 + 2 * layers
    assert (out / "config.yaml").exists() and final["git_commit"]
    assert (out / "connections.pt").exists() == (connection != "residual")
    assert not (tmp_path / "runs" / out.name / "last.pt").exists()                    # scratch checkpoint is cleaned up
    train(cfg)                                                                        # a complete run is skipped


def test_interrupted_run_resumes(tmp_path, monkeypatch):
    cfg = fake_cfg(tmp_path)
    out, last = tmp_path / "raw" / run_id_of(cfg), tmp_path / "runs" / run_id_of(cfg) / "last.pt"
    real, calls = trainer.evaluate, []

    def killed_in_epoch_2(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(trainer, "evaluate", killed_in_epoch_2)
    with pytest.raises(KeyboardInterrupt):
        train(cfg)
    assert last.exists() and not (out / "final.json").exists()                        # epoch 1 was checkpointed
    monkeypatch.setattr(trainer, "evaluate", real)
    train(cfg)
    final = json.loads((out / "final.json").read_text())
    assert final["epochs_done"] == 2
    epochs = [line.split(",")[0] for line in (out / "metrics.csv").read_text().splitlines()[1:]]
    assert epochs == ["1", "2"]                                                       # epoch 1 kept, epoch 2 added once


def test_changed_settings_are_refused_not_skipped(tmp_path):
    """A pilot (30 epochs) must never be 'skipped' as a finished 100-epoch run that happens to share the run id."""
    train(fake_cfg(tmp_path))
    longer = fake_cfg(tmp_path)
    longer["train"]["epochs"] = 3
    with pytest.raises(RuntimeError, match="different settings.*train.epochs"):
        train(longer)
    same = fake_cfg(tmp_path)                                  # where / how a run executes is not a setting
    same["train"]["num_workers"], same["train"]["checkpoint_every"] = 0, 5
    train(same)                                                # complete and identical: skipped, no error


def test_checkpoint_every_sets_the_resume_point(tmp_path, monkeypatch):
    cfg = fake_cfg(tmp_path)
    cfg["train"]["epochs"], cfg["train"]["checkpoint_every"] = 3, 2
    rid = run_id_of(cfg)
    real, calls = trainer.evaluate, []

    def killed_in_epoch_3(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(trainer, "evaluate", killed_in_epoch_3)
    with pytest.raises(KeyboardInterrupt):
        train(cfg)
    assert torch.load(tmp_path / "runs" / rid / "last.pt")["epoch"] == 2        # written after epoch 2 only (every 2)
    monkeypatch.setattr(trainer, "evaluate", real)
    train(cfg)
    epochs = [line.split(",")[0] for line in (tmp_path / "raw" / rid / "metrics.csv").read_text().splitlines()[1:]]
    assert epochs == ["1", "2", "3"]


def test_run_pauses_at_the_time_limit_and_resumes(tmp_path):
    """train.stop_at (a session limit) pauses a run after a finished epoch, with a resume point whatever checkpoint_every
    says; the next session has another deadline, and that is not a change of the experiment."""
    cfg = fake_cfg(tmp_path)
    cfg["train"]["epochs"], cfg["train"]["checkpoint_every"], cfg["train"]["stop_at"] = 3, 5, 1       # 1 = long past
    rid = run_id_of(cfg)
    out = tmp_path / "raw" / rid
    with pytest.raises(trainer.Paused):
        train(cfg)
    assert torch.load(tmp_path / "runs" / rid / "last.pt")["epoch"] == 1 and not (out / "final.json").exists()
    cfg["train"]["stop_at"] = None
    train(cfg)
    epochs = [line.split(",")[0] for line in (out / "metrics.csv").read_text().splitlines()[1:]]
    assert epochs == ["1", "2", "3"] and (out / "final.json").exists()


def test_a_missed_deadline_still_makes_progress_and_never_pauses_the_last_epoch(tmp_path):
    cfg = fake_cfg(tmp_path)
    cfg["train"]["epochs"], cfg["train"]["stop_at"] = 3, 1
    rid = run_id_of(cfg)
    for expected in (1, 2):                              # one epoch per call, even though the deadline has passed
        with pytest.raises(trainer.Paused):
            train(cfg)
        assert torch.load(tmp_path / "runs" / rid / "last.pt")["epoch"] == expected
    train(cfg)                                           # epoch 3 is the last one: it is finished, not paused
    assert (tmp_path / "raw" / rid / "final.json").exists()


def test_schedule_shape():
    lrs = [lr_at(s, 100, 10, 0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.01) and max(lrs) == pytest.approx(0.1)           # linear warmup to the base lr
    assert all(a <= b for a, b in zip(lrs[:10], lrs[1:10])) and lrs[-1] < 1e-3        # then cosine to ~0
    assert lr_at(0, 100, 0, 0.1) == pytest.approx(0.1)                                # no warmup: starts at the base lr
    assert lr_at(5, 3, 10, 0.1) > 0                                                   # warmup longer than the run: no crash


def test_checkpoint_epochs():
    assert checkpoint_epochs(200, [0.25, 0.5, 1.0]) == {1, 50, 100, 200}
    assert checkpoint_epochs(2, [0.25, 0.5, 1.0]) == {1, 2}


@pytest.mark.parametrize("model,decay_norm", [("resnet20", True), ("vit_tiny_d6", False)])
def test_weight_decay_groups(model, decay_norm):
    """D9/D17: static HC parameters are never decayed, dynamic projections always are; backbone rule per recipe."""
    net = build_model(load_config(str(ROOT / "configs/experiments/smoke.yaml"), model, "dhc_n4", "cifar100_p100"))
    decay, no_decay = split_decay(net, decay_norm)
    ids = lambda ps: {id(p) for p in ps}
    assert len(decay) + len(no_decay) == len(list(net.parameters())) and not ids(decay) & ids(no_decay)
    conn = net.conns[0]
    assert all(id(getattr(conn, n)) in ids(no_decay) for n in conn.no_decay)
    assert all(id(getattr(conn, n)) in ids(decay) for n in ("w_m", "w_r", "w_b"))
    if not decay_norm:
        assert id(net.pos_embed) in ids(no_decay) and id(net.cls_token) in ids(no_decay)
        assert id(net.head.bias) in ids(no_decay) and id(net.head.weight) in ids(decay)
    else:
        assert id(net.fc.bias) in ids(decay)
