"""Subsets are class-balanced, deterministic, nested and stored once; the test split is never touched."""
import numpy as np
import pytest

from hcsmall.data.subsets import class_balanced_indices, load_or_make_subset

LABELS = np.repeat(np.arange(100), 500)          # CIFAR-100 train layout: 100 classes x 500 images
rng = np.random.default_rng(0)
LABELS = LABELS[rng.permutation(len(LABELS))]    # not sorted, like the real data


@pytest.mark.parametrize("percent,per_class", [(10, 50), (25, 125), (100, 500)])
def test_class_balanced(percent, per_class):
    idx = class_balanced_indices(LABELS, percent, seed=0)
    assert len(idx) == len(np.unique(idx)) == 100 * per_class
    assert (np.bincount(LABELS[idx], minlength=100) == per_class).all()
    assert idx.min() >= 0 and idx.max() < len(LABELS)        # indexes the training split only


def test_deterministic_and_seed_dependent():
    a, b = class_balanced_indices(LABELS, 10, 0), class_balanced_indices(LABELS, 10, 0)
    assert (a == b).all()
    assert (a != class_balanced_indices(LABELS, 10, 1)).any()


def test_nested_across_fractions():
    p10, p25, p100 = (set(class_balanced_indices(LABELS, p, 0)) for p in (10, 25, 100))
    assert p10 <= p25 <= p100


def test_saved_once_and_reused(tmp_path):
    path = tmp_path / "subsets" / "p10.json"
    first = load_or_make_subset(path, LABELS, 10, seed=0)
    assert path.exists()
    # a second call must return the stored indices even if asked with another seed (the file is the truth)
    assert (load_or_make_subset(path, LABELS, 10, seed=99) == first).all()


def test_rejects_a_stored_subset_that_is_not_balanced(tmp_path):
    path = tmp_path / "bad.json"
    load_or_make_subset(path, LABELS, 10, seed=0)
    with pytest.raises(AssertionError):
        load_or_make_subset(path, LABELS, 25, seed=0)         # stored file is a 10% subset, 25% requested
