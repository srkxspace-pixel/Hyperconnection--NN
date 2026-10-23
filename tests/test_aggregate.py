"""Tables and figures from run folders: statistics against independent computations, LaTeX marks, edge cases."""
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from synth import GAIN, make_cost, make_raw, make_run

from hcsmall.analysis import figures, runs, tables

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def raw(tmp_path):
    return make_raw(tmp_path / "raw")


def test_loader_reads_the_run_folders(raw):
    df = runs.load_runs(raw)
    assert len(df) == 2 * 3 * 2 * 3 and set(df.conn) == set(GAIN) and set(df.data) == {"cifar100_p100", "cifar100_p10"}
    r = df[(df.model == "resnet20") & (df.conn == "dhc_n2") & (df.data == "cifar100_p100") & (df.seed == 1)].iloc[0]
    assert r.final_acc == pytest.approx(61.0) and r.epoch_steps == 10 and r.family == "resnet"


def test_paired_statistics_match_independent_computation(raw):
    df = runs.load_runs(raw)
    p = runs.paired(df, "vit_tiny_d6", "cifar100_p100", "mhc_n4")           # differences +0.5, -0.5, +0.1 (by seed)
    d = np.array([0.5, -0.5, 0.1])
    assert p["n"] == 3 and p["delta"] == pytest.approx(d.mean()) and p["sd"] == pytest.approx(d.std(ddof=1))
    assert p["p"] == pytest.approx(stats.ttest_rel([50.3, 49.5, 50.3], [49.8, 50.0, 50.2]).pvalue)   # 50+off+gain vs 50+off
    assert d.min() <= p["lo"] <= p["hi"] <= d.max() and not p["sig"]
    q = runs.paired(df, "resnet20", "cifar100_p100", "dhc_n2")             # +1.0 on every seed: zero variance
    assert q["delta"] == pytest.approx(1.0) and q["p"] == 0.0 and q["lo"] == q["hi"] == pytest.approx(1.0) and q["sig"]


def test_pairs_use_only_common_seeds_and_diverged_runs_are_excluded(tmp_path):
    raw = make_raw(tmp_path / "raw")
    shutil.rmtree(raw / "resnet20_residual_cifar100p100_s2")                 # residual lacks seed 2
    shutil.rmtree(raw / "resnet20_dhc_n2_cifar100p100_s0")
    make_run(raw, "resnet20", "dhc_n2", 100, 0, diverged=True)                # and dhc_n2 seed 0 diverged
    df = runs.load_runs(raw)
    assert runs.paired(df, "resnet20", "cifar100_p100", "dhc_n2")["n"] == 1   # only seed 1 is in both
    t, _ = tables.t1(df, "cifar100_p100")
    row = t[t.model == "resnet20"].iloc[0]
    assert row.dhc_n2_n == 2 and row.residual_n == 2                          # diverged / missing runs are not averaged
    long4, _ = tables.t4(df)
    c = long4[(long4.model == "resnet20") & (long4.conn == "dhc_n2") & (long4.data == "cifar100_p100")].iloc[0]
    assert c.n_runs == 3 and c.n_diverged == 1


def test_t1_values_best_in_bold_and_significance_marks(raw):
    df = runs.load_runs(raw)
    t, tex = tables.t1(df, "cifar100_p100")
    row = t[t.model == "resnet20"].iloc[0]
    assert row.residual_mean == pytest.approx(60.0) and row.residual_std == pytest.approx(0.2)
    assert row.dhc_n2_mean == pytest.approx(61.0) and row.dhc_n2_sig and not row.mhc_n4_sig
    line = next(l for l in tex.splitlines() if l.startswith("ResNet-20"))
    cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
    assert cells[2] == r"\textbf{61.00}$\pm$0.20" + "$^{*}$"                  # best of the row (dhc_n2), significant
    assert cells[1] == "60.00$\\pm$0.20" and "textbf" not in cells[1] + cells[3] and "^{*}" not in cells[1] + cells[3]
    assert tex.count(r"\textbf") == 2                                         # one best per row, two rows


def test_t2_gain_table(raw):
    long, tex = tables.t2(runs.load_runs(raw))
    r = long[(long.model == "resnet20") & (long.data == "cifar100_p10") & (long.conn == "dhc_n2")].iloc[0]
    assert r.delta == pytest.approx(1.0) and r.significant and r.n_pairs == 3
    assert len(long) == 2 * 2 * 2                                              # models x data x non-residual connections
    assert "+1.00 [+1.00, +1.00]$^{*}$" in tex                                 # signed numbers, CI in brackets, star


def test_mixed_training_budgets_are_refused(tmp_path):
    raw = make_raw(tmp_path / "raw", seeds=(0,), percents=(100,), conns=("residual",))
    make_run(raw, "resnet20", "dhc_n2", 100, 0, epochs=5)                      # same family and data, different epochs
    with pytest.raises(ValueError, match="mixes training budgets"):
        tables.build_all(raw, tmp_path / "t", None, ROOT / "configs")
    with pytest.raises(ValueError, match="mixes training budgets"):
        figures.make_all(raw, tmp_path / "f", None)


def test_build_all_writes_every_table(raw, tmp_path):
    cost = make_cost(tmp_path / "cost_Test_GPU.csv")
    out = tmp_path / "tables"
    names = {p.name for p in tables.build_all(raw, out, cost, ROOT / "configs")}
    for want in ("runs.csv", "coverage.csv", "T1_cifar100_p100.csv", "T1_cifar100_p100.tex", "T1_cifar100_p10.tex", "T2_gain.csv",
                 "T2_gain.tex", "T3_cost.csv", "T3_cost.tex", "T4_stability.csv", "T4_cifar100_p100.tex", "T0_setup.tex", "macros.tex"):
        assert want in names, want
    assert r"\newcommand{\NRuns}{36}" in (out / "macros.tex").read_text() and r"\NSeeds}{3}" in (out / "macros.tex").read_text()
    t3 = pd.read_csv(out / "T3_cost.csv")
    r = t3[(t3.model == "vit_tiny_d6") & (t3.connection == "mhc_n4")].iloc[0]
    assert r.mem_x == pytest.approx(2.0) and r.train_slowdown_x == pytest.approx(2.0) and r.infer_slowdown_x == pytest.approx(2.0)
    setup = (out / "T0_setup.tex").read_text()
    assert "SGD" in setup and "AdamW" in setup and "0.0005" in setup           # recipes are read from the YAML configs
    assert "nan" not in (out / "T1_cifar100_p100.tex").read_text().lower()


@pytest.mark.parametrize("with_cost", [True, False])
def test_figures_render(raw, tmp_path, with_cost):
    cost = make_cost(tmp_path / "cost.csv") if with_cost else None
    out = tmp_path / "figs"
    stems = {p.stem for p in figures.make_all(raw, out, cost)}
    want = {"F1_accuracy_vs_depth", "F2_gain_vs_data_fraction", "F3_representation_collapse", "F4_gradient_norms",
            "F6_connection_patterns", "F7_training_curves"} | ({"F5_accuracy_vs_cost"} if with_cost else set())
    assert stems == want
    for s in want:
        for ext in ("pdf", "png"):
            assert (out / f"{s}.{ext}").stat().st_size > 1000, (s, ext)


def test_single_seed_and_single_fraction_degrade_gracefully(tmp_path):
    """The smoke run has one seed and one data fraction: no std, no tests, no F2, but nothing crashes."""
    raw = make_raw(tmp_path / "raw", seeds=(0,), percents=(10,))
    df = runs.load_runs(raw)
    t, tex = tables.t1(df, "cifar100_p10")
    assert math.isnan(t.iloc[0].residual_std) and "$\\pm$" not in tex
    assert math.isnan(runs.paired(df, "resnet20", "cifar100_p10", "dhc_n2")["p"])
    stems = {p.stem for p in figures.make_all(raw, tmp_path / "f", None)}
    assert "F2_gain_vs_data_fraction" not in stems and "F1_accuracy_vs_depth" in stems
