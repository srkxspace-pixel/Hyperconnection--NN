"""The launcher expands Tier 1 to exactly the 144 runs of the original plan (seed 0 first) and the smoke tier to every model x connection."""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = re.compile(r"^(resnet(20|56|110)|vittiny(6|12|24))_(residual_n1|dhc_n2|dhc_n4|mhc_n4)_cifar100p(100|10)_s[012]$")


def todo(tier: str, *extra: str) -> list:
    out = subprocess.run([sys.executable, str(ROOT / "scripts/run_matrix.py"), "--tier", tier, "--dry-run",
                          "--set", f"paths.results_dir={ROOT / 'nonexistent_results'}", *extra],
                         capture_output=True, text=True, check=True, cwd=ROOT).stdout.splitlines()
    return [line.split()[-1] for line in out if line.startswith("  todo")]


def test_sub_grid_flags_select_exactly_the_requested_runs():
    ids = todo("1", "--models", "resnet20", "--connections", "residual", "dhc_n4", "--data", "cifar100_p10", "--seeds", "0", "1")
    assert ids == ["resnet20_residual_n1_cifar100p10_s0", "resnet20_dhc_n4_cifar100p10_s0",
                   "resnet20_residual_n1_cifar100p10_s1", "resnet20_dhc_n4_cifar100p10_s1"]


def test_tier1_is_the_144_runs_of_goal_md():
    ids = todo("1")
    assert len(ids) == len(set(ids)) == 144 == 6 * 4 * 2 * 3
    assert all(RUN_ID.match(i) for i in ids), [i for i in ids if not RUN_ID.match(i)]
    assert {i.split("_")[-1] for i in ids[:48]} == {"s0"}                  # seed 0 of every cell first
    assert {i.rsplit("_", 1)[0].rsplit("_", 1)[1] for i in ids} == {"cifar100p100", "cifar100p10"}


def test_smoke_covers_every_model_connection_pair():
    ids = todo("smoke")
    assert len(ids) == len(set(ids)) == 24 and all(RUN_ID.match(i) for i in ids)


def run_matrix(*args: str) -> str:
    return subprocess.run([sys.executable, str(ROOT / "scripts/run_matrix.py"), *args],
                          capture_output=True, text=True, check=True, cwd=ROOT).stdout


ONE_RUN = ["--tier", "smoke", "--models", "resnet20", "--connections", "residual", "--seeds", "0"]


def test_claim_is_exclusive(tmp_path):
    spec = importlib.util.spec_from_file_location("run_matrix", ROOT / "scripts/run_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.claim(None, "a") and mod.claim(None, "a")                       # no claim folder: every launcher may run it
    assert mod.claim(tmp_path / "c", "a") and not mod.claim(tmp_path / "c", "a") and mod.claim(tmp_path / "c", "b")


def test_a_run_claimed_by_another_launcher_is_not_started(tmp_path):
    """One launcher per GPU shares a claim folder, so no run is trained twice."""
    claims, raw = tmp_path / "claims", tmp_path / "raw"
    claims.mkdir()
    (claims / "resnet20_residual_n1_cifar100p10_s0").write_text("")            # the other launcher took it
    out = run_matrix(*ONE_RUN, "--claim-dir", str(claims), "--set", f"paths.results_dir={raw}")
    assert "1 runs, 0 already complete" in out and "ok" not in out and "FAILED" not in out
    assert not raw.exists()                                                    # nothing was started


def test_time_budget_stops_before_a_run_is_started(tmp_path):
    raw = tmp_path / "raw"
    out = run_matrix(*ONE_RUN, "--stop-after-hours", "0.01", "--set", f"paths.results_dir={raw}")   # 36 s < 10 min left
    assert "time budget reached" in out and not raw.exists()
