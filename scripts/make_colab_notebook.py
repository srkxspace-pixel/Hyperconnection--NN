"""Build notebooks/nn09_colab.ipynb: a self-contained Google Colab notebook that embeds a snapshot of this repo's code.
The same script also writes notebooks/nn09_demo_weights.ipynb: the same notebook with PLAN preset to demo (D20), so that
nothing has to be edited to get the weights for the live prediction page.

  python scripts/make_colab_notebook.py            # (re)build the notebooks
  python scripts/make_colab_notebook.py --check    # exit 1 if a notebook embeds code that differs from the repo

Needs only git and the standard library (no torch, no GPU). Regenerate and re-upload after any code change.
The notebook runs the tests, profiles the GPU, launches the resumable runs, builds the tables / figures and packages the
results on Kaggle or Colab, so nothing heavy runs on the local machine. Study numbers come from the run folders it writes, never from notebook cells.
"""
import base64
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ("src/", "scripts/", "configs/", "tests/", "docs/references.md", "pyproject.toml", "requirements.txt")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def code_files() -> list[str]:
    """Tracked and new-but-not-ignored files under INCLUDE (so unstaged new files are embedded too)."""
    names = git("ls-files", "--cached", "--others", "--exclude-standard").split("\n")
    return sorted(f for f in names if f.startswith(INCLUDE) and (ROOT / f).is_file())


def content(f: str) -> bytes:
    """The file with LF line endings: Git for Windows checks out CRLF, which would make a fresh clone look stale."""
    return (ROOT / f).read_bytes().replace(b"\r\n", b"\n")


def snapshot() -> tuple[str, str, int]:
    """(base64 zip of the code, commit label, file count). The zip is reproducible: fixed timestamps and modes."""
    files = code_files()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            info = zipfile.ZipInfo(f, date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            z.writestr(info, content(f), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    dirty = git("status", "--porcelain", "--", *INCLUDE).strip()       # only the embedded code counts, not the notebook itself
    commit = git("rev-parse", "HEAD").strip() + ("-dirty" if dirty else "")
    return base64.b64encode(buf.getvalue()).decode("ascii"), commit, len(files)


NOTEBOOKS = ("nn09_colab.ipynb", "nn09_demo_weights.ipynb")      # the second one has PLAN preset to demo


def check() -> int:
    """0 if every notebook embeds exactly the current code, else 1 (and says which files differ)."""
    stale = 0
    for name in NOTEBOOKS:
        path = ROOT / "notebooks" / name
        if not path.exists():
            print(f"{name} is MISSING; run the script without --check to build it")
            stale = 1
            continue
        cell = next(c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if "SNAPSHOT_B64" in "".join(c["source"]))
        ns: dict = {}
        exec("".join(cell["source"]), ns)
        z = zipfile.ZipFile(io.BytesIO(base64.b64decode(ns["SNAPSHOT_B64"])))
        embedded, current = set(z.namelist()), set(code_files())
        differ = sorted((embedded ^ current) | {f for f in embedded & current if z.read(f) != content(f)})
        print(f"{name} is up to date" if not differ else f"{name} is STALE; differing files: " + ", ".join(differ))
        stale |= int(bool(differ))
    return stale


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str, form: bool = False, collapsed: bool = False) -> dict:
    meta = {"cellView": "form"} if form else {}
    if collapsed:
        meta["collapsed"] = True
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


INTRO = """
# NN-09 · Hyper-Connections at small scale — GPU runner (Kaggle or Colab)

Runs the study's code on a **free cloud GPU** and keeps the results in that service's storage. Nothing here needs the project
repo: the code is embedded in this notebook (snapshot below).

**Where to run it (both are free)**
- **Kaggle (recommended).** A weekly GPU quota (about 30 hours is commonly reported; your profile shows yours), **two T4 GPUs at
  once**, and *Save Version ▸ Save & Run All (Commit)* runs on Kaggle's servers for up to 12 h, so you can close the browser.
  It needs a phone-verified account. In the right-hand *Session options* set **Accelerator = GPU T4 x2** and **Internet = On**.
- **Colab.** Also works (results go to your Google Drive), but the free tier has no background mode and fewer, less
  predictable hours. One GPU only.

**How to use**
1. **First run = a cost check, about 15 minutes, nothing big is trained.** Leave the settings as they are (`PLAN = pilot`,
   `LAUNCH` off) and press *Run all*. It runs the test-suite, one tiny training run, and a profile of your GPU(s) (step time,
   memory and speed of every model × connection). Then it prints how many GPU-hours the pilot plan needs and what other epoch /
   seed choices would cost. Send back the printed output, then **stop the session** so it does not use up quota.
2. **Real run, only after the plan is agreed:** tick **LAUNCH** (and set EPOCHS / SEEDS). Kaggle: *Save Version ▸ Save & Run All
   (Commit)*, which switches LAUNCH on by itself. Colab: *Run all*. Every GPU gets its own worker, and the notebook pauses cleanly after about **MAX_HOURS** hours so
   nothing is lost when the 12 h session limit hits.
3. **Continuing:** finished runs are skipped and paused ones resume. Colab: just *Run all* again. Kaggle: first add the last
   version's output as input (*Add Input ▸ Notebook Output Files ▸ this notebook*); the notebook copies the earlier progress
   back. `PLAN = smoke` repeats the 2-epoch pipeline check, `tier1` is the original 144-run matrix (not part of the compact study),
   and `demo` trains the models of the live prediction page with their weights saved (D20; it took 83 minutes on Kaggle's two T4s,
   including the CIFAR-100 download).

**What you get back**
- the results zip `<DRIVE_FOLDER>/<plan>_raw.zip` (Colab: in your Drive; Kaggle: in the notebook's *Output* tab): unzip it into
  the project's `results/raw/` (for `demo`: `results/demo/`);
- the cost table `<DRIVE_FOLDER>/timing/cost_<gpu>__<plan>.csv` (it is also printed) and, once runs have finished, the tables and
  figures built from them (`<plan>_analysis.zip`).

**Good to know**
- `EPOCHS = 30` is the length of the compact study (D19). The 200 epochs of the original recipes (D7/D8) belong to the 144-run
  matrix and would need hundreds of GPU-hours.
- **P10_STEPS = same_steps** (D6, approved): the 10 % runs get the same number of training steps as the runs on all the data.
- Training is fp32 (no mixed precision).
- The two-GPU workers, the Kaggle restore and the time-limit pause were used in two real Kaggle runs: the 24-run study (`pilot`)
  and the demo run. If a cell prints a red error, the output above it says where.
- The numbers of the report come from the run folders this notebook writes, never from notebook cells.

"""

DEMO_NOTE = """

**This copy is preset to `PLAN = demo`: change nothing in the settings.** It trains ResNet-20 with the four connection types (standard
shortcut, HC x2, HC x4, mHC x4) on all of CIFAR-100 for 30 epochs, seed 0, and saves the trained networks for the live prediction page
(`python app/server.py`, decision D20). It took 83 minutes on 2026-10-22: about 40 minutes for Kaggle to download CIFAR-100 from a slow
server, about 33 minutes of training and about 6 minutes of checks, with nothing to watch.

1. In the right-hand *Session options*: **Accelerator = GPU T4 x2** and **Internet = On**.
2. Add no input: the earlier 24-network run kept CIFAR-100 outside its saved output, so adding it would not save the download. This run
   downloads CIFAR-100 itself and keeps it in its own output for later runs.
3. *Save Version ▸ Save & Run All (Commit)*. You can close the browser.
4. When it has finished: the version's *Output* tab ▸ `nn09/demo_e30_raw.zip` ▸ download it, and unzip it into the project's `results/demo/`.

The rest of this text describes the notebook in general.
"""

PARAMS = '''
#@title 1 · What to run  { display-mode: "form" }
PLAN = "pilot"            #@param ["smoke", "pilot", "tier1", "demo"]
EPOCHS = 30               #@param {type:"integer"}
SEEDS = "0"               #@param {type:"string"}
DEMO_MODELS = "resnet20"  #@param {type:"string"}
P10_STEPS = "same_steps"  #@param ["same_steps", "same_epochs"]
CKPT_EVERY = 1            #@param {type:"integer"}
RUN_TESTS = True          #@param {type:"boolean"}
RUN_TIMING = True         #@param {type:"boolean"}
LAUNCH = False            #@param {type:"boolean"}
MAX_HOURS = 11            #@param {type:"number"}
USE_ALL_GPUS = True       #@param {type:"boolean"}
WEEKLY_GPU_HOURS = 30     #@param {type:"number"}
RUN_ANALYSIS = True       #@param {type:"boolean"}
DRIVE_FOLDER = "nn09"     #@param {type:"string"}
# PLAN: smoke = all 24 model x connection pairs for 2 tiny epochs; pilot = ResNet-20/56 + ViT-Tiny-6, 4 connections,
#       CIFAR-100 100% and 10%, SEEDS, EPOCHS; tier1 = the full 6 models x 4 connections x 2 data settings, SEEDS, EPOCHS;
#       demo = the weights for the live prediction page (app/server.py): DEMO_MODELS (space-separated, e.g. "resnet20" or
#       "resnet20 resnet56") x 4 connections x CIFAR-100 100% x SEEDS x EPOCHS, with train.save_weights on (the study runs never save weights).
# LAUNCH: off = only check and price the plan (the safe default); on = start the real runs.
# MAX_HOURS: both services end a session after 12 h; the runs pause cleanly (progress saved) once this many hours have passed
#       since the notebook started, leaving time to package the results.   USE_ALL_GPUS: one worker per GPU (Kaggle T4 x2 = 2).
# WEEKLY_GPU_HOURS: your free quota per week (Kaggle shows it on your profile); only used to print how many weeks a plan takes.
# SEEDS: "0" or "0,1,2".   CKPT_EVERY: resume-checkpoint cadence in epochs (raise to 5 if Drive writes are slow).
# RUN_ANALYSIS: after the runs, build tables T1-T4 and figures F1-F7 from the finished runs.
# DRIVE_FOLDER: no spaces; the Drive folder on Colab, the output folder on Kaggle. One results folder is used per plan and
#       epoch count (e.g. pilot_e30) so plans never mix.

'''

SETUP = '''
# 3 · Setup: GPUs, storage, project code, helper
import base64, glob, io, json, os, pathlib, shutil, subprocess, sys, time, urllib.request, zipfile
import torch

SESSION_START = time.time()
if os.path.isdir("/kaggle/working"):               # test for Kaggle first: its image also ships a google.colab package that cannot mount Drive
    PLATFORM = "kaggle"
else:
    try:
        from google.colab import drive
        PLATFORM = "colab"
    except ImportError:
        PLATFORM = "other"
IN_COLAB = PLATFORM == "colab"
# Kaggle "Save & Run All (Commit)" is the real run (papermill writes __notebook__.ipynb into the working folder); the
# interactive "Run all" stays a cost check
KAGGLE_BATCH = PLATFORM == "kaggle" and (os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch" or os.path.exists("__notebook__.ipynb"))
if KAGGLE_BATCH and not LAUNCH:
    LAUNCH = True
    print("Kaggle 'Save & Run All' detected: LAUNCH is switched on for this run.")

assert torch.cuda.is_available(), ("No GPU found. Colab: Runtime > Change runtime type > pick a GPU. "
                                   "Kaggle: Session options > Accelerator > GPU T4 x2. Then run again.")
N_GPU = torch.cuda.device_count()
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | platform: {PLATFORM} | {N_GPU} GPU(s)")
for k in range(N_GPU):
    props = torch.cuda.get_device_properties(k)
    print(f"  GPU {k}: {props.name} | {props.total_memory / 2**30:.1f} GiB")


def restore_progress(input_root, base, folder):
    """Kaggle: the output of an earlier version, added as an input, holds the finished runs and the resume checkpoints.
    Copies every folder called `folder` that contains plan results from `input_root` into `base`; returns how many."""
    n, root = 0, pathlib.Path(input_root)
    if not root.is_dir():
        return n
    for cur, dirs, _ in os.walk(root):
        cur = pathlib.Path(cur)
        if len(cur.relative_to(root).parts) > 4:
            dirs[:] = []
        elif cur.name == folder and any(cur.glob("*/raw")):
            shutil.copytree(cur, base, dirs_exist_ok=True)
            n += 1
            dirs[:] = []
    return n


def has_internet():
    try:
        urllib.request.urlopen("https://pypi.org", timeout=10)
        return True
    except Exception:
        return False


if PLATFORM == "colab":
    drive.mount("/content/drive")
    BASE = pathlib.Path("/content/drive/MyDrive") / DRIVE_FOLDER
    WORK = pathlib.Path("/content")
elif PLATFORM == "kaggle":
    BASE = pathlib.Path("/kaggle/working") / DRIVE_FOLDER      # everything under /kaggle/working is saved as the notebook's output
    WORK = pathlib.Path("/kaggle/temp") if os.path.isdir("/kaggle/temp") else pathlib.Path("/tmp/nn09_work")
else:                                              # running elsewhere (e.g. a cloud VM): use local folders
    BASE = pathlib.Path(os.environ.get("NN09_OUT", "nn09_out")).resolve()
    WORK = pathlib.Path("nn09_work").resolve()
BASE.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
if PLATFORM == "kaggle":
    n_restored = restore_progress("/kaggle/input", BASE, DRIVE_FOLDER)
    print(f"earlier progress restored from {n_restored} input folder(s)" if n_restored
          else "no earlier output among the inputs: starting fresh")
    if not has_internet():
        raise RuntimeError("Internet is off. Open Session options (right-hand panel), set Internet = On (it needs a "
                           "phone-verified account), then run again.")

CODE_DIR = WORK / "nn09_code"
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)
zipfile.ZipFile(io.BytesIO(base64.b64decode(SNAPSHOT_B64))).extractall(CODE_DIR)
(CODE_DIR / ".snapshot_commit").write_text(SNAPSHOT_COMMIT, encoding="utf-8")
print("project code:", SNAPSHOT_COMMIT, "->", CODE_DIR)

pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest", "pyyaml", "pandas", "scipy", "matplotlib"])
if pip.returncode:
    print("WARNING: pip install failed (no internet?); continuing with the packages that are already installed.")
ENV = dict(os.environ, PYTHONPATH=str(CODE_DIR / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONUNBUFFERED="1")


def sh(args, cwd=CODE_DIR):
    """Run a command, stream its output into the cell, raise if it fails."""
    p = subprocess.Popen([str(a) for a in args], cwd=cwd, env=ENV, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
    if p.wait():
        raise RuntimeError(f"command failed (exit code {p.returncode}): {' '.join(str(a) for a in args)}")

'''

PLAN_CELL = '''
# 4 · Resolve the plan
seeds = [int(s) for s in SEEDS.replace(",", " ").split()]
PLANS = {
    "smoke": dict(tier="smoke", models=None, connections=None, data=None, seeds=None, epochs=None),
    "pilot": dict(tier="1", models=["resnet20", "resnet56", "vit_tiny_d6"],
                  connections=["residual", "dhc_n2", "dhc_n4", "mhc_n4"],
                  data=["cifar100_p100", "cifar100_p10"], seeds=seeds, epochs=EPOCHS),
    "tier1": dict(tier="1", models=None, connections=None, data=None, seeds=seeds, epochs=EPOCHS),
    "demo": dict(tier="1", models=DEMO_MODELS.split(), connections=["residual", "dhc_n2", "dhc_n4", "mhc_n4"],
                 data=["cifar100_p100"], seeds=seeds, epochs=EPOCHS),
}
plan = PLANS[PLAN]
RUN_NAME = PLAN if plan["epochs"] is None else f"{PLAN}_e{plan['epochs']}"
# On Kaggle the CIFAR-100 download is slow (about an hour), so it is kept in the saved output and restored by the next session
DATA_DIR = (BASE if PLATFORM == "kaggle" else WORK) / "data"
RAW, CKPT = BASE / RUN_NAME / "raw", BASE / RUN_NAME / "runs"
for d in (RAW, CKPT, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

SETS = ["--set", f"paths.results_dir={RAW}", "--set", f"paths.runs_dir={CKPT}",
        "--set", f"paths.data_dir={DATA_DIR}", "--set", f"train.checkpoint_every={CKPT_EVERY}"]
if plan["epochs"]:
    SETS += ["--set", f"train.epochs={plan['epochs']}"]
    if P10_STEPS == "same_steps":                      # D6(b): 390 = one pass over 50,000 images at batch 128
        SETS += ["--set", "train.epoch_steps=390"]
if PLAN == "demo":                                     # also write <run folder>/model.pt; the folder name keeps these apart from the study's raw/
    SETS += ["--set", "train.save_weights=true"]
GRID = []
for flag, key in (("--models", "models"), ("--connections", "connections"), ("--data", "data")):
    if plan[key]:
        GRID += [flag, *plan[key]]
if plan["seeds"]:
    GRID += ["--seeds", *map(str, plan["seeds"])]

listing = subprocess.run([sys.executable, "scripts/run_matrix.py", "--tier", plan["tier"], *GRID, *SETS, "--dry-run"],
                         cwd=CODE_DIR, env=ENV, capture_output=True, text=True, check=True).stdout.splitlines()[0]
print(f"plan {PLAN!r}: {listing}")
print(f"epochs: {plan['epochs'] or '2 (smoke)'} | seeds: {plan['seeds'] or 'smoke default'} | 10%-data budget: {P10_STEPS}")
print(f"results -> {RAW}\\ncheckpoints -> {CKPT}")
'''

TESTS = '''
# 5 · Test-suite (about 3-5 minutes): equivalence at init, parameter counts, subsets, trainer, launcher, ...
#     The checks that were already proven on a real GPU must pass (the run stops otherwise). The newer checks (time-limit
#     pause, one launcher per GPU, tables / figures consistency) only warn, so a problem in them cannot throw
#     away a long run; their output is kept in the log. The demo page's checks (tests/test_predict.py: saved weights reload and
#     prepare an image like the training evaluation; tests/test_app.py: its server, skipped here) are in that second group too,
#     except that tests/test_predict.py is in the first group when PLAN = demo.
NEW_TESTS = ["tests/test_trainer.py::test_run_pauses_at_the_time_limit_and_resumes",
             "tests/test_trainer.py::test_a_missed_deadline_still_makes_progress_and_never_pauses_the_last_epoch",
             "tests/test_matrix.py::test_claim_is_exclusive",
             "tests/test_matrix.py::test_a_run_claimed_by_another_launcher_is_not_started",
             "tests/test_matrix.py::test_time_budget_stops_before_a_run_is_started"]
if RUN_TESTS:
    # the demo plan is useless without working saved weights, so there tests/test_predict.py must pass before anything is trained
    WARN_ONLY = ["tests/test_aggregate.py", "tests/test_app.py"] + ([] if PLAN == "demo" else ["tests/test_predict.py"])
    sh([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *[f"--ignore={t}" for t in WARN_ONLY],
        *[f"--deselect={t}" for t in NEW_TESTS]])
    try:
        sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *WARN_ONLY, *NEW_TESTS])
    except RuntimeError:
        print("WARNING: some newer checks failed (see above). The run continues; send me this output.")
else:
    print("skipped (RUN_TESTS is off)")
'''

SMOKE = '''
# 6 · One tiny training run: catches setup problems (dataset download, paths) before a long launch
#     (demo plan: with the weights saved, then reloaded and used to classify one image)
if PLAN != "smoke":
    scratch = WORK / "smoke_scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    sh([sys.executable, "scripts/train.py", "--config", "configs/experiments/smoke.yaml",
        "--set", f"paths.results_dir={scratch}/raw", "--set", f"paths.runs_dir={scratch}/runs",
        "--set", f"paths.data_dir={DATA_DIR}", *(["--set", "train.save_weights=true"] if PLAN == "demo" else [])])
    if PLAN == "demo":                                 # the saved weights must reload and classify before the long run starts
        sys.path.insert(0, str(CODE_DIR / "src"))
        from torchvision import datasets
        from hcsmall.predict import classify, load_run

        run = load_run(next((scratch / "raw").glob("*")), "cpu")
        probs, _ = classify(run, datasets.CIFAR100(DATA_DIR, train=False)[0][0])
        print(f"saved weights reload and classify: {len(probs)} probabilities, sum {sum(probs):.4f}")
else:
    print("not needed: the smoke plan below runs the same thing for every model x connection")
'''

TIMING = '''
# 7 · Profile every model x connection in the plan (step time, peak memory, inference speed, MACs; about 5 min) and
#     price the plan in GPU-hours. The cost table is also what tables T3 and figure F5 are made from.
if RUN_TIMING and PLAN not in ("smoke", "demo"):       # the demo models are already in the saved cost table
    models = plan["models"] or ["resnet20", "resnet56", "resnet110", "vit_tiny_d6", "vit_tiny_d12", "vit_tiny_d24"]
    conns = plan["connections"] or ["residual", "dhc_n2", "dhc_n4", "mhc_n4"]
    data = [d.replace("cifar100_", "") for d in (plan["data"] or ["cifar100_p100", "cifar100_p10"])]
    try:
        sh([sys.executable, "scripts/profile_cost.py", "--models", *models, "--connections", *conns])
        prefix = "cost"
    except RuntimeError:
        print("WARNING: profile_cost.py failed; using the plain timing script instead (no inference speed / MACs).")
        sh([sys.executable, "scripts/time_epoch.py", "--models", *models, "--connections", *conns])
        prefix = "timing"
    csv_path = max(glob.glob(str(CODE_DIR / f"results/tables/{prefix}_*.csv")), key=os.path.getmtime)
    (BASE / "timing").mkdir(exist_ok=True)
    cost_copy = BASE / "timing" / f"{pathlib.Path(csv_path).stem}__{PLAN}.csv"
    shutil.copy(csv_path, cost_copy)
    print("\\nCost table (if you cannot send the file, paste this):\\n" + pathlib.Path(csv_path).read_text(encoding="utf-8"))
    budget = [sys.executable, "scripts/budget.py", csv_path, "--epochs", plan["epochs"], "--seeds", len(plan["seeds"]),
              "--models", *models, "--connections", *conns, "--data", *data, "--steps", P10_STEPS,
              "--gpus", N_GPU if USE_ALL_GPUS else 1]
    if WEEKLY_GPU_HOURS:
        budget += ["--weekly-hours", WEEKLY_GPU_HOURS]
    sh(budget)
    print("\\nCost table also saved to", cost_copy)
else:
    print("skipped")

'''

LAUNCH = '''
# 8 · Run, one worker per GPU. Safe to run again after a disconnect or a session limit: finished runs are skipped, paused
#     or interrupted ones resume.
if LAUNCH:
    import signal

    t0 = time.time()
    left_h = MAX_HOURS - (t0 - SESSION_START) / 3600          # the session ends at 12 h; leave time to package the results
    if left_h < 0.3:
        print("Too little session time left to start (MAX_HOURS counts from the start of the notebook). Start a fresh session.")
    else:
        n_workers = N_GPU if USE_ALL_GPUS else 1
        if n_workers > 1:                                      # download CIFAR-100 once, not twice at the same time
            sh([sys.executable, "-c", "from torchvision import datasets; import sys; "
                "[datasets.CIFAR100(sys.argv[1], train=t, download=True) for t in (True, False)]", DATA_DIR])
        claims = WORK / f"claims_{int(t0)}"                    # a run is taken by creating a file here, so no run is started twice
        workers = max(1, (os.cpu_count() or 2) // n_workers)   # data-loader processes per GPU worker
        logs, procs = [], []
        for k in range(n_workers):
            cmd = [sys.executable, "scripts/run_matrix.py", "--tier", plan["tier"], *GRID, *SETS,
                   "--set", f"train.num_workers={min(4, workers)}", "--claim-dir", claims, "--stop-after-hours", f"{left_h:.3f}"]
            logs.append(open(BASE / RUN_NAME / f"launch_gpu{k}.log", "a", encoding="utf-8"))
            procs.append(subprocess.Popen([str(c) for c in cmd], cwd=CODE_DIR, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(k)),
                                          stdout=logs[-1], stderr=subprocess.STDOUT, text=True, start_new_session=True))
        total = int(listing.split(" runs")[0])
        print(f"{n_workers} worker(s) started; the plan pauses itself after about {left_h:.1f} h. Progress every 10 minutes:", flush=True)
        next_print = t0 + 600
        try:
            while any(p.poll() is None for p in procs):
                time.sleep(30)
                if time.time() >= next_print:
                    next_print += 600
                    print(f"  {(time.time() - t0) / 3600:.1f} h | {len(list(RAW.glob('*/final.json')))}/{total} runs finished", flush=True)
        except BaseException:                                  # stop button or disconnect: leave no orphan training processes
            for p in procs:
                try:
                    os.killpg(p.pid, signal.SIGTERM) if hasattr(os, "killpg") else p.terminate()
                except ProcessLookupError:
                    pass
            raise
        finally:
            for log in logs:
                log.close()
        print(f"finished in {(time.time() - t0) / 3600:.2f} h | worker exit codes: {[p.returncode for p in procs]}")
        for k in range(n_workers):
            tail = (BASE / RUN_NAME / f"launch_gpu{k}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
            print(f"--- GPU {k}, last lines of its log:\\n" + "\\n".join(tail))
else:
    print("skipped (LAUNCH is off)")

'''

PROGRESS = '''
# 9 · What is finished (a progress view only: the study's numbers come from the run folders, not from this table)
import pandas as pd

rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(RAW.glob("*/final.json"))]
if rows:
    df = pd.DataFrame(rows)
    df["hours"] = df["train_seconds"] / 3600
    pd.set_option("display.max_rows", 300)
    display(df[["run_id", "epochs_done", "final_acc", "best_acc", "diverged", "hours", "peak_mem_mb"]].round(4))
    print(f"{len(df)} runs complete | {int(df['diverged'].sum())} diverged | {df['hours'].sum():.1f} GPU-hours of training")
else:
    print("no finished runs yet")
failed = RAW / "_failures.txt"
if failed.exists():
    print("FAILURES (see each run's train.log):\\n" + failed.read_text(encoding="utf-8")[-3000:])
'''

PACKAGE = '''
# 10 · Package the results: one zip, to unzip into the project's results/raw/
if any(RAW.glob("*/final.json")):
    zip_path = shutil.make_archive(str(BASE / f"{RUN_NAME}_raw"), "zip", RAW)
    print(f"{zip_path}  ({os.path.getsize(zip_path) / 2**20:.1f} MB)")
    if IN_COLAB:
        print("Optional direct download to your computer:")
        print(f"    from google.colab import files; files.download({zip_path!r})")
    elif PLATFORM == "kaggle":
        print("On Kaggle the zip appears in this notebook's Output tab once the version has finished; download it from there.")
    if PLAN == "demo":
        print("DEMO WEIGHTS: unzip this zip into the project's results/demo/ (NOT results/raw/: the folder names are the study's).")
else:
    print("nothing to package yet")

'''

DEMO_CHECK = '''
# 11 · (PLAN = demo) Check the saved weights the way the demo page uses them: reload every model.pt and classify the 10,000 test
#      images one by one through hcsmall.predict (PIL image -> crop / resize -> normalise -> network), then compare the accuracy
#      with the one logged at the end of training. About a minute per model. Only prints: a problem here cannot lose the zip.
if PLAN == "demo" and any(RAW.glob("*/model.pt")):
    try:
        sys.path.insert(0, str(CODE_DIR / "src"))
        from torchvision import datasets
        from hcsmall.predict import classify, load_run

        test = datasets.CIFAR100(DATA_DIR, train=False)
        for f in sorted(RAW.glob("*/model.pt")):
            run = load_run(f.parent, "cuda" if torch.cuda.is_available() else "cpu")
            hits = 0
            for img, label in test:
                probs = classify(run, img)[0]
                hits += probs.index(max(probs)) == label
            logged = json.loads((f.parent / "final.json").read_text(encoding="utf-8"))["final_acc"]
            same = abs(hits / len(test) - logged) < 0.002
            print(f"{'OK  ' if same else 'DIFF'} {f.parent.name}: reloaded {hits / len(test):.4f} | logged at the end of training {logged:.4f}")
    except Exception as e:
        print(f"WARNING: the weights check failed ({type(e).__name__}: {e}). The weights are saved anyway; send me this output.")
else:
    print("skipped (not the demo plan, or no saved weights yet)")
'''

ANALYSIS = '''
# 12 · Tables T1-T4 and figures F1-F7 from the finished runs (no GPU needed; a minute or two)
if RUN_ANALYSIS and PLAN != "demo" and any(RAW.glob("*/final.json")):
    try:
        import pandas as pd
        from IPython.display import Image, display

        tables_dir, figs_dir = CODE_DIR / "results/tables", CODE_DIR / "results/figures"
        costs = sorted((BASE / "timing").glob(f"cost_*__{PLAN}.csv"), key=os.path.getmtime) \\
            or sorted((BASE / "timing").glob("cost_*.csv"), key=os.path.getmtime)
        COST = ["--cost", costs[-1]] if costs else []
        print("cost table:", costs[-1].name if costs else "none (tables T3 and figure F5 need cell 7 with RUN_TIMING on)")
        sh([sys.executable, "scripts/aggregate.py", "--raw", RAW, "--out", tables_dir, *COST])
        sh([sys.executable, "scripts/make_figures.py", "--raw", RAW, "--out", figs_dir, *COST])
        for f in sorted(tables_dir.glob("T1_*.csv")) + sorted(tables_dir.glob("T2_gain.csv")):
            print("\\n" + f.name + " (the CSV the report table is made from)")
            display(pd.read_csv(f).round(3))
        for name in ("F1_accuracy_vs_depth", "F7_training_curves"):
            if (figs_dir / f"{name}.png").exists():
                display(Image(filename=str(figs_dir / f"{name}.png")))
        analysis_zip = shutil.make_archive(str(BASE / f"{RUN_NAME}_analysis"), "zip", CODE_DIR / "results")
        print(f"{analysis_zip}  ({os.path.getsize(analysis_zip) / 2**20:.1f} MB): tables/ and figures/, to unzip into the project's results/")
    except Exception as e:
        print(f"WARNING: the analysis step failed ({type(e).__name__}: {e}). The run folders are untouched; send me this output.")
else:
    print("skipped (RUN_ANALYSIS is off, this is the demo plan, or no finished runs yet)")
'''

OUTRO = """
## Bring it back into the project
1. **Executed notebook:** download it (*File ▸ Download*) and send it back; it is kept under `notebooks/executed/`.
2. **Results:** unzip `<plan>_raw.zip` (Colab: `MyDrive/<DRIVE_FOLDER>/`, Kaggle: the notebook's *Output* tab) into `results/raw/`
   (one folder per run). Keep one plan and epoch count per `results/raw/`: the analysis scripts assume all runs share their settings.
3. **Cost table:** `<DRIVE_FOLDER>/timing/cost_<gpu>__<plan>.csv` goes into `results/tables/` (T3 and F5 are made from it, and
   `python scripts/budget.py` uses it to price a plan).
4. **Tables and figures (optional):** `<plan>_analysis.zip` unzips into `results/`; they can always be rebuilt from `results/raw/`
   with `python scripts/aggregate.py && python scripts/make_figures.py`.
5. **Demo weights (`PLAN = demo`):** unzip `demo_e30_raw.zip` into `results/demo/` (not `results/raw/`), then start the page with
   `python app/server.py`. Cell 11 prints OK for every model whose reloaded accuracy equals the one logged at the end of training.

If anything fails, the cell output above it says where; the first thing to try is *Run all* again (runs resume).

"""


def main() -> None:
    if "--check" in sys.argv:                       # python scripts/make_colab_notebook.py --check
        sys.exit(check())
    b64, commit, n_files = snapshot()
    chunks = [b64[i:i + 4000] for i in range(0, len(b64), 4000)]
    snap = ('#@title Project code snapshot (generated by scripts/make_colab_notebook.py; do not edit)  { display-mode: "form" }\n'
            f'SNAPSHOT_COMMIT = "{commit}"\n'
            "SNAPSHOT_B64 = \"\".join([\n" + "".join(f'    "{c}",\n' for c in chunks) + "])\n")
    for name in NOTEBOOKS:
        demo = name == "nn09_demo_weights.ipynb"
        intro = INTRO.replace("# NN-09 · Hyper-Connections at small scale — GPU runner (Kaggle or Colab)",
                              "# NN-09 · Demo weights for the live prediction page" + DEMO_NOTE, 1) if demo else INTRO
        params = PARAMS.replace('PLAN = "pilot"', 'PLAN = "demo" ', 1) if demo else PARAMS
        nb = {
            "cells": [md(intro), code(params, form=True), code(snap, form=True, collapsed=True), code(SETUP),
                      code(PLAN_CELL), code(TESTS), code(SMOKE), code(TIMING), code(LAUNCH), code(PROGRESS),
                      code(PACKAGE), code(DEMO_CHECK), code(ANALYSIS), md(OUTRO)],
            "metadata": {"colab": {"name": name, "provenance": [], "gpuType": "T4"},
                         "accelerator": "GPU",
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0,
        }
        out = ROOT / "notebooks" / name
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")   # LF on every OS
        print(f"wrote {out.relative_to(ROOT)}: {n_files} files embedded ({len(b64) / 1024:.0f} KiB base64), code snapshot {commit}")


if __name__ == "__main__":
    main()
