# NN-09 — Hyper-Connections at Small Scale

**Depth Scaling and Data Efficiency of Multi-Stream Residual Connections in Compact CNNs and Vision Transformers**

This is a controlled study of whether **hyper-connections (HC)** and **manifold-constrained hyper-connections (mHC)** improve compact vision models, compared with the ordinary residual shortcut. The finished **compact study** (decision D19, sized for a B.Tech final-year project and for free GPUs) trains ResNet-20, ResNet-56 and a 6-block ViT-Tiny with the normal shortcut, HC with 2 and 4 lanes and mHC with 4 lanes, on CIFAR-100 (all the data and 10% of it), for 30 epochs, one run per setting: 24 networks, about 6 hours on two free Kaggle T4 GPUs. The original, larger plan (more depths, 25% data, Tiny-ImageNet, three seeds) is not part of the compact study.

**Start here:** read `docs/PROJECT_REPORT.docx` (the same text is in `docs/PROJECT_REPORT.md`). A student who has to present and defend the project should then read `docs/STUDENT_GUIDE.md` (also `.docx`): a 10-minute demo script, likely viva questions with honest answers, the limits and a glossary.

**Try the live prediction page** (any computer, no GPU, no dataset download): clone the repository, create a Python environment, run `pip install torch torchvision pyyaml`, then `python app/server.py`. You upload a picture and get the top-5 answers of ResNet-20 with the normal shortcut, HC or mHC, a bar chart, a CSV download and a prediction history. The four trained networks it uses are in `results/demo/` (4.6 MB). Full steps: `README_DEMO.md`, section "Quick start".

**Results in one paragraph.** Every setting was trained once, so the outcomes are observations, not proven effects (no standard deviations or significance tests exist yet). With all the training data, HC with 2 or 4 lanes did better than the normal shortcut for both ResNets, mHC helped less, and every connection type did worse for the ViT; with 10% of the data there was no consistent gain; HC and mHC cost extra training time and memory. The numbers are in `docs/PROJECT_REPORT.md` (section 5) and in `results/tables/`.

| File | Purpose |
|---|---|
| `docs/PROJECT_REPORT.docx` / `.md` | The final report, generated from the data by `scripts/build_report.py` (text in `docs/templates/`) |
| `docs/STUDENT_GUIDE.docx` / `.md` | Demo script, viva questions, limits, glossary |
| `README_DEMO.md` | Step-by-step commands for the live demonstration on a student laptop, with a one-line explanation of each command |
| `app/` | The live image-prediction page for the demonstration (D20): `python app/server.py`. The class names and sample pictures are bundled here; the four networks are in `results/demo/` (`README_DEMO.md`, "Quick start" and step 8) |
| `docs/experiment_log.md` | What was run, when and on what; what went wrong; short answers to RQ1-RQ5; open items |
| `results/` | `raw/` (the 24 run folders of the study), `tables/` (T0-T4 as CSV and LaTeX, and the measured cost table), `figures/` (F1-F7), `demo/` (the four ResNet-20 networks of the live page, with their weights), `smoke/raw/` (the stored 2-epoch pipeline check of all 24 model x connection pairs, not study results) |
| `docs/references.md` | Verified references with links |
| `docs/method_notes.md` | HC / mHC equations as implemented, init, channel-first design, implementation details, and the design decisions D1-D20 (section 12) |
| `notebooks/nn09_colab.ipynb` | The notebook that runs everything heavy on a free GPU (see below) |
| `notebooks/nn09_demo_weights.ipynb` | The same notebook with the plan preset to `demo`: re-creates the four networks of the live page (`README_DEMO.md`, "Get the demo weights") |
| `notebooks/executed/` | The executed notebooks: the Kaggle study run, the Kaggle demo run and the earlier Colab smoke run |

## Status
The project is finished: the compact study, its analysis, the written report and student guide, and the live prediction page.
- [x] P0 Setup
- [x] P1 Method notes (approved 2026-10-07; the design decisions are in `docs/method_notes.md`, section 12)
- [x] P2 Connections
- [x] P3 Models + data (all 24 model x connection pairs passed the 2-epoch smoke run)
- [x] P4 Timing (cost measured on a Tesla T4: `results/tables/cost_Tesla_T4__pilot.csv`; compact plan D19 approved 2026-10-16)
- [x] P5 Study runs (compact plan: 24 runs, seed 0, 30 epochs, Kaggle 2 x T4, 2026-10-17; replaces the 144-run Tier 1; none diverged)
- [x] P6 Analysis (tables T0-T4, figures F1-F7, `docs/experiment_log.md`, report and student guide)
- [x] D20 Live prediction page (`app/`, four ResNet-20 networks trained on Kaggle on 2026-10-22, `results/demo/`)

Not part of this project (outside the compact study, D19): the original 144-run plan, Tier 2 (25% subset, Tiny-ImageNet, static-HC ablation), ResNet-110 and ViT-Tiny with 12 or 24 blocks (the code and configs exist and are tested, but no run was made), and an IEEE paper (the project report replaces it). Seeds 1 and 2, for error bars and significance tests, are the first thing to add (about 6 GPU-hours of Kaggle time per seed; see `docs/experiment_log.md`, section 5).

## Where things run
Training, timing and the test suite run on a **free cloud GPU**, through `notebooks/nn09_colab.ipynb`: **Kaggle** (recommended:
a weekly GPU quota, two T4 GPUs used in parallel, background runs of up to 12 h) or **Google Colab** (one GPU, no background
mode). Nothing heavy runs on the author's laptop (the analysis and the report are pandas / text only; the live page classifies one picture at a time on the CPU). The notebook is self-contained: it embeds a snapshot of this repo's code, so you only upload the `.ipynb`.
It checks the GPU, runs the tests and a smoke run, profiles your GPU (time, memory, inference speed, MACs) and prints the
GPU-hours of the plan you choose, launches the (resumable) runs, keeps everything on Google Drive (Colab) or in the notebook output (Kaggle), zips the
results, and builds the tables and figures. One worker runs per GPU; the runs pause cleanly
before the 12 h session limit (`MAX_HOURS`), and a later session continues them. On Kaggle: Session options > Accelerator =
GPU T4 x2, Internet = On (phone-verified account), then Save Version > Save & Run All (Commit); to continue, add the last
version's output as an input (Add Input > Notebook Output Files) and the notebook copies the progress back. Open the notebook in one browser tab only (a second tab makes Kaggle refuse to save).

To bring the results back into this repo:
1. save the executed notebook (File > Download > .ipynb) under `notebooks/executed/`;
2. unzip the results zip (Colab: from Drive; Kaggle: from the notebook's Output tab) into `results/raw/` (it contains one folder per run: `config.yaml`, `metrics.csv`,
   `final.json`, `grad_norms.csv`, `similarity.csv`, `connections.pt`, `train.log`);
3. put `timing/cost_<gpu>__<plan>.csv` (Drive / Output tab) into `results/tables/` (tables T3 and figure F5 are made from it).

The study run (plan `pilot`, Kaggle, 2 x Tesla T4, 24 of 24 runs complete, none diverged) is stored as
`notebooks/executed/nn09_kaggle_pilot_e30_T4x2.ipynb`, its run folders are in `results/raw/`. The earlier Colab smoke run
(Tesla T4, 148 tests passed, 24 of 24 tiny runs complete) is `notebooks/executed/nn09_colab_smoke_T4.ipynb`; its run logs are in `results/smoke/raw/` (they are not study
results). `python scripts/aggregate.py --raw results/smoke/raw --out results/smoke/tables` shows the tables on them (the output folder is git-ignored scratch).
The demo run of 2026-10-22 (four ResNet-20 networks with their weights saved, for the live page) is stored as `notebooks/executed/nn09_kaggle_demo_weights_e30_T4x2.ipynb` and `results/demo/`.
The three executed notebooks are kept exactly as they ran, with their saved output. The code each of them embeds is the version of that day, so it differs slightly from the final code in this repository (for example in comments and in the file layout); the numbers always come from the run folders.

The notebook is generated from the code in this repo: after any code change run `python scripts/make_colab_notebook.py`
(it needs only git and the standard library) and re-upload it; `--check` says whether the notebook is stale. Study numbers
always come from the run folders, never from notebook cells.

## Commands (the notebook runs these for you)
```bash
pytest                                        # equivalence at init for every model, parameter counts, subsets, trainer, ...
python scripts/train.py --config configs/experiments/smoke.yaml        # one tiny run; downloads CIFAR-100 (169 MB) on first use
python scripts/run_matrix.py --tier smoke     # every model x connection, 2 tiny epochs
python scripts/profile_cost.py                # cost table: params, MACs, step time, peak memory, train / inference speed
python scripts/budget.py results/tables/cost_<gpu>.csv --epochs 30 --seeds 3       # GPU-hours for a plan
python scripts/run_matrix.py --tier 1 --dry-run    # the 144 Tier 1 runs; drop --dry-run to launch (resumable)
python scripts/aggregate.py && python scripts/make_figures.py          # results/raw -> results/tables (T0-T4, csv + tex) and results/figures (F1-F7)
python scripts/build_report.py && python scripts/build_report_docx.py  # docs/PROJECT_REPORT.* and STUDENT_GUIDE.* from results/ (pandas, PyYAML, python-docx; no torch)
python scripts/build_report.py --check        # exit 1 if the written report / guide no longer match the templates and results/
python app/server.py                          # the live prediction page (needs torch, torchvision, pyyaml; the four networks are in results/demo/)
```
Install (outside Colab): `pip install -r requirements.txt` (a large download because of PyTorch). On a laptop without a GPU a lighter route is
`pip install torch torchvision numpy pandas scipy matplotlib pyyaml fvcore pytest` followed by `pip install -e . --no-deps`. The Word export also needs `pip install python-docx`.
On Colab keep its preinstalled torch (the notebook does).

Useful flags of `run_matrix.py`: `--models/--connections/--data/--seeds` select a sub-grid; `--set KEY=VALUE` overrides any
config value (`train.epochs=30`, `train.epoch_steps=390`, `paths.results_dir=...`, `train.checkpoint_every=5`).
A finished run (`final.json`) is skipped and an interrupted one continues from its last finished epoch, so after a disconnect
simply run the same command again. `--stop-after-hours H` pauses cleanly (progress saved, exit code 75) before a session
limit; `--claim-dir DIR` lets several launchers, one per GPU, share the work without starting a run twice. A results folder
that already holds a run with different settings is refused: use one results folder per plan. Seed 0 of every cell runs first, so a partial campaign still covers every cell.
The recipes' 200 epochs (D7/D8) belong to the full plan; the compact study used 30 epochs (D19).

**Base paper:** D. Zhu et al., "Hyper-Connections," ICLR 2025 — https://arxiv.org/abs/2409.19606
