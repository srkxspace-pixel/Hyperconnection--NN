# Experiment log

What was run, on what, and what went wrong. This file records provenance and short answers only; the numbers are in `results/` and in `docs/PROJECT_REPORT.md` (section 5), so nothing is typed in twice.

## 1. The study run

| Item | Value |
|---|---|
| Date | 2026-10-17 |
| Where | Kaggle notebook, *Save & Run All (Commit)*, version 1 of `notebooks/nn09_colab.ipynb`, plan `pilot` |
| Hardware | 2 x Tesla T4 (14.6 GiB each), one worker per GPU |
| Software | Python 3.13, PyTorch 2.11.0+cu128 (CUDA 12.8) |
| Code | commit `481ea54`; the notebook embeds a snapshot of the repository and the label is stored in every `final.json` |
| Plan | ResNet-20, ResNet-56, ViT-Tiny (6 blocks) x residual, DHC x2, DHC x4, mHC x4 x CIFAR-100 (all data, 10%) = 24 runs; seed 0; 30 epochs of 390 steps; batch 128; fp32; the 10% runs get the same number of steps (D6) |
| Outcome | 24 of 24 runs finished, none diverged, 10.8 GPU-hours of training; about 6 hours of wall-clock on two GPUs, plus about 1 hour to download CIFAR-100 |
| Checks on the same machine, before training | 148 core tests passed (5 deselected), 29 newer tests passed, a 2-epoch run worked, the cost table was measured |
| Stored | `notebooks/executed/nn09_kaggle_pilot_e30_T4x2.ipynb` (the executed notebook), `results/raw/` (the 24 run folders), `results/tables/cost_Tesla_T4__pilot.csv` (measured cost) |

Everything after the run is rebuilt from the run folders:

```bash
python scripts/aggregate.py && python scripts/make_figures.py       # results/raw -> results/tables, results/figures
python scripts/build_report.py && python scripts/build_report_docx.py   # docs/PROJECT_REPORT.*, docs/STUDENT_GUIDE.*
```

The tables were built on Kaggle and again on the laptop from the same run folders and came out identical. Figure F6 needs PyTorch to read `connections.pt`; F7 was regenerated after the figure code was changed to show one ResNet and one ViT.

## 1b. The demo run (D20)

The four networks behind the live prediction page. A re-run for the demonstration, not part of the study: the report uses only `results/raw/`.

| Item | Value |
|---|---|
| Date | 2026-10-22 |
| Where | Kaggle notebook, *Save & Run All (Commit)*, `notebooks/nn09_demo_weights.ipynb` (plan `demo`) |
| Hardware | 2 x Tesla T4 (14.6 GiB each), one worker per GPU |
| Software | Python 3.13, PyTorch 2.11.0+cu128 (CUDA 12.8) |
| Code | `src/` and `configs/` exactly as in commit `bb33f74` (checked file by file against the snapshot embedded in the executed notebook); the run used them before they were committed, on top of `becffc7`, so the label ends in `-dirty` in every `final.json` |
| Plan | ResNet-20 x residual, DHC x2, DHC x4, mHC x4 x CIFAR-100 (all data); seed 0; 30 epochs of 390 steps; batch 128; fp32; `train.save_weights=true` |
| Outcome | 4 of 4 runs finished, none diverged, 0.9 GPU-hours of training; 83 minutes in total, of which about 40 minutes were Kaggle downloading CIFAR-100 |
| Checks on the same machine | 152 core tests and 29 newer tests passed (one skipped: the page's server test needs `app/`); a 2-epoch run saved its weights, reloaded them and classified an image; after training every `model.pt` was reloaded and classified the 10,000 test images one by one through `hcsmall.predict`, and all four gave exactly the test accuracy logged at the end of training |
| Stored | `notebooks/executed/nn09_kaggle_demo_weights_e30_T4x2.ipynb` (the executed notebook); the weights zip `demo_e30_raw.zip` (4.5 MB, from the Output tab) is unpacked into `results/demo/`. Git tracks only what the page reads (`model.pt`, `config.yaml`, `final.json`, `metrics.csv` of each run, 4.6 MB), so a clone can run the page; the other logs of the runs stay out of the repository |
| Note | Same recipe and seed as the study, but a new run: GPU non-determinism makes the accuracies differ a little from the same settings in `results/raw/`. The page shows each demo network's own logged accuracy. |

## 2. Timeline

| Date | What |
|---|---|
| 2026-10-06 / 07 | P0-P3 built and tested (connections, models, data, trainer); design decisions D1-D17 approved on 2026-10-07 |
| 2026-10-11 | First Colab smoke run on a free T4: 148 tests passed and all 24 model x connection pairs trained for two tiny epochs, none diverged (`notebooks/executed/nn09_colab_smoke_T4.ipynb`); analysis tools, a paper skeleton (removed again on 2026-10-24) and a prior-art search |
| 2026-10-16 | Compact study size (D19) approved; Kaggle, two-GPU and clean-pause support added to the notebook |
| 2026-10-17 | The study run above; D11 changed to one seed; results, report and student guide |
| 2026-10-20 / 22 | Live prediction page built (D20); the demo weights trained on Kaggle (section 1b) |
| 2026-10-23 / 24 | Made ready for the hand-over to students: the page needs no dataset and the demo networks are in the repository; placeholders and the unfinished paper skeleton removed; report and student guide describe the page |

## 3. What went wrong, and what was done

1. **The free Colab quota is too small for the original plan.** The full plan (144 runs at 200 epochs) is hundreds of GPU-hours. The study was cut to the compact plan (D19) and moved to Kaggle (a weekly quota of about 30 GPU-hours, two T4 GPUs).
2. **Kaggle needs a phone-verified account for the GPU and for internet access.** The first account's number was already tied to another account, so the run used the older account that was already verified.
3. **The notebook mistook Kaggle for Colab.** Kaggle's image contains a `google.colab` package, so a version failed when the notebook tried to mount Google Drive. The check now looks for Kaggle first (commits `746a895`, `a736fc3`).
4. **Two browser tabs on one notebook.** Kaggle then refuses to save ("ConcurrencyViolation"). Keep one editor tab and watch the run from the log page.
5. **A first "Save & Run All" created no saved version** (only an interactive session that uses quota). The commit was started from *Save Version*.
6. **Downloading CIFAR-100 on Kaggle took about an hour** (roughly 50 kB/s). The data is now kept in the saved output, so a continuation session skips the download (commit `ede321e`).
7. **The newer checks could have stopped a long run.** They only warn now (commit `481ea54`); the core tests still block.
8. **The study run saved no network weights** (the trainer deletes its resume file when a run ends; only the small `connections.pt` stays), so the live page had nothing to predict with. The demo run (section 1b) re-trains four ResNet-20 networks with `train.save_weights=true`.
9. **A freshly imported Kaggle notebook has Accelerator = None**, so the notebook would stop at its first check. Set it in the editor's *Settings* menu (Accelerator, GPU T4 x2) before *Save & Run All*.
10. **Adding the study run's output as an input does not save the CIFAR-100 download.** That run (code `481ea54`) kept the data in a temporary folder, not in its saved output, so the demo run downloaded it again (about 40 minutes). The demo run's own output has an `nn09/data` folder, which a later run can reuse.

## 4. Short answers to RQ1-RQ5

One run per setting, so these are observations, not established effects. The numbers and figures are in `docs/PROJECT_REPORT.md`, section 5.

- **RQ1, depth.** With all the data, HC with 2 or 4 lanes gained about the same on ResNet-20 and ResNet-56; the gain from mHC shrank. With only two ResNet depths and one ViT depth, no trend can be claimed.
- **RQ2, data.** The gain was not larger with 10% of the data: the ResNet gains seen with all the data disappeared there.
- **RQ3, mechanism.** In the ResNets, HC with 4 lanes lowered the similarity of consecutive layers and flattened the gradient profile of ResNet-56; mHC behaved like the normal shortcut; the ViT hardly changed. The link to accuracy is partial.
- **RQ4, cost.** 4 lanes cost more than 2 lanes without more gain; mHC cost the most and gained the least; no connection type paid off for the ViT.
- **RQ5, family.** The ResNets gained from HC on all the data and the ViT lost. The reasons were not tested; the short training and the ViT warm-up are candidates.

## 5. Not done (optional next steps)

- **More seeds** (the first thing to do): about 6 GPU-hours of Kaggle time per extra seed. Re-import `notebooks/nn09_colab.ipynb`, set `SEEDS = "0,1,2"`, add the previous output as an input so that the finished runs are skipped, and run again. Then `aggregate.py` reports standard deviations and significance (D10) and the report states them.
- **Not run:** ResNet-110, ViT-Tiny with 12 and 24 blocks, Tier 2 (25% subset, Tiny-ImageNet, the static-HC ablation). They are outside the compact study (D19).
- **IEEE paper:** not part of this project (D19); the project report replaces it. The unfinished skeleton of 2026-10-11 was removed on 2026-10-24 and is in the git history (`git log --stat -- paper`).
