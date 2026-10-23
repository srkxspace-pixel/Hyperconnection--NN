# NN-09 demo README - step-by-step commands for a student laptop

This file lists, in order, every command needed to prepare and give the live demonstration of the project, with a one-line explanation above each command. Nothing here trains the 24 networks of the study: that run (about 6 hours on two free Kaggle T4 GPUs) is finished and its logs are in `results/raw/`. On the laptop you only show those results, run a short check, run a 20-step tiny training, and rebuild the tables from the saved logs.

The project in full: `README.md` and `docs/PROJECT_REPORT.docx`. Likely viva questions: `docs/STUDENT_GUIDE.docx`.

## Quick start - only the live prediction page (for anyone who cloned the repository)

The four trained networks and everything else the page needs are in the repository, so nothing has to be trained or downloaded except the Python libraries (about 200 MB). No GPU is needed. You need Python 3.10 or newer and a terminal; the commands below are for Windows PowerShell, with the macOS / Linux form in the notes.

**Q.1** Downloads the project (use the address of the repository you were given).

```powershell
git clone <repository-address>
```

**Q.2** Moves into the folder that was created (the one that contains `README.md`, `app` and `src`).

```powershell
cd <the-folder-that-was-created>
```

**Q.3** Creates a private Python environment inside the project folder.

```powershell
python -m venv .venv
```

**Q.4** Turns the environment on; the prompt then starts with `(.venv)`. On macOS / Linux use `source .venv/bin/activate`.

```powershell
.\.venv\Scripts\Activate.ps1
```

**Q.5** Installs the three libraries the page needs. On Linux add `--index-url https://download.pytorch.org/whl/cpu` to avoid a 2.5 GB download.

```powershell
pip install torch torchvision pyyaml
```

**Q.6** Starts the page and opens it in your browser (it listens on your computer only); Ctrl+C stops it. Expect `Models found in results/demo: 4`. What to click is in step 8 below.

```powershell
python app/server.py
```

## Read this first

- **Terminal:** PowerShell on Windows 10 or 11 (in VS Code: Terminal > New Terminal).
- **Folder:** run every command from the project folder, the one that contains `README.md`, `src` and `scripts` (on the author's laptop it is the *inner* `NN-09_HyperConnections_SmallScale` folder; the folder one level up is an empty wrapper).
- **`python`** means the project's own Python (the `.venv` folder). If activating it fails (step A.2), type `.\.venv\Scripts\python.exe` instead of `python` in every command below.
- **Internet** is needed only to install libraries (another laptop) and, for steps 4 and 5 only, if the folder `data\cifar-100-python` is missing, to download CIFAR-100 (169 MB) once. The live page (step 8) needs no dataset.
- **Rehearse once, the day before:** do the setup, then steps 3, 4 and 5. They load PyTorch, so how long they take depends on your laptop. The expected outputs below were worked out from the code and from the stored Kaggle and Colab outputs; trust what your own screen shows.
- **Before step 4:** plug the laptop in and close other heavy programs.
- **Never remove `--dry-run`** from the step 6 command and never change `--seeds` there: without `--dry-run`, a missing run would start real training on your laptop.
- If `code` is not recognised, open the file from VS Code's File menu. If `start` opens the wrong program, open the file from File Explorer.

## The order at a glance

| Part | What you do | Talking time |
|---|---|---|
| Setup A or B | Get the environment ready (once) | - |
| Step 1 | Show the result (charts and report) | 2 min |
| Step 2 | Show that the results are real (run folders, log, Kaggle notebook) | 1 min |
| Step 3 | Show the guarantee: hyper-connections start identical to the normal network | 2 min |
| Step 4 | Run a live 20-step tiny training | 3 min |
| Step 5 | Rebuild tables and figures from the saved logs | 2 min |
| Step 6 | Show how the study was planned and launched | 1 min |
| Step 7 | Show what happens inside the model, then the code | 3 min |
| Step 8 | Live image prediction page (the four networks are in `results\demo\`; no dataset needed) | 3 min |
| Step 9 | Close with the limits | 1 min |

Short version (about 10 minutes): steps 1, 3, 4 and 7. Add step 8 if you have 3 more minutes; it needs only the libraries of the setup.

---

## Setup A - the author's laptop (the environment already exists)

The `.venv` folder there is Python 3.11.9 with PyTorch 2.9.1 and the project installed, and CIFAR-100 is already in `data\`.

**A.1** Moves the terminal into the project folder (change the path to where the project is on your computer); every other command assumes you are here.

```powershell
cd "C:\path\to\NN-09_HyperConnections_SmallScale"
```

**A.2** Turns on the project's Python environment; the prompt then starts with `(.venv)`. If this prints any error, skip it and use `.\.venv\Scripts\python.exe` in place of `python` from now on.

```powershell
.\.venv\Scripts\Activate.ps1
```

**A.3** Checks that PyTorch and the project package load; it prints two version numbers and no error.

```powershell
python -c "import torch, hcsmall; print(torch.__version__, hcsmall.__version__)"
```

## Setup B - another computer (one time, with internet)

**B.1** Gets the project onto the computer: clone the repository (Q.1), or copy the project folder (USB stick or similar) and leave out `.venv`, which only works on the machine that made it. Steps 4 and 5 also need the dataset folder `data\cifar-100-python` (169 MB, not in the repository): copy `data\` along, or download it once with `python -c "from torchvision import datasets; datasets.CIFAR100('data', download=True)"` after B.6 (a slow download). This step has no other command.

**B.2** Checks that Python is version 3.10 or newer.

```powershell
python --version
```

**B.3** Moves the terminal into the copied folder (change the path to where you put it).

```powershell
cd "C:\path\to\NN-09_HyperConnections_SmallScale"
```

**B.4** Creates a new private Python environment inside the project folder.

```powershell
python -m venv .venv
```

**B.5** Turns the new environment on.

```powershell
.\.venv\Scripts\Activate.ps1
```

**B.6** Installs PyTorch and the libraries the project uses (a large download).

```powershell
pip install torch torchvision numpy pandas scipy matplotlib pyyaml pytest
```

**B.7** Installs the project package `hcsmall` from this folder so the scripts and tests can import it.

```powershell
pip install -e . --no-deps
```

**B.8** Repeat check A.3; it must print two version numbers without an error.

Alternative to B.6 and B.7: `pip install -r requirements.txt` installs the exact pinned versions, but it pulls a very large CUDA build of PyTorch (`fvcore` is listed there and no code uses it).

---

## Step 1 - Show the result (about 2 min, no computing)

*Say:* Same layers and same recipe; only the shortcut differs. Hyper-connections with two lanes helped both ResNets when all the data was used, the ViT got worse, and with 10% of the data there was no consistent gain. One run per setting, so these are observations, not proven effects.

**1.1** Opens the chart of accuracy against depth for each connection type.

```powershell
start results\figures\F1_accuracy_vs_depth.png
```

**1.2** Opens the chart of the gain over the normal shortcut, with all the data and with 10% of it.

```powershell
start results\figures\F2_gain_vs_data_fraction.png
```

**1.3** Opens the final report in Word; scroll to section 5.1, the accuracy table.

```powershell
start docs\PROJECT_REPORT.docx
```

## Step 2 - Show that the results are real (about 1 min, no computing)

*Say:* These 24 folders are the raw evidence; every number in the report is computed from them.

**2.1** Lists the 24 run folders, one per trained network.

```powershell
ls results/raw
```

**2.2** Lists the 7 files saved for one run: settings, per-epoch metrics, final summary, gradient log, similarity log, learned lane weights and training log.

```powershell
ls results/raw/resnet56_dhc_n2_cifar100p100_s0
```

**2.3** Prints that run's summary: final accuracy, parameters, steps, GPU used and code version.

```powershell
cat results/raw/resnet56_dhc_n2_cifar100p100_s0/final.json
```

**2.4** Prints the last four lines of that run's training log (epochs 27 to 30).

```powershell
Get-Content results/raw/resnet56_dhc_n2_cifar100p100_s0/train.log -Tail 4
```

**2.5** Opens the notebook exactly as it ran on Kaggle, with all its outputs.

```powershell
code notebooks/executed/nn09_kaggle_pilot_e30_T4x2.ipynb
```

In that notebook point at: cell 3 (two Tesla T4 GPUs), cell 5 (the tests passed), cell 8 (progress from 0 to 24 of 24 runs and the finish time), cell 9 (the table of all 24 runs).

If the notebook opens as plain text, VS Code's Jupyter extension is missing. Either install that extension, or use the next two optional commands.

**2.6** (Optional) Installs the Jupyter notebook viewer.

```powershell
pip install notebook
```

**2.7** (Optional) Opens the executed notebook in your browser.

```powershell
jupyter notebook notebooks/executed/nn09_kaggle_pilot_e30_T4x2.ipynb
```

## Step 3 - Show the guarantee (about 2 min)

*Say:* At step 0 the hyper-connection network gives the same output as the normal network, so any difference after training was learned, not built in.

**3.1** Runs the quick checks that HC and mHC layers equal the normal shortcut at the start, and that the connection weights really change the output. Expect a row of dots and `30 passed` (24 plus 6 checks).

```powershell
python -m pytest tests/test_equivalence_init.py -q -k "test_equals_residual_at_init or test_equivalence_is_not_vacuous"
```

**3.2** (Optional, slower) Runs the same check on every complete network, ResNet-20/56/110 and ViT-6/12/24; time it before the demo.

```powershell
python -m pytest tests/test_equivalence_init.py -q
```

**3.3** (Optional, slowest) Runs the whole test suite; it passed completely on Kaggle (cell 5 of the executed notebook); time it before the demo.

```powershell
python -m pytest -q
```

**3.4** Opens the test file so you can show the "equal at the start" test (the function at line 38).

```powershell
code tests/test_equivalence_init.py
```

## Step 4 - Run a live tiny training (about 3 min)

*Say:* This is exactly what ran 24 times on Kaggle, only 20 steps instead of the full 30 epochs.

**4.1** Trains ResNet-20 with 2-lane hyper-connections for 2 mini-epochs of 10 steps on the 5,000-image subset and saves a run folder. Expect two lines (`epoch 1/2` and `epoch 2/2`) with test accuracy close to chance, about 1 to 3%, because it is only 20 steps; the point is that the whole pipeline works.

```powershell
python scripts/train.py --config configs/experiments/smoke.yaml --seed 1
```

**4.2** Lists the 6 files the tiny run wrote: settings, metrics, final summary, gradient log, similarity log and learned lane weights (there is no `train.log` here, that file only comes from the launcher).

```powershell
ls results/smoke/resnet20_dhc_n2_cifar100p10_s1
```

**4.3** Prints the tiny run's summary (2 epochs, 20 steps, 5,000 training images).

```powershell
cat results/smoke/resnet20_dhc_n2_cifar100p10_s1/final.json
```

**4.4** Prints the loss, accuracy and learning rate of each of the 2 epochs.

```powershell
cat results/smoke/resnet20_dhc_n2_cifar100p10_s1/metrics.csv
```

Notes:
- The seed picks the folder name (`..._s1`). Running the same seed twice prints `[skip] ... final.json exists` and trains nothing, so use `--seed 2`, `--seed 3` and so on for another live run.
- For the rehearsal run 4.1 without `--seed` (seed 0), so the demo run with `--seed 1` is still fresh.
- If `data\cifar-100-python` is missing, the first run downloads CIFAR-100 (169 MB, needs internet). Do this at home.
- With no GPU it runs on the CPU and takes longer; time it during the rehearsal.

## Step 5 - Rebuild everything from the saved logs (about 2 min)

*Say:* Nobody typed these numbers; the tables and figures are computed from the 24 run folders. Outputs go into `results\smoke\`, which git ignores, so your saved tables and figures are not touched.

**5.1** Rebuilds the tables T0-T4 from the 24 run folders into a scratch folder, using the saved cost table. Expect `24 runs from ...`, a grid of 1s (one run per setting) and `wrote 15 files`.

```powershell
python scripts/aggregate.py --out results/smoke/demo_tables --cost results/tables/cost_Tesla_T4__pilot.csv
```

**5.2** Rebuilds the figures F1-F7 from the 24 run folders into a scratch folder (a minute or two; PyTorch is needed only to read the saved lane weights).

```powershell
python scripts/make_figures.py --out results/smoke/demo_figures
```

**5.3** Checks that the written report and student guide still match their templates and the results. Expect `reports are up to date`.

```powershell
python scripts/build_report.py --check
```

**5.4** Lists the table files that were just rebuilt.

```powershell
ls results/smoke/demo_tables
```

**5.5** Prints the accuracy table for the all-data runs, the table the report is built from (the standard-deviation columns are empty because there is one run per setting).

```powershell
cat results/smoke/demo_tables/T1_cifar100_p100.csv
```

**5.6** Opens a rebuilt chart; it should look the same as the saved one.

```powershell
start results\smoke\demo_figures\F1_accuracy_vs_depth.png
```

## Step 6 - Show how the study was planned and launched (about 1 min)

*Say:* Before launching, the speed of every model was measured on a Kaggle T4 and the cost of the plan was predicted; the launcher skips finished runs, so after a disconnect you just run it again.

**6.1** Prices the 24-run plan in GPU-hours from the measured speed of each model; expect the plan line `3 models x 4 connections x 2 data settings x 1 seeds = 24 runs, 30 epochs` followed by its GPU-hours and the session time on two GPUs.

```powershell
python scripts/budget.py results/tables/cost_Tesla_T4__pilot.csv --epochs 30 --seeds 1 --models resnet20 resnet56 vit_tiny_d6 --connections residual dhc_n2 dhc_n4 mhc_n4 --data p100 p10 --steps same_steps --gpus 2 --weekly-hours 30
```

**6.2** Lists the planned runs and how many are already finished, without starting any training; expect `24 runs, 24 already complete`.

```powershell
python scripts/run_matrix.py --tier 1 --models resnet20 resnet56 vit_tiny_d6 --seeds 0 --set train.epochs=30 --set train.epoch_steps=390 --dry-run
```

## Step 7 - Show what happens inside, then the code (about 3 min)

*Say:* In the ResNets, hyper-connections made consecutive layers change the representation more; mHC stayed close to the normal shortcut; in the learned lane-mixing maps the diagonal dominates, so each lane mostly keeps its own content.

**7.1** Opens the chart of how much each layer changes its input (lower similarity means the layer does more).

```powershell
start results\figures\F3_representation_collapse.png
```

**7.2** Opens the chart of the gradient size in each layer.

```powershell
start results\figures\F4_gradient_norms.png
```

**7.3** Opens the heat-maps of the learned lane weights.

```powershell
start results\figures\F6_connection_patterns.png
```

**7.4** Opens the chart of accuracy against the memory cost.

```powershell
start results\figures\F5_accuracy_vs_cost.png
```

**7.5** Opens the training curves (test accuracy per epoch).

```powershell
start results\figures\F7_training_curves.png
```

**7.6** Opens the layer interface: `expand`, `forward_block` (read, layer, write; line 46) and `reduce`.

```powershell
code src/hcsmall/connections/base.py
```

**7.7** Opens hyper-connections for tokens, with the dynamic weights (the `weights` function, line 40).

```powershell
code src/hcsmall/connections/hyper_connection.py
```

**7.8** Opens the image version, where each lane's feature map is averaged to compute the weights (line 17).

```powershell
code src/hcsmall/connections/hyper_connection_conv.py
```

**7.9** Opens mHC, with the Sinkhorn step that makes the mixing matrix doubly stochastic (line 13).

```powershell
code src/hcsmall/connections/mhc.py
```

**7.10** Opens the ResNet and shows how a connection is plugged into each block (the `forward` function, line 54).

```powershell
code src/hcsmall/models/resnet_cifar.py
```

**7.11** Opens the training loop for one run, from the config to the run folder (the `train` function, line 78).

```powershell
code src/hcsmall/training/trainer.py
```

## Step 8 - Live image prediction (about 3 min)

*Say:* Same network code as the study, trained once more with its weights saved. I give it a picture and it answers with its five best guesses; the four models were trained with exactly the same recipe, only the shortcut differs. It knows only the 100 CIFAR-100 classes (no dog or cat: those are CIFAR-10 names) and it is right on about two out of three test images, so a real photo is often wrong. That is expected, not a bug.

The four networks are part of the repository (`results\demo\`, 4.6 MB), and the class names and sample pictures are bundled in `app\`, so there is nothing to download or train here. ("Get the demo weights" below is only for re-creating the networks.)

**8.1** Starts the page's local server and opens the page in your browser; it listens on this computer only. Expect `Models found in results/demo: 4`. Leave the terminal open (Ctrl+C stops the server).

```powershell
python app/server.py
```

**8.2** In the page, panel 1: keep Connection Type "Standard (1 stream)" and press **Load Model**. The green box says `Model loaded successfully!` and shows the model's accuracy from its training log.

**8.3** In the page, panel 2: drag in a picture or press **Browse Files**. Use a close-up of something CIFAR-100 knows (panel 6, Class Information, lists all 100 classes: fox, wolf, bus, rose, apple, bear, ...). Panels 3 and 4 show the top-5 answers as a list with confidence bars and as a bar chart; the small square labelled "model input" is the 32 x 32 picture the network really sees.

**8.4** In the page, panel 1: choose another Connection Type (Hyper-Connection 2 streams, 4 streams, mHC) and press **Load Model**. The same picture is predicted again by the new model, and every prediction is added to panel 5, Prediction History (model, connection, checkpoint, top answer, confidence, time).

**8.5** In the page: press **Download Results (CSV)** (the top-5 of the current picture) and **Download history (CSV)** (all predictions); the files go to your Downloads folder.

**8.6** In the terminal: press Ctrl+C to stop the server.

Notes:
- The history lives in the browser, not on disk; **Clear** empties it. The page writes nothing into the project folder.
- If the port is busy, start with `python app/server.py --port 8001` and open that address.
- Rehearse once with the real weights: the first **Load Model** imports PyTorch and takes a few seconds, and you will see how your own photos behave.

## Step 9 - Close (about 1 min)

*Say:* Limits: one run per setting, 30 epochs, three models, one dataset. Next step: more seeds, about 6 Kaggle GPU-hours each.

**9.1** Opens the student guide with the likely viva questions and honest answers.

```powershell
start docs\STUDENT_GUIDE.docx
```

---

## If a step fails (Plan B) - do not debug in front of the audience

**P.1** If the tests (step 3) fail or are too slow: opens the executed Kaggle notebook; show cell 5.

```powershell
code notebooks/executed/nn09_kaggle_pilot_e30_T4x2.ipynb
```

**P.2** If the tiny run (step 4) fails: lists the stored tiny run from the earlier Colab test; it has the same files plus a `train.log`.

```powershell
ls results/smoke/raw/resnet20_dhc_n2_cifar100p10_s0
```

**P.3** Prints the summary of that stored tiny run.

```powershell
cat results/smoke/raw/resnet20_dhc_n2_cifar100p10_s0/final.json
```

**P.4** If the rebuild (step 5) fails: lists the saved tables, which are the same ones.

```powershell
ls results/tables
```

**P.5** Lists the saved figures F1-F7 (open them with `start`, as in steps 1 and 7).

```powershell
ls results/figures
```

**P.6** If the prediction page (step 8) will not start or the weights are missing: skip step 8. The accuracy of each demo model is in its summary file.

```powershell
cat results/demo/resnet20_dhc_n2_cifar100p100_s0/final.json
```

## Common problems

| You see | Do this |
|---|---|
| An error when activating `.venv` | Skip activation and type `.\.venv\Scripts\python.exe` instead of `python`. |
| `No module named 'torch'` or `'hcsmall'` | You are not using the project's Python: repeat A.2 (or B.5), or use the `.venv` path above. |
| `[skip] ... final.json exists` in step 4 | That seed was already run; use a new `--seed`. |
| `code` is not recognised | Open the file from VS Code's File menu or from File Explorer. |
| The first tiny run starts a download | `data\cifar-100-python` is missing; connect to the internet once, or copy `data\` from this laptop. |
| The page says "No trained models found" | `results\demo\` is empty: you started the server from another folder, or the clone is incomplete (`git pull`). Otherwise re-create the networks: see "Get the demo weights". |
| `Cannot start: ... Address already in use` | Another program uses port 8000: `python app/server.py --port 8001`. |
| The page opens but shows another program | You opened a different address; use exactly the one the server prints. |

---

## Not for the laptop - the full study runs on Kaggle

These commands train real networks. The Kaggle notebook runs them for you; do not run them on the laptop.

| Command | What it does (one line) |
|---|---|
| `python scripts/profile_cost.py --models ... --connections ...` | Measures parameters, MACs, peak memory and speed of every model and connection on the GPU you run it on. |
| `python scripts/time_epoch.py --models ... --connections ...` | Older, simpler timing of a training step; the notebook's fallback for the line above. |
| `python scripts/run_matrix.py --tier smoke` | Trains every model and connection (24 combinations, including ResNet-110 and ViT-24) for 2 tiny epochs. |
| `python scripts/run_matrix.py --tier 1 --models ... --connections ... --data ... --seeds ... --set KEY=VALUE --claim-dir DIR --stop-after-hours H` | Launches the real runs; finished runs are skipped, interrupted ones resume, and one launcher per GPU shares the list through `--claim-dir`. |
| `python scripts/train.py --config configs/experiments/tier1.yaml --model resnet56 --connection dhc_n4 --data cifar100_p10 --seed 1 --set train.epochs=30` | Trains one full run by hand. |

To redo the full study on Kaggle (about 6 hours, plus about 1 hour to download CIFAR-100 the first time):

1. Sign in to Kaggle with a phone-verified account (the GPU and internet access need it).
2. Create a new notebook and import `notebooks/nn09_colab.ipynb` into it.
3. In the right-hand Session options set Accelerator = GPU T4 x2 and Internet = On.
4. Leave the settings cell as it is (`PLAN = pilot`, `EPOCHS = 30`, `SEEDS = 0`).
5. Keep this the only open browser tab of the notebook (two tabs make Kaggle refuse to save).
6. Press Save Version > Save & Run All (Commit); Kaggle switches the launch on by itself.
7. Wait; you can close the browser and watch progress from the version's log page.
8. When it finishes, download from the version's Output tab the results zip (`pilot_e30_raw.zip`), the cost table (`timing/cost_Tesla_T4__pilot.csv`) and the executed notebook (File > Download).
9. In the project: unzip the zip into `results/raw/`, put the cost table into `results/tables/` and the notebook into `notebooks/executed/`.
10. Rebuild the tables, figures and report with the Maintenance commands below.

For more seeds: in the settings cell set `SEEDS = "0,1,2"` and add the previous version's output as an input (Add Input > Notebook Output Files) so finished runs are skipped.

### Get the demo weights (once; about 1.5 hours on Kaggle, you do the clicks)

The four networks of the live page are already in the repository (`results/demo/`); do this only to re-create them, for example with another seed or with ResNet-56 as well. The 24 networks of the study saved no weights, so the page needed a short re-run: ResNet-20 with the four connection types, the same recipe, seed 0, all the data, 30 epochs, this time with the weights saved. It is separate from the study: the results go to `results\demo\`, never to `results\raw\`, so no number in the report changes. It took 83 minutes on 2026-10-22: about 40 minutes for Kaggle to download CIFAR-100 from a slow server, about 33 minutes of training on two T4 GPUs and about 6 minutes of checks. You can close the browser while it runs.

1. Sign in to Kaggle with the phone-verified account and create a new notebook; import **`notebooks/nn09_demo_weights.ipynb`** (File > Import notebook). This copy is already set to the demo plan: change nothing in its settings. (`notebooks/nn09_colab.ipynb` is the general notebook and starts the 24-run study plan by default: do not use it for this.)
2. In the editor's Settings menu: Accelerator > **GPU T4 x2** (a new notebook starts with None, and the notebook then stops at its first check) and check that Internet is on (the menu offers "Turn off internet" when it is on).
3. Add no input. (The earlier 24-network run kept CIFAR-100 outside its saved output, so adding it would not save the download. This run downloads CIFAR-100 itself, about 40 minutes, and its output has an `nn09/data` folder, so a later run can reuse it through Add Input > Notebook Output Files.)
4. Keep this the only open tab of the notebook and press Save Version > Save & Run All (Commit). You can close the browser.
5. When it has finished, open the version's log: the weights check (cell 11) should print `OK` once per model. Download `demo_e30_raw.zip` (Output tab, folder `nn09`; a few MB).

Optional: to get ResNet-56 as well, change `DEMO_MODELS = "resnet20"` to `"resnet20 resnet56"` in the settings cell before step 4 (adds about 2.5 GPU-hours).

**W.1** Unpacks the downloaded zip into `results\demo\` (one folder per model, each with its `model.pt`); change the path if the file is elsewhere.

```powershell
Expand-Archive -Path "$HOME\Downloads\demo_e30_raw.zip" -DestinationPath results\demo
```

**W.2** Lists the four model folders; each holds `model.pt`, `final.json` (accuracy) and the training log.

```powershell
ls results\demo
```

## Maintenance commands (not part of the demo)

| Command | What it does (one line) |
|---|---|
| `python scripts/aggregate.py` | Rebuilds the saved tables in `results/tables/` from `results/raw/` (same files again). |
| `python scripts/make_figures.py` | Rebuilds the saved figures in `results/figures/` from `results/raw/`. |
| `python scripts/build_report.py` | Rewrites `docs/PROJECT_REPORT.md` and `docs/STUDENT_GUIDE.md` from the templates in `docs/templates/` and the results. |
| `pip install python-docx` | Installs the library that makes the Word files (needed only for the next line). |
| `python scripts/build_report_docx.py` | Converts the report and the student guide to `.docx` files. |
| `python scripts/prepare_data.py` | Downloads CIFAR-100, measures its mean and standard deviation, and writes the saved subset and probe index files (already done and committed; running it again rewrites the probe file). |
| `python scripts/make_colab_notebook.py` | Regenerates `notebooks/nn09_colab.ipynb` and `notebooks/nn09_demo_weights.ipynb` from the project code (needs git). |
| `python scripts/make_colab_notebook.py --check` | Says whether the two notebooks are out of date with the code (exit code 1 if one is). |
| `python scripts/make_app_assets.py` | Rewrites `app/classes.json` and `app/classes.png` (the class names and sample pictures of the live page) from the CIFAR-100 files in `data\` (already done and committed). |

## After the demo

The demo wrote three folders: `results\smoke\demo_tables`, `results\smoke\demo_figures` and `results\smoke\resnet20_dhc_n2_cifar100p10_s1` (plus `s0` if you rehearsed). Git ignores all of them, so leaving them is harmless; delete them in File Explorer if you like, but keep `results\smoke\raw` (the stored Colab tiny runs, your Plan B).
