# Method notes — HC and mHC as implemented in this project

**Status:** implemented in `src/hcsmall/connections/` and tested in `tests/`; the design decisions D1-D20 are listed in section 12.
**Authority:** the two papers, not the lucidrains code.

| Tag | Source | Version read |
|---|---|---|
| **[HC]** | D. Zhu et al., "Hyper-Connections," ICLR 2025, arXiv:2409.19606 | v3 (2025-03-18), arXiv HTML |
| **[mHC]** | Z. Xie et al., "mHC: Manifold-Constrained Hyper-Connections," arXiv:2512.24880 | v2 (2026-01-05), arXiv HTML |

> **Provenance caveat.** The two papers were read on 2026-10-06 through an HTML-to-text fetch tool, not as PDFs. Equation numbers, shapes and settings below are as that tool reported them. The HC equations agree with the paper's own PyTorch listing (Alg. 2, App. J). Items the implementation depends on are marked ⚠: those are the ones to check against the PDFs first.
> Statements marked **[verified]** were checked numerically on a toy model (scratch script, not project code) on 2026-10-06.

---

## 1. Notation

- `n` = expansion rate (number of streams). `k` = layer index (0-based). `T_k` = the layer function ("branch"), *including its own pre-normalisation*.
- Streams: token layout `H ∈ R^{B×T×n×d}`; conv layout `H ∈ R^{B×n×C×h×w}`. `h_in` is the single tensor the branch reads.
- **Residual baseline (n = 1):** `x_{k+1} = x_k + T_k(x_k)`.
  - ViT, Pre-Norm: `T = Attn∘LN` or `MLP∘LN`. Each attention and each MLP sublayer is one layer, so `k = 2·block + {0,1}` ([HC] Alg. 3 uses a separate HC module per sublayer).
  - Pre-act ResNet basic block: `T = conv∘ReLU∘BN∘conv∘ReLU∘BN`. One layer per block.

## 2. Static HC — [HC] §2.1

- Eq. (1) `HC = [[0, B],[A_m, A_r]] ∈ R^{(n+1)×(n+1)}`, with `B ∈ R^{1×n}` (depth connection), `A_m ∈ R^{n×1}` (width connection, layer input), `A_r ∈ R^{n×n}` (stream mixing).
- Eq. (2) `Ĥ = B^T · T(H^T A_m)^T + A_r^T H`. Per stream:
  ```
  h_in   = Σ_i A_m[i] · H_i                         (read)
  H'_j   = B_j · T(h_in) + Σ_i A_r[i,j] · H_i       (write + mix)
  ```
- Eq. (6)/(7): `WC = (A_m A_r) ∈ R^{n×(n+1)}`, `DC = (B; diag(A_r)) ∈ R^{2×n}`. Residual connections are the non-trainable n = 1 special case ([HC] §2.1).
- Parameters: `n(n+2)` per layer ([HC] App. B).
- **Expand:** copy `h⁰` n times, `H⁰ = (h⁰ … h⁰)^T`. **Reduce:** sum the n rows, then the final norm ([HC] Alg. 1, lines 14–16). See §5 on why we use a mean (D14).

## 3. Dynamic HC (DHC) — [HC] §2.2

```
(8)   HC(H) = [[0, B(H)],[A_m(H), A_r(H)]]            (9)  Ĥ = HC(H)(T, H)
(10)  H̄ = norm(H)
(11)  B(H)   = s_β ∘ tanh(H̄ W_β)^T + B          ∈ R^{1×n}
(12)  A_m(H) = s_α ∘ tanh(H̄ W_m)   + A_m        ∈ R^{n×1}
(13)  A_r(H) = s_α ∘ tanh(H̄ W_r)   + A_r        ∈ R^{n×n}
```
- `H̄ ∈ R^{n×d}`: the norm acts on each stream's d-vector. `W_m, W_β ∈ R^{d×1}`, `W_r ∈ R^{d×n}`. Row i of `[A_m | A_r]` comes from stream i.
- Paper code (Alg. 2): `dynamic_alpha_fn = [W_m | W_r]` (d×(n+1)); `dynamic_beta_fn = W_β`; `s_α`, `s_β` are learnable scalars initialised to 0.01; the norm is `LayerNorm(dim)`.
- Dynamic extras: `d(n+2) + 2` per layer ([HC] App. B). That count has no norm affine parameters, so **we use a non-affine LayerNorm** ⚠.
- **Weight decay** ([HC] §4): "the static component … does not utilize weight decay, whereas the dynamic component does." Our D9 says "no WD on HC connection scalars". These agree on static parameters; D9 is silent on the dynamic matrices. → D17.

## 4. Initialisation — [HC] §2.3

Eq. (14): `B^k = 1_{1×n}`, `A_m^k = e_{k mod n}`, `A_r^k = I_n`. Dynamic `W_β = W_m = W_r = 0`, `s_α = s_β = 0.01`.

## 5. Equivalence at init

**Claim.** If the streams start as copies of `x`, then under Eq. (14) with zero dynamic weights: `h_in = H_{k mod n} = x`, and `H'_j = T(x) + H_j = x + T(x)` for every j. All streams stay equal and equal to the residual model's state. ✔ **[verified]**: max |Δoutput| = 0.0 in float64.

**Problem: the sum readout.** The reduced state is `Σ_j H_j = n·x`, not `x`. **[verified]**: gap 6.4 on the toy model. In a real network a norm follows the readout. For LayerNorm (ViT) and for BatchNorm in train mode the gap shrinks to the norm's ε, but it is not zero. For BatchNorm in eval mode (pre-act ResNet's final BN, which uses running statistics) the input is simply scaled by n and the outputs differ. A float64 test at `atol=1e-5` therefore cannot be exact with a sum.

**D14.** Read out the **mean** `(1/n)Σ_j H_j`. A scale-invariant norm follows immediately, so in LayerNorm and in train-mode BatchNorm `Norm(Σ) = Norm(mean)` except for ε: same function class, same gradients. With the mean, equality at init is exact in train and eval mode. This deviates from the literal "sum" in [HC] Alg. 1; it was approved on 2026-10-07.

**Symmetry breaking.** `A_m = e_{k mod n}` makes consecutive layers read different streams, so gradients differ across streams from the first step ([verified]: stream spread 1.8e-2 after one SGD step on the toy model). The mHC init must also break this symmetry (§8).

## 6. Channel-first HC for conv nets (this project's adaptation)

Streams `H ∈ R^{B×n×C×h×w}`. Every connection weight is a **per-sample scalar broadcast over (C, h, w)**:
```
read   h_in[b]   = Σ_i A_m[b,i] · H[b,i]                          einsum 'bi,bichw->bchw'
write  H'[b,j]   = Σ_i A_r[b,i,j] · S(H[b,i]) + B[b,j] · T(h_in)   einsum 'bij,bichw->bjchw'
```
`S` is the shortcut (identity, or a projection at stage transitions — §7). Static HC: the weights are parameters shared across the batch (the leading `b` is broadcast).

**Dynamic weights — D3(a) (chosen).** Pool each stream to a vector, then apply [HC] Eqs. (10)–(13) to the pooled vector:
`g = GAP_{h,w}(H) ∈ R^{B×n×C}`, `H̄ = LN_C(g)` (non-affine), `W_m, W_β ∈ R^{C×1}`, `W_r ∈ R^{C×n}`. This is the exact analogue of the per-token rule with "token feature" → "pooled channel vector", and it costs `C(n+2)+2` parameters per layer, the same formula as in the token case with d = C.

**D3(b) (not implemented).** A per-pixel 1×1 conv computes weights at every spatial position. See §10 item 1: this is what lucidrains' channel-first file does.

BatchNorm sits *inside* `T` (pre-act), so HC never normalises the streams themselves; the dynamic branch uses its own LN.

## 7. ResNet stage transitions — D4

A stage-transition block changes `C → 2C` and stride 2, so its shortcut is a learned **1×1 stride-2 conv `S`** (no BN, no bias; `C·2C` parameters). Elsewhere the shortcut is the identity.

- **D4(a), chosen:** apply the same `S` to every stream: `H'_j = B_j·T(h_in) + Σ_i A_r[i,j]·S(H_i)`. Keeps n streams end to end; no extra parameters per stream.
- D4(b), Tier-3 ablation: reduce → standard transition block → re-expand.

**Convention to fix (D15).** Common pre-act implementations apply `S` to the *pre-activated* input (after the block's first BN-ReLU); so, as far as is recalled, does He et al.'s released Torch code (⚠ not re-checked). With n streams that needs a BN-ReLU per stream, and equality at init would need those BNs to share statistics. Instead: **apply `S` to the raw block input in every arm** (baseline and HC alike). Then at init `S(x) + T(x)` is the same function in both, the comparison stays one-variable-at-a-time, and parameter counts are the standard ones (closed form in `test_param_counts.py`). The cost is a small, documented deviation from the reference pre-act unit.

## 8. mHC — [mHC] §3–4

```
(3)  x_{l+1} = H^res x_l + H^post^T · F(H^pre x_l, W_l)         x_l ∈ R^{n×C}
(6)  P_M(H^res) := { H ∈ R^{n×n} : H1 = 1, 1^T H = 1^T, H ≥ 0 }   (doubly stochastic)
(7)  x̄_l = vec(x_l) ∈ R^{1×nC};   x̄' = RMSNorm(x̄)
     H̃^pre  = α^pre  · (x̄' φ^pre)        + b^pre          ∈ R^{1×n}
     H̃^post = α^post · (x̄' φ^post)       + b^post         ∈ R^{1×n}
     H̃^res  = α^res  · mat(x̄' φ^res)     + b^res          ∈ R^{n×n}
(8)  H^pre = σ(H̃^pre);   H^post = 2σ(H̃^post);   H^res = Sinkhorn-Knopp(H̃^res)
(9)  M^(0) = exp(H̃^res);   M^(t) = T_r( T_c( M^(t-1) ) )        (T_c: column-normalise, T_r: row-normalise)
```
Shapes ([mHC] §4.2): `φ^pre, φ^post ∈ R^{nC×n}`, `φ^res ∈ R^{nC×n²}` (stored together as `φ ∈ R^{nC×(n²+2n)}`), `b ∈ R^{1×(n²+2n)}`, α scalars.

**Settings stated in the paper (D12) ⚠:** `t_max = 20` (§4.2, "a practical value"); `α` init 0.01 and `RMSNorm ε = 1e-20` (Table 5); `n = 4` (§5.1); weight decay 0.1 on all parameters in their LLM runs (Table 5). No Sinkhorn ε.

**Not stated in the paper (what D12 cannot "follow exactly"):** the initialisation of `φ` and `b`; how the n streams are created at the input and read out at the end; whether mHC equals the residual network at init. The paper is silent on all three.

**Our adaptation.**
- Token layout: per token, flatten its `n×d` streams.
- Conv layout, D3(a)-style: `x̄ = vec(GAP(H)) ∈ R^{B×nC}`, then the same maps per sample.
- Compute the maps in **float32** (`exp` and RMSNorm ε = 1e-20 under fp16 would underflow), whatever the autocast dtype is.
- Parameters: `nC(n²+2n) + (n²+2n) + 3` per layer. That is ~16× DHC's `d(n+2)+n(n+2)+2` at n = 4, d = 192 (18,459 vs 1,178 per layer; +8% of ViT-Tiny-12, ~+12% of ResNet-20). This is the paper's design; it is a cost-table item (T3), not a bug.

**Init D16 (so that equivalence at init holds and the streams can differentiate).**
- `φ = 0` (as [HC] does for its dynamic weights); `α = 0.01`.
- `b^post = 0` → `H^post = 2σ(0) = 1` exactly.
- `b^pre = logit(p^k)` with `p^k` = soft one-hot on stream `k mod n`: `p = 1−(n−1)ε` there and `ε` elsewhere (ε = 0.05 → 0.85/0.05/0.05/0.05). Then `Σ_i H^pre_i = 1` exactly and layers read different streams, mirroring `A_m = e_{k mod n}`.
- `b^res = log M*` with `M* = (1−λ)I + (λ/n)J`, λ = 0.1. `M*` is already doubly stochastic, so Sinkhorn leaves it unchanged and `H^res = M*` exactly. (The identity itself is a vertex of the Birkhoff polytope and unreachable with finite logits; large negative off-diagonals would have dead gradients.)
- ε and λ are fixed init constants, not tuned on any result.
- Readout: mean (D14).

**[verified]** on the toy model: this init equals the residual output to 1.6e-15; zero-bias + `φ = 0` (the "paper-literal" biases with no random φ) keeps all streams **exactly identical forever** (spread 0.0 for 200 SGD steps) and has `ΣH^pre = 2`, so it equals the residual model only up to norm ε (4.9e-6 here).

**Inherent consequence.** With identical streams at init, the doubly-stochastic projection annihilates row- and column-constant gradients on `H^res`. These are exactly the gradients that let HC rescale streams individually. So mHC streams differentiate through `H^post` and the dynamic `φ` only, and more slowly than HC's (the gradient asymmetry across streams was only ~3× smaller on the toy model, but the forward stream spread was ~500× smaller after one step). The study should expect "mHC starts close to a plain residual net" and say so.

## 9. Mapping to code

| Module | Implements | Test |
|---|---|---|
| `connections/base.py` | `expand(x)→H`, `forward_block(H, branch, shortcut=None)`, `reduce(H)→x` | — |
| `connections/residual.py` | n = 1 | `test_shapes`, `test_param_counts` |
| `connections/hyper_connection.py` | [HC] Eqs. 1–2, 8–14, token layout `(B,T,n,d)` | `test_equivalence_init`, `test_shapes` |
| `connections/hyper_connection_conv.py` | §6, layout `(B,n,C,h,w)`, D3(a), D4(a) | same |
| `connections/mhc.py` | [mHC] Eqs. 3, 6–9, D16 init | `test_mhc_constraint` (rows/cols sum to 1, entries ≥ 0, incl. after random perturbation of φ, b) |

Equivalence tests: float64, fixed batch, **both train and eval mode**, every model × depth × n × {static, dynamic, mHC}.

## 10. Findings from the literature check

1. **The lucidrains repo already has a channel-first HC.** `hyper_connections/hyper_connections_channel_first.py` (`HyperConnections(..., channel_first = True)`) computes dynamic weights with a per-pixel `nn.Conv2d(dim, n+1, 1, bias=False)` and expands/reduces with repeat/sum. The same package also ships `mHCv2.py`, `manifold_constrained_hyper_connections.py` (`sinkhorn_iters = 20`), `triton_sinkhorn.py` and `vit.py`. (Seen on GitHub, 2026-10-06; the README does not mention any of this.) An earlier statement of the project brief said the reference implementation "does not support channel-first conv feature maps" — that is outdated. The CNN *evaluation* is still unclaimed by any paper found, but a "channel-first module" is no longer a novel engineering contribution by itself; our D3(a) (pooled weights) differs from their D3(b) (per-pixel). → D18 (claim wording).
2. **D9 vs the paper:** see §3, D17.
3. **mHC settings the paper leaves open:** D16 and D14.
4. **Cost asymmetry:** mHC's projection is ~16× DHC's parameters per layer (§8). T3 must show it.
5. **Prior-art scan (2026-10-06)** is logged in `docs/references.md`. The OpenReview paper `Zu3B2TKOPV` could not be opened (bot-verification page) and is not cited; check it by hand for completeness.

## 11. Implementation details beyond the decisions D1-D17

Everything below is identical for every connection type (one variable at a time). None of it was named in D1-D17.

| Topic | What the code does |
|---|---|
| Status | P2 (connections) and P3 (models, data, trainer) are implemented; `pytest` covers equivalence at init for every model x depth x connection, parameter counts, subsets, the trainer end to end (offline) and its resume. |
| Training length (D6, approved 2026-10-16) | `train.epochs` x `train.epoch_steps` optimizer steps. `epoch_steps: null` means one pass over the training set, so a 10% subset would get 10x fewer steps. D6(b) "same steps" is `train.epoch_steps=390` (one pass over 50,000 images at batch 128 with `drop_last`) for every data fraction; the compact study uses it, so *less data* is not mixed up with *less training*. |
| LR schedule | Per optimizer step: linear warmup (D8: 10 epochs; D7: none) then cosine to 0. |
| Weight decay groups | D9/D17: static HC/mHC parameters never decayed, dynamic projections always. Backbone: ResNet recipe decays every parameter (standard CIFAR-ResNet practice); ViT recipe does not decay biases, LayerNorm weights, `pos_embed`, `cls_token` (standard ViT practice). |
| RandAugment (D8) | D8 gives no parameters: torchvision defaults (2 ops, magnitude 9), applied on the PIL image after crop and flip. |
| Precision | fp32 everywhere, in every run of the study. (Mixed precision was not used. On tensor-core GPUs it may be much faster, but the streams and the connection maps would have to stay fp32, otherwise the HC arms would accumulate their streams in lower precision than the residual arm.) `cudnn.benchmark = True`, so runs are seeded but not bit-exact. |
| Seeds | The model is built right after `set_seed`, before any data RNG, and connections own no random numbers: equal seeds give equal backbones across connection types (tested). The loader shuffle uses a generator seeded with the run seed. |
| Subsets | Per class, the first k entries of a seeded permutation: nested (10% within 25% within 100%), class-balanced, one fixed subset seed (0) for all runs. Indices are saved in `configs/data/subsets/` (committed) and validated on load. |
| Normalisation | Per-channel mean/std measured on the full 50,000-image training set at load time when the data config leaves them null. |
| Probe set | 1,000 class-balanced test images (10 per class), fixed by `analysis.seed`. Used for similarity / CKA and for the effective connection weights in `connections.pt`. |
| Gradient norms | 20 fixed batches of unaugmented images from the run's own training subset, model in train mode (BatchNorm statistics restored afterwards). Per layer = branch + shortcut parameters only, so profiles compare across connection types. |
| Similarity | Layer k compares the reduced (mean over streams) hidden state before and after layer k: per-sample cosine (nan where the shape changes, i.e. ResNet stage transitions) and linear CKA. Eval mode. |
| Layer index | Layer k is the k-th connection layer: a ResNet block, or a ViT attention / MLP sublayer. |
| Peak memory | Max allocated during the training steps of an epoch (analysis passes excluded). |
| Divergence | A non-finite training loss stops the run; `final.json` has `diverged: true` and no final accuracy (feeds table T4). |
| Resume and session limits | After each epoch `experiments/runs/<run_id>/last.pt` (git-ignored) is written; an interrupted run continues from it. `scripts/run_matrix.py` skips runs with a `final.json`. `--stop-after-hours H` sets `train.stop_at`: after an epoch the trainer pauses (checkpoint written first, `Paused`, exit code 75) if another epoch of the same length plus 30% would not end before it, and the launcher starts no run in the last 10 minutes; at least one epoch is always finished, so repeated sessions always progress, and the last epoch is never paused. `train.stop_at` is not part of the experiment (ignored by the settings check). `--claim-dir` lets one launcher per GPU (`CUDA_VISIBLE_DEVICES`) share the run list through atomically created claim files. |
| Parameter counts | The ResNets add two 1x1 projection shortcuts (D4a/D15a), so they sit within 3% above the published option-A counts (0.27M / 0.85M / 1.7M, He et al. CVPR 2016 Table 6). ViT-Tiny with patch 16 at 224 px and 1000 classes has exactly the standard 5,717,416 parameters. |
| Cost measurement (deviations from the original plan) | `scripts/profile_cost.py` replaces the `scripts/profile.py` of the original plan: a file called `profile.py` shadows the stdlib `profile` module that `cProfile` imports. Forward MACs come from `torch.utils.flop_counter.FlopCounterMode` (1 MAC = 2 FLOPs; matmul / conv / attention only) instead of fvcore or thop: fvcore traces with `torch.jit`, is not part of the Colab image and breaks easily across torch versions, the built-in counter is what the run has. The elementwise work that dominates the overhead of HC and mHC is not counted by any FLOP counter, which is why memory and throughput are measured, not derived. One batch size (128), fp32, after warm-up. |
| Aggregation and statistics (D10 approved; D11 changed to one seed on 2026-10-17) | `scripts/aggregate.py` reads the run folders only (`config.yaml`, `final.json`) and refuses a results folder whose runs mix training budgets. Accuracy is the final-epoch top-1 in percent. Pairs are formed by seed over the seeds both arms finished (diverged runs are excluded and counted in T4). Gain = mean of per-seed differences; p-value = paired t-test (`scipy.stats.ttest_rel`); CI = percentile bootstrap, 10,000 resamples over seeds, generator seeded with 0; significant = p < 0.05 **and** the CI excludes 0. With 3 seeds the bootstrap CI cannot be finer than the 3 differences, so a missing star is weak evidence of no effect. With one seed per cell (the compact study) the standard deviation and p-value columns are empty (NaN) and nothing is marked significant; the report then states every difference as an observation from a single run. |
| Figures | `scripts/make_figures.py`: one validated categorical colour per connection in a fixed order, a second encoding by marker shape, legends on every multi-series panel, a log-scale axis for cost, layer-wise panels at the deepest model of each family. F2 needs at least two data fractions, F5 a cost CSV, F6 `connections.pt`; a figure the runs cannot support is skipped with a message. |
| Report and student guide | `scripts/build_report.py` fills the prose in `docs/templates/*.md` from `results/tables/*.csv` and the run folders (`config.yaml`, `similarity.csv`, `grad_norms.csv`): every number in the generated `docs/PROJECT_REPORT.md` and `docs/STUDENT_GUIDE.md` is a placeholder filled from the logs (no result is typed by hand), an unknown placeholder is an error, and `--check` fails when the written files are stale. Similarity is the mean over layers at the last epoch of `similarity.csv`; the gradient ratio is last layer / first layer at the first logged epoch of `grad_norms.csv`. `scripts/build_report_docx.py` converts both files to .docx (python-docx). |

## 12. Design decisions (D1-D20)

The decisions were taken by the project lead: D1-D5, D7-D9 and D12-D17 on 2026-10-07; D6, D10, D11 and D19 on 2026-10-16 (D11 changed on 2026-10-17: one seed); D20 on 2026-10-20, extended on 2026-10-23.

| ID | Decision | What was chosen | Why |
|---|---|---|---|
| D1 | ResNet variant | Pre-activation ResNet (He et al., ECCV 2016) | Its clean identity path matches HC's Pre-Norm-equivalent initialisation, so the "equal at init" test is exact. |
| D2 | HC variant of the main comparison | Dynamic HC (DHC) with 2 and 4 lanes, and mHC with 4 lanes; static HC only as an ablation (not run) | DHC is the best-performing variant in the base paper. |
| D3 | Dynamic weights for convolutional feature maps | Computed from the global-average-pooled per-stream features, shape (B, n, C) | Cheap, and the closest analogue of per-token DHC. A per-pixel 1x1-convolution version was not built. |
| D4 | Streams at ResNet stage transitions (channel / stride change) | The shared projection shortcut is applied to every stream | Keeps n streams end to end and adds no parameters per stream. |
| D5 | ViT-Tiny configuration | Patch 4 on 32 x 32 images (64 tokens), width 192, 3 heads, MLP ratio 4, class token, Pre-Norm | A standard small ViT for CIFAR. |
| D6 | Training budget for data subsets | The same number of optimizer steps for every data fraction (390 steps per epoch at batch size 128) | A 10% run with equal epochs would get ten times fewer updates, which would confound data size with training length. |
| D7 | ResNet recipe | SGD, momentum 0.9, Nesterov, learning rate 0.1, weight decay 5e-4, cosine schedule, batch size 128, random crop and horizontal flip; 30 epochs in the study (the recipe's own length is 200) | Identical for all connection types. |
| D8 | ViT recipe | AdamW, learning rate 1e-3, weight decay 0.05, 10-epoch warm-up then cosine, batch size 128, RandAugment, random crop and flip, label smoothing 0.1, no mixup or cutmix; 30 epochs in the study (200 in the recipe) | Identical for all connection types. |
| D9 | Weight decay on connection weights | None on the static HC scalars | The common treatment of norm and bias parameters; the same rule for every HC run. |
| D10 | Significance test | Paired t-test and bootstrap 95% confidence interval, paired by seed | Not usable with one seed (D11). |
| D11 | Seeds | Three were planned; the study stopped at one (seed 0): 24 runs | With one run per setting there are no standard deviations or tests, so differences are reported as observations. |
| D12 | mHC details | As in arXiv:2512.24880: 20 Sinkhorn rounds, alpha 0.01, RMSNorm epsilon 1e-20, 4 lanes; the maps are computed in at least float32 | Faithful to the base paper. |
| D13 | Experiment tracking | CSV and JSON files are the source of truth | No external service is needed. |
| D14 | Stream readout at the end of the network | Mean over the streams (the paper sums them) | A sum gives n times the input at initialisation; a norm follows immediately, so mean and sum are the same function class, and the mean makes the equivalence test exact. |
| D15 | Projection shortcut at stage transitions | A 1 x 1 stride-2 convolution on the raw block input, in the baseline and in every stream | Keeps the baseline and HC identical at initialisation and the parameter counts standard; a small, documented deviation from the reference pre-activation unit. |
| D16 | mHC initialisation | Static and exactly equivalent to the residual network: phi = 0, H^post = 1, H^pre a soft one-hot (0.85 / 0.05 / 0.05 / 0.05), H^res = 0.9 I + 0.1 J / n | The paper does not specify it. This choice makes mHC equal the residual network at the start and breaks the symmetry between streams; the constants are fixed, never tuned. |
| D17 | Weight decay on the dynamic HC projections | Static scalars and biases: none; dynamic projections: weight decay like any weight matrix | Matches [HC] section 4. |
| D18 | Wording of the novelty claim | "The first systematic evaluation of HC and mHC in CNNs", noting that a public implementation has a per-pixel channel-first version | An earlier statement that the reference implementation has no channel-first support is outdated. This is the wording the report uses. |
| D19 | Size of the study | Compact: ResNet-20, ResNet-56 and ViT-Tiny-6 x 4 connection types x CIFAR-100 (all the data and 10%) x 30 epochs, seed 0: 24 runs, about 6 GPU-hours on free GPUs | Keeps the CNN-versus-transformer and shallow-versus-deeper contrasts and the data-size question. The original 144-run plan needs hundreds of GPU-hours. |
| D20 | Live prediction page | A small local page (`app/`) with four ResNet-20 networks trained once more with their weights saved (`results/demo/`); the weights and the class assets are in the repository, so a clone runs it | The study's runs saved no weights. Not built: Grad-CAM, Tiny-ImageNet, training from the page. |
