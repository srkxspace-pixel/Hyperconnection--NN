# Verified References

Only sources listed here may be cited in code comments, docs or the report. Add new ones only after you open the source page and confirm the title, authors and venue.

## Base paper
- D. Zhu, H. Huang, Z. Huang, Y. Zeng, Y. Mao, B. Wu, Q. Min, X. Zhou, "Hyper-Connections," in *Proc. ICLR*, 2025.
  - arXiv: https://arxiv.org/abs/2409.19606
  - OpenReview: https://openreview.net/forum?id=9FqARW7dwB
  - Proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f1e8ff97057e0c3f6bcb7018c78d4df1-Abstract-Conference.html
  - Unofficial reference implementation: https://github.com/lucidrains/hyper-connections

## Closest prior work (cite and differentiate)
- Z. Xie, Y. Wei, H. Cao, C. Zhao, C. Deng, J. Li, D. Dai, H. Gao, J. Chang, K. Yu, L. Zhao, S. Zhou, Z. Xu, Z. Zhang, W. Zeng, S. Hu, Y. Wang, J. Yuan, L. Wang, W. Liang, "mHC: Manifold-Constrained Hyper-Connections," arXiv:2512.24880, 2025 (v1 31 Dec 2025, v2 5 Jan 2026). https://arxiv.org/abs/2512.24880
- D. Zhu, H. Huang, J. Zhou, Z. Huang, Y. Zeng, B. Wu, Q. Min, X. Zhou, "Frac-Connections: Fractional Extension of Hyper-Connections," arXiv:2503.14125, 2025 (18 Mar 2025). https://arxiv.org/abs/2503.14125 — splits the hidden state into segments instead of widening it, to cut the memory-access cost of HC.
- S. Mishra, "mHC-GNN: Manifold-Constrained Hyper-Connections for Graph Neural Networks," arXiv:2601.02451, 2026 (v1 5 Jan 2026). https://arxiv.org/abs/2601.02451 — graph benchmarks only.
- Z. Liu, H. Zhang, A. Li, "Spectral-Sphere-Constrained Hyper-Connections," arXiv:2603.20896, 2026 (v1 21 Mar 2026, v2 25 Sep 2026). https://arxiv.org/abs/2603.20896 — language-model pre-training only (HTML text read 2026-10-11).
- T. Dandachi, S. Diggs-Galligan, "go-mHC: Direct Parameterization of Manifold-Constrained Hyper-Connections via Generalized Orthostochastic Matrices," arXiv:2604.02309, 2026 (2 Apr 2026). https://arxiv.org/abs/2604.02309 — synthetic stream-mixing tasks and a 30M-parameter GPT-style model.

## Foundations
- K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE CVPR*, 2016.
- K. He, X. Zhang, S. Ren, J. Sun, "Identity Mappings in Deep Residual Networks," in *Proc. ECCV*, 2016.
- A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. ICLR*, 2021.
- S. Kornblith, M. Norouzi, H. Lee, G. Hinton, "Similarity of Neural Network Representations Revisited," in *Proc. ICML*, 2019.
- A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Tech. Rep., Univ. of Toronto, 2009.

## Training components (added 2026-10-11; arXiv pages opened and title / authors checked)
- E. D. Cubuk, B. Zoph, J. Shlens, Q. V. Le, "RandAugment: Practical Automated Data Augmentation with a Reduced Search Space," arXiv:1909.13719, 2019. https://arxiv.org/abs/1909.13719 (the arXiv page names no venue, so cite it as a preprint)
- I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. ICLR*, 2019. https://arxiv.org/abs/1711.05101 (AdamW)
- I. Loshchilov, F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," in *Proc. ICLR*, 2017. https://arxiv.org/abs/1608.03983 (cosine schedule)
- C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna, "Rethinking the Inception Architecture for Computer Vision," arXiv:1512.00567, 2015 (also CVPR 2016). https://arxiv.org/abs/1512.00567 (label smoothing; the abstract page does not name it, the method is its §7 — cite for that only)

## Datasets
- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- Tiny-ImageNet (200 classes, 64×64): http://cs231n.stanford.edu/tiny-imagenet-200.zip

## Prior-art check log
| Date | Queries | New overlapping work found | Action |
|---|---|---|---|
| 2026-09-23 | HC + CNN / small data / depth sweep / vision | None for CNNs or small-data vision. One ICLR 2026 workshop paper (OpenReview id Zu3B2TKOPV) could not be opened — check by hand | — |
| 2026-10-06 | "hyper-connections ResNet CIFAR CNN multi-stream residual"; "manifold-constrained hyper-connections mHC vision image classification small dataset"; abstracts read on arxiv.org | None for CNNs / small-data / depth-sweep vision. Screened, no overlap: xHC (arXiv:2607.14530, LLM MoE); TBP-mHC (arXiv:2605.21724, LLM pre-training; names mHC-lite and KromHC as further mHC variants); "Ablate and Rescue" (arXiv:2603.14833, mHC language model, mechanistic analysis); ES-mHC (arXiv:2601.15757) and mHC-HSI (arXiv:2603.03418), both hyperspectral imaging with Mamba. OpenReview `Zu3B2TKOPV` still behind a bot-verification page — **still check by hand** | Not cited yet. Re-run at P8 |
| 2026-10-06 | Code inspection of https://github.com/lucidrains/hyper-connections (file listing + `hyper_connections_channel_first.py` + `manifold_constrained_hyper_connections.py` on GitHub) | The unofficial reference implementation **does** include a channel-first HC (per-pixel `Conv2d(dim, n+1, 1)` dynamic weights, `channel_first=True`), two mHC files, a Triton Sinkhorn and a `vit.py`. an earlier statement of the project brief said the opposite | Narrow the "channel-first module" claim — see D18 and `docs/method_notes.md` §10 |
| 2026-10-11 | Three web searches (HC / mHC with CNNs, CIFAR, limited data, depth sweeps, vision), then the arXiv abstract page or HTML text of every candidate below | **Nothing that evaluates HC or mHC on CNNs, on CIFAR-scale / limited-data vision, or in a depth sweep.** Read: (1) the HC paper itself, arXiv:2409.19606v3, **Appendix E**: its only vision experiments are DiT on ImageNet 256×256 (SHC, n=2; E.1) and ViT/16 Base and Large on ImageNet 224×224 (SHC and DHC, n=2, 300 epochs; E.2, Table 11), plus a DHC-weight visualisation (E.3); no CNN / ResNet / CIFAR experiment; the text notes that the Large-model DHC gain shrinks as training continues (Fig. 11) — the nearest vision prior, cite and differentiate. (2) mHC-GNN, arXiv:2601.02451 (S. Mishra): graph benchmarks, depth 2–128 layers; no images. (3) s²HC, arXiv:2603.20896: LM pre-training only (Qwen3-0.6B-variant, Gemma3-1B, Qwen2.5-1.5B on FineWebEdu). (4) go-mHC, arXiv:2604.02309: synthetic tasks + 30M GPT-style LM; its related-work sentence "applying HC to vision and graph neural networks (Zhu et al., 2025a; Mishra, 2026)" resolves to (1) and (2), so it is **not** a CNN / small-data study. Also seen, not HC studies: arXiv:2605.11526 (Yang, Gu, Yuan; training with linear-constraint projection layers; abstract does not mention HC or mHC, full text not read); xHC arXiv:2607.14530 (LLM, screened 2026-10-06). OpenReview `Zu3B2TKOPV` could not be opened (bot check, not bypassed) — **check it by hand for completeness** | The paper cites (1)–(4) in related work; novelty wording follows D18 ("to our knowledge", scoped to CNNs / small data / depth sweeps, not "first to apply HC to vision"). Re-run at P8 |
