# Do Multi-Lane Shortcuts Help Small Image Networks?

**A controlled study of hyper-connections on CIFAR-100**

B.Tech final-year project NN-09 (full title: *Hyper-Connections at Small Scale: Depth Scaling and Data Efficiency of Multi-Stream Residual Connections in Compact CNNs and Vision Transformers*)

> **How strong is the evidence?** Every setting was trained [[once]] ([[n_runs]] networks in total). The numbers below are observations from those runs. They are not statistically proven effects: with one run per setting there is no way to tell luck from a real difference when the gap is small.

## Abstract

A deep network passes information forward along a "shortcut" (the residual connection), a single lane that runs through all the layers. *Hyper-connections* (HC) replace that single lane by several parallel lanes whose weights are learned, and *mHC* adds a mathematical constraint on how the lanes mix, to keep training stable. Earlier work showed gains for very large language models and for ImageNet-scale transformers. We asked whether HC and mHC also help *small* image networks trained on *small* data, and what they cost. We implemented both for convolutional networks and for vision transformers, checked that each such network computes exactly the same function as the ordinary network before training starts, and trained [[n_runs]] networks on CIFAR-100: ResNet-20, ResNet-56 and a 6-block ViT-Tiny, each with the normal shortcut, HC with 2 or 4 lanes and mHC with 4 lanes, on all of the training data and on 10% of it. With all the data, HC with 2 or 4 lanes improved both ResNets by [[agg.resnet_hc.p100.lo]] to [[agg.resnet_hc.p100.hi]] points of accuracy, whereas every connection type (HC and mHC) lowered the accuracy of the ViT, by [[agg.vit_all.p100.abslo]] to [[agg.vit_all.p100.abshi]] points. With 10% of the data there was no consistent gain. HC needed [[range.hoursx.hc.lo]] to [[range.hoursx.hc.hi]] times as long to train and mHC [[range.hoursx.mhc_n4.lo]] to [[range.hoursx.mhc_n4.hi]] times as long, and the memory needed grew by up to a factor of [[range.mem.all.hi]]. For the ResNets, HC with 4 lanes made consecutive layers change the representation more than the normal shortcut, and mHC did not.

## 1. Introduction

Very deep networks are hard to train because the learning signal fades as it travels through many layers. The *residual connection* [3] fixes this: each layer adds its output to a copy of its input, so the signal can skip layers along a shortcut. Almost every modern vision network uses it.

*Hyper-connections* (HC) [1] generalise the shortcut. Instead of one copy of the signal, the network keeps *n* parallel copies (we call them **lanes**; the papers call them streams). Each layer reads a learned mixture of the lanes, writes its result back into the lanes with learned weights, and mixes the lanes among themselves. In the *dynamic* version the weights are computed from the data. *mHC* [2] forces the lane-mixing matrix to be "doubly stochastic" (non-negative, every row and column summing to 1) so that training stays stable. (Further variants exist, for example Frac-Connections [8]; they are not part of this study.)

HC was reported to help large language models and, in vision, ViT-Base/Large and DiT on ImageNet with 2 lanes [1]. To our knowledge, when we checked the literature (2026-10-11, see `docs/references.md`), nobody had tested HC or mHC in convolutional networks, in small-data image classification, or across network depth. Small models matter in practice because they are trained on limited data and run on limited hardware, so both the benefit and the cost of extra lanes are worth measuring.

We ask five questions:

1. **RQ1, depth.** Does the gain from HC/mHC grow, stay flat or shrink from a shallower to a deeper network?
2. **RQ2, data.** Is the gain larger when only 10% of the training data is available?
3. **RQ3, mechanism.** Do HC/mHC change how similar consecutive layers are, or the size of the gradients across depth, and does that go together with the accuracy change?
4. **RQ4, cost.** What do 2 and 4 lanes cost in memory and time, and is any setting worth it?
5. **RQ5, family.** Do convolutional networks and vision transformers respond differently?

**Scope.** This is a compact study sized for a final-year project and for free GPUs: three models, [[runword]] per setting, [[epochs]] epochs of training.

## 2. Background

**Residual connection.** One layer computes `x_next = x + F(x)`, where `F` is the layer's own transformation.

**Hyper-connection (HC) [1].** The state is kept as *n* lanes `H_1 ... H_n`. A layer does three things:

- *read*: `h_in = a_1·H_1 + ... + a_n·H_n` (a learned mixture of the lanes is fed to the layer);
- *write*: `H_j_new = b_j·F(h_in) + (mixing of the old lanes)`, so the layer output is added to each lane with its own weight `b_j`;
- *mix*: the old lanes are combined with a learned `n x n` matrix `A` before the output is added.

Static HC learns these numbers once. Dynamic HC (DHC) adds a small correction that depends on the current data (`tanh` of a projection of the lane features). With the right starting values (`A` = identity, `b` = 1, each layer reading a different lane) all lanes start out equal to the residual network's state, so HC starts as an ordinary residual network.

**mHC [2].** The mixing matrix is computed from the data and pushed through the Sinkhorn-Knopp procedure (20 rounds of normalising rows and columns) so that it is doubly stochastic. The paper does not say how to initialise it; our choice is explained in section 3.

## 3. What we built

- **HC for convolutional networks.** A lane is a feature map. Every weight is one number per lane (per image), applied to the whole map. The dynamic weights are computed from the *average of each lane's feature map*, which adds under 1% to the number of parameters. Where a ResNet stage halves the image size and doubles the channels, one shared 1x1 convolution carries every lane across. Before the classifier the lanes are averaged. *(A different design, with weights computed at every pixel, exists in a public unofficial implementation [9]; we did not compare it.)*
- **HC and mHC for transformers.** Each attention block and each MLP block is one connection layer, as in the original paper.
- **mHC start values.** If every mHC weight started at zero, all lanes would stay identical forever, so we start the mixing at 0.9 times the identity plus 0.1 times uniform averaging (a matrix that Sinkhorn leaves unchanged) and let each layer read a different lane. These values are fixed constants, not tuned. The mHC results depend on this choice.
- **Equal at the start.** For every model, depth and connection type, the network computes exactly the same output as the ordinary network before training (tested in double precision, in training and evaluation mode). The tests passed on the GPU machine used for the study (148 core tests plus 29 newer checks).
- **A reproducible pipeline.** One command trains a run and writes its logs; a launcher runs the whole list, resumes after interruptions, and pauses cleanly before a session limit; scripts rebuild every table and figure from the stored logs. The same notebook runs on Kaggle and Colab.
- **A live prediction page.** A small local web page, built for the demonstration and not part of the study, loads four trained ResNet-20 networks and classifies a picture the user uploads (Appendix C).

## 4. Experimental setup

**Models.** ResNet-20 and ResNet-56 (pre-activation ResNets [4] for CIFAR with 16/32/64 channels) and a small vision transformer [5], "ViT-Tiny", with 6 blocks (width 192, 3 attention heads, patch size 4 on 32x32 images).

**Connections compared.** The normal shortcut; dynamic HC with 2 lanes; dynamic HC with 4 lanes; mHC with 4 lanes. Only the connection differs between the networks of a model: the same layers, the same starting weights (same random seed), the same data order and the same training recipe. Nothing was tuned for HC.

**Data.** CIFAR-100 [7]: 50,000 training and 10,000 test images in 100 classes. We trained on all training images and on a fixed, class-balanced 10% subset (50 images per class). The 10% runs got the same number of training steps as the full-data runs ([[steps_per_epoch]] steps per epoch, batch size [[batch_size]]), so that *less data* is not mixed up with *less training*.

**Training.** [[epochs]] epochs per network. The recipes are identical for every connection type; both families use the cosine schedule [11]; AdamW [10], RandAugment [12] and label smoothing [13] are used for the ViT only:

[[table.setup]]

**Measurements.** *Accuracy* is the top-1 accuracy on the 10,000 test images after the last epoch (the last epoch, not the best one, so that no epoch is picked by looking at the test set). *Similarity* is the cosine similarity and the linear CKA [6] between the hidden states before and after each layer, on a fixed set of 1,000 test images. *Gradient size* is the size of the weight gradients of each layer, averaged over 20 fixed batches. *Cost* is the number of parameters, the multiply-accumulate operations, the peak memory of a training step and the time per step, measured on the GPU at batch size [[batch_size]], plus the real wall-clock time of each run.

**Hardware.** Kaggle, two [[device]] GPUs working in parallel, [[gpu_hours]] GPU-hours in total. All [[n_runs]] runs finished; [[diverged]] of them diverged.

## 5. Results

### 5.1 Accuracy

The table gives the final top-1 test accuracy in percent; the number in brackets is the difference from the normal shortcut of the same model and data (positive = better than normal).

[[table.accuracy]]

![Figure 1 (file F1_accuracy_vs_depth). Final test accuracy against depth for each connection type (one run per point).](../results/figures/F1_accuracy_vs_depth.png)

![Figure 2 (file F2_gain_vs_data_fraction). Gain over the normal shortcut, in percentage points, for 10% and for all of the training data (one run per point).](../results/figures/F2_gain_vs_data_fraction.png)

![Figure 3 (file F7_training_curves). Test accuracy during training for ResNet-56 and ViT-Tiny-6, on all the data (top) and on 10% (bottom).](../results/figures/F7_training_curves.png)

With all the data, both ResNets did better with HC than with the normal shortcut ([[agg.resnet_hc.p100.lo]] to [[agg.resnet_hc.p100.hi]] points for 2 or 4 lanes), mHC helped ResNet-20 ([[gain.resnet20.mhc_n4.p100]]) but hardly changed ResNet-56 ([[gain.resnet56.mhc_n4.p100]]), and every connection type was worse than the normal shortcut for the ViT ([[agg.vit_all.p100.lo]] to [[agg.vit_all.p100.hi]]). With 10% of the data the differences are small and point in both directions ([[agg.resnet_all.p10.lo]] to [[agg.resnet_all.p10.hi]] for the ResNets, [[agg.vit_all.p10.lo]] to [[agg.vit_all.p10.hi]] for the ViT).

### 5.2 Why? Similarity and gradients

![Figure 4 (file F3_representation_collapse). Similarity between the input and the output of each layer at the end of training (linear CKA on top, cosine below; lower = the layer changes the representation more). ResNet-56 on the left, ViT-Tiny-6 on the right, all the data.](../results/figures/F3_representation_collapse.png)

![Figure 5 (file F4_gradient_norms). Size (L2 norm) of the weight gradient in each layer at the end of the first epoch. ResNet-56 on the left, ViT-Tiny-6 on the right, all the data.](../results/figures/F4_gradient_norms.png)

![Figure 6 (file F6_connection_patterns). Learned connection weights of HC with 4 lanes after training on all the data (ResNet-56 above, ViT-Tiny-6 below): lane-mixing matrices of six layers (left; rows are the source lane, columns the target lane) and read and write weights of every layer (right).](../results/figures/F6_connection_patterns.png)

Summary numbers for the networks trained on all the data (similarity at the end of training; gradient size after epoch 1):

[[table.mechanism]]

In the ResNets, HC with 4 lanes lowered the CKA between consecutive layers (ResNet-20: [[cka.resnet20.residual.p100]] with the normal shortcut, [[cka.resnet20.dhc_n4.p100]] with HC; ResNet-56: [[cka.resnet56.residual.p100]] and [[cka.resnet56.dhc_n4.p100]]): the layers change the representation more. HC also made the gradient profile of ResNet-56 flatter (last layer / first layer = [[gratio.resnet56.dhc_n4.p100]] against [[gratio.resnet56.residual.p100]]). mHC behaved almost like the normal shortcut in both measures ([[cka.resnet56.mhc_n4.p100]] and [[gratio.resnet56.mhc_n4.p100]] for ResNet-56), which matches its design: it starts as a plain residual network and its mixing is tightly constrained. In the ViT none of the connection types changed these measures much (CKA between [[cka.vit_tiny_d6.range.p100.lo]] and [[cka.vit_tiny_d6.range.p100.hi]] for all four). In Figure 6 the diagonal dominates every learned lane-mixing matrix: each lane mostly keeps its own content, and the mixing between lanes is small and sometimes negative.

### 5.3 Cost

![Figure 7 (file F5_accuracy_vs_cost). Final accuracy against the peak memory of a training step; the arrow in each panel runs from the normal shortcut to the same network with HC or mHC.](../results/figures/F5_accuracy_vs_cost.png)

[[table.cost]]

"Step time" and "memory" come from a dedicated measurement on the GPU; "Real training time" is the wall-clock time of the run on all the data (it includes loading and augmenting the images, so the small ResNet-20 shows a smaller slowdown than its step time suggests; the two GPUs also worked at the same time and shared four CPU cores, so these times contain some noise, for example ResNet-20 with 4 lanes finished before ResNet-20 with 2 lanes). Our implementation uses plain PyTorch operations without fused GPU kernels, so these overheads are those of this implementation.

### 5.4 Answers to the research questions

- **RQ1, depth.** For the ResNets with all the data the gain from HC is about the same at depth 20 and depth 56 ([[gain.resnet20.dhc_n2.p100]] and [[gain.resnet56.dhc_n2.p100]] for 2 lanes); the gain from mHC shrinks ([[gain.resnet20.mhc_n4.p100]] to [[gain.resnet56.mhc_n4.p100]]). With 10% of the data, HC with 4 lanes goes from [[gain.resnet20.dhc_n4.p10]] to [[gain.resnet56.dhc_n4.p10]]. With only two ResNet depths and one ViT depth we cannot describe a trend; for transformers the depth question is open.
- **RQ2, data.** HC did **not** help more when data was scarce. For the ResNets the gain seen with all the data (HC with 2 lanes: [[gain.resnet20.dhc_n2.p100]], [[gain.resnet56.dhc_n2.p100]]) disappears with 10% (HC with 2 lanes: [[gain.resnet20.dhc_n2.p10]], [[gain.resnet56.dhc_n2.p10]]).
- **RQ3, mechanism.** In the ResNets, HC changed the similarity of consecutive layers and the gradient profile, mHC did not, and in the ViT neither did. The accuracy gain goes together with this change for the ResNets on all the data, but not with 10% of the data (HC with 4 lanes lowered the CKA of ResNet-20 there as well: [[cka.resnet20.dhc_n4.p10]] against [[cka.resnet20.residual.p10]], yet its accuracy fell by [[absgain.resnet20.dhc_n4.p10]] points). So the link is partial, and with three models and one run each we cannot establish it.
- **RQ4, cost.** HC with 2 lanes needed [[range.slow.dhc_n2.lo]] to [[range.slow.dhc_n2.hi]] times the step time and [[range.mem.dhc_n2.lo]] to [[range.mem.dhc_n2.hi]] times the memory of the normal shortcut; with 4 lanes [[range.slow.dhc_n4.lo]] to [[range.slow.dhc_n4.hi]] times the time and [[range.mem.dhc_n4.lo]] to [[range.mem.dhc_n4.hi]] times the memory; mHC [[range.slow.mhc_n4.lo]] to [[range.slow.mhc_n4.hi]] times the time and [[range.mem.mhc_n4.lo]] to [[range.mem.mhc_n4.hi]] times the memory, plus about [[range.paramsup.mhc_n4.lo]]% to [[range.paramsup.mhc_n4.hi]]% more parameters (HC adds under [[range.paramsup.dhc_n4.hi]]%). In this study HC with 2 lanes gave the best return on the ResNets (about as much gain as 4 lanes at lower cost); mHC cost the most and gained the least; for the ViT no connection type paid off.
- **RQ5, family.** The two families responded differently: the ResNets gained from HC on all the data, the ViT lost. HC changed the layer-to-layer behaviour of the ResNets but not of the ViT. Why this is so was not tested (see section 6).

## 6. Discussion

**What the data supports.** On this setup, HC with 2 lanes is a modest improvement for small ResNets trained on all of CIFAR-100, it is not an improvement for a small ViT, and it does not help with 10% of the data. mHC, as we implemented and initialised it, behaves like the normal shortcut but costs the most.

**What it does not support.** Every number comes from one training run. We did not measure how much accuracy varies when the same network is trained again, so we cannot say which of the differences are real. Differences of one to two points could partly be luck. Repeating each run with two more random seeds would settle this (about six more hours of GPU time per seed on the setup we used).

**Possible explanations (not tested).**

- *ResNets with all the data:* extra lanes give the network more ways to route information, which showed up as layers that change the representation more. With 10% of the data (the same number of steps, so many passes over few images) the extra freedom may simply overfit.
- *ViT:* its recipe was written for 200 epochs and uses a [[vit_warmup]]-epoch warm-up, a third of our [[epochs]]-epoch budget, and its accuracy was still rising at the end (Figure 3). A longer run might change the comparison. HC also starts as an identity-like network, so it may need more training to pay off.
- *mHC:* its mixing is strongly constrained and starts close to the identity, so it stays close to a normal network; a different start would be a different experiment.

**Threats to validity.** Only [[runword]] per setting; short training ([[epochs]] epochs); three models and one dataset; one training recipe per family (no tuning for HC, which could favour or hurt it); our own mHC start values; timing measured with unfused PyTorch operations; the ViT is only a single depth.

**Practical guidance (from this study only).** For a small ResNet trained on all the data, HC with 2 lanes is worth trying if the overhead is acceptable (for ResNet-56: [[hoursx.resnet56.dhc_n2]]x the training time and [[mem.resnet56.dhc_n2]]x the memory), and 4 lanes add cost without extra gain. With little data, a small ViT, or a tight compute budget, the normal shortcut was at least as good. Treat these as hypotheses to confirm with repeated runs.

## 7. Conclusion and future work

We implemented hyper-connections for convolutional networks and vision transformers, verified that they start out identical to ordinary residual networks, and compared them on CIFAR-100 with a complete, reproducible pipeline. In this first study HC helped small ResNets on all the data, hurt a small ViT, did not help with 10% of the data, and needed [[range.hoursx.all.lo]] to [[range.hoursx.all.hi]] times as long to train. The next steps are: repeat the runs with more seeds to put error bars on every number; add deeper networks (ResNet-110, ViT-Tiny with 12 and 24 blocks) to answer the depth question properly; train the ViT for longer; test per-pixel weights for the convolutional version; use fused GPU kernels to measure the best-case cost; and try Tiny-ImageNet.

## References

1. D. Zhu, H. Huang, Z. Huang, Y. Zeng, Y. Mao, B. Wu, Q. Min, X. Zhou, "Hyper-Connections," in *Proc. ICLR*, 2025. arXiv:2409.19606.
2. Z. Xie et al., "mHC: Manifold-Constrained Hyper-Connections," arXiv:2512.24880, 2025.
3. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE CVPR*, 2016.
4. K. He, X. Zhang, S. Ren, J. Sun, "Identity Mappings in Deep Residual Networks," in *Proc. ECCV*, 2016.
5. A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. ICLR*, 2021.
6. S. Kornblith, M. Norouzi, H. Lee, G. Hinton, "Similarity of Neural Network Representations Revisited," in *Proc. ICML*, 2019.
7. A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Tech. Rep., University of Toronto, 2009.
8. D. Zhu et al., "Frac-Connections: Fractional Extension of Hyper-Connections," arXiv:2503.14125, 2025.
9. lucidrains, "hyper-connections" (unofficial PyTorch implementation), https://github.com/lucidrains/hyper-connections.
10. I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. ICLR*, 2019.
11. I. Loshchilov, F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," in *Proc. ICLR*, 2017.
12. E. D. Cubuk, B. Zoph, J. Shlens, Q. V. Le, "RandAugment: Practical Automated Data Augmentation with a Reduced Search Space," arXiv:1909.13719, 2019.
13. C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna, "Rethinking the Inception Architecture for Computer Vision," arXiv:1512.00567, 2015.

## Appendix A. How to reproduce

1. Open `notebooks/nn09_colab.ipynb` on Kaggle (Session options: GPU T4 x2, Internet on; needs a phone-verified account) or on Colab.
2. Press *Save & Run All (Commit)* on Kaggle (or *Run all* on Colab with `LAUNCH` ticked). The plan `pilot` with seed 0 and 30 epochs is exactly the study reported here; it takes about 6 hours on two T4 GPUs.
3. Unzip the results into `results/raw/`, then run `python scripts/aggregate.py` and `python scripts/make_figures.py` to rebuild every table and figure, and `python scripts/build_report.py` to rebuild this report.

## Appendix B. All runs

Peak memory here is the highest GPU memory seen during the run, including the evaluation passes between epochs, so it is larger than the dedicated measurement of a training step in section 5.3.

[[table.runs]]

## Appendix C. The live prediction page

For the demonstration the project includes a small local web page (folder `app/`) that shows trained networks at work. It is a demonstration tool, not part of the study. The user uploads a picture, chooses one of four ResNet-20 networks (the normal shortcut, HC with 2 lanes, HC with 4 lanes, mHC with 4 lanes) and gets the five most likely CIFAR-100 classes as a list and as a bar chart. The page can download the answer for the current picture and the whole prediction history (model, connection type, checkpoint file, top answer, confidence and time) as CSV files, shows the training curves of the loaded network from its training log, and lists the 100 classes with a sample picture each.

**The networks.** The runs of the study saved only their logs, not the trained networks. The four networks of the page were therefore trained once more on Kaggle with the same recipe and seed, on all of CIFAR-100, with the weights saved (folder `results/demo/`, plan `demo` of `notebooks/nn09_demo_weights.ipynb`). A new run with the same settings does not give bit-identical numbers (the GPU is not fully deterministic), so the accuracies the page shows, taken from the training logs of these four runs, differ slightly from the table in section 5.1; the report itself uses only the runs in `results/raw/`. After training, every saved network was reloaded and classified all the test pictures one by one through the code the page uses, and gave exactly the test accuracy that had been logged at the end of its training.

**How a picture is prepared.** The middle square of the picture is cut out, shrunk to 32x32 pixels and normalised with the mean and standard deviation of the CIFAR-100 training images, exactly as the test pictures of the study were. The page shows the shrunken picture that the network really saw.

**Limits.** The networks know only the 100 CIFAR-100 classes (there is no dog or cat among them), were trained on 32x32 pictures and are small: they are right for most test pictures but not for all of them, and a photo whose subject is small or whose background looks unlike CIFAR is often misclassified. This is a property of small networks on a small dataset, not a defect of the page. The page runs on the user's own computer, on the CPU, one picture at a time; it needs PyTorch but no GPU and no dataset download.

**How to run it.** From the project folder, in a Python environment with `torch`, `torchvision` and `pyyaml`: `python app/server.py`. The step-by-step instructions are in `README_DEMO.md`.
