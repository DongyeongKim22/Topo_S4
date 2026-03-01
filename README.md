# Topo_S4 (WIP)

**Frequency-localized sensitivity** in sequence models (S4/S4D) and simple mitigations via **PC-band** (phase-coherent bandlimiting) and **PC-weight** (phase-coherent weighting).

- **Project homepage:** https://dongyeongkim22.github.io/Topo_S4/
- **Overleaf draft:** https://www.overleaf.com/read/qvvrpygjhbvv#15a2ec
- **Repo:** https://github.com/DongyeongKim22/Topo_S4

## Motivation

A working hypothesis is that **near-Nyquist digital frequencies** (`Ω -> π`) can become numerically over-sensitive under bilinear/Tustin discretization due to frequency warping:

$$
\omega(\Omega)=\frac{2}{\Delta}\tan(\Omega/2), \qquad
\frac{d\omega}{d\Omega}=\frac{1}{\Delta}\sec^2(\Omega/2)
$$

so sensitivity grows rapidly as `Ω` approaches `π`. If inputs contain strong near-Nyquist components, training and evaluation can exhibit sharper degradation.

## What this repo contains

- **Single-mode Fourier injection benchmark**: sweep normalized frequency `ρ in [0, 1]` (`ρ=1` is Nyquist) and measure accuracy drop vs frequency.
- **Filtering baselines**: guard band and low-pass filtering.
- **Two-track evaluation protocol** to disentangle “removing the perturbation” vs “changing sensitivity”:
  - `filter_both`: evaluate `F(u + δ)`
  - `filter_then_pert`: evaluate `F(u) + δ`
- **PC-band (phase-coherent bandlimiting)** preprocessing: offline estimation of an effective bandwidth plus a cached dataset-level hard mask (`--pc_keep_ratio`).
- **PC-weight (phase-coherent weighting)** preprocessing: a soft mask with a keep-budget target (`--pc_target_mean_w`).
- **preproc_scope ablation** (`train` / `test` / `both`) to study distribution shift and scope-dependent generalization.

> The PC mask is computed once and cached, then reused during training/evaluation.

## Results (current snapshot)

### 1) sCIFAR10 / CIFAR10: accuracy vs frequency (`ρ` sweep)

Setting: `s4d | cifar10 | target relΔ=0.05 | preproc=lpf(test) | track=filter_then_pert`

![CIFAR10 rho sweep](assets/eval_lpf070_072_track2_rel005_s4d_cifar10_rho_sweep_acc.png)

**Observation:** clean accuracy stays relatively stable, while perturbed accuracy drops sharply near **high `ρ`** (near Nyquist).

### 2) DTD96: PC-band improves mean best test accuracy (3 seeds)

S4D, seeds 0/1/2 (preliminary).

| Setting | keep_ratio | Mean Best Test (%) | Delta vs Raw (pp) |
|---|---:|---:|---:|
| Raw | -- | 36.42 | 0.00 |
| PC-band | 0.90 | 37.41 | +0.99 |

### 3) LRA Image / sequential CIFAR-10: full-epoch scope ablation (200 epochs, seed 0)

This is the strongest completed scope-ablation result so far on the image-side benchmark.

| Setting | Scope | Best Val (%) | Test @ Best Val (%) | Best Test (%) |
|---|---|---:|---:|---:|
| Raw | both | 86.72 | **85.73** | 85.73 |
| PC-band (`keep_ratio=0.95`) | train | 86.56 | 85.62 | **85.95** |
| PC-band (`keep_ratio=0.95`) | test | 85.86 | 84.62 | 84.98 |
| PC-band (`keep_ratio=0.95`) | both | 86.00 | 84.75 | 84.75 |
| PC-weight (`target_mean_w=0.60`) | train | 83.42 | 81.86 | 82.11 |
| PC-weight (`target_mean_w=0.60`) | both | 86.16 | 85.31 | 85.32 |

**Takeaways**
- **Raw is still a very strong baseline**: it gives the best validation accuracy and the best `test @ best val` among the completed runs.
- **PC-band with `preproc_scope=train` gives the best peak/final test accuracy** among the tested PC settings.
- Relative to applying the same PC-band mask at `both` or `test`, **`scope=train` gives about +0.9 to +1.0 pp test-side gain**.
- **PC-weight is highly scope-sensitive**: `scope=both` remains competitive, while `scope=train` drops sharply.

A working interpretation is that **sequential CIFAR-10 is comparatively low-frequency dominated**, so dev and test remain close overall. Even in that milder regime, scope choice still changes test-side generalization by about 1 pp.

### 4) Pathfinder32: pilot sweeps (5 epochs)

Some datasets and long-sequence settings are still expensive in my environment, so I use **short 5-epoch sweeps** to select promising keep-budgets before running longer confirmations.

Pathfinder32 (seed 0, **5 epochs**; metric = **test @ best dev**):

| Setting | Scope | Budget | Test @ Best Dev (%) |
|---|---|---:|---:|
| Raw | both | -- | 77.00 |
| PC-band | train | `keep_ratio=0.90` | 77.37 |
| PC-band | test | `keep_ratio=0.90` | 76.10 |
| PC-band | both | `keep_ratio=0.90` | 76.79 |
| PC-weight | train | `target_mean_w=0.40` | 52.40 |
| PC-weight | test | `target_mean_w=0.40` | 49.93 |
| PC-weight | both | `target_mean_w=0.40` | 77.76 |

**Pilot takeaway:** evaluation-only preprocessing can introduce distribution shift, and scope effects are especially strong for PC-weight.

**Current engineering status:** Pathfinder32 full runs through a pickle-based pipeline were taking too long in my environment. I am now moving to a **direct dataset download/loading path** instead of the pickle packaging route before launching the next Pathfinder32 round.

## This week's progress update

- Added **full-epoch LRA Image / sequential CIFAR-10** scope-ablation runs for `PC-band (keep_ratio=0.95)` and `PC-weight (target_mean_w=0.60)` against the raw baseline.
- Confirmed that **raw remains the strongest checkpoint-selection baseline** on LRA Image, while **PC-band + `preproc_scope=train`** gives the best peak/final test accuracy among the tested PC settings.
- Interpreted the LRA Image / CIFAR-10 result as a **comparatively low-frequency-dominated** case: dev and test stay fairly close, but scope choice still changes test-side generalization by about **1 percentage point**.
- Reconfirmed that **PC-weight is much more fragile under scope mismatch** than PC-band.
- Started migrating **Pathfinder32** away from the pickle-based pipeline because it is too slow for the next stage of experiments; direct dataset download/loading is now in progress.
- The next research direction will likely be decided by **Pathfinder32**. If the higher-frequency-heavy setting shows clearer gains, a natural next step is **learned PC optimization**, for example a **GNN-based mask optimizer** over frequency-bin relations.

## Next steps

- Finish the **Pathfinder32 direct dataset pipeline** and rerun the next pilot/full-epoch settings.
- Use Pathfinder32 to decide whether fixed PC masks are sufficient or whether to move to **learned PC optimization**.
- If the higher-frequency hypothesis keeps holding, explore **GNN-based PC optimization** instead of a fixed dataset-level mask.
- Revisit **SC35** and other longer-sequence settings after narrowing the keep-budget search.

## Repro (minimal notes)

- Track definitions:
  - `filter_both`: `F(u+δ)`
  - `filter_then_pert`: `F(u)+δ`
- See scripts/flags in the repo for dataset and model configs.

## Status

Work in progress.
