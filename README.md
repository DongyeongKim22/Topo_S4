# Topo_S4 (WIP)
**Frequency-localized sensitivity** in sequence models (S4/S4D) and a simple mitigation via
**PC-band: phase-coherent bandlimiting** (offline → cache).

Project homepage: https://dongyeongkim22.github.io/Topo_S4/  
Overleaf draft: https://www.overleaf.com/read/qvvrpygjhbvv#15a2ec  
Repo: https://github.com/DongyeongKim22/Topo_S4

---

## Motivation (high level)
A working hypothesis is that **near-Nyquist digital frequencies** (Ω → π) can become numerically
over-sensitive under bilinear/Tustin discretization due to frequency warping:

\[
\omega(\Omega)=\frac{2}{\Delta}\tan(\Omega/2), \quad
\frac{d\omega}{d\Omega}=\frac{1}{\Delta}\sec^2(\Omega/2),
\]

so sensitivity grows rapidly as Ω approaches π. If inputs contain strong near-Nyquist components,
training/evaluation can exhibit sharper degradation.

---

## What this repo contains
- **Single-mode Fourier injection** benchmark: sweep normalized frequency ρ ∈ [0, 1] (ρ=1 is Nyquist)
  and measure accuracy drop vs frequency.
- **Filtering baselines** (guard band / low-pass).
- **Two-track evaluation protocol** to disentangle “removing perturbation” vs “changing sensitivity”:
  - `filter_both`: evaluate `F(u + δ)` (filter can remove the injected perturbation)
  - `filter_then_pert`: evaluate `F(u) + δ` (perturbation stays intact → probes post-filter sensitivity)
- **PC-band (phase-coherent bandlimiting)** preprocessing:
  offline estimation of an effective bandwidth + cached smooth roll-off mask.

---

## PC-band method (offline → cache)
1. Compute FFT/STFT over training data.
2. Apply amplitude gating (phase is unreliable at low magnitude).
3. Measure inter-frame **phase coherence** per frequency bin (phasor coherence).
4. Derive dataset-level cutoff Ω_max < π and build a smooth roll-off mask W(Ω)
   (optionally with a guard band near π).
5. Preprocess once and cache → **no per-step overhead**.

---

## Results (Preliminary)

### 1) sCIFAR10 / CIFAR10: accuracy vs frequency (ρ sweep)
Setting: `s4d | cifar10 | target relΔ=0.05 | preproc=lpf(test) | track=filter_then_pert`

![CIFAR10 rho sweep](assets/eval_lpf070_072_track2_rel005_s4d_cifar10_rho_sweep_acc.png)

Observation: clean accuracy stays stable; perturbed accuracy drops sharply near **high ρ** (near Nyquist).

---

### 2) DTD96: PC-band improves mean best test accuracy (3 seeds)
S4D, seeds 0/1/2 (preliminary). PC-band keep_ratio=0.90 improves mean best test accuracy.

| Setting | keep_ratio | Mean Best Test (%) | Δ vs Raw (pp) |
|---|---:|---:|---:|
| Raw | -- | 36.42 | 0.00 |
| PC-band | 0.90 | 37.41 | +0.99 |

> If you want the per-seed breakdown in the README, add the run logs (or a CSV summary) and we can auto-generate the table.

---

### 3) sCIFAR10 sanity check (seed 0)
This is mainly to validate the preprocessing pipeline end-to-end; effects are small here.

| Setting | Best Val (%) | Test @ Best Val (%) | Best Test (%) |
|---|---:|---:|---:|
| Raw | 89.08 | 88.69 | 88.78 |
| PC-band (keep_ratio=0.80) | 89.16 | 88.54 | 88.58 |

---

## Repro (minimal notes)
- Track definitions:
  - `filter_both`: `F(u+δ)`
  - `filter_then_pert`: `F(u)+δ`
- See scripts/flags in the repo for dataset + model configs.

---

## Status
Work in progress. Next steps:
- Complete keep_ratio sweep for PC-band.
- Add stability metrics (loss spikes, NaN/Inf, grad-norm percentiles).
- Compare S4D vs Transformer on multiple datasets.
