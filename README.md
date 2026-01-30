# PH-Band (WIP): Phase-Coherent Bandlimiting for Stable S4 Training

This repo explores a simple idea:

> **Near-Nyquist digital frequencies (Ω → π) can become numerically over-sensitive under bilinear/Tustin discretization (frequency warping), which may destabilize S4-family training.**  
> We mitigate this by estimating a **phase-coherent “effective bandwidth”** from data and applying a cached, smooth low-pass mask.

## Motivation (high level)
Bilinear/Tustin warping maps discrete to continuous frequency as
\[
\omega(\Omega)=\frac{2}{\Delta}\tan(\Omega/2), \quad
\frac{d\omega}{d\Omega}=\frac{1}{\Delta}\sec^2(\Omega/2),
\]
so sensitivity grows rapidly as Ω approaches π. If inputs contain strong near-Nyquist components, training can become unstable.

## Method (offline → cache)
1. Compute STFT (or FFT) over training data.
2. Use amplitude gating (phase is unreliable at low magnitude).
3. Measure inter-frame **phase coherence** per frequency bin (phasor coherence).
4. Derive a dataset-level cutoff \( \Omega_{\max}<\pi \) and build a **smooth roll-off** mask \(W(\Omega)\),
   optionally with a fixed **guard band** near π.
5. Preprocess inputs once and cache → **no per-step overhead**.

## Experiments (planned / in progress)
- Sequential CIFAR (sCIFAR10/100) + controlled synthetic near-Nyquist injection.
- Metrics: accuracy + stability (NaN/Inf, loss spikes, grad-norm percentiles) + throughput.
- Ablations: raw, fixed LPF, energy cutoff, PH cutoff, hard vs smooth, guard-only vs PH(+guard).

Status: **Work in progress** (solo project). Results/preprint will be added later.
