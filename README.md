# Topo_S4: Reliability-Weighted PH Conditioning for S4D/DFouT

## Current framing

The current version of this project is no longer centered on a learnable PH
encoder plus learnable `C_head` adapter. The main paper framing is now:

> **DFouT provides frequency coverage; PH provides sample-specific reliable-frequency selection.**

S4D/DFouT already places poles across a broad Fourier grid, so the main issue is
not simply missing high-frequency basis functions. The remaining problem is that
high-frequency evidence is mixed with noise-like fluctuation, while optimization
and gradient SNR are still biased toward low and mid frequencies. The role of PH
is therefore not to add a large learnable module. Its role is to select
**topologically persistent and phase-reliable high-frequency supports** and use
them to reweight the existing S4D/DFouT `C` residues per sample.

The current main method is a deterministic, zero-parameter gate:

```text
magnitude FFT → H0 superlevel PH → half-height support
              → group-delay reliability post-weighting
              → PH-support / S4D-pole overlap
              → high-band contrastive z-score
              → train-set mean residual
              → per-sample C-residue gate
```

One-sentence project summary:

> **DFouT provides the spectral basis; PH identifies stable high-frequency supports; group-delay reliability selects phase-coherent speech peaks; mean-subtracted contrastive C-gating turns these into zero-parameter sample-specific spectral conditioning.**

## What changed from the previous README

The previous README described **operator-level PH conditioning with a learnable
PH encoder and learnable zero-init `C_head`**. That branch remains important as a
historical diagnostic path, but it is no longer the main result.

Current interpretation:

- **PH-Sobolev forward mask**: useful theory and baseline; too easy for the
  scalar per-sample gain to be absorbed into the LTI output scale.
- **Learnable PH encoder + `C_head`**: exposed valuable learning-dynamics
  pathologies, but added trainable machinery that could collapse, shrink, or
  become a global shift.
- **Current main method**: zero-parameter PH/GD post-weighted residual gate on
  `C`, with train-set mean subtraction to isolate sample-specific PH evidence.

The older pathologies are now treated as lessons for why the final method should
be deterministic and residue-level:

1. **Encoder rank collapse**: post-pool MLP/LayerNorm erased per-sample PH
   diversity.
2. **`z` magnitude shrinkage**: even a repaired encoder could shrink the
   conditioning code while bias or weight scale absorbed the signal.
3. **`C_head` rank-1 drift**: the adapter matrix could map diverse `z` values
   into nearly the same kernel-correction direction.

## Method

### 1. Spectrum and 1D superlevel `H0` PH

For each sample `x_i`, compute a 1D FFT magnitude spectrum:

```text
X_i(ω) = FFT[x_i](ω)
m_i(ω) = log(1 + |X_i(ω)|)
```

Run 1D superlevel-set `H0` persistent homology on `m_i`. In this setting, PH is
essentially a stable spectral peak detector. Each peak `k` has

```text
(ν_{i,k}, b_{i,k}, d_{i,k}, p_{i,k}, I_{i,k})
p_{i,k} = b_{i,k} - d_{i,k}
```

where `ν` is peak frequency, `b` is birth, `d` is death, `p` is persistence, and
`I` is the connected support interval. The current main method uses half-height
support:

```text
θ_{i,k} = d_{i,k} + α_s (b_{i,k} - d_{i,k})
I_{i,k}(α_s) = {ω : m_i(ω) ≥ θ_{i,k}}
α_s = 0.5
```

### 2. Group-delay reliability post-weighting

Magnitude PH finds stable peaks, but magnitude alone cannot always distinguish
speech structure from noise-like artifacts. The current method keeps the PH
topology from magnitude, then reweights persistence using group-delay reliability
inside each PH support:

```text
p_eff_{i,k} = p_{i,k} · [max(r_GD_{i,k}, f)]^γ
r_GD_{i,k} = Agg_{ω ∈ I_{i,k}} R_GD_i(ω)
```

Main SC35 setting:

```text
Agg = support_mean
γ = 1.0
f = 0.25
```

This preserves the magnitude-PH support while trusting phase-coherent peaks more
than phase-unstable peaks.

### 3. PH support to S4D pole overlap

For each S4D/DFouT pole `n` with normalized frequency `ω̃_n` and bandwidth
surrogate `b_n`, compute Lorentzian overlap with PH support:

```text
M_{i,k,n} = 1/π · [atan((r_{i,k} - ω̃_n)/(b_n + ε))
                 - atan((l_{i,k} - ω̃_n)/(b_n + ε))]
```

The sample-pole PH evidence score is

```text
s_{i,n} = Σ_k p_eff_{i,k} M_{i,k,n}
```

### 4. Contrastive high-band log-gate

The method does not apply a positive-only high-frequency boost. Instead, it
redistributes evidence within the high band. For high-band pole set `H`, compute

```text
s̄_i = mean_{n ∈ H} s_{i,n}
σ_i = std_{n ∈ H} s_{i,n}
z_{i,n} = tanh((s_{i,n} - s̄_i)/(σ_i + ε))
```

Then apply a smooth high-band window and Sobolev-style frequency weight:

```text
h_n(c, τ) = sigmoid((ω̃_n - c)/τ)
q_n = (1 + ω̃_n)^β
ℓ_{i,n} = α q_n h_n z_{i,n}
```

Main SC35 setting:

```text
c = 0.25
τ = 0.10
α = 0.75
β = 0.5
```

The measured SC35 spectral slope was approximately `γ_spectrum ≈ 1.1375`, so
`β ≈ γ_spectrum / 2 ≈ 0.5688`; the current `β = 0.5` is close to this estimate.

### 5. Train-set mean residual C gate

Early real-vs-shuffle diagnostics showed that a raw PH gate could carry a
strong dataset-level spectral prior. The current method removes this prior by
subtracting the train-set mean log-gate per pole:

```text
μ_n = E_{i ∈ train}[ℓ_{i,n}]
r_{i,n} = ℓ_{i,n} - μ_n
g_{i,n} = exp(ρ r_{i,n})
C^{(i)}_n = clip(g_{i,n}, g_max) C_n
```

Main SC35 setting:

```text
ρ = 1.5
g_max = 3.0
n_ph_params = 0
```

Final conditioning rule:

```text
C^{(i)}_n = clip(exp(ρ(ℓ_{i,n} - μ_n)), g_max) · C_n
```

## Why condition `C`, not `A` or `dt`?

DFouT already provides pole placement and frequency coverage. Strongly changing
`A` or pole frequencies with PH is therefore redundant and can destabilize the
frequency basis. Modulating `dt` also mixes frequency warping with memory length,
so the sign of the intervention is less interpretable.

The `C` residue is the clean target:

```text
K_i(Ω) = Σ_n g_{i,n} C_n η_n v_n(Ω)
```

PH does not create new poles. It answers:

> **Which existing frequency modes should be emphasized for this sample?**

## Main empirical picture

### SC35: current main result

| Method | PH params | Reliability | Filtration | Test | Interpretation |
|---|---:|---|---|---:|---|
| S4D/DFouT baseline | 0 | none | none | 95.9109 test@best-dev | strong baseline |
| Magnitude PH residual | 0 | magnitude persistence | magnitude | modest | sample signal present but weak |
| **v19 GD post-only** | **0** | `p · r_GD` | magnitude | **96.075** | **current main result** |
| v20 GD filtration-only | 0 | none post | `m + λ log R_GD` | 95.729 | real > shuffle, but real ≈ zero |

The most important evidence is not only final accuracy; it is the eval-time
causal diagnostic. For v19 at the ep30 checkpoint on SC35:

| Mode | Test accuracy |
|---|---:|
| real PH | 94.648 |
| shuffled PH | 94.386 |
| zero PH | 94.366 |

Gaps:

```text
real - shuffle = +0.262 pp
real - zero    = +0.282 pp
```

This is the strongest current evidence that the PH signal is sample-aligned on
SC35 rather than merely acting as a dataset-level prior.

### v20: why filtration-only is secondary

v20 modifies the landscape before PH:

```text
m̃_i(ω) = m_i(ω) + λ_GD log(R_GD_i(ω) + ε)
```

With `λ_GD = 0.25`, the setting is aggressive. At the ep40 checkpoint:

| Mode | Test accuracy |
|---|---:|
| real PH | 95.729 |
| shuffled PH | 95.427 |
| zero PH | 95.702 |

The real-shuffle gap survives (`+0.302 pp`), but the real-zero gap nearly
vanishes (`+0.027 pp`). Interpretation:

> **Phase filtration can create sample alignment, but perturbing the PH landscape
> too strongly can erase the benefit. Post-weighting is more robust.**

### CIFAR/sCIFAR: negative transfer is task-alignment evidence

Flattened 1D FFT PH is not well aligned with image semantics. CIFAR/LRA-image
needs spatial layout, edge orientation, texture locality, and object shape; a
global flattened 1D spectrum discards too much of that structure.

| Method | Test | Interpretation |
|---|---:|---|
| LRA-image S4D/DFouT baseline none | 89.18 | strong baseline |
| `cifar_v17_resid15_max3` | 87.38 | residual signal appears, but under baseline |
| `cifar_v18_resid125_cut005_phdrop03` | 88.03 | improved, still under baseline |

Conclusion:

> **Speech is a native 1D spectral task; flattened grayscale CIFAR is not.**

Future image versions should use 2D FFT radial/angular PH, local patch PH, edge
PH, or spatial/texture-aware topological summaries rather than flattened 1D FFT
PH.

## Real / shuffle / zero diagnostic

For a checkpoint, evaluate four modes:

```text
real:     f(x_i, PH_i)
shuffle:  f(x_i, PH_{π(i)})
zero:     f(x_i, 0)
none:     f(x_i, no PH branch)
```

Interpretation table:

| Pattern | Meaning |
|---|---|
| `real > shuffle > zero` | sample-specific PH signal |
| `real ≈ shuffle > zero` | distributional PH prior |
| `real ≈ shuffle ≈ zero` | PH ignored |
| `zero > real` | PH perturbation cost exceeds benefit |

v19 on SC35 currently shows `real > shuffle` and `real > zero`.

## Hyperparameter calibration

The PH cache is dataset-specific, so several hyperparameters can be calibrated
from train-set PH/GD statistics.

### Spectral slope for `β`

```text
E(f) = E_i[|X_i(f)|²]
log E(f) ≈ -γ log(1 + f) + b
β_auto ≈ γ / 2
```

For SC35, `γ ≈ 1.1375`, so `β_auto ≈ 0.5688`; the current `β = 0.5` is close.

### GD power and floor

```text
w_{i,k} = p_{i,k} [max(r_GD_{i,k}, f)]^γ
η_M = E_i [Σ_k w_{i,k} / (Σ_k p_{i,k} + ε)]
ESS_i = (Σ_k w_{i,k})² / (Σ_k w_{i,k}² + ε)
```

Suggested targets:

```text
η_M ∈ [0.65, 0.85]
median(ESS) ∈ [8, 16]
```

### Residual scale and gain budget

```text
u_{i,n} = ρ(ℓ_{i,n} - μ_n)
ρ_auto = log(g_95) / (Q95(|ℓ - μ|) + ε)
g_max  = exp(Q99(|u|))
```

Target `Q95(|u|) ≈ log(1.4–1.6)` and keep clipping fraction small.

### Weak phase-filtration lambda

The v20 `λ = 0.25` setting is too aggressive for the main claim. A safer rule is
peak-preserving calibration:

```text
m̃(ω) = m(ω) + λ a(ω),  a(ω) = log(R_GD(ω) + ε)
p̃_k ≈ p_k + λ[a(ν_k) - a(d_k)]
λ_auto = Q0.2 [ η p_k / (a(d_k) - a(ν_k) + ε) ]
```

Use this only for the secondary weak-filtration ablation.

## Quick-start configuration target

The exact script name may differ by branch. The current main scientific
configuration should match the following target settings:

```bash
python <train_script.py> \
  --task sc35 \
  --paper dfout \
  --condition_target C \
  --ph_gate c_residue \
  --ph_learnable_params 0 \
  --ph_support magnitude_h0 \
  --support_alpha 0.5 \
  --ph_gd_post_weight \
  --ph_gd_power 1.0 \
  --ph_gd_floor 0.25 \
  --ph_gd_agg support_mean \
  --ph_pole_overlap lorentzian \
  --ph_high_cut 0.25 \
  --ph_high_tau 0.10 \
  --sobolev_beta 0.5 \
  --ph_gate_alpha 0.75 \
  --ph_residual_mean train \
  --ph_residual_scale 1.5 \
  --ph_gain_max 3.0
```

Baseline comparison target:

```bash
python <train_script.py> \
  --task sc35 \
  --paper dfout \
  --condition_target none
```

Eval-time causal diagnostic target:

```bash
python <eval_script.py> --ckpt <checkpoint.pt> --ph_mode real
python <eval_script.py> --ckpt <checkpoint.pt> --ph_mode shuffle
python <eval_script.py> --ckpt <checkpoint.pt> --ph_mode zero
python <eval_script.py> --ckpt <checkpoint.pt> --ph_mode none
```

## File structure

The repository contains both historical and current research code. The exact
entry-point names may differ by branch. The existing legacy files are still
useful because they document the pivot away from learned PH adapters. The current
main method should be organized around PH/GD caching, deterministic pole scoring,
real/shuffle/zero evaluation, and C-residue gating.

Known legacy / diagnostic files from the previous branch:

```text
s4d_ph_conditioned.py               # Learnable PHEncoder + per-layer PHModulationHead
phase4_train_ph_conditioned.py      # Older operator-level adapter training script
phase4d_encoder_collapse_audit.py   # Encoder pairwise-cosine diagnostic
phase4e_chead_weight_audit.py       # C_head SVD + Wz/b diagnostic
s4d_ph_sobolev_runner_consistent.py # Original forward-mask PH-Sobolev baseline
variance_analysis.py                # Gradient variance analysis for PH-Sobolev
empirical_bridge.py                 # γ measurement and gap analyses
```

Suggested current-code responsibilities, regardless of filename:

```text
FFT magnitude + H0 PH cache          # sample-wise supports and persistence
Group-delay reliability cache        # phase-derived reliability over frequency
PH-to-pole overlap module            # Lorentzian support/pole overlap
Mean-residual C-gate module          # ℓ_i,n - μ_n and clipped exp gate
Eval-mode diagnostic                 # real / shuffle / zero / none
Calibration utilities                # β, GD floor/power, residual scale, high cut
```

## Roadmap

1. **v19 auto-calibrated GD post-only replication**  
   Fix the current main method and repeat with calibrated `β`, GD floor/power,
   residual scale, and high-band cutoff.

2. **v21 cepstral envelope/fine additive stream**  
   Keep GD post-only as the main stream and add cepstral envelope/fine PH as
   additive evidence, not multiplicative filtering.

3. **v20 weak calibrated filtration**  
   Replace aggressive `λ = 0.25` with peak-preserving weak filtration, likely in
   the `0.05–0.10` range after calibration.

4. **Three seeds on SC35**  
   Run v19 and the most promising v21 variant across three seeds before making a
   strong paper claim.

5. **Class-conditional shuffle diagnostic**  
   Compare real, same-class shuffle, and different-class shuffle to separate
   class-level spectral signatures from instance-level hard-example cues.

## Project materials

- Project page: <https://dongyeongkim22.github.io/Topo_S4/>
- Overleaf draft: <https://www.overleaf.com/read/qvvrpygjhbvv#15a2ec>
- GitHub repository: <https://github.com/DongyeongKim22/Topo_S4>

## Acknowledgments

This work builds on:

- S4/S4D: Gu et al. (2022)
- S4D-DFouT: Solozabal et al. (2025)
- Tuning Frequency Bias of SSMs: Yu et al. (2024)
- PH-Sobolev: the earlier forward-mask starting point for this project
