# PH-Conditioned S4D: Operator-Level Persistent-Homology Conditioning for State Space Models

## Overview

State space models (SSMs) are linear-time-invariant (LTI) by construction, so
any per-sample modulation must enter through either (i) a scalar
frequency-domain gain on the layer output or (ii) a modification of the SSM
kernel itself.

Our earlier work, **PH-Sobolev**, explored (i): a persistent-homology (PH)--guided
Sobolev filter applied to each S4D layer's output. That forward-mask approach
was theoretically clean (Parseval-based gradient decomposition, variance bound,
bottleneck stability) but produced only a +0.09pp improvement on sCIFAR-10, for
a *structural* reason: a scalar per-sample gain acts on an already-learned
global output scale, and batch-averaged gradients absorb the per-sample
variation back into LTI parameters.

This project, **PH-Conditioned S4D**, develops (ii) instead. A lightweight PH
encoder produces a per-sample code *z*, and per-layer zero-init adapter heads
map *z* to an additive kernel correction *ΔC*, giving each sample its own
SSM kernel `C_eff = C + ΔC(x)`. The mechanism is complementary to—and cannot
be collapsed into—any scalar forward mask.

The PH-Sobolev work motivates the pivot to operator-level conditioning and is
retained in the paper as the starting point and in this codebase as a baseline.

### Why operator-level?

Writing an S4 layer's output as ŷ(ω; x) = K(ω) · û(ω; x):

- **Forward mask**: `ŷ = H(ω;x) · K(ω) · û(ω;x)`. Scalar gain on existing
  output. Under batch gradient averaging, absorbed into the layer's existing
  output scale. Expressive power ≤ baseline + per-sample scalar reweighting.
- **Operator-level (ours)**: `ŷ = (K(ω) + ΔK(ω;x)) · û(ω;x)`. The kernel
  itself differs per sample, and the adapter parameters do not share a global
  output scale with the SSM kernel. Per-sample gradient signal is preserved.

### Key findings

Moving the conditioning signal from the output to the operator exposes three
learning-dynamics pathologies that the forward-mask setting did not have:

1. **Encoder rank collapse.** Standard PointNet-style set encoders
   (φ → maxpool → ρ MLP + LayerNorm) drift toward rank-1 in depth. The encoder
   code *z* loses per-sample diversity before reaching the adapter.
2. **z magnitude shrinkage.** Even with a rank-repaired encoder, ||z|| shrinks
   3–7× during training while the adapter's bias inflates, so per-sample
   signal is replaced by a learned global shift.
3. **C_head rank-1 drift.** The per-layer weight matrix W in each
   C_head: ℝ^64 → ℝ^64 drifts toward rank-1 under plain training, so every
   sample receives kernel correction in the same direction regardless of *z*.

All three are diagnosable by short 20-epoch audits and repairable by single
architectural or regularization changes. The current best configuration
combines:

- **PH-native pooling**: parameter-free persistence-weighted sum over peaks
  replaces maxpool. Every peak receives gradient, blocking the φ drift
  observed under maxpool (where φ-per-peak cosine rises from 0.5 to 0.8
  across training).
- **Maxpool-only encoder**: no post-pool learnable layers. The pool output
  *is* z, sidestepping z magnitude shrinkage.
- **Bias-free zero-init C_head**: removes the bias that absorbs per-sample
  capacity into a global kernel shift.
- **Scale-invariant orthogonality penalty** on C_head W: shapes the
  singular-value distribution toward uniformity without pushing |W| to
  grow or shrink.

## Installation

```bash
git clone <repo-url>
cd topo_s4

pip install torch torchvision torchaudio numpy matplotlib --break-system-packages

# S4D model (place in models/s4/)
# Requires the S4D implementation from https://github.com/state-spaces/s4
```

## Quick Start

### 1. Train the current best-believed configuration

```bash
python phase4_train_ph_conditioned.py \
    --runner ./s4d_ph_sobolev_runner_consistent.py \
    --task lra_image --paper dfout \
    --stage C --seed 2222 \
    --cifar_root ./data/cifar \
    --out_dir ./runs/phase4v9b_persweighted_orth \
    --support_alpha 0.6 \
    --ph_input_dropout 0.1 --ph_modality_dropout 0.15 \
    --ph_transform_persistence --ph_transform_freq \
    --ph_recode_support --ph_standardize \
    --ph_arctan_constant 4.0 \
    --no_head_bias \
    --ph_encoder_type maxpool_only \
    --ph_pool_type persistence_weighted \
    --chead_orth_coef 0.0005 \
    --epochs 200 \
    --save_every_epoch 20
```

### 2. Baseline (no PH conditioning) for comparison

```bash
python phase4_train_ph_conditioned.py \
    --runner ./s4d_ph_sobolev_runner_consistent.py \
    --task lra_image --paper dfout \
    --stage baseline --seed 2222 \
    --cifar_root ./data/cifar \
    --out_dir ./runs/phase4_baseline \
    --epochs 200
```

### 3. Run structural audits on checkpoints

```bash
# Stage-by-stage pairwise cosine (encoder rank collapse diagnostic)
python phase4d_encoder_collapse_audit.py \
    --runner ./s4d_ph_sobolev_runner_consistent.py \
    --ckpt ./runs/phase4v9b_persweighted_orth/ckpt_ep20.pt \
    --cifar_root ./data/cifar --support_alpha 0.6 \
    --out_json ./runs/phase4v9b_persweighted_orth/audit_ep20.json

# C_head weight + output structure (rank-1 drift diagnostic)
python phase4e_chead_weight_audit.py \
    --runner ./s4d_ph_sobolev_runner_consistent.py \
    --ckpt ./runs/phase4v9b_persweighted_orth/ckpt_ep20.pt \
    --cifar_root ./data/cifar --support_alpha 0.6 \
    --out ./runs/phase4v9b_persweighted_orth/chead_audit_ep20.json
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--stage` | `baseline` | `baseline` / `dt` / `C` / `dt_C`. `C` is the primary setting used in this paper. |
| `--ph_encoder_type` | `full` | `full` (φ + maxpool + proj + ρ_branch) or `maxpool_only` (φ → pool → z, no post-pool layers). |
| `--ph_pool_type` | `maxpool` | `maxpool` / `persistence_weighted` / `mean`. The v9b default is `persistence_weighted`. |
| `--no_head_bias` | off | Remove bias in C_head (forces ΔC = Wz). |
| `--chead_orth_coef` | 0.0 | Coefficient for the scale-invariant orthogonality penalty on C_head W. `5e-4` is the current best-believed value. |
| `--ph_transform_persistence` | off | arctan(4·p) transform on persistence. |
| `--ph_transform_freq` | off | log1p(f) transform on peak frequency. |
| `--ph_recode_support` | off | Replace (left, right) with (center, log1p(width)). |
| `--ph_standardize` | off | Per-coordinate z-score on transformed PH features. |
| `--support_alpha` | 0.0 | PH support width: 0 widest (at death level), 1 tightest (at birth). |
| `--ph_max_peaks` | 15 | Maximum PH peaks to consider. |
| `--save_every_epoch` | 0 | If > 0, save a checkpoint every N epochs for trajectory analysis. |

## Ablation Ladder

The configuration search proceeded as follows (all 20-epoch checkpoints on
sCIFAR-10; 200-epoch runs in progress):

| Variant | encoder | pool | C_head | test (ep20) |
|---|---|---|---|---|
| Baseline (no PH)                 | --             | --          | --                | 80.67 |
| v7 full                          | proj + ρ_br    | max         | bias on           | 80.55 |
| v7 full + no bias                | proj + ρ_br    | max         | no bias           | (diagnosed) |
| v8 no bias + orth 5e-2           | proj + ρ_br    | max         | no bias + orth    | 79.11 |
| v8 no bias + orth 5e-3           | proj + ρ_br    | max         | no bias + orth    | (diagnosed) |
| v9 maxpool-only                  | φ only         | max         | no bias           | (running) |
| v9 maxpool-only + orth 5e-4      | φ only         | max         | no bias + orth    | (running) |
| **v9b persistence-weighted + orth** | **φ only**  | **pers-w**  | **no bias + orth** | **(running)** |

The baseline 20-epoch number (80.67) is the fair comparison target at this
training budget; the published S4D-DFouT baseline test accuracy at 200 epochs
is 89.18.

## File Structure

```
s4d_ph_conditioned.py                # Model: PHEncoder + per-layer PHModulationHead
phase4_train_ph_conditioned.py       # Training script (adds CLI for all ablations)
phase4b_diagnose.py                  # Loader helpers + PH heads vs kernel diagnostic
phase4d_encoder_collapse_audit.py    # Stage-by-stage pairwise cosine diagnostic
phase4e_chead_weight_audit.py        # C_head W SVD + Wz/b + z statistics diagnostic
models/s4/s4d.py                     # S4D model (external dependency)

# Older PH-Sobolev code, kept as the forward-mask baseline
s4d_ph_sobolev_runner_consistent.py  # Original forward-mask runner
variance_analysis.py                 # Gradient statistics for the forward-mask paper
empirical_bridge.py                  # γ measurement, gap analyses
```

## Status

- Theory: operator-level vs forward-mask separation, scale-invariant
  orthogonality penalty, backward-only non-integrability result.
- Structural diagnostics: rank collapse, z shrinkage, and W rank-1 drift
  identified, each repaired in turn.
- End-to-end training: 20-epoch checkpoint numbers collected for the full
  ablation ladder; 200-epoch runs for the current best configuration are
  in progress.

## Retained Theory from PH-Sobolev

The Parseval-based gradient decomposition, the selective-vs-uniform variance
bound, the SNR corollary, and the bottleneck stability corollary from the
PH-Sobolev paper all transfer to the operator-level setting (see the
Appendix of the current paper draft). In particular:

- Gradient magnitude at each frequency bin is the product of the residual
  error and the input energy at that bin (Parseval + chain rule).
- PH support identifies bins where input energy stably exceeds the noise
  floor, so boosting outside this set amplifies only noise-level gradient.

The γ–β guideline (`β ≈ γ/2`) is retained as a diagnostic:
measure γ on the input spectrum; if γ is near zero, expect no task room
for improvement. Reference values: sCIFAR γ ≈ 0.70, SC35 γ ≈ 1.08,
Pathfinder γ ≈ 0.30.

## Acknowledgments

This work builds on:

- S4/S4D: Gu et al. (2022)
- S4D-DFouT: Solozabal et al. (2025)
- Tuning Frequency Bias of SSMs: Yu et al. (2024)
- PH-Sobolev (prior work, this project): the forward-mask starting point
