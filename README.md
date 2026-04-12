# PH-Sobolev: Sample-Adaptive Frequency Boost for State Space Models

## Overview

State space models (SSMs) exhibit an implicit spectral bias toward low-frequency components. Even with full spectral coverage (e.g., S4D-DFouT initialization), the *learning dynamics* remain biased: high-frequency parameters receive weaker gradient signal due to the 1/f spectral decay of natural inputs.

**PH-Sobolev** addresses this by applying a Sobolev-weighted frequency boost *selectively* to topologically stable spectral peaks identified via persistent homology (PH), rather than uniformly across all frequencies.

### Key Ideas

1. **Sample-adaptive PH mask**: For each input, H₀ superlevel persistence identifies stable spectral peaks and their frequency support regions
2. **Selective Sobolev boost**: The filter `(1+f²)^β` is applied only to PH support bins; noise bins remain at identity
3. **Forward-consistent**: The filter is applied in the forward pass, producing well-defined gradients via standard autograd
4. **γ–β guideline**: The input spectral slope γ predicts the optimal boost exponent: `β ≈ γ/2`

### Theoretical Results

- **Proposition**: Selective boost yields lower gradient variance than uniform boost: `tr(D_PH Σ D_PH*) ≤ tr(D_uni Σ D_uni*)`
- **SNR Corollary**: When PH support aligns with signal-rich bins, gradient SNR improves
- **PH Stability**: Peaks with persistence > 2ε survive ε-perturbations (bottleneck stability theorem)
- **Nested Support**: α₂ ≥ α₁ ⟹ S(α₂) ⊆ S(α₁), giving support_alpha a structural interpretation

## Installation

```bash
git clone <repo-url>
cd topo_s4

# Dependencies
pip install torch torchvision torchaudio numpy matplotlib --break-system-packages

# S4D model (place in models/s4/)
# Requires the S4D implementation from https://github.com/state-spaces/s4
```

## Quick Start

### 1. Measure spectral slope γ

```bash
python empirical_bridge.py --task lra_image --experiment gap2 \
    --cifar_root ./data/cifar --cifar_download \
    --out_dir ./analysis/gamma
```

This outputs `input_spectral_slope` = γ. Use `β ≈ γ/2` as starting point.

### 2. Run training

```bash
python s4d_ph_sobolev_runner_consistent.py \
    --task lra_image \
    --preproc ph_support \
    --paper dfout \
    --sobolev_beta 0.5 \
    --boost_layer all \
    --support_alpha 0.6 \
    --freq_cut 0.6 \
    --ph_weighting persistence \
    --ph_max_peaks 15 \
    --cifar_download \
    --amp \
    --name ph_sobolev_run \
    --out_dir runs/experiments
```

### 3. Run baselines for comparison

```bash
# No boost (baseline)
python s4d_ph_sobolev_runner_consistent.py \
    --task lra_image --preproc none --paper dfout \
    --cifar_download --amp --name baseline --out_dir runs/experiments

# Uniform Sobolev (Yu et al. style)
python s4d_ph_sobolev_runner_consistent.py \
    --task lra_image --preproc uniform --paper dfout \
    --sobolev_beta 0.5 --freq_cut 0.6 --boost_layer all \
    --cifar_download --amp --name uniform --out_dir runs/experiments
```

## SLURM Sweep

```bash
# Phase 1: Scout best β and α (single seed)
chmod +x run_ph_sobolev_fwd.sh
./run_ph_sobolev_fwd.sh

# Phase 2: Best config × 3 seeds
SEEDS="2222 3333 4444" BETAS="0.5" ALPHAS="0.6" ./run_ph_sobolev_fwd.sh

# Cross-task
TASKS="lra_image lra_pathfinder sc35" SEEDS="2222 3333 4444" BETAS="0.5" ./run_ph_sobolev_fwd.sh
```

## Analysis Scripts

### Gradient Statistics Validation

```bash
# Exp A: PH mask variance (no model needed)
python variance_analysis.py --task lra_image --experiment exp_a --cifar_download

# Exp B: Gradient variance under different boost modes
python variance_analysis.py --task lra_image --experiment exp_b \
    --boost_layer all --sobolev_beta 1.0 --cifar_download

# Exp C: Gradient direction consistency (cosine similarity)
python variance_analysis.py --task lra_image --experiment exp_c \
    --boost_layer all --sobolev_beta 1.0 --cifar_download
```

### Empirical Bridge Experiments

```bash
# Gap 1: PH persistence vs cross-sample consistency
python empirical_bridge.py --task lra_image --experiment gap1 --cifar_download

# Gap 2: Spectral bias propagation (measures γ)
python empirical_bridge.py --task lra_image --experiment gap2 --cifar_download

# Gap 3: Per-frequency gradient SNR
python empirical_bridge.py --task lra_image --experiment gap3 --cifar_download
```

## γ–β Reference Table

| Dataset | γ | Predicted β | HF Importance | Expected Effect |
|---|---|---|---|---|
| sCIFAR-10 | 0.702 | 0.35 – 0.53 | Low (shape/color) | Marginal |
| SC35 | 1.080 | 0.54 – 0.81 | High (phonemes) | Strong |
| Pathfinder | 0.298 | 0.15 – 0.22 | Low (spatial) | Minimal |

**Rule of thumb**: `β ≈ γ/2` for uniform, `β ∈ [γ/2, 0.75γ]` for PH-selective.

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--preproc` | `none` | `none` / `uniform` / `ph_support` |
| `--sobolev_beta` | 1.0 | Boost exponent. Set via γ/2 guideline |
| `--boost_layer` | `last` | `all` (recommended) or `last` (no effect in practice) |
| `--support_alpha` | 0.0 | PH support width: 0=widest, 0.6=validated, 1.0=tightest |
| `--freq_cut` | 0.25 | Normalized freq cutoff. Bins ≤ freq_cut are not boosted. Use 0.6 |
| `--ph_weighting` | `binary` | `binary` or `persistence` (recommended) |
| `--ph_max_peaks` | 15 | Maximum PH peaks to consider |

## File Structure

```
s4d_ph_sobolev_runner_consistent.py  # Main runner (forward-consistent)
variance_analysis.py                  # Gradient statistics validation (Exp A/B/C)
empirical_bridge.py                   # Empirical analysis (Gap 1-4, γ measurement)
run_ph_sobolev_fwd.sh                 # SLURM sweep script
collect_sweep.py                      # Sweep result aggregation
run_sweep.sh                          # Variance sweep script
models/s4/s4d.py                      # S4D model (external dependency)
```

## Citation

```bibtex
@article{kim2025phsobolev,
  title={Sample-Adaptive Frequency Boost for State Space Models via Persistent Homology},
  author={Kim, Dongyeong},
  year={2025}
}
```

## Acknowledgments

This work builds on:
- S4/S4D: Gu et al. (2022)
- S4D-DFouT: Solozabal et al. (2025)
- Tuning Frequency Bias: Yu et al. (2024)
