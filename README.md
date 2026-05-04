# Topo_S4: Cubical-GUDHI PH-Guided Sobolev C-Modulation for DFouT

## Current framing

The current version of this project is no longer centered on 1D FFT PH, group-delay
post-weighting, or a learnable PH encoder / `C_head` adapter. The main paper
framing is now:

> **DFouT provides frequency coverage; 2D cubical PH provides sample-specific
> frequency selection.**

S4D/DFouT already places poles across a broad Fourier grid. The remaining issue is
not simply the absence of high-frequency basis functions, but the difficulty of
selecting which frequency regions are reliable for each sample. Uniform Sobolev
or high-frequency boosting can amplify useful speech structure and noise together.
The role of PH is therefore to identify **topologically persistent frequency
regions** in the time-frequency plane and use those regions to selectively
modulate the existing DFouT/S4D `C` residues.

The current main method is deterministic and zero-parameter:

```text
waveform
→ log-mel spectrogram
→ 2D cubical filtration
→ GUDHI cubical persistence pairs and birth/death cofaces
→ persistence-weighted coface saliency S_PH(f,t)
→ frequency marginal s_PH(f)
→ top-P PH-salient pseudo-peaks
→ mel-bin to Hz/Nyquist mapping
→ PH-to-DFouT-pole Lorentzian overlap
→ contrastive Sobolev C-residue gate
```

One-sentence project summary:

> **DFouT supplies the spectral basis; cubical PH localizes persistent
> time-frequency topology; coface saliency converts that topology into
> PH-salient frequency pseudo-peaks; and a zero-parameter Sobolev C-gate
> selectively modulates DFouT residues.**

## Current empirical status

The current accuracy numbers are **20% effective-data SC35 results**, not the
final full-dataset claim. Full 100% dataset experiments will be added separately.

| Method | Setting | Best epoch | Train acc | Dev acc | Test acc | Time |
|---|---|---:|---:|---:|---:|---:|
| DFouT-init baseline | lr=0.001 | 30 | 98.71 | 90.70 | 88.78 | 53.6m |
| PH-guided C/Sobolev | P96, support=3.0, beta=0.5, lr=0.001 | 39 | 99.36 | **91.03** | **89.21** | 107.4m |
| PH-guided C/Sobolev | P96, support=3.0, beta=1.0, lr=0.001 | 34 | 99.24 | 90.78 | 89.10 | 93.6m |

Current 20%-data interpretation:

```text
beta=0.5 PH-guided C/Sobolev vs baseline:
  dev  +0.33 percentage points
  test +0.43 percentage points

beta=1.0 PH-guided C/Sobolev vs baseline:
  dev  +0.08 percentage points
  test +0.32 percentage points
```

The current main choice is therefore:

```text
P = 96
support_bins = 3.0
Sobolev beta = 0.5
lr = 0.001
effective training data = 20%
```

## Why the framing changed

Previous project versions tried several PH-conditioning routes:

- **1D FFT PH**: useful class signal, but incremental gain on top of S4D/DFouT
  was unstable.
- **PH-Sobolev forward mask**: useful theoretical baseline, but easy for a
  scalar per-sample gain to be absorbed into the LTI output scale.
- **Learnable PH encoder + `C_head`**: exposed valuable learning-dynamics
  pathologies, but added trainable machinery that could collapse or become a
  global shift.
- **Logit residual / feature FiLM**: showed PH alignment, but did not reliably
  improve baseline generalization.

The current method moves PH to the operator level while keeping it
zero-parameter. It uses 2D log-mel cubical PH to locate persistent
time-frequency topology, then modulates the existing DFouT `C` residues.

## Method

### 1. Log-mel cubical filtration

For each waveform `x_i`, compute a log-mel spectrogram:

```text
X_i(f,t) ∈ R^{F×T}
```

Then normalize it per sample:

```text
X̃_i(f,t) = (X_i(f,t) - min X_i) / (max X_i - min X_i + ε)
```

This normalized 2D image defines the cubical filtration. The filtration is the
topological problem on which cubical PH is computed.

Important distinction:

```text
Log-mel cubical filtration defines the 2D PH problem.
Betti curves summarize the resulting PH globally.
Cofaces recover where the persistent events happened.
```

### 2. Betti curves are global summaries

A Betti curve can be computed from the cubical PH output by counting how many
topological features are alive at each filtration threshold.

Betti curves showed that 2D log-mel PH contains class-relevant signal:

| Representation | Dev | Test |
|---|---:|---:|
| PH Betti only | 13.77 | 13.30 |
| Raw log-mel | — | 34.30 |
| Raw log-mel + PH | — | 36.57 |
| Raw + PH, global shuffle | — | 33.97 |
| Raw + PH, within-class shuffle | — | 37.60 |

However, Betti curves discard location. They tell us that persistent topology
exists, but not which frequency band produced it. That is why the current method
uses GUDHI persistence pair cofaces.

### 3. Persistence pair coface saliency

GUDHI cubical persistence can provide the birth and death coface coordinates for
each persistence pair.

For a persistence pair `k`:

```text
birth coface: u^b_{i,k} = (f^b_{i,k}, t^b_{i,k})
death coface: u^d_{i,k} = (f^d_{i,k}, t^d_{i,k})
persistence: p_{i,k} = |birth - death|
```

The method deposits persistence mass at those cofaces to form a saliency map:

```text
S_PH_i(f,t) = sum_k p_{i,k} [K((f,t)-u^b_{i,k}) + K((f,t)-u^d_{i,k})]
```

Then it collapses time to get a frequency saliency curve:

```text
s_PH_i(f) = sum_t S_PH_i(f,t)
```

Interpretation:

> `s_PH_i(f)` is not a raw spectral peak curve. It is the frequency marginal of
> persistent 2D time-frequency topology.

### 4. PH-salient pseudo-peaks

The current direct-PH C-modulation path expects pseudo-peaks:

```text
(pf, pp, sl, sr)
```

where:

- `pf`: normalized center frequency
- `pp`: PH saliency weight
- `sl`, `sr`: left and right support boundaries

The current main setting keeps the top 96 pseudo-peaks:

```text
ph_max_peaks = 96
```

This does **not** mean 96 raw GUDHI persistence pairs. It means the top 96
frequency pseudo-peaks after constructing a lossless frequency pseudo-peak cache.

### 5. Hz/Nyquist frequency mapping

Earlier versions used normalized mel-bin coordinates:

```text
pf = mel_bin / (n_mels - 1)
```

The current method maps mel-bin locations to physical frequency first:

```text
mel bin → mel center Hz → Hz / Nyquist
```

Support intervals are mapped the same way:

```text
mel-bin ± support_bins → Hz → Hz / Nyquist
```

Current main setting:

```text
cubical_freq_map = hz_nyquist
cubical_pseudo_support_bins = 3.0
```

### 6. PH-to-DFouT-pole overlap

For pseudo-peak `k` with support `[sl_{i,k}, sr_{i,k}]` and DFouT pole `n` with
normalized frequency `ω_n`, compute a Lorentzian overlap:

```text
M_{i,k,n} =
  1/π · [atan((sr_{i,k} - ω_n)/(b_n + ε))
       - atan((sl_{i,k} - ω_n)/(b_n + ε))]
```

Pole evidence is:

```text
e_{i,n} = Σ_k pp_{i,k} M_{i,k,n}
```

This turns PH-salient frequency regions into pole-wise evidence.

### 7. Contrastive PH-guided Sobolev C-gate

The gate is contrastive within the high band:

```text
z_{i,n} = tanh((e_{i,n} - mean_H(e_i)) / (std_H(e_i) + τ_s + ε))
```

A smooth high-band window and Sobolev factor are applied:

```text
h_n = sigmoid((ω_n - c) / τ_c)
q_n = (1 + ω_n)^β
ℓ_{i,n} = α q_n h_n z_{i,n}
```

Current main values:

```text
direct_ph_alpha = 0.10
direct_ph_peak_sobolev_beta = 0.5
direct_ph_high_cut = 0.15
direct_ph_cut_tau = 0.10
direct_ph_score_tau = 0.05
direct_ph_max_gain = 1.25
```

The final residue modulation is:

```text
C^{(i)}_n = g_{i,n} C_n
```

with `g_{i,n}` derived from the PH-guided Sobolev log-gate.

## Why condition `C`, not `A` or `dt`?

DFouT already provides pole placement and broad Fourier coverage. Strongly
changing `A` or pole frequencies with PH is redundant and can destabilize the
basis. Modulating `dt` also mixes frequency warping with memory length.

The `C` residue is the clean target because it controls how much each existing
pole contributes:

```text
K_i(Ω) = Σ_n g_{i,n} C_n η_n v_n(Ω)
```

PH does not create new poles. It answers:

> **Which existing frequency modes should be emphasized for this sample?**

## Uniform Sobolev vs PH-guided Sobolev

The current evidence does not support the claim that PH-guided Sobolev simply
maximizes Fisher separation. Instead, PH-guided Sobolev is best interpreted as a
selective high-frequency modulation that improves downstream accuracy while
reducing the variance/noise amplification of a uniform Sobolev boost.

### Variance and Fisher diagnostic

| Representation | Within | Between | Fisher |
|---|---:|---:|---:|
| Raw | 15.519 | 0.728 | 0.04688 |
| Uniform Sobolev | 20.017 (+28.98%) | 0.977 (+34.30%) | 0.04881 (+4.12%) |
| PH-guided Sobolev | 17.554 (+13.11%) | 0.819 (+12.52%) | 0.04663 (-0.52%) |

Interpretation:

```text
Uniform Sobolev:
  increases class separation,
  but also strongly increases within-class variance.
  Fisher improves slightly.

PH-guided Sobolev:
  suppresses within-class variance amplification compared with uniform Sobolev,
  but also reduces between-class separation gain.
  Fisher remains approximately raw-level.
```

### Noise sensitivity diagnostic

| Representation | MSE | Mean abs |
|---|---:|---:|
| Raw | 172,076 | 2.468 |
| Uniform Sobolev | 235,691 (+36.97%) | 2.870 (+16.28%) |
| PH-guided Sobolev | 193,368 (+12.37%) | 2.619 (+6.11%) |

Interpretation:

> PH-guided Sobolev is less aggressive than uniform Sobolev. It keeps the
> useful high-frequency inductive bias while suppressing much of the noise and
> within-class variance amplification.

## P and support calibration

### P sweep

| P | Retained mass | Retained HF mass | Cosine vs full | Effective poles |
|---:|---:|---:|---:|---:|
| 8 | 0.0969 | 0.0768 | 0.4909 | 9.24 |
| 16 | 0.1853 | 0.1533 | 0.5271 | 10.09 |
| 24 | 0.2671 | 0.2306 | 0.5790 | 11.40 |
| 32 | 0.3436 | 0.3074 | 0.6351 | 12.98 |
| 48 | 0.4846 | 0.4535 | 0.7382 | 16.49 |
| 64 | 0.6127 | 0.5887 | 0.8337 | 20.14 |
| 96 | 0.8373 | 0.8282 | 0.9627 | 25.50 |
| 128 | 1.0000 | 1.0000 | 1.0000 | 28.95 |

Current interpretation:

```text
P32 is too sparse.
P128 is too broad and risks becoming dense boosting.
P96 preserves most PH/HF saliency while retaining selectivity.
```

### Support sweep with P96

| support_bins | Support coverage |
|---:|---:|
| 2.0 | 0.645 |
| 3.0 | 0.843 |
| 4.0 | 0.951 |
| 5.0 | 0.998 |
| 6.0 | 1.000 |
| 8.0 | 1.000 |

Current interpretation:

```text
support=2.0 undercovers poles.
support=3.0 gives enough coverage while preserving selectivity.
support>=4.0 starts to become too broad.
```

## Current main configuration target

```bash
python <train_script.py> \
  --task sc35 \
  --paper dfout \
  --lr 0.001 \
  --stage C \
  --c_mod_type direct_ph \
  --direct_ph_mode contrastive \
  --direct_ph_mean_mode residual \
  --direct_ph_residual_scale 1.0 \
  --direct_ph_alpha 0.10 \
  --direct_ph_static_alpha 0.0 \
  --direct_ph_peak_sobolev_beta 0.5 \
  --direct_ph_high_cut 0.15 \
  --direct_ph_cut_tau 0.10 \
  --direct_ph_score_tau 0.05 \
  --direct_ph_max_gain 1.25 \
  --direct_ph_overlap_mode lorentzian \
  --direct_ph_mean_batch_size 2048 \
  --ph_source cubical_gudhi \
  --threshold_mode fixed01 \
  --ph_norm per_sample_minmax \
  --auto_params \
  --f_min 1000 \
  --delta_f_min 50 \
  --tau_event_ms 20 \
  --sigma_noise 0.02 \
  --n_mels_cap 128 \
  --cubical_freq_map hz_nyquist \
  --cubical_saliency_topk 0 \
  --ph_max_peaks 96 \
  --cubical_peak_weight_norm none \
  --cubical_pseudo_support_bins 3.0 \
  --cubical_saliency_dims 0,1 \
  --cubical_saliency_sigma_f 2.0 \
  --cubical_saliency_norm max
```

Baseline comparison target:

```bash
python <train_script.py> \
  --task sc35 \
  --paper dfout \
  --lr 0.001 \
  --condition_target none
```

For 20% effective-data runs, the scheduler must also be corrected to the
effective subset size:

```text
scheduler_total_steps: 200000 → approximately 40000
```

## Real / shuffle / zero diagnostic

For final validation, evaluate a fixed checkpoint under:

```text
real:          f(x_i, PH_i)
global_shuffle f(x_i, PH_{π(i)})
within_class: f(x_i, PH_{π_c(i)})
zero / none:  f(x_i, 0) or baseline path
```

Interpretation table:

| Pattern | Meaning |
|---|---|
| real > none | PH contributes beyond no-PH ablation |
| real > global shuffle | sample-aligned PH frequency saliency matters |
| real > within-class shuffle | instance-level PH matters beyond class prototype |
| real ≈ global/within > none | PH mostly acts as a distributional or class prior |
| zero > real | PH perturbation cost exceeds benefit |

## Roadmap

1. **Add full-dataset SC35 results**  
   The current numbers are 20% effective-data results only. The 100% dataset
   results should be reported separately.

2. **Run core ablations**  
   Compare baseline, PH real, PH global shuffle, PH within-class shuffle, no
   Sobolev, residual center, and no-center.

3. **Evaluate P/support variants**  
   Compare P32/P96/P128 and support bins 2/3/4 under the same lr=0.001 setup.

4. **Multi-seed validation**  
   Repeat the best 20% and full-data settings over multiple seeds before making a
   strong final claim.

5. **Class-conditional shuffle diagnostic**  
   Separate instance-level PH signal from class-level topological prototypes.

## Project materials

- Project page: <https://dongyeongkim22.github.io/Topo_S4/>
- Overleaf draft: <https://www.overleaf.com/read/qvvrpygjhbvv#15a2ec>
- GitHub repository: <https://github.com/DongyeongKim22/Topo_S4>

## Acknowledgments

This work builds on:

- S4/S4D: Gu et al. (2022)
- S4D-DFouT: Solozabal et al. (2025)
- Tuning Frequency Bias of SSMs: Yu et al. (2024)
- Persistent homology and cubical persistence tooling including GUDHI
