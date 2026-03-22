# Topo_S4 (WIP)

**Frequency-localized sensitivity** in state space models (**S4 / S4D / S4ND**) and a now-narrowed main method family: **PH-guided inverse-tan spectral normalization** (`ph_norm` / `ph_inv_tan`).

- **Project homepage:** https://dongyeongkim22.github.io/Topo_S4/
- **Overleaf draft:** https://www.overleaf.com/read/qvvrpygjhbvv#15a2ec
- **Repo:** https://github.com/DongyeongKim22/Topo_S4

## Motivation

A working hypothesis is that **near-Nyquist digital frequencies** (`Ω -> π`) can become numerically over-sensitive under bilinear/Tustin discretization due to frequency warping:

$$
\omega(\Omega)=\frac{2}{\Delta}\tan(\Omega/2), \qquad
\frac{d\omega}{d\Omega}=\frac{1}{\Delta}\sec^2(\Omega/2)
$$

so sensitivity can grow rapidly as `Ω` approaches `π`. The current question is not only whether high-frequency vulnerability exists, but whether a **sample-wise controller** can decide when attenuation is helpful and when it should simply back off.

## Current direction: PH-guided inverse-tan normalization

The repository started with **dataset-level PC masks** and later moved through **dynamic PH masking**. The current line fixes the frequency intervention to a simpler and cleaner final form:

- compute **exact PH descriptors** on each sample's spectrum,
- predict a nonnegative normalization strength `λ` (currently through either a global scalar or a lightweight PH-conditioned controller),
- apply the inverse-tan normalizer
  $$
  W_{\lambda}(\Omega)=\beta + \frac{1-\beta}{1+\lambda\tan^2(\Omega/2)},
  $$
- and feed the result to the otherwise unchanged **S4 / S4D** backbone.

Why this form?

- **Exact identity fallback:** if `λ = 0`, then `W = 1` everywhere.
- **Targeted high-frequency control:** larger `λ` suppresses near-Nyquist components more strongly.
- **Cleaner implementation-to-theory match:** exact PH features + a one-parameter inverse-tan controller align the code path more closely with the intended mathematical guarantee.
- **Cleaner story:** the method is best viewed as **adaptive stabilization**, not as a filter that must always improve clean accuracy.

## This week's empirical snapshot

### 1) Global `λ` is a useful intermediate baseline

This week I first trained a **shared global `λ`** version of the inverse-tan normalizer. Even without sample-wise conditioning, it already produced small clean-set gains:

- **SC35:** best dev `96.57 -> 96.73`, test@bestdev `95.91 -> 96.27`
- **CIFAR10-gray:** best dev `88.50 -> 88.74`, test@bestdev `87.70 -> 88.31`

This is useful because it shows that the inverse-tan pathway can help without immediately relying on a complex controller.

### 2) CIFAR-10-C shows the clearest robustness gain

Using the best global-`λ` CIFAR10-gray checkpoint, I evaluated **CIFAR-10-C** and compared it to the raw baseline checkpoint:

- mean corruption accuracy improved from **67.21** to **69.11** (`+1.90`)
- **68 / 75** corruption-severity conditions improved
- gains increased with severity: `+1.23`, `+1.54`, `+1.80`, `+2.09`, `+2.86` from severity 1 to 5
- the strongest average gains appeared on **noise** corruptions (`+4.42`) and remained positive on blur, weather, and digital corruptions

So the global-`λ` line is not only stable; it is already a meaningful robustness baseline.

### 3) E0-E7 screening narrowed the sample-wise controller family

I then ran a **short single-seed E0-E7 screening** to test how sample-wise `λ` should be controlled:

- **E0:** Raw / no preprocessing
- **E1:** Global learned `λ`
- **E2:** PH-rule using top lifetimes only
- **E3:** PH-rule using top lifetimes + peak frequency
- **E4:** PH-linear `Δλ`
- **E5:** PH-tiny-MLP `Δλ`
- **E6:** Spectrum-MLP `Δλ`
- **E7:** Shuffled-PH sanity

On **lra_image** (flattened image-sequence screening), the best held-out test result was **E4 = 82.22**, slightly above **E6 = 82.13** and raw **E0 = 81.53**. On **SC35** screening, **E4 = 10.67** also led **E3 = 10.27** and **E1 = 10.12** on the current held-out test metric.

The pattern is informative:

- **E2** is weak, so **lifetime alone is not enough**
- **E3** recovers much of the gap, so **peak location matters**
- **E4** is currently the **best overall candidate**
- **E5** does not outperform the simpler heads yet
- **E7** causes only a modest drop, so the nonlinear controller still does not appear to exploit PH strongly enough

## Current interpretation

The main progress this week is not a final SOTA claim. It is that the controller design space has become much clearer:

- the inverse-tan normalization path is stable,
- a shared global `λ` already helps clean performance and corruption robustness,
- sample-wise control is worth pursuing,
- and the current best controller is a **low-capacity PH-linear readout** rather than a larger MLP.

That is a good outcome for the paper story because it keeps the method **interpretable, lightweight, and closer to the original PH-guided motivation**.

## What this repo contains

- **Frequency sensitivity probes** and simple filtering baselines from earlier stages.
- **Archived dataset-level PC masks** (`PC-band`, `PC-weight`) from the initial global-prior direction.
- **Earlier dynamic PH masking** experiments, kept as historical stepping stones.
- **Current main line: PH-guided inverse-tan normalization**
  - global-`λ` baseline
  - PH-rule / PH-linear / PH-MLP sample-wise controllers
  - controller statistics such as `λ` and actual pass ratio
  - robustness-oriented evaluation on shifted and corrupted conditions

## Next steps

The remaining work is now mostly empirical:

- promote **PH-linear (`E4`)** to the main sample-wise controller line,
- run **multi-seed** comparisons for `E1`, `E4`, and `E6`,
- add stronger **PH-dependence checks** (including shuffled-feature sanity beyond the current quick screen),
- extend corruption-style evaluation beyond the current CIFAR-10-C study,
- and keep logging controller behavior (`λ`, pass ratio, gradient statistics) alongside accuracy.

## Status

The project is still a work in progress, but the method family is now much narrower than before. The key remaining question is no longer *what the controller should roughly look like*; it is whether the **PH-linear sample-wise controller** remains strongest under longer training, multi-seed evaluation, and broader robustness tests.
