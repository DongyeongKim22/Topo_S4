# Topo_S4 (WIP)

**Frequency-localized sensitivity** in state space models (**S4 / S4D / S4ND**) and a now-stabilized final method line: **PH-guided inverse-tan spectral normalization** (`ph_norm` / `ph_inv_tan`).

- **Project homepage:** https://dongyeongkim22.github.io/Topo_S4/
- **Overleaf draft:** https://www.overleaf.com/read/qvvrpygjhbvv#15a2ec
- **Repo:** https://github.com/DongyeongKim22/Topo_S4

## Motivation

A working hypothesis is that **near-Nyquist digital frequencies** (`Ω -> π`) can become numerically over-sensitive under bilinear/Tustin discretization due to frequency warping:

$$
\omega(\Omega)=\frac{2}{\Delta}\tan(\Omega/2), \qquad
\frac{d\omega}{d\Omega}=\frac{1}{\Delta}\sec^2(\Omega/2)
$$

so sensitivity can grow rapidly as `Ω` approaches `π`. The current question is not only whether high-frequency vulnerability exists, but whether a **sample-wise controller** can decide when attenuation is helpful and when it should simply back off to identity.

## Current direction: PH-guided inverse-tan normalization

The repository started with **dataset-level PC masks** and later moved through **dynamic PH masking**.  
The current active line fixes the method to a simpler and mathematically cleaner final form:

- compute **exact PH descriptors** on each sample's spectrum,
- predict a nonnegative normalization strength `λ` with a lightweight controller,
- apply the inverse-tan normalizer
  $$
  W_{\lambda}(\Omega)=\beta + \frac{1-\beta}{1+\lambda\tan^2(\Omega/2)},
  $$
- and feed the result to the otherwise unchanged **S4 / S4D** backbone.

Why this form?

- **Exact identity fallback:** if `λ = 0`, then `W = 1` everywhere.
- **Targeted high-frequency control:** larger `λ` suppresses near-Nyquist components more strongly.
- **Cleaner implementation-to-theory match:** exact PH features + a one-parameter inverse-tan controller align the code path more closely with the intended mathematical guarantee.
- **Cleaner story:** the method is now best viewed as **adaptive stabilization**, not as a filter that must always improve clean accuracy.

## Current evidence (this week's snapshot)

### 1) Pathfinder32: the controller safely backs off

In the current Pathfinder32 run, the controller starts in a filtering regime but quickly collapses toward identity:

- `λ` drops from about `3.9` at epoch 0 to effectively `0` by roughly epoch 20,
- the actual pass ratio rises from about `0.34` to `~1.0`,
- and the run still reaches **>95% test accuracy** while training is ongoing.

This is a useful result. It means the controller can **decline to intervene** when strong filtering is not beneficial, rather than forcing a harmful mask.

### 2) SC35: native 1D audio keeps the controller active

On **SC35**, the controller behaves very differently. In the current run (16 kHz train/dev, 8 kHz zero-shot evaluation test), it stays active throughout training:

- final `λ(train/dev/test) ≈ 1.58 / 1.49 / 3.21`
- final `actual(train/dev/test) ≈ 0.54 / 0.55 / 0.41`

So the controller is strongest on the shifted test condition. This is the clearest current evidence that the normalization line is **task-adaptive** and not just collapsing everywhere.

### 3) Accuracy reading: mechanism validated, calibration still open

The current **mechanistic story is strong**, but the final robustness numbers are not yet where they need to be.

- **SC35 + `ph_inv_tan`**: best dev `96.06%` with linked 8 kHz test `85.63%`
- **SC35 raw baseline**: best dev `96.14%` with linked 8 kHz test `92.51%`

So the method is now behaving as intended, but the current SC35 calibration still trails the raw baseline on zero-shot 8 kHz accuracy.

## This week's progress update

**300-char summary**

> This week I finalized the PH-guided inverse-tan code path for a mathematically cleaner controller: exact PH features, exact identity fallback at λ=0, and task-adaptive normalization. Pathfinder32 already exceeds 95% while backing off to identity; SC35 keeps the controller active under 8k evaluation.

### What changed

- Fixed the project's **main method** to `ph_norm` / `ph_inv_tan`.
- Strengthened the **implementation-to-theory match** with exact PH extraction and a one-parameter inverse-tan controller that has an explicit identity guarantee.
- Reframed the contribution as **adaptive stabilization** rather than universal clean-accuracy gain.
- Confirmed two clearly different controller regimes:
  - **Pathfinder32:** identity fallback
  - **SC35:** persistent nontrivial normalization
- Moved the project closer to a final paper story: **the theory/method are mostly fixed; what remains is empirical consolidation**.

## What this repo contains

- **Frequency sensitivity probes** and simple filtering baselines from earlier stages.
- **Archived dataset-level PC masks** (`PC-band`, `PC-weight`) from the initial global-prior direction.
- **Earlier dynamic PH masking** experiments, kept as historical stepping stones.
- **Current main line: PH-guided inverse-tan normalization**
  - native **1D S4 / S4D + 1D PH**
  - controller statistics such as `λ` and actual pass ratio
  - robustness-oriented evaluation on shifted conditions

## Next steps

The remaining work is mainly empirical, not theoretical:

- finish the ongoing **Pathfinder32 200-epoch** run cleanly,
- add final **multi-seed** summaries,
- include **gradient-volatility** logs (norm mean/std/CV, max spikes, clipping rate),
- report **SC35** with both matched-rate and shifted-rate evaluation,
- tune the SC35 controller without changing the core method.

## Status

The project is still a work in progress, but the **core theory/method now looks fixed enough** to move toward final experiments and paper writing. The main remaining question is not what the method is, but how strongly and under which evaluation protocol it should be calibrated.
