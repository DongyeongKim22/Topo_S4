# Sample-wise controller follow-up addendum

## 0. Status of last week's RGB CIFAR result (unchanged)

The RGB CIFAR result from last week still provides the cleanest **confirmed** story for the inverse-tan line: **E1 remains the strongest completed global baseline**, and the earlier **E4** run is still best interpreted as a viable sample-wise pilot rather than a replacement for E1.

| Exp | Best dev | Epoch | Test@best-dev | Best test | Final test | CIFAR-C mean | Severity-5 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 91.32 | 175 | 90.86 | 91.42 | 91.41 | 74.36 | 60.30 |
| E1 | 91.42 | 195 | 91.38 | 91.43 | 91.36 | 77.15 | 64.29 |
| E4 | 91.66 | 191 | 91.30 | 91.40 | 91.36 | 75.57 | 61.63 |

**Carry-over interpretation.**  
E1 is still the strongest confirmed RGB clean+robust model. E4 remains competitive and improves over E0, but it still does not beat E1 on the main held-out criteria. So this part of the write-up should stay cautious.

## 1. New long-run follow-up in the paper setting

The newer evidence below comes from the long-sequence paper setting rather than the RGB CIFAR-C suite. It is therefore best read as **controller-selection / mechanism evidence**, not as a direct replacement of the RGB main table.

### 1.1 Flattened CIFAR-10 (\texttt{lra_image}, single seed)

| Exp | Controller | Best dev | Epoch | Test@best-dev | Best test | Final test |
|---|---|---:|---:|---:|---:|---:|
| E0 | Raw / no preprocessing | 86.72 | 189 | 85.34 | 85.69 | 85.52 |
| E1 | Global learned \(\lambda\) | 86.44 | 186 | 85.63 | 86.00 | 85.85 |
| E2 | PH lifetime rule | 85.28 | 191 | 84.92 | 84.99 | 84.95 |
| E3 | PH lifetime + peak rule | 84.92 | 185 | 84.57 | 85.11 | 85.09 |
| E5 | PH-tiny-MLP \(\Delta\lambda\) | 86.54 | 199 | 85.63 | 85.78 | 85.63 |
| E6 | Spectrum-MLP \(\Delta\lambda\) | 86.56 | 190 | 85.69 | 85.93 | 85.83 |

### 1.2 Pathfinder (single seed)

| Exp | Controller | Best dev | Epoch | Test@best-dev | Best test | Final / last test |
|---|---|---:|---:|---:|---:|---:|
| E0 | Raw / no preprocessing | 93.36 | 193 | 93.36 | 93.40 | 93.40 |
| E1 | Global learned \(\lambda\) | 93.27 | 196 | 93.11 | 93.21 | 93.19 |
| E5 | PH-tiny-MLP \(\Delta\lambda\) | 92.18 | 192 | 92.25 | 92.28 | 92.16* |

\* The uploaded Pathfinder E5 log ends at epoch 197, so the E5 row is based on the visible portion of that run rather than a completed epoch-199 summary.

## 2. Main interpretation

The new follow-up weakens the old “sample-wise PH head is the next main model” story.

On flattened CIFAR-10, the gap between the useful runs is very small. Relative to E0, **E1 improves test@best-dev by +0.29**, **E5 also improves by +0.29**, and **E6 improves by +0.35**. That is not enough to claim a meaningful sample-wise advantage, especially because **E2** and **E3** are clearly worse.

More importantly, the current PH-conditioned head does not show convincing PH dependence. In **E5**, the shuffled-PH evaluation gives the **same best-dev and the same test@best-dev** as the normal evaluation. So the current PH-MLP controller is not yet extracting a measurable benefit from the PH input itself.

Pathfinder points in the same direction. In the currently available logs, **E0 remains best on best-dev and test@best-dev**, while **E5** stays below both **E0** and **E1** in the visible part of the run. So there is still no positive evidence that the current PH-conditioned scalar controller improves generalization in the paper setting.

The strongest positive statement that remains is therefore narrower: **the inverse-tan line itself still looks useful as a stabilizing / filtering prior, but the current sample-wise scalar controllers do not yet show a clear interpretability or accuracy gain over the simpler baselines.**

## 3. Controller dynamics pointing to collapse

The current failure mode looks less like “PH is useless” and more like **scalar-\(\lambda\) controller collapse**.

On flattened CIFAR-10:

- **E5** ends at \(\lambda=0.1769\) with \(\text{base\_lambda}=2.1769\), so \(\Delta\lambda \approx -2\).
- **E6** ends at \(\lambda=0.2658\) with \(\text{base\_lambda}=2.2658\), so \(\Delta\lambda \approx -2\) again.

On Pathfinder:

- The visible end of **E5** is \(\lambda \approx 7.6790\) with \(\text{base\_lambda} \approx 5.6790\), so \(\Delta\lambda \approx +2\).

So the current sample-wise heads are not behaving like rich per-sample controllers. They are mostly behaving like **boundary-seeking scalar perturbations** around a learned base \(\lambda\).

This also explains why “filtering helped training, but test accuracy stayed similar” does **not** yet imply a clean interpretability win. A reversible inverse-tan front-end can still stabilize the optimization path, but if the downstream S4 kernel can reconstruct what it needs, and if the controller itself collapses to near-global or saturated behavior, then the final test gap can remain very small.

## 4. What this means for the write-up

The write-up should **not** currently say that PH-conditioned or spectrum-conditioned sample-wise controllers have already demonstrated better high-frequency interpretability.

A better phrasing is:

> Recoverable inverse-tan filtering appears to stabilize the S4D training path, but the current scalar sample-wise controllers mostly collapse to near-global or boundary-saturated solutions. The next step is therefore not to claim that PH is already stronger, but to redesign the controller so that PH peak information can act locally rather than only through a single scalar \(\lambda\).

This makes the current evidence much more coherent:
- global inverse-tan remains the strongest confirmed result,
- current sample-wise controllers do not yet beat the bottleneck,
- and the next method should target the bottleneck directly.

## 5. Next method direction: PH-peak residual boost

The cleanest next move is to **keep the global inverse-tan low-pass prior**, but add a **localized PH-peak release / boost** instead of relying on a single scalar \(\Delta\lambda\).

One useful form is

\[
W(\omega;x)=W_{\mathrm{inv}}(\omega;\lambda_{\mathrm{global}})
+\bigl(1-W_{\mathrm{inv}}(\omega;\lambda_{\mathrm{global}})\bigr)\,B_{\mathrm{PH}}(\omega;x),
\]

where \(B_{\mathrm{PH}}(\omega;x)\in[0,1]\) is a sum of narrow bumps centered at the top-\(k\) PH peak frequencies of sample \(x\).

Why this is a better next hypothesis:

1. It keeps the **global inverse-tan stability prior** that already seems useful.
2. It avoids forcing all PH information through **one scalar**.
3. It can reopen gradients in frequency regions that are effectively dead under the current scalar controller.
4. It gives a clearer interpretability target: **global high-frequency suppression with local PH-guided release**.

## 6. Recommended headline for this week

A safe and strong summary sentence is:

> The current evidence confirms inverse-tan filtering as a useful stabilizing prior, but does not support a meaningful gain from the current PH- or spectrum-conditioned scalar sample-wise controllers; the next step is a PH-peak localized residual boost that preserves global stability while releasing selected high-frequency bands.

