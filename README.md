# E4 initial-setting addendum

## CIFAR-10 RGB clean + CIFAR-C summary

| Exp | Best dev | Epoch | Test@best-dev | Best test | Final test | CIFAR-C mean | Severity-5 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 91.32 | 175 | 90.86 | 91.42 | 91.41 | 74.36 | 60.30 |
| E1 | 91.42 | 195 | 91.38 | 91.43 | 91.36 | 77.15 | 64.29 |
| E4 | 91.66 | 191 | 91.30 | 91.40 | 91.36 | 75.57 | 61.63 |

## Main interpretation

- The initial full-length **E4** run is **clearly better than E0**, but it **does not yet beat E1** on the main held-out criteria.
- Relative to E0, E4 improves **test@best-dev** by **0.44pt** and **CIFAR-C mean** by **1.21pt**.
- Relative to E1, E4 is **-0.08pt** lower on **test@best-dev** and **-1.58pt** lower on **CIFAR-C mean**.
- E4 has the **highest best-dev accuracy** among the three (**91.66%**), but its **test@best-dev** is only **91.30%**, so the default setting is not yet the strongest main-line model.

## Robustness detail

- Severity-wise, E4 stays between E0 and E1 at every CIFAR-C severity.

| Severity | E0 | E1 | E4 | E4-E0 | E4-E1 |
|---|---:|---:|---:|---:|---:|
| 1 | 85.03 | 86.59 | 85.96 | 0.93 | -0.62 |
| 2 | 80.91 | 83.15 | 82.16 | 1.24 | -0.99 |
| 3 | 75.91 | 78.80 | 77.31 | 1.41 | -1.49 |
| 4 | 69.65 | 72.91 | 70.79 | 1.14 | -2.12 |
| 5 | 60.30 | 64.29 | 61.63 | 1.32 | -2.66 |

- E4 improves most over E0 on **gaussian_noise, shot_noise, pixelate, zoom_blur, and frost**.
- E4 still trails E1 most on **gaussian_noise, glass_blur, and shot_noise**.

## Controller dynamics from the initial E4 run

- For the first **31 epochs (epochs 1-31)**, the dev-side effective lambda stayed below **0.05**, so the controller behaved very close to the identity path.
- After that transition, E4 moved to a moderate operating regime; over the last 20 epochs the dev-side effective lambda averaged **0.316** and the actual pass ratio averaged **0.695**.
- Late in training, the train-side sample-wise variation also became very small (last-20 mean train lambda-batch std **0.0060**, delta-lambda-batch std **0.0058**), so the initial E4 run behaves more like a lightly adaptive near-global controller than a strongly varying per-sample controller.
- Optimization is not fully smooth yet: the maximum recorded total gradient norm reached **85.90** during early training, much higher than E0 (**9.52**) or E1 (**10.66**).

## Why this matters for the write-up

- The current write-up should **not** say that E4 has already replaced E1.
- A better phrasing is: **E4 remains the most promising sample-wise direction, and the initial full-length RGB run confirms that it is viable and improves over E0, but it has not yet surpassed the strong E1 baseline.**
- This also fits the partial tuning sweep: the current full-length run uses the default **top-5 / gamma=1.0 / l2** setting, while the short incomplete sweep suggests there is room to improve the E4 operating point further.
