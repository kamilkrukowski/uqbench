# Roadmap

## Image-scale shift study (CIFAR-10 → CIFAR-10-C)

**Goal.** Promote the image track from a plumbing prototype to a headline result: the same method matrix and metric suite as the toy study, run on competitive CNN backbones, evaluated under corruption shift in the style of Ovadia et al. (2019).

**Why it isn't done yet.** The current backbones ([`experiments/results/*_cifar10_*`](experiments/results)) are small convnets trained on a downsampled subset and reach only ~30–37% accuracy. Calibration metrics on a classifier that is wrong two-thirds of the time are not a meaningful demonstration, so these are kept as a pipeline check, not presented as results.

### Definition of done
- Deterministic CNN backbone at **≥ 90% clean CIF-10 test accuracy** (competitive baseline).
- Full method matrix evaluated on **clean + CIFAR-10-C** (15+ corruptions × 5 severities).
- An **Ovadia-style figure**: accuracy / NLL / ECE vs. corruption severity (mean over corruptions, with error bands), one line per method.
- A committed `experiments/results/cifar10c_benchmark.json` (mean ± std over ≥ 3 seeds), mirroring `toy_benchmark.json`.

### Method matrix (image-adapted)
| Method | Status | Notes |
|---|---|---|
| Deterministic CNN | backbone needed | ResNet-20/18 or tuned VGG-style in Flax |
| Deep Ensemble | cheap once backbone exists | N independently-seeded CNNs — expected strong baseline |
| MC-Dropout | reuse `DropoutCNN` | tune dropout rate |
| Temperature scaling | post-hoc | trivial once logits exist |
| Last-layer Laplace | post-hoc | last-layer only at this scale (full Laplace infeasible) |
| Bayes-by-Backprop (conv) | uses `BayesianConv2D` | mean-field conv VI is unstable — budget tuning time |
| MCMC (NUTS) | **out of scope** | does not scale to CNN parameter counts; note in writeup |

### Work breakdown
1. **Data** — CIFAR-10-C loader (Hendrycks & Dietterich): the 19 corruption tensors × 5 severities + clean. Wire into `uqbench/data`.
2. **Backbone** — `scripts/train_cifar.py`: ResNet-20 in Flax, full train set, standard augmentation (random crop + flip), cosine LR, ~100–200 epochs. Checkpoint via Orbax.
3. **Method wrappers** — extend `method_registry` with CNN-backed entries (ensemble, dropout, temp-scale, last-layer Laplace, BBB-conv) sharing the toy interface.
4. **Eval harness** — `scripts/eval_cifar10c.py`: run the metric suite per (corruption, severity), aggregate, write JSON. Add an OOD split (SVHN or CIFAR-100) for OOD-AUROC.
5. **Plot** — `scripts/plot_cifar10c.py`: severity-vs-metric curves from the JSON (same pattern as `plot_pareto.py`).
6. **README** — replace the Roadmap note with the results section once the above lands.

### Compute & constraints
- **GPU required.** Training ≥ 5 CNNs plus MC inference over CIFAR-10-C's ~950k eval images is not feasible on CPU. Estimate ~1–2 GPU-hr per backbone, ensemble ×N, plus the corruption sweep (the dominant cost). Multi-seed multiplies this.
- The toy study runs on CPU; this track is the reason a GPU box is needed.

### Risks
- **BBB-conv instability / posterior collapse** (already observed in the toy BBB) — may need KL warmup and prior tuning, or honest reporting of failure.
- **MC-dropout calibration** sensitive to dropout rate under shift.
- CIFAR-10-C eval cost — may sample a corruption subset for iteration speed, but report the full set for the final number (and log any subsetting, no silent caps).

---

## Smaller follow-ups
- Mutual-information / BALD as an epistemic-only OOD score (current OOD-AUROC uses total predictive entropy, which under-credits the posterior methods — see the OOD discussion).
- CI: GitHub Actions running `pytest` on the calibration metrics + a green badge.
- API: implement the FastAPI predictive-uncertainty endpoint (with epistemic/aleatoric decomposition) or remove it from the package and description.
