# Explainable AI Under Distribution Shift

**Research protocol (pre-registration)** · Visual roadmap: [docs/roadmap.html](docs/roadmap.html)

> **Pre-registration status: FROZEN on 2026-11-25** — this is the first public commit (see the
> [Protocol Change Log](#protocol-change-log)). From here on, any change to the primary hypothesis, primary metric
> or evaluation protocol is logged with a date and a reason. Nothing is edited silently. Values marked **[freeze]**
> are design choices fixed before the main experiments and may not be changed without a change-log entry.

## Working Paper Title

When Does Heatmap Drift Add Value for Error Detection? An Empirical Study Under Distribution Shift on CIFAR-10-C

## Research Questions

**Primary (RQ1).** Does perturbation-based local heatmap instability provide incremental information for detecting
individual classification errors under image distribution shift, beyond maximum softmax confidence and predictive
entropy, and does this information generalise to unseen corruption families?

**Secondary (RQ2).** Does aggregate attribution-profile drift track condition-level model degradation, and does it
provide information complementary to conventional performance indicators?

## Research Motivation

Masood et al. (2026) studied confidence-based monitoring and explanation-derived audits under controlled
distribution shift on tabular data. They separated two kinds of explanation monitoring:

- **Local explanation volatility:** whether one input's attribution ranking changes under small perturbations.
- **Batch-level attribution drift:** whether the aggregate attribution profile of an evaluation batch moves away
  from a clean reference batch.

They report that local volatility was a weak per-sample error detector, while batch-level drift tracked realised
error across shift conditions.

This project tests whether that distinction transfers to image classifiers under natural image corruption, and
whether either form of heatmap drift adds information **beyond** confidence, entropy and corruption severity.

> Reference: F. Masood, S. Masood, A. R. Faridi, A. Zoha, A. Khan, A. H. Shamsan. *Explanation audits reveal silent
> failures of machine learning models under distribution shift.* Discover Artificial Intelligence 6, 890 (2026).
> https://doi.org/10.1007/s44163-026-02012-6 (DOI checked against Crossref on 2026-11-23.)

Related vision work is listed in [prior_work.md](prior_work.md). The two closest studies have been checked (see
the verified notes there). Hovakimyan & Bravo (2026) show that Grad-CAM degrades under drift, but do not use that drift
for error detection. Kalita & Singh (2026, arXiv 2610.36640) test output-level perturbation consistency beyond softmax
confidence on CIFAR-10-C. Its gain vanished on unseen shift families, and they use no attributions. Whether
**explanation-level** instability adds failure information on unseen families is the open question here.

---

## Primary Hypothesis

**H1.** Operational local heatmap drift (defined in [§ Local drift](#local-heatmap-drift-track-a)) adds
incremental information for detecting individual classification errors on **unseen corruption families**, beyond
maximum softmax confidence and predictive entropy.

*Test:* **E4 vs E3** (see table below), evaluated with leave-one-corruption-family-out.

*Supported if:* the pooled held-out ΔAUROC = AUROC(E4) − AUROC(E3) has a cluster-bootstrap 95% CI that excludes
zero **and** is positive in at least 3 of the 4 held-out families. Primary configuration: ResNet-18, Integrated
Gradients, predicted-class target, averaged over 3 training seeds.

*Not supported if:* the CI includes zero, or the gain appears in fewer than 3 families. This outcome will be
reported as a negative result.

## Secondary Hypotheses

- **H2.** Batch-level attribution drift is positively associated with condition-level error rate (Spearman ρ > 0 across the 75 corruption conditions, CI excluding zero).
- **H3.** Local drift still adds information after accounting for corruption severity (**E6 vs E5**).
- **H4.** The incremental value of drift differs between Integrated Gradients and Grad-CAM (H1 test repeated per method; difference in ΔAUROC with CI).
- **H5.** Rejecting by the E4 score gives lower selective risk than rejecting by the E3 score at 90%, 80%, 70% and 60% coverage (risk-coverage curves, AURC).
- **H6.** Local heatmap drift adds information beyond **output-level** perturbation consistency (**E8 vs E7**). This tests whether heatmap drift is more than output instability in disguise (cf. Kalita & Singh, 2026).

## Monitoring Experiments (Track A — per image)

All scores are logistic regressions on standardised features (scikit-learn `LogisticRegression`, L2, C = 1.0)
**[freeze]**, except E1 and E2, which are used as raw scores.

| ID | Inputs | Role |
|----|--------|------|
| E1 | Confidence (max softmax) | Standard baseline |
| E2 | Entropy | Alternative baseline |
| E3 | Confidence + entropy | **Conventional baseline** for H1 |
| E4 | Confidence + entropy + local drift | **H1 model** |
| E5 | Confidence + entropy + severity | Severity-adjusted baseline |
| E6 | Confidence + entropy + severity + local drift | Full model (H3) |
| E7 | Confidence + entropy + output consistency (LS, CD) | Output-consistency baseline |
| E8 | Confidence + entropy + output consistency + local drift | Drift beyond output instability (H6) |

Each pair being compared differs by **one** feature group (E4 − E3, E6 − E5 and E8 − E7 all differ by drift alone).
E7 is reported against E3 as well, to replicate Kalita & Singh's output-consistency result on this setup. If E6 does not beat
E5, drift is most likely measuring the corruption itself rather than adding information about failure.

**Severity is an oracle variable.** At deployment the corruption severity is unknown, so E5 and E6 are control
analyses, not deployable monitors. The corruption *type* is not used as a feature, because the held-out family is
unseen by construction.

Track B (batch-level drift against condition-level error rate) is analysed separately. A batch drift score is
shared by every image in its batch, so it is never used to rank individual images.

---

## Local Heatmap Drift (Track A)

### Primary: operational drift (no clean image needed)

For an observed (possibly corrupted) image *x*, let *A(x)* be its attribution map and *T₁ … T_K* be random
perturbations of the same image. The drift score is

```
D(x) = (1/K) · Σ_k  d( A(x), A(T_k(x)) )
```

| Component | Choice **[freeze]** |
|---|---|
| Perturbation *T_k* | Additive Gaussian pixel noise, σ = 0.01 in [0, 1] pixel space, clipped to [0, 1]. This is 4× smaller than CIFAR-10-C's mildest Gaussian-noise severity (σ = 0.04). |
| Number of perturbations *K* | 8 |
| Random seeds for *T_k* | Fixed per image (seed = image index), identical across models and methods |
| Attribution target | The class predicted for **x** (the same target for all *K* perturbed copies) |
| Map *A* | Absolute attribution summed over colour channels → 32 × 32 map |
| Distance *d* (primary) | 1 − Spearman rank correlation between the two flattened maps |
| Distances (secondary) | Normalised L1 (maps scaled to sum to 1), cosine distance |
| Aggregation | Mean over the *K* perturbations |

*Sensitivity analysis (reported, not used to choose the primary setting):* σ ∈ {0.005, 0.01, 0.02} and
K ∈ {4, 8, 16}. If the H1 conclusion flips within this grid, that is reported as a limitation.

*Why this design:* drift must be computable at inference time. Perturbations are kept much milder than the
benchmark corruptions so the score measures the model's local explanation stability, not its sensitivity to an
extra artificial corruption.

### Output-consistency baseline (same perturbations)

Computed from the **same K = 8 perturbed copies** used for drift, so it needs no extra forward passes. Let *ŷ* be
the class predicted for *x* and *p(·)* the softmax output:

| Feature | Definition **[freeze]** |
|---|---|
| Label stability LS(x) | Fraction of the K copies whose predicted class equals ŷ |
| Confidence deviation CD(x) | Mean over k of \|p_ŷ(x) − p_ŷ(T_k(x))\| |

These two features follow the idea of Kalita & Singh (2026). It is our implementation, not a reproduction of their
code: they use six operators, and we use only the Gaussian-noise one (σ = 0.01), so that drift and consistency
see identical inputs.

### Secondary: diagnostic drift (needs the clean image)

`D_diag(x) = d( A(x_clean), A(x_corrupted) )`, with the same *d* and target rule. This measures explanation
degradation and is used to interpret results. It is **never** used as a monitoring feature, because the clean
image does not exist at deployment.

### Attribution methods **[freeze]**

| Method | Settings |
|---|---|
| Integrated Gradients (primary) | Captum, black-image baseline (all zeros in [0, 1] space), 32 steps, Gauss–Legendre |
| Grad-CAM (secondary, CNN only) | `layer4` output (4 × 4), ReLU, bilinear upsampling to 32 × 32 |

Before any drift analysis, both methods must pass the model-parameter randomisation sanity check (Adebayo et al., 2018).

---

## Batch-Level Drift (Track B)

| Component | Choice **[freeze]** |
|---|---|
| Per-image profile | Absolute IG map summed over channels, average-pooled to an **8 × 8 grid**, normalised to sum to 1 |
| Batch profile *P* | Mean of the per-image profiles in the batch, plus ε = 1e-8, renormalised |
| Clean reference *P_clean* | 1,000 images from the CIFAR-10 **validation** split (100 per class, fixed indices). The same images are used for every seed; the profile is recomputed for each model. |
| Evaluation batch | For each of the 75 conditions: the corrupted versions of the 2,000-image explanation subset (below), which is disjoint from the reference |
| Variability | 200 bootstrap batches of 500 images per condition |
| Drift | Jensen–Shannon **distance** (base 2), `D_JS(P_clean, P_condition)` |
| Outcome | Error rate of the condition on the full 10,000 corrupted test images |
| Sensitivity | Class-stratified profiles (one reference per predicted class, averaged JS); 4 × 4 and 16 × 16 grids |

*Analysis:* Spearman ρ between drift and error rate across the 75 conditions. Complementarity is tested by a
condition-level regression error ~ mean confidence + JS drift, compared with error ~ mean confidence, using
leave-one-family-out prediction of condition error.

---

## Data Splits and Leakage Controls

### CIFAR-10 **[freeze]**
- Train is split 45,000 / 5,000 (train / validation), seed 0. The indices are saved with each checkpoint.
- Model selection uses validation accuracy only. The clean test set is evaluated once per trained model.

### Explanation subset **[freeze]**
- 2,000 CIFAR-10 test images, stratified 200 per class, fixed indices (seed 0).
- Split into two disjoint halves of 1,000 by **clean image identity**: half **F** (fitting) and half **H** (held-out evaluation).
- Every corrupted version of an image inherits its half. This stops the same underlying image from being used to fit a monitor and to evaluate it.

### Corruption families **[freeze]** (as in `src/data.py`)

| Family | Corruptions |
|---|---|
| Noise | gaussian_noise, shot_noise, impulse_noise |
| Blur | defocus_blur, glass_blur, motion_blur, zoom_blur |
| Weather | snow, frost, fog, brightness |
| Digital | contrast, elastic_transform, pixelate, jpeg_compression |

### Leave-one-corruption-family-out (primary evaluation) **[freeze]**

| Fold | Fit on (half F) | Evaluate on (half H) |
|---|---|---|
| 1 | Noise, weather, digital | Blur |
| 2 | Blur, weather, digital | Noise |
| 3 | Noise, blur, digital | Weather |
| 4 | Noise, blur, weather | Digital |

- Each fold fits E3–E8 on half-F images from the three training families (all 5 severities), then evaluates on half-H images from the held-out family.
- Held-out predictions from the 4 folds are pooled for the primary metric and also reported per fold.
- A random split of corrupted images is **not** used as a generalisation test.

---

## Statistical Analysis **[freeze]**

- **Unit of resampling:** clean image identity (a cluster contains every corrupted version of one image). 2,000 cluster-bootstrap resamples of half H.
- **Paired comparisons:** ΔAUROC and ΔAUPRC are computed on the same resampled images for both models.
- **Seeds:** 3 training seeds (0, 1, 2). The primary estimate is the mean across seeds; per-seed results are reported. A conclusion that holds in fewer than 2 of 3 seeds is reported as unstable.
- **Prevalence:** the error rate is reported for every evaluation set, next to AUPRC.
- **Multiplicity:** H1 is the only confirmatory test. H2–H6 are reported with CIs and labelled as secondary.

## Evaluation Metrics

- AUROC and AUPRC (error detection), with error prevalence
- AURC and selective risk at 90 / 80 / 70 / 60% coverage
- Expected Calibration Error (15 equal-width bins) and Brier score
- Local drift: Spearman distance (primary), normalised L1, cosine distance
- Batch drift: Jensen–Shannon distance

## Experimental Setup

- Primary dataset: CIFAR-10 / CIFAR-10-C (15 corruptions × 5 severities)
- Generalisation dataset: CIFAR-100-C (after the primary study is complete)
- Primary architecture: ResNet-18 (CIFAR variant: 3×3 stem, no max-pool)
- Additional architecture: ViT-Tiny (after the primary study is complete)
- Primary attribution method: Integrated Gradients; secondary: Grad-CAM (CNN only)
- Primary attribution target: **predicted class**. Ground-truth-class attribution is a secondary diagnostic.

## Main Ablations

1. E3 vs E4: drift beyond confidence + entropy (H1).
2. E5 vs E6: drift beyond severity as well (H3).
3. Predicted-class vs ground-truth-class attribution.
4. Integrated Gradients vs Grad-CAM (H4).
5. E7 vs E8: drift beyond output-level consistency (H6).
6. Operational vs diagnostic drift (interpretation only).
7. CNN vs Vision Transformer (extension).
8. CIFAR-10-C vs CIFAR-100-C (extension).

## Order of Work

1. Freeze this protocol (first public commit).
2. Train ResNet-18 (3 seeds); evaluate all 75 CIFAR-10-C conditions.
3. Implement IG and Grad-CAM; pass the randomisation sanity check.
4. Compute local and batch drift (kept separate).
5. Run the incremental-value comparisons with leave-one-family-out.
6. Only then: ViT-Tiny and CIFAR-100-C.

## Scientific Integrity

The primary hypothesis and evaluation protocol are frozen before the main experiments.
Negative results will be reported. No claim of incremental predictive value will be made unless it is supported by
held-out results and appropriate uncertainty estimates.

## Protocol Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2026-11-23 | Initial draft | — |
| 2026-11-23 (pre-freeze) | Experiments renumbered E1–E6. Added the confidence + entropy baseline (E3). H1 is now E4 vs E3, H3 is E6 vs E5. | H1 claimed "beyond confidence and entropy" but the old E3 only controlled for confidence. Each comparison now differs by the drift feature alone. |
| 2026-11-23 (pre-freeze) | Defined operational local drift (perturbation-based) as primary and clean-vs-corrupted drift as diagnostic only. | The clean image is not available at deployment. |
| 2026-11-23 (pre-freeze) | Added leave-one-corruption-family-out evaluation and an image-identity split (halves F / H). | Kalita & Singh (2026) report that a comparable output-level signal did not generalise to unseen shift families. The image split prevents leakage through shared clean images. |
| 2026-11-23 (pre-freeze) | Defined the batch attribution profile, clean reference, batch construction and JS distance. | "JS drift" was underspecified. |
| 2026-11-23 (pre-freeze) | Specified cluster bootstrap, seed rule, prevalence reporting and H1 decision rule. | Make H1 falsifiable in advance. |
| 2026-11-23 (pre-freeze) | Added output-consistency features LS and CD (same K perturbations), experiments E7 / E8 and secondary hypothesis H6. H1 is unchanged. | Full-text check of Kalita & Singh (2026): output consistency is the nearest competing explanation for any drift gain. |
| **2026-11-25** | **Protocol frozen — first public commit.** Baseline already trained (3 seeds, mean 94.1% clean test accuracy) and CIFAR-10-C verified before freezing; no confirmatory experiment has been run. | Freeze the analysis plan before the drift experiments (Weeks 4+). |

---

## Repository Layout

```
src/models.py            ResNet-18 (CIFAR variant) with normalisation inside the model
src/data.py              CIFAR-10 loaders, fixed train/val split, CIFAR-10-C loader, corruption families
src/train_resnet18.py    Baseline training (Week 1)
src/verify_cifar10c.py   Checks CIFAR-10-C files and evaluates a checkpoint on one corruption
prior_work.md            Literature comparison table
docs/roadmap.html        Visual roadmap: stages, tech stack, diagrams, 12-week plan
```

The model takes inputs in **[0, 1] pixel space** and normalises internally. Integrated Gradients baselines
(e.g. a black image), the drift perturbations and the corruptions therefore all live in the same space as the raw images.

## Week 1 Quick Start

```bash
pip install -r requirements.txt

# Train the baseline (about 35–60 min on a T4 or RTX 3050, 40 epochs)
python src/train_resnet18.py --epochs 40 --seed 0

# CIFAR-10-C: download CIFAR-10-C.tar from https://zenodo.org/records/2535967 (~2.9 GB) and extract to data/CIFAR-10-C/
python src/verify_cifar10c.py --c10c-dir data/CIFAR-10-C --corruption gaussian_noise --ckpt checkpoints/resnet18_seed0.pt
```

On Windows with several Python versions installed, use `py -3.12` instead of `python`. On Google Colab, set
*Runtime → Change runtime type → T4 GPU*, clone the repository, and run the same commands prefixed with `!`.
