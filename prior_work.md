# Prior Work Comparison

Search date: 2026-11-23. Sources: Consensus (Semantic Scholar / OpenAlex index) and a Crossref check of the anchor DOI.
Queries: (1) saliency stability under common corruptions, CIFAR-10-C, Grad-CAM, IG; (2) attribution features for
misclassification detection beyond softmax confidence under shift. **Still to do:** a Google Scholar / arXiv pass with
"CIFAR-10-C" + "saliency" / "attribution" + "error detection" / "failure prediction", and a forward-citation check on
Hovakimyan & Bravo and Kalita & Singh.

> **Verification status.** This is a literature-search result, not a verified systematic review. Only Masood et al.
> has been checked beyond its index entry: Crossref metadata plus an abstract that matches the reported numbers. Every
> other row is based on the indexed abstract only. Before Hovakimyan & Bravo and Kalita & Singh are used in the novelty
> argument, read their full texts and confirm their setups, datasets and claims (tracked in the checklist below).
>
> **Update 2026-11-23:** Kalita & Singh was read in full (arXiv HTML, Sections 1–7). For Hovakimyan & Bravo, the
> landing page, abstract and code README were read; the PDF body could not be text-extracted. Details are in
> [Verified notes](#verified-notes-on-the-two-closest-papers).

| Paper | Metadata checked | Full text read | Claims confirmed |
|---|---|---|---|
| Masood et al. 2026 | ✅ Crossref | ☐ | ◐ abstract only |
| Hovakimyan & Bravo 2026 | ✅ journal page, DOI | ◐ abstract + code README; PDF body not read | ◐ no detection experiment in abstract or README |
| Kalita & Singh 2026 (arXiv: "Kalitaa") | ✅ arXiv abs page | ✅ Sections 1–7 (HTML) | ✅ see notes |
| Subramaniakuppusamy et al. 2026 (FASS) | ☐ | ☐ | ☐ |
| Others in the matrix | ☐ | ☐ | ☐ |

## Comparison matrix

Legend: ✅ = yes, ◐ = partly, ✗ = no

| Paper | Domain | Shift | Attribution | Measures drift | Tests drift → per-image error detection | Controls for confidence / severity | Batch-level drift | Relation to this project |
|---|---|---|---|---|---|---|---|---|
| **Masood et al. 2026**, *Explanation audits reveal silent failures…*, Discover AI 6:890 ([doi](https://doi.org/10.1007/s44163-026-02012-6)) | Tabular (3 real + synthetic) | Noise, masking, mean shift, shortcut reversal | Occlusion, TreeSHAP | ✅ | ✅ (AUROC 0.575, weak) | ◐ compares against confidence, no nested incremental test | ✅ JS drift, Spearman 0.68 with error | **Anchor.** We test whether its local-vs-batch split transfers to vision and add an incremental-value test. |
| **Hovakimyan & Bravo 2026**, *When Pixels Lie: The Hidden Toll of Concept Drift*, J. Operations Intelligence 4(1):65–94 ([doi](https://doi.org/10.31181/jopi41202670), [code](https://github.com/GurgenHovakimyan/hidden_toll_of_visual_drift)) | CIFAR-10, Tiny ImageNet; ImageNet-pretrained ResNet-18, DenseNet-121, ShuffleNet V2 | Own transforms: noise, blur, rotation, brightness (not CIFAR-10-C) | Grad-CAM, Grad-CAM++ (predicted class) | ✅ SSIM / IoU / Pearson / cosine / MSE, clean vs drifted heatmap of the same image | ✗ (not in abstract or README) | ✗ | ◐ pixel-distribution tests (KS, AD), diagnostic only | **Closest vision prior work.** Shows Grad-CAM degrades under drift. Does not use drift to detect errors or compare it with confidence. Must be cited; our contribution must not be "heatmaps degrade under corruption". |
| **Kalita & Singh 2026**, *When Does Perturbation Consistency Add Information Beyond Softmax Confidence?*, arXiv 2610.36640 ([abs](https://arxiv.org/abs/2610.36640)) | CIFAR-10 / CIFAR-10-C (19 corruptions), OCTMNIST, others; 14 CIFAR CNNs | CIFAR-10-C, own corruptions, PGD | **None** (label stability + confidence deviation of outputs) | ✗ (output consistency, not heatmaps) | ✅ | ✅ MSP, margin, entropy, energy, DOCTOR; cross-validated logistic combiners | ✗ | **Methodological twin for H1** using outputs instead of heatmaps. Consistency is worse than MSP on clean data, helps under noise when that noise is seen in training, and the gain **disappears on unseen shift families**. Strong reason for our leave-one-family-out design. |
| **Subramaniakuppusamy et al. 2026**, *FASS: How Stable Are Post-Hoc Attributions?*, arXiv ([link](https://consensus.app/papers/details/w:3a050ce4-a9b1-4181-bbb9-902baa4d6e74/)) | ImageNet, COCO, CIFAR-10 | Geometric, photometric, compression | IG, GradientSHAP, Grad-CAM, LIME | ✅ SSIM, rank corr., top-k Jaccard | ✗ | ✗ (filters to prediction-preserving pairs) | ✗ | Stability benchmark. Borrow its metrics. Its prediction-invariance filtering is the opposite of what we need: we want the cases where predictions change. |
| **Sarkar et al. 2025**, *Noise robustness of Class Activation Maps*, Image and Vision Computing ([link](https://consensus.app/papers/details/w:93343c94-cc2f-4aa3-a253-9ecee320464e/)) | Multiple | Noise | CAM family | ✅ consistency / responsiveness | ✗ | ✗ | ✗ | Stability metric for CAMs. Could serve as an extra drift metric. |
| **Skliarov et al. 2025**, *Comparative evaluation of explainability techniques for image data*, Sci. Reports ([link](https://consensus.app/papers/details/w:ca58f6ab-19b7-4876-b909-b2edc3ad5d21/)) | CIFAR-10, Imagenette, SVHN | Small perturbations | LIME, SHAP, Grad-CAM(++), IG, SmoothGrad | ✅ stability | ✗ | ✗ | ✗ | Reports IG more stable than Grad-CAM (FASS reports the reverse). Supports H4 that the method matters. |
| **Shin et al. 2026**, *CANARY: Catching an image classifier's silent, confident failures*, arXiv ([link](https://consensus.app/papers/details/w:4a62088e-08e0-484d-bda5-946be3c244e0/)) | ImageNet shifts, 5 backbones | Natural shifts | None (feature kNN) | ✗ | ✅ | ✅ fused with MSP | ✗ | Shows most feature-space detectors are redundant with MSP. Drift must be tested the same way. A possible strong non-heatmap baseline. |
| **Xia & Bouganis 2024**, *SIRC*, IJCV ([link](https://consensus.app/papers/details/w:74701588-d2bf-4df9-814d-881f1c9c477a/)) | ImageNet-scale | OOD | None | ✗ | ✅ selective classification | ✅ softmax + secondary score | ✗ | Template for combining confidence with a secondary score (H5). |
| **Zhu et al. 2023**, *Revisiting Confidence Estimation*, TPAMI ([link](https://consensus.app/papers/details/w:a0f8f1ec-ead0-4837-98cb-16506dfa8988/)) | CIFAR, ImageNet | Includes covariate shift | None | ✗ | ✅ | — | ✗ | Failure-prediction baselines and metrics (AURC, etc.). |
| **Ghorbani et al. 2019**, *Interpretation of Neural Networks Is Fragile*, AAAI ([link](https://consensus.app/papers/details/w:1dde519c-2b4a-4c69-bbc9-415bf3aef082/)) | ImageNet, CIFAR-10 | Adversarial | Gradients, IG, DeepLIFT | ✅ | ✗ | ✗ | ✗ | Foundational: explanations can change a lot while the label stays the same. Background for "drift ≠ error". |
| **Adebayo et al. 2018**, *Sanity Checks for Saliency Maps*, NeurIPS ([link](https://consensus.app/papers/details/w:a3e8719d-709e-494f-a1db-4b6e8f2b10f5/)) | Vision | — | Many | — | ✗ | ✗ | ✗ | Run the model-randomisation sanity check on our IG and Grad-CAM before using them. |
| **Kindermans et al. 2019**, *The (Un)reliability of Saliency Methods* ([link](https://consensus.app/papers/details/w:91e3384b-da59-46fc-844e-89c95cbfde2e/)) | Vision | Constant input shift | Many | ✅ | ✗ | ✗ | ✗ | IG depends on the baseline choice. Fix and report the IG baseline in the protocol. |

## Verified notes on the two closest papers

### Kalita & Singh (2026), arXiv 2610.36640 — read in full (Sections 1–7)

- **Authors:** Amit Kalita and Moirangthem Tiken Singh (Dibrugarh University). The arXiv metadata spells the first author "Kalitaa"; the paper body says "Kalita". Submitted 27 Oct 2026, v1, no venue. **Cite as Kalita & Singh (2026)** and keep the arXiv ID.
- **Signal:** label stability and confidence deviation of the *outputs* under six mild operators: 2° rotation, brightness ×1.1, **Gaussian noise σ = 0.01**, contrast ×1.1, 3×3 Gaussian blur (σ = 0.5), JPEG q75. **No heatmap or attribution method anywhere** in the paper.
- **Stronger operators made it worse:** consistency AUROC fell from 0.903 to 0.755 across five strength levels, while MSP stayed at 0.920.
- **Clean data:** consistency is worse than MSP. In 48 frozen-CNN runs, CS − MSP = −0.057 AUROC (95% CI −0.073 to −0.043). Adding label stability to a logistic model on MSP + margin gave +0.001.
- **CIFAR-10-C pooled (19 × 5):** CS 0.828, MSP 0.835, negative entropy 0.837.
- **Generalisation, the key result for us.** Gains are from adding consistency features to confidence features:
  - Fitted on the same corruption: +0.13 on noise.
  - Fitted on the other noise corruptions (leave-one-corruption-out): +0.067.
  - Fitted on clean data only: +0.009.
  - Fitted on all *other* families, tested on noise: **+0.002 (not significant)**.
  - Blur: −0.046. Digital: −0.012.
  - Their conclusion is that consistency helps only under a shift that has been observed and labelled.
- **Statistics:** seeds 42–44, hierarchical bootstrap (configurations → seeds → images, 500 replicates), Wilcoxon tests with Holm correction, pre-specified predictions.
- **Not stated in the text:** the fold structure of the combiners and whether CIFAR-10-C images were split by image identity.

### Hovakimyan & Bravo (2026), J. Operations Intelligence 4(1):65–94 — abstract and code README only

- **Authors:** Gurgen Hovakimyan and Jorge Miguel Bravo (NOVA IMS, Lisbon). Published 21 Sep 2026, CC BY 4.0, DOI 10.31181/jopi41202670.
- **Setup:** their own drift transforms (noise, blur, brightness, rotation) at 12 intensity levels, not CIFAR-10-C. ImageNet-pretrained backbones fine-tuned with Adam, no augmentation. Clean validation accuracy on CIFAR-10: ResNet-18 0.841, DenseNet-121 0.859, ShuffleNet V2 0.794.
- **Explanations:** Grad-CAM and Grad-CAM++ on the predicted class. Similarity (SSIM, IoU at 0.3 / 0.5 / 0.7, Pearson, cosine, MSE) is measured against the clean heatmap of the same image.
- **No error detection.** Neither the abstract nor the README uses heatmap drift to detect misclassifications, compares it with confidence or entropy, or reports AUROC. There is no batch monitoring, only pixel-level distribution tests.
- **Internal inconsistency to note:** the README says CIFAR-10 last-block Grad-CAM maps are "effectively degenerate" (final feature map about 1×1), so heatmap analysis is meaningful only on Tiny ImageNet. Yet the abstract reports CIFAR-10 SSIM / IoU falling to 18% / 25%. This supports our concern about Grad-CAM resolution on 32×32 inputs.
- **Not read:** the PDF body. A detection experiment could exist there; this is unlikely given the abstract and README, but unverified. Before submission, read the PDF directly.

## Novelty assessment (updated after reading)

- **Already done:** "heatmaps degrade under image corruption" (Hovakimyan & Bravo 2026; FASS; Sarkar et al.). This cannot be our claim.
- **Already done for outputs:** "does perturbation consistency add information beyond MSP on CIFAR-10-C, and does it transfer to unseen families?" (Kalita & Singh 2026). Their answer for **output** consistency is: barely, and not to unseen families.
- **Still open:** the same question for **attribution** consistency. Kalita & Singh use no heatmaps at all, and Hovakimyan & Bravo do no error detection.

**Conclusion:** H1 is still open, and the positioning is now sharper. Our study answers: *does explanation-level
instability carry failure information that output-level instability and confidence do not, on unseen corruption
families?* Kalita & Singh's negative transfer result is the natural prior. A negative H1 result would extend their
finding from outputs to explanations. A positive one would show that explanations carry something outputs don't.
Either outcome is publishable.

**Implications for the protocol:**
1. Our perturbation (Gaussian noise σ = 0.01) is identical to one of Kalita & Singh's operators. This makes the two studies directly comparable, and their strength sweep supports keeping it mild.
2. *(Done: README E7 / E8, H6.)* Add their output-consistency score (label stability + confidence deviation under the same K perturbations) as an **extra baseline feature**. This costs almost nothing: the K forward passes are already computed for IG. It answers the obvious reviewer question, "is heatmap drift just output instability?"
3. Our image-identity split (halves F / H) fixes a detail that Kalita & Singh leave unstated.

## Notes on practical concerns raised by prior work

1. IG needs a fixed baseline (black image is the default; record it). See Kindermans et al.
2. Grad-CAM on a CIFAR ResNet-18 uses the last block's 4×4 feature map, so it is very coarse. Expect it to look "stable" partly because it is low-resolution. Report the map resolution. Hovakimyan & Bravo's README says their CIFAR-10 last-block maps were degenerate (about 1×1). Our CIFAR-variant ResNet keeps 4×4, but consider `layer3` (8×8) as a Grad-CAM sensitivity check.
3. The drift definition needs a clean reference. At deployment, the clean version of a corrupted image is **not** available. *Resolved in the README:* operational drift is perturbation-based, and clean-vs-corrupted drift is diagnostic only.
