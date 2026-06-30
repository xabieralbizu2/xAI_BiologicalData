# Explainable AI for Biological Data

### Interpretable, Uncertainty-Aware Classification of Primate Dental Microtexture

**B.S. Thesis — Mathematical Engineering and Artificial Intelligence**
ICAI, Universidad Pontificia Comillas · 2025–2026
Author: **Xabier Albizu Arias** · Advisors: Simón Rodríguez Santana, Jaime Pizarroso Gonzalo

---

## Overview

This project asks a deceptively simple question about dental microwear texture analysis (DMTA) in African cercopithecids:

> **How far can the signal actually be trusted?**

Rather than chasing the best classifier, it measures the *informational ceiling* of the data itself and quantifies the uncertainty of every claim. The core contribution is not a new method but an **architecture**: a three-layer Bayesian pipeline in which each layer answers a question whose answer determines whether the next one is even meaningful.

The dataset comprises **160 specimens** (125 extant from 7 species with known diet, 35 fossils from extinct taxa) described by **107 continuous microtexture variables**.

---

## The three-layer architecture

Each layer closes one door and opens the next.

### Layer 1 — Supervised informational ceiling

*Question: to what taxonomic level does microtexture discriminate?*

Multinomial Bayesian logistic regression on the extant specimens (the only set with verifiable diet), comparing **four shrinkage priors** (Gaussian, Laplace, horseshoe, spike-and-slab) under homogeneous leave-one-out cross-validation against trivial baselines.

**Finding:** the signal resolves *group* (~0.56 accuracy) but **not species** (~0.30, barely above chance). The ceiling reproduces across every prior and every data-derived hierarchy — it is a property of the **signal, not the model**. A revealing calibration-vs-accuracy dissociation emerges: the best-calibrated priors (best ELPD-LOO) are the worst classifiers — the prior reorders the prudence–decision trade-off but creates no information the data does not contain.

### Layer 2 — Extant–fossil comparability

*Question: are fossils even comparable to extant specimens?*

A Gaussian Mixture Model fit on each population, measuring **cross-likelihood** (how well each population fits the other's density), with the internal likelihood computed via LOO to avoid overfitting inflation.

**Finding:** the two populations occupy **disjoint regions** of the space. Each population assigns the other a practically null likelihood (median cross log-likelihood ≈ −10⁷), and a Mann–Whitney test confirms the separation (*p* < 10⁻¹⁸, robust across three dataset variants). This **domain shift** means projecting fossils onto an extant-trained classifier would be out-of-distribution extrapolation — forcing the unsupervised approach of Layer 3.

### Layer 3 — Fossil affinities (unsupervised)

*Question: what do the fossils resemble, and how confidently?*

A variational Bayesian **GPLVM** (Matérn-5/2 kernel, 75 inducing points) embeds all 160 specimens **without labels** into a shared latent space, where each specimen is a *distribution*, not a point. Structure is validated by comparing **Bayesian Hierarchical Clustering** against a Dirichlet-Process baseline.

**Finding:** BHC recovers the cercopithecid phylogeny fully unsupervised (**NMI = 0.85**, ARI = 0.58), while the Dirichlet Process collapses 82% of specimens into one component — the morphospace is a *continuum*, not discrete clusters. Projecting the fossils reveals **real but low-resolution affinities**: *Parapapio* concentrates consistently (10/15 specimens in one cluster), whereas *Theropithecus oswaldi* disperses across four — the same approximate-not-taxonomic pattern seen in the extant data.

---

## Repository structure

```
.
├── transformations/        # Preprocessing: Yeo–Johnson, standardization, one-hot encoding
├── eda/                    # Exploratory analysis + Layer 2 (GMM cross-likelihood, EMD, Mann–Whitney)
├── feature_engineering/    # Bayesian feature selection (horseshoe credible-interval analysis)
├── probabilistic_models/   # Layer 1: Bayesian logistic regression (blr.py) + hierarchical cascade
├── ml_models/              # Classical baselines + hierarchical model comparison
├── visualization/          # Layer 3: GPLVM, BHC, latent-space discovery; PCA/LDA projections
└── previous_work/          # Reference materials the thesis builds on
```

Key scripts:

| File | Role |
|------|------|
| `probabilistic_models/blr.py` | Multinomial Bayesian logistic regression (4 priors, NUTS) |
| `probabilistic_models/blr_hierarchical.py` | Hierarchical cascade classifier |
| `feature_engineering/bayesian_feature_selection.py` | Horseshoe-based variable importance |
| `eda/variable_analysis.py` | GMM cross-likelihood + domain-shift tests (Layer 2) |
| `visualization/gplvm/` | Variational GPLVM + BHC latent-structure discovery (Layer 3) |

---

## Methodological highlights

- **Leakage-free validation:** every fitted preprocessing step (Yeo–Johnson λ, standardization, feature selection) is re-estimated *inside* each LOO fold, so the held-out specimen never informs preprocessing.
- **Inference:** NUTS (4 chains, 1000+1000), with $\hat{R}$, ESS and PSIS-LOO diagnostics; non-centered reparametrization for the horseshoe to avoid funnel pathologies.
- **Imbalance-robust metrics:** balanced accuracy, macro-F1 and Cohen's $\kappa$ throughout, always reported against majority-class and random baselines.
- **Uncertainty everywhere:** posterior credible intervals for variable importance, variational uncertainty per specimen in the latent space, full predictive distributions rather than point labels.

---

## Tech stack

**Python** · PyMC (NUTS/MCMC) · GPyTorch (variational GPLVM) · scikit-learn · ArviZ · NumPy · Pandas · SciPy · Matplotlib

---

## Key results at a glance

| Result | Value |
|--------|-------|
| Species-level accuracy (best prior) | 0.32 (chance ≈ 0.14, majority ≈ 0.19) |
| Group-level accuracy | ~0.56 |
| Extant–fossil domain shift | *p* < 10⁻¹⁸ |
| Unsupervised phylogeny recovery (BHC) | NMI = 0.85, ARI = 0.58 |
| Dirichlet-Process baseline | NMI = 0.11 (collapses to 1 cluster) |

---

## Citation

If you reference this work:

```
Albizu Arias, X. (2026). Explainable AI for Biological Data:
Interpretable, Uncertainty-Aware Classification of Primate Dental Microtexture.
B.S. Thesis, ICAI, Universidad Pontificia Comillas.
```
