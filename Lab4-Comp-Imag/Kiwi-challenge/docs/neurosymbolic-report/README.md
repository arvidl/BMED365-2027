# Differentiable Neurosymbolic AI for Semi-Verifiable 3D Biomedical Instance Segmentation ("Thesis + Code" Package)

> **BMED365 (Spring 2027) — Lab 4 Technical Report Series**  
> University of Bergen (UiB) & Mohn Medical Imaging and Visualization Centre (MMIV)

This folder contains a self-contained, Master's-thesis-level **IMRAD technical report** in LaTeX / BibLaTeX (`bibtex` backend) together with the **five modular PyTorch companion notebooks** and shared library (`kiwi_ns_lib.py`) that generate **every single figure, metric, ablation table, and whole-organ biomarker** in the report.

- **Compiled Technical Report (Clickable PDF, TOC, DOIs, Back-References & Appendices A–F):** [`neurosymbolic-kiwi-report.pdf`](./neurosymbolic-kiwi-report.pdf)
- **LaTeX Source & Verified Peer-Reviewed Bibliography (33 entries):** [`neurosymbolic-kiwi-report.tex`](./neurosymbolic-kiwi-report.tex) | [`references.bib`](./references.bib)
- **Single-Notebook Fast Walkthrough (Course Companion):** [`../../notebooks/07-kiwi-neurosymbolic-ai.ipynb`](../../notebooks/07-kiwi-neurosymbolic-ai.ipynb) ([printable PDF](../../notebooks/07-kiwi-neurosymbolic-ai_ipynb.pdf))

---

## Thesis–Code Reproducibility Map

Every figure (`figures/fig01..09_*.png`) and quantitative table (`results/nb01..05_*.json`) in `neurosymbolic-kiwi-report.pdf` is generated directly by one of the five notebooks below importing [`notebooks/kiwi_ns_lib.py`](./notebooks/kiwi_ns_lib.py):

| Report Section | Companion Notebook | Launch in Colab | Figures & Tables | JSON Output | Measured Runtime (MPS) |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **Sec. 2.1.1 & 3.1** (GT Audit & Priors) | [`01_data_geometry_and_prior_audit.ipynb`](./notebooks/01_data_geometry_and_prior_audit.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/BMED365-2027/blob/main/Lab4-Comp-Imag/Kiwi-challenge/docs/neurosymbolic-report/notebooks/01_data_geometry_and_prior_audit.ipynb) | Figs. 1–3, Table 3 | [`nb01_prior_audit.json`](./results/nb01_prior_audit.json) | `27.4 s` |
| **Sec. 3.2** (Theory & Math) | [`02_instancenorm_trap_and_ltn_math.ipynb`](./notebooks/02_instancenorm_trap_and_ltn_math.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/BMED365-2027/blob/main/Lab4-Comp-Imag/Kiwi-challenge/docs/neurosymbolic-report/notebooks/02_instancenorm_trap_and_ltn_math.ipynb) | Figs. 4–5, Table 4 | [`nb02_instancenorm_and_ltn.json`](./results/nb02_instancenorm_and_ltn.json) | `7.2 s` |
| **Sec. 3.3** (Multi-Seed Ablation) | [`03_multiseed_ablation_and_calibration.ipynb`](./notebooks/03_multiseed_ablation_and_calibration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/BMED365-2027/blob/main/Lab4-Comp-Imag/Kiwi-challenge/docs/neurosymbolic-report/notebooks/03_multiseed_ablation_and_calibration.ipynb) | Figs. 6–7, Tables 5–6 | [`nb03_multiseed_ablation.json`](./results/nb03_multiseed_ablation.json) | `555.1 s` |
| **Sec. 3.4** (Interpretability & Whole-Organ) | [`04_mechanistic_interpretability_and_biomarkers.ipynb`](./notebooks/04_mechanistic_interpretability_and_biomarkers.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/BMED365-2027/blob/main/Lab4-Comp-Imag/Kiwi-challenge/docs/neurosymbolic-report/notebooks/04_mechanistic_interpretability_and_biomarkers.ipynb) | Figs. 8–9, Tables 7–8 | [`nb04_mechanistic_and_whole_organ.json`](./results/nb04_mechanistic_and_whole_organ.json) + [`kiwi_neurosymbolic_pred_labels.nii.gz`](./results/kiwi_neurosymbolic_pred_labels.nii.gz) | `99.3 s` |
| **Sec. 3.5** (Mass-Effect Stress Tests) | [`05_axiom_stress_tests_and_clinical_translation.ipynb`](./notebooks/05_axiom_stress_tests_and_clinical_translation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/BMED365-2027/blob/main/Lab4-Comp-Imag/Kiwi-challenge/docs/neurosymbolic-report/notebooks/05_axiom_stress_tests_and_clinical_translation.ipynb) | Fig. 10, Table 9 | [`nb05_stress_test.json`](./results/nb05_stress_test.json) | `55.4 s` |

> **Runtime & Colab note:** All five notebooks were executed end-to-end locally on Apple Silicon MPS via `uv` (`744.5 s` / `12.4 min` total). Each notebook includes an automatic bootstrap cell that fetches `kiwi_ns_lib.py` and the 7 T NIfTI data from GitHub `main` when opened in Google Colab (`CUDA -> MPS -> CPU`); live Colab cloud execution has not yet been tested directly.

---

## Technical Appendices Inside the Report

To keep the main IMRAD text accessible to biomedicine MSc students and high-school/sophomore readers alike, **Appendices A–F** explain every conceptual and mathematical notion from first principles with worked numerical examples:

- **Appendix A:** 7 T MRI Acquisition Physics, $T_1$/$T_2$ Relaxation, RARE Spin-Echo Contrast, $B_1$ Shading, and Anisotropic Partial-Volume Averaging
- **Appendix B:** 3D U-Nets, BatchNorm vs. `InstanceNorm3d` (Worked Proof of Coordinate Erasure), 3D SSL, `AdamW`/Cosine Annealing, and Additive Bayesian Log-Odds Fusion
- **Appendix C:** 3D Grayscale Mathematical Morphology (`grey_closing`), Anisotropic Distance Transforms, Marker-Controlled Watershed, $K$-Means, and Matérn Hard-Core Packing
- **Appendix D:** Continuous Real Logic, Fuzzy Triangular Norms (Product / Łukasiewicz / Gödel), and Exact Derivative Derivation of LTN $p$-Mean Quantifiers ($25\times$ Hard-Counterexample Mining)
- **Appendix E:** 3D Dice vs. IoU, Hungarian Bipartite Matching, Panoptic Quality ($\text{PQ} = \text{SQ} \times \text{RQ}$), Error Taxonomy, Calibration Shift, and Whole-Organ Ratio Estimation
- **Appendix F:** Software Artifact Layout, `kiwi_ns_lib.py` API Reference, and Reproducibility Guide

---

## Reproducing Locally

### 1. Run the Companion Notebooks (`uv`)
From the repository root (`BMED365-2027/`):
```bash
uv sync
uv run jupyter lab Lab4-Comp-Imag/Kiwi-challenge/docs/neurosymbolic-report/notebooks/
```
Select the `bmed365-2027` (`python3`) kernel.

### 2. Interactive 3D Superimposition in FreeSurfer 8.2.0 `freeview` (macOS)
Both `data/kiwi_micro_gt_labels.nii.gz` (silver-standard 98-seed reference) and `results/kiwi_neurosymbolic_pred_labels.nii.gz` (unmasked whole-fruit 749-seed prediction) share the exact `256 × 256 × 128` lattice and bit-identical `sform` affine as the raw 7 T MRI. Launch them interactively with a custom high-contrast ColorLUT (`scripts/kiwi_seeds_ColorLUT.txt`):
```bash
# Default: 7T MRI + Silver Micro-GT (toggleable Neurosymbolic prediction & yellow outline)
./Lab4-Comp-Imag/Kiwi-challenge/scripts/view_kiwi_freeview.sh

# Show both Silver Micro-GT and Whole-Fruit Neurosymbolic 3D prediction + 3D seed isosurfaces
./Lab4-Comp-Imag/Kiwi-challenge/scripts/view_kiwi_freeview.sh --with-pred --3d

# Start in crisp outline-only mode (unobstructed dark seed cores)
./Lab4-Comp-Imag/Kiwi-challenge/scripts/view_kiwi_freeview.sh --outline

# Edit a personal working copy in git-ignored MyStuff/kiwi-gt/ (and re-run any notebook via gt_nii_override=...)
./Lab4-Comp-Imag/Kiwi-challenge/scripts/view_kiwi_freeview.sh --gt MyStuff/kiwi-gt/kiwi_gt_manual_v1.nii.gz
```

### 3. Recompile the LaTeX / BibLaTeX Report (TeX Live 2025)
```bash
cd Lab4-Comp-Imag/Kiwi-challenge/docs/neurosymbolic-report
pdflatex neurosymbolic-kiwi-report.tex
bibtex neurosymbolic-kiwi-report
pdflatex neurosymbolic-kiwi-report.tex
pdflatex neurosymbolic-kiwi-report.tex
```
