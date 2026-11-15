# Lab 5: The Computational Kiwi Challenge — Morphogenesis, Inverse Problems & a 7T MRI Digital Twin

*BMED365 — Computational Imaging, Modeling and AI in Biomedicine (Spring 2027)*  
*Arvid Lundervold — runs locally with `uv` (`bmed365-2027` kernel) or in Google Colab*

---

## Overview: Closing the Loop Between Computational Imaging (`Lab 4`) and Mechanistic Modeling (`Lab 5`)

In **[Lab 4 (`Lab4-Comp-Imag/Kiwi-challenge`)](../../Lab4-Comp-Imag/Kiwi-challenge/README.md)**, we solved the **observational imaging problem**: reconstructing, segmenting, and counting ~1,440 seeds in a 2005 **7 Tesla Bruker Pharmascan** 3D MRI scan of an intact `'Hayward'` kiwifruit (*Actinidia chinensis* var. *deliciosa*, hexaploid $2n = 6x = 174$).

Here in **Lab 5 (`Computational Modeling`)**, we turn the problem upside down and ask the **generative & mechanistic question**:

> **Can we write down the mathematical laws of *how biology built that 3D organ* — from sex-determining genes and Turing symmetry breaking on the floral ring, to turgor-driven growth, paracrine seed hormones, and 7T Bloch spin-echo MRI contrast?**

Together, the Lab 4 and Lab 5 Kiwi modules form a complete **Imaging ↔ Mechanistic Digital Twin** pipeline — mirroring how modern biomedical engineering combines patient MRI scans with biophysical ODE/PDE simulations in cardiology, nephrology, oncology, and *in silico* clinical trials.

---

## Progressive Modeling Ladder & Explicit Human Biomedical Parallels

The modeling starts with zero-calculus algebraic equations and steps progressively up to coupled ODEs, spatial reaction–diffusion PDEs, 3D spatial point processes, ill-posed inverse problems, and synthetic MRI physics:

| Stage / Module | Botanical Mechanism in *Actinidia* | Mathematical Formulation | Direct Human Biomedical Parallel | Empirical Check on Our 2005 7T MRI |
|:---|:---|:---|:---|:---|
| **Stage 1** *(NB 01)* | Seed-derived auxin/cytokinin trophic scaling (*Pyke & Alspach, 1986*) | **Algebraic saturation:** $V(N) = V_{\text{basal}} + V_{\max}\frac{N}{K_N + N}$ | **Pharmacodynamic $E_{\max}$ / Hill curves**; nephron endowment vs. kidney mass; placental perfusion | Predicts **`86.1 cm³`** at $\hat{N}=1{,}440$ vs. measured **`85.97 cm³`**; slice area $A(z)$ tracks local seed density (**`r = 0.851`**, $p = 5.8\times 10^{-30}$) |
| **Stage 2A** *(NB 01)* | `XY` floral sex switch: Y-linked ***Shy Girl* (`SyGI`)** represses carpel fate (*Akagi et al., 2018, 2019*) | **Bistable nonlinear ODEs** (Hill repression + positive feedback) | **Mammalian gonadal fate (`SRY`/`SOX9` vs. `WNT4`/$\beta$-catenin)**; toggle switches ([`05-cell-signaling`](../notebooks/05-cell-signaling.ipynb)) | Explains why our fruit-bearing scan literally had DICOM tag `PatientSex = F` (`XXXXXX`) |
| **Stage 2B** *(NB 01)* | 160-day berry growth, Stage II starch fill, Stage III sugar osmosis (*Hall et al., 2013*) & seed dehydration | **Lockhart viscoplastic ODEs** + **7T Bloch $T_2$ spin-echo** (`TE = 63.9 ms`) | **Starling filtration & edema** ([`06-kidney-filtration`](../notebooks/06-kidney-filtration.ipynb)), `ADPKD` renal cyst expansion, & tumor growth ([`02-tumor-growth`](../notebooks/02-tumor-growth.ipynb)) | Reproduces mature volume (**`86.9 cm³`**) and explains why lignified seeds turn **jet-black** while mucilage gel stays **`~190×` brighter** |
| **Stage 3A** *(NB 01 & 02)* | Spontaneous emergence of **`~33` radial locules**; ploidy shift ($2x \to 4x \to 6x$) | **1D Ring Schnakenberg Turing PDE** & Jacobian dispersion $\text{Re}\,\lambda(q)$ | **Embryonic digit spacing & polydactyly** (`BMP`–`SOX9`–`WNT` Turing network); polyploid cardiomyocyte/hepatocyte hypertrophy | Polar-unwrapped `z = 64` 7T ring exhibits **`33` radial chambers**, matching hexaploid $6x$ Turing mode **`K* = 33`** (vs. `22` in $2x$, `28` in $4x$) |
| **Stage 3B** *(NB 01)* | Local pericarp expansion from diffusing seed hormones; flat-sided poorly pollinated fruit | **2D Screened-Poisson PDE:** $(\nabla^2 - \ell^{-2})c = -\frac{\alpha}{D}S(x,y)$ | **Tumor `VEGF` angiogenesis gradients** & **wound-healing growth-factor PDEs** ([`08-wound-healing`](../notebooks/08-cybernetics-of-wound-healing.ipynb)) | Shows **developmental canalization** (`±3.5%` circularity) in our well-pollinated 2005 berry vs. **`ΔR = 9.4 mm`** flat-sided asymmetry under unilateral seed loss |
| **Module 2** *(NB 02)* | Ovule steric exclusion & placental spacing in 3D locules | **3D Spatial Point Processes:** Clark–Evans $R$, NN CDF $G(r)$, Ripley's $\Delta L(r)$, Matérn Hard-Core | **Tumor–immune spatial point-process statistics** (`CD8+` T-cell exclusion/clustering) in **IMC** ([`Lab4/03-imc-intro`](../../Lab4-Comp-Imag/notebooks/03-imc-intro.ipynb)) | Curated 98 Micro-GT seeds exhibit a **`0.945 mm` hard-core floor** and **Clark–Evans `R = 1.192`** ($p = 0.0025$), rejecting 3D Poisson CSR |
| **Module 3** *(NB 02)* | Non-invasive reconstruction of hidden seed deficits from outer fruit contour | **Ill-posed Fredholm / Green's operator `SVD`** + **Tikhonov ($L_2$) L-curve** | **Electrocardiographic Imaging (`ECGi`)**, **EEG/MEG brain source localization**, & MR elastography | `cond(G) = 1.3e6`: naive $\mathbf{G}^{-1}\mathbf{y}$ blows up (`>470,000%` error), whereas **Morozov Tikhonov** reconstructs hidden deficits to **`5.2%` error** |
| **Module 4** *(NB 02)* | Whole-organ synthetic ground-truth benchmark (`Lab 5 → Lab 4`) | **3D Generative Anatomy + Bruker 7T RARE Bloch equation + Rician k-space noise** | **`BrainWeb` MS lesion phantoms**, **`XCAT` cardiac phantoms**, & *in silico* FDA validation of clinical AI | Generates a `192 × 192 × 88` synthetic 7T MRI with exact $N_{\text{true}} = 750$ seeds (`3D F1 = 0.99`) and validates Lab 4's sector ratio calibration |

---

## Notebooks

No data files are duplicated in this folder: locally, both notebooks read the 2005 7T NIfTI volume and curated 3D Micro-GT directly from [`../../Lab4-Comp-Imag/Kiwi-challenge/data/`](../../Lab4-Comp-Imag/Kiwi-challenge/data/); on Google Colab, `resolve_lab4_kiwi_file(...)` downloads them automatically from `main`.

| Notebook | Description | 1-Click Notebook |
|:---------|:------------|:-----------------|
| [01-kiwi-growth-and-morphogenesis.ipynb](https://github.com/arvidl/BMED365-2027/blob/main/Lab5-Comp-Mod/Kiwi-challenge/notebooks/01-kiwi-growth-and-morphogenesis.ipynb) | **Part 1 (Algebra → ODEs → Spatial PDEs):** Explicit MSc/PhD learning outcomes & Botanical ↔ Biomedical glossary; intuitive + rigorous derivations with parameter/unit tables; algebraic seed-hormone saturation & 128-slice axial MRI scaling (`r = 0.85`, `0.94` in seed zone); `XY` floral sex-determination switch (`SyGI` vs. carpel); Hall's 160-day turgor/starch/sugar ODEs + 7T Bloch $T_2$ contrast (`190×` gel/seed ratio); 1D Ring Turing PDE (`K = 33` locules); 2D screened-Poisson paracrine hormone PDE; static 4-regime pollination snapshot + interactive `ipywidgets` lab & `<details>` Q&A checkpoints | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/BMED365-2027/blob/main/Lab5-Comp-Mod/Kiwi-challenge/notebooks/01-kiwi-growth-and-morphogenesis.ipynb) |
| [02-kiwi-digital-twin-and-inverse-problems.ipynb](https://github.com/arvidl/BMED365-2027/blob/main/Lab5-Comp-Mod/Kiwi-challenge/notebooks/02-kiwi-digital-twin-and-inverse-problems.ipynb) | **Part 2 (Advanced Math, Inverse Problems & Digital Twin):** Explicit MSc/PhD learning outcomes & quantitative glossary; analytical Turing Jacobian dispersion $\text{Re}\,\lambda(q)$ across $2x/4x/6x$ ploidy (`22 → 28 → 33` carpels); 3D spatial point-process statistics (`R = 1.19`, $G(r)$, Ripley's $\Delta L(r)$) on the 98 Micro-GT seeds; Hadamard ill-posedness, `SVD` filter factors & Morozov Tikhonov L-curve; **Generative 3D Bloch-MRI Digital Twin** with Rician k-space noise, Hungarian 3D instance matching, **Sim-to-Real gap** analysis, `<details>` Q&As, & Block 2 project ideas | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/BMED365-2027/blob/main/Lab5-Comp-Mod/Kiwi-challenge/notebooks/02-kiwi-digital-twin-and-inverse-problems.ipynb) |

> **Runtime Notes:** Both notebooks are pure CPU (`NumPy`, `SciPy`, `scikit-image`, `nibabel`, `matplotlib`, `ipywidgets`) and execute end-to-end in **`~12–18 seconds`** each (`< 1.5 GB` RAM).

---

## Ideas for Block 2 Individual Projects (`BMED365`)

Students looking for a self-contained **Computational Modeling + Imaging** topic for their [Individual Project](../../Individual-project/README.md) can extend these notebooks in several directions (detailed in Section 6 of Notebook `02`):

1. **Finite-Strain Morphoelasticity ($\mathbf{F} = \mathbf{F}_e\mathbf{F}_g$):** Couple differential pericarp growth with stiff *columella* mechanics — directly analogous to **human cerebral cortical gyrification** (brain folding) and cardiac remodeling.
2. **Physics-Informed Neural Networks (`PINNs`) in PyTorch:** Estimate unknown hormone diffusion length $\ell$ and receptor saturation $K_c$ directly from 3D MRI contours by minimizing PDE + boundary losses.
3. **Zero-Annotation `Sim-to-Real` 3D Deep Learning:** Train a 3D `U-Net` purely on synthetic Bloch-MRI Digital Twins (`Module 4`) and evaluate zero-shot transfer on the real 2005 7T MRI scan (`Lab 4`).
4. **Comparative Ploidy Modeling ($2x$ Golden vs. $6x$ Green Kiwifruit):** Predict differences in locule count, seed load, carbohydrate dynamics, and MRI $T_2$ contrast across *Actinidia* ploidy series.
