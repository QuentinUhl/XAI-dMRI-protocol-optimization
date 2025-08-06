# XAI-dMRI-protocol-optimization

This repository contains all code used in the paper:

> **"Reduced NEXI protocol for the quantification of human gray matter microstructure on the CONNECTOM 2.0 scanner"**  
> Quentin Uhl\*, Tommaso Pavan, Julianna Gerold, Kwok-Shing Chan, Yohan Jun, Shohei Fujita, Aneri Bhatt, Yixin Ma, Qiaochu Wang, Hong-Hsi Lee, Susie Y. Huang, Berkin Bilgic\*, Ileana Jelescu\*  
> _\*Joint last authorship_  
>  
> [Preprint / Journal link](https://github.com/QuentinUhl/XAI-dMRI-protocol-optimization) (to be updated)

---

## 📌 Overview

This work introduces a **data-driven framework** for optimizing NEXI (Neurite Exchange Imaging) protocols using **explainable machine learning** to reduce scan time without compromising accuracy.

We propose and validate two complementary strategies:

- ✅ **SHAP-RFE optimization** using XGBoost regression, recursive feature elimination, and SHAP values.
- ✅ **FIM-based optimization** using the determinant of the Fisher Information Matrix (D-optimality) for protocol reduction.

From an initial 15-feature ($b$, $\Delta$) protocol, we derived an optimal 8-feature subset. This reduced protocol cuts scan time by nearly **50%**, while retaining estimation accuracy, anatomical contrast, and test-retest reliability across all NEXI parameters ($t_{ex}$, $f$, $D_i$, $D_e$).

---

## 📂 Repository structure

```bash
XAI-dMRI-protocol-optimization/
├── fim_optimization.py          # Fisher Information Matrix-based protocol reduction
├── shap-rfe_optimization.py         # SHAP-RFE-based protocol optimization using XGBoost
├── C2_complex_all_sigma.npz     # Empirical distribution of noise levels (sigma)
├── C2_complex_fim_optimization.png   # Visualization of FIM-reduced protocol
├── C2_complex_xgb_optimization.png   # Visualization of SHAP-reduced protocol
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE                      # MIT License
