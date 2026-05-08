# Section-Aware SciBERT for Citation Intent Classification

> A stability and explainability analysis using LIME on the SciCite benchmark.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Section_Aware_SciBERT.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: SciBERT](https://img.shields.io/badge/model-SciBERT-orange)](https://huggingface.co/allenai/scibert_scivocab_uncased)
[![Dataset: SciCite](https://img.shields.io/badge/dataset-SciCite-green)](https://github.com/allenai/scicite)

---

## Overview

This repository contains the code and paper for **Section-Aware SciBERT**, a lightweight input-augmentation approach for citation intent classification. We prepend a canonical section tag (e.g., `[METHOD]`, `[INTRODUCTION]`) to each citation sentence prior to fine-tuning SciBERT, with **no architectural changes**.

Across **5 random seeds** on the **SciCite** benchmark, our method:

- Achieves **statistically equivalent mean performance** to a SciBERT baseline (paired *t*-test, all *p* > 0.73).
- Reduces **95% confidence-interval widths on F1 macro by ~40%**, indicating substantially more stable predictions.
- Improves **`background` F1 by +2.40%** while costing **`result` F1 −3.45%** — a trade-off we diagnose in detail.
- Provides **interpretable predictions** via LIME, including 4 detailed case studies showing where section context helps and where it hurts.

> **Key takeaway:** Section context is a **stabilizing prior**, not an accuracy booster. LIME analysis reveals that systematic failures occur when the section tag conflicts with sentence-level intent (e.g., a result-stating sentence appearing in an experiment section).

---

## Repository Contents

```
.
├── Section_Aware_SciBERT.ipynb       # Main reproducible notebook (Colab-ready)
├── paper/
│   ├── Section_Aware_SciBERT_Paper.pdf   # Final IEEE conference paper (9 pages)
│   ├── Section_Aware_SciBERT_Paper.tex   # LaTeX source
│   └── figures/                          # All paper figures (PNG, 300 DPI)
├── results/                          # CSVs and JSON of all experimental results
├── lime_outputs/                     # Interactive HTML LIME explanations
├── requirements.txt                  # Pinned Python dependencies
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## Quick Start

### Option A: Run on Google Colab (recommended)

1. Click the **"Open in Colab"** badge at the top.
2. Set runtime: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells (`Runtime → Run all`). The full pipeline takes about 40 minutes.

### Option B: Run locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Section_Aware_SciBERT.ipynb
```

A CUDA-capable GPU is strongly recommended (CPU runtime is ~30× slower).

---

## Method

The full pipeline differs from a standard SciBERT fine-tuning run in only **two places**:

1. **Section canonicalization** — heterogeneous raw `sectionName` strings (50+ unique values) are mapped via a deterministic keyword-based rule into 8 canonical tags: `INTRODUCTION`, `RELATED_WORK`, `METHOD`, `EXPERIMENT`, `RESULT`, `DISCUSSION`, `CONCLUSION`, `OTHER`.

2. **Input prefixing** — for every citation, the canonical tag is prepended in square brackets:

   ```
   Baseline:       "This method outperforms previous work [1]"
   Section-aware:  "[METHOD] This method outperforms previous work [1]"
   ```

The model architecture and tokenizer are unchanged. Class-weighted cross-entropy is used to handle SciCite's class imbalance (method 59% / background 28% / result 13%).

---

## Results

### Test-set performance across 5 seeds (mean ± std)

| Metric       | Baseline SciBERT      | Section-Aware SciBERT |
|--------------|-----------------------|------------------------|
| Accuracy     | 0.8598 ± 0.0078       | 0.8608 ± 0.0058        |
| F1 Macro     | 0.8457 ± 0.0097       | 0.8443 ± 0.0058        |
| F1 Weighted  | 0.8615 ± 0.0072       | 0.8635 ± 0.0057        |

### Statistical tests

| Test                           | Accuracy | F1 Macro | F1 Weighted |
|--------------------------------|----------|----------|-------------|
| Paired *t*-test (means)        | p=0.874  | p=0.839  | p=0.734     |
| F-test for variance reduction  | 46.2% ↓  | 64.1% ↓  | 38.3% ↓     |
| 95% CI width reduction         | 27%      | 40%      | 21%         |

### Per-class F1 (seed = 42)

| Class      | Baseline | Section-Aware | Δ |
|------------|----------|---------------|---|
| background | 0.8659   | **0.8898**    | **+0.0240** ✅ |
| method     | 0.8817   | 0.8747        | −0.0070       |
| result     | 0.8020   | 0.7675        | **−0.0345** ❌ |

---

## LIME Case Studies

We present 4 case studies that diagnose the trade-off:

| Case | Scenario | Insight |
|------|----------|---------|
| **1** | Section WIN on background | `[METHOD]` tag (weight +0.57) overrides misleading content cues like "supported", "previous"; corrects the baseline's wrong `result` prediction. |
| **2** | Section LOSS on result ⭐ | `[EXPERIMENT]` tag (weight +0.40) overrides correct content cues like "similar", "accuracy 96%"; produces an over-confident wrong `background` prediction. **This explains the −3.45% result F1 cost.** |
| **3** | Confidence boost | Both models predict `method` correctly; section-aware confidence jumps 0.58 → 0.97 (+39pp). |
| **4** | Robustness to noisy tag | Misleading `[METHOD]` tag (true label is `background`); model still predicts background correctly because content signals dominate. |

Interactive HTML visualizations of all four cases are in `lime_outputs/`.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{your_paper_2026,
  title     = {Section-Aware SciBERT for Citation Intent Classification:
               A Stability and Explainability Analysis using LIME},
  author    = {Your Name and Co-authors},
  booktitle = {Proceedings of <Conference Name>},
  year      = {2026},
  url       = {https://github.com/YOUR_USERNAME/YOUR_REPO}
}
```

---

## Acknowledgments

- **Dr. Salman Taseer** for supervision, technical guidance, and continuous support throughout this research.
- **Allen Institute for AI** for releasing the [SciCite](https://github.com/allenai/scicite) dataset and the original [SciBERT](https://github.com/allenai/scibert) model.
- **Hugging Face** for hosting the dataset and model weights, and for the `transformers` and `datasets` libraries.
- The authors of [LIME](https://github.com/marcotcr/lime), [PyTorch](https://pytorch.org/), and [scikit-learn](https://scikit-learn.org/), without whose open-source tools this work would not be possible.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

The SciCite dataset is distributed by the Allen Institute for AI under its own license; please refer to the [SciCite repository](https://github.com/allenai/scicite) for terms of use.

---

## Contact

For questions about the paper or the code, please open an issue on this repository or contact the corresponding author.
