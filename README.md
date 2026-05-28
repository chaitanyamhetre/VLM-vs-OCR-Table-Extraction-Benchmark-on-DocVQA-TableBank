# VLM vs OCR: Table Extraction Benchmark on DocVQA Dataset

Traditional OCR pipelines struggle with complex document layouts, dense tables, and structured text extraction. This project benchmarks Tesseract OCR against a Vision Language Model (Qwen2-VL-2B) on structured information extraction from document images — a core challenge in digitising historical and archival documents for large-scale data analysis.

---

## Motivation

Traditional OCR pipelines struggle with complex document layouts, tables, and context-dependent information extraction. This project investigates whether modern Vision Language Models (VLMs) can outperform OCR-based methods on document question answering — a core challenge in building AI systems for historical and archival document analysis.

---

## Research Questions

1. Does a Vision LLM outperform Tesseract OCR on document Q&A accuracy?
2. Does OCR preprocessing (denoising, binarization) meaningfully improve accuracy?
3. How do the methods compare in inference speed on identical hardware?
4. Where does each method fail — and what does that reveal about its limitations?

---

## Dataset

**DocVQA** — Document Visual Question Answering  
- Source: `lmms-lab/DocVQA` via HuggingFace Datasets  
- Split: validation set, 300 samples  
- Format: Document page image + question + ground truth answer
- Evaluation method: answer extraction accuracy used as a proxy for text extraction quality  
- Document types: financial reports, scientific papers, forms, tables, advertisements  

---

## Methods

### Baseline 1 — Tesseract OCR (raw)
- Raw Tesseract OCR on original document images
- Answer found by searching OCR text output for ground truth string

### Baseline 2 — Tesseract OCR + Preprocessing
- OpenCV preprocessing pipeline: Gaussian denoising → Otsu binarization
- Same answer extraction as Baseline 1

### Method 3 — Qwen2-VL-2B (Vision Language Model)
- Model: `Qwen/Qwen2-VL-2B-Instruct` via HuggingFace Transformers
- Inference: float16, GPU (NVIDIA T4, 15GB)
- Prompt: document image + natural language question → direct answer generation
- No fine-tuning — zero-shot inference only

---

## Results

### Overall Accuracy

| Method | Accuracy | Avg Time / Image |
|---|---|---|
| OCR raw | 64.7% | 3.99s |
| OCR + preprocessing | 66.7% | 3.97s |
| **Qwen2-VL 2B (VLM)** | **77.0%** | **0.74s** |

### Accuracy by Answer Complexity

| Complexity | OCR raw | OCR + prep | Qwen2-VL 2B |
|---|---|---|---|
| Simple | ~65% | ~69% | higher |
| Medium | ~64% | ~64% | higher |
| Complex | ~15% | ~15% | significantly higher |

### Key Findings

- **VLM outperforms OCR by +10.3 percentage points** overall (77.0% vs 66.7%)
- **VLM is 5x faster** than Tesseract on GPU (0.74s vs 3.99s per image)
- **OCR preprocessing provides marginal improvement** (+2%) — not worth the added complexity for most use cases
- **Complex answers are where VLM wins most decisively** — OCR drops to 15% on multi-word answers while VLM maintains substantially higher accuracy
- **77% is a lower bound** — the strict exact-match metric penalises VLM answers that are richer than the ground truth (e.g. "UNIVERSITY OF CALIFORNIA, SAN DIEGO" vs ground truth "university of california")

### Sample Cases — VLM Beats OCR

| Question | Ground Truth | VLM Answer |
|---|---|---|
| What time is the coffee break? | 11:14 to 11:39 a.m. | 11:14 to 11:39 a.m. ✓ |
| What time is the Q&A session? | 12:25 to 12:58 p.m. | 12:25 to 12:58 p.m. ✓ |
| What is the name of the choco fills? | dark fantasy | Dark Fantasy ✓ |
| To whom is the document sent? | Paul | Paul ✓ |

---

## Repo Structure

```
vlm-ocr-table-benchmark/
├── notebooks/
│   ├── 01_data_exploration.ipynb       # dataset loading, inspection, metadata
│   ├── 02_ocr_baseline.ipynb           # Tesseract OCR baseline + preprocessing
│   ├── 03_vlm_extraction.ipynb         # Qwen2-VL inference + evaluation
│   └── 04_analysis.ipynb               # plots, comparison, findings
├── results/
│   ├── metadata.csv                    # sample metadata + complexity labels
│   ├── ocr_results.csv                 # OCR accuracy + runtime per sample
│   ├── vlm_results.csv                 # VLM answers + accuracy + runtime
│   ├── comparison_table.csv            # merged results, all methods
│   ├── final_summary.csv               # aggregate summary statistics
│   └── figures/
│       ├── sample_grid.png             # dataset sample visualisation
│       ├── accuracy_comparison.png     # overall accuracy bar chart
│       ├── accuracy_by_complexity.png  # accuracy breakdown by complexity
│       ├── runtime_comparison.png      # inference time comparison
│       └── outcome_distribution.png    # VLM vs OCR win/loss pie chart
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Reproducibility

All notebooks run on **Google Colab free tier (T4 GPU)**.  
No paid APIs or subscriptions required.  
All models and datasets are freely available on HuggingFace.

```bash
pip install transformers accelerate datasets pillow pandas \
            matplotlib pytesseract opencv-python-headless
```

Tesseract system dependency:
```bash
apt-get install tesseract-ocr
```

---

## Limitations

- **Exact-match evaluation** — strict string matching underestimates VLM accuracy; richer correct answers are penalised. A fuzzy match or ANLS metric would be more appropriate.
- **Zero-shot only** — Qwen2-VL was not fine-tuned; fine-tuning on domain-specific documents would likely push accuracy significantly higher.
- **Sample size** — 300 samples from the validation set; larger evaluation would improve statistical reliability.
- **Single VLM** — only Qwen2-VL-2B tested; comparing multiple VLMs (PaliGemma, LLaVA, moondream2) is a natural next step.
- **Document type mix** — DocVQA contains diverse document types; performance on historical or degraded documents specifically is not yet evaluated.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-green)
![Colab](https://img.shields.io/badge/Google-Colab_T4-red)

- **Model:** Qwen2-VL-2B-Instruct (HuggingFace)
- **OCR:** Tesseract 5.x + pytesseract
- **Preprocessing:** OpenCV
- **Dataset:** DocVQA via HuggingFace Datasets
- **Hardware:** NVIDIA T4 GPU (Google Colab free tier)

---
