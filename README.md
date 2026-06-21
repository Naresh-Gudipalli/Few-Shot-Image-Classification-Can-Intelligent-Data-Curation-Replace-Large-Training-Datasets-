# Few-Shot Image Classification: Can Intelligent Data Curation Replace Large Training Datasets?

## Overview

This dissertation investigates whether intelligently curated small subsets of training data can approach the performance of models trained on full datasets. A fixed SimpleCNN architecture is trained across five phases from full-data baselines down to subsets as small as 0.2% of the original training set with the only variable being *how* the training examples are selected.

---

## Architecture - SimpleCNN (Fixed Across All Phases)

The same model is used in every phase. Nothing changes except the training data.

```
Input (1×28×28 MNIST / 3×32×32 CIFAR-10)
        │
        ▼
Conv Block 1: Conv2d(in_ch→32, 3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)
        │
        ▼
Conv Block 2: Conv2d(32→64, 3×3, pad=1)    → BatchNorm2d → ReLU → MaxPool2d(2×2)
        │
        ▼
Conv Block 3: Conv2d(64→128, 3×3, pad=1)   → BatchNorm2d → ReLU → MaxPool2d(2×2)
        │
        ▼
AdaptiveAvgPool2d(2×2)  →  Flatten  →  512 features
        │
        ▼
Linear(512→256) → ReLU → Dropout(0.5) → Linear(256→10)
        │
        ▼
   Logits (10 classes)
```

**Why fixed?** Any accuracy difference between phases must come from the data selection strategy, not the model. Keeping the architecture identical is what makes the comparison scientifically valid.

---

## Fixed Hyperparameters

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Dropout | 0.5 |
| Batch size | 128 |
| Seeds | 42, 123, 456, 789, 1011 |
| MNIST epochs | 20 |
| CIFAR-10 epochs | 30 |
| Platform | Kaggle GPU T4 |

---

## Datasets & Preprocessing

### MNIST
- 60,000 training and 10,000 test images (greyscale handwritten digits 0–9)
- Normalised using a mean of 0.1307 and standard deviation of 0.3081.
- No augmentation is applied

### CIFAR-10
- 50,000 training and 10,000 test images across 10 natural image categories
- Normalisation: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
- Training augmentation: RandomHorizontalFlip + RandomCrop(32, padding=4)

---

## Experimental Phases

| Phase | Description | Subset Size | Status |
|---|---|---|---|
| 1 | Full-data baseline | 100% | ✅ Complete |
| 2 | Random subset sampling | 0.2%, 0.5%, 1%, 2%, 5% | 🔄 In Progress |
| 3 | Stratified subset sampling | 0.2%, 0.5%, 1%, 2%, 5% | ⏳ Next Up |
| 4 | Intelligent curation strategies | 0.2%, 0.5%, 1%, 2%, 5% | ⏳ Pending |
| 5 | Hard-example mining | 0.2%, 0.5%, 1%, 2%, 5% | ⏳ Pending |

---

## Results

### Phase 1 — Full Baseline (5 seeds, GPU T4)

| Dataset | Top-1 Accuracy | Macro F1 |
|---|---|---|
| MNIST | **98.98% ± 0.43** | **0.9898 ± 0.0042** |
| CIFAR-10 | **80.93% ± 0.33** | **0.8084 ± 0.0028** |

These are the **ceiling lines**- the best the fixed architecture can do with all available data.

#### CIFAR-10 Per-Class F1 (Phase 1)

| Class | F1 ± Std | Class | F1 ± Std |
|---|---|---|---|
| airplane | 0.8102 ± 0.0169 | frog | 0.8508 ± 0.0081 |
| automobile | 0.9055 ± 0.0100 | horse | 0.8468 ± 0.0114 |
| bird | 0.7262 ± 0.0060 | ship | 0.8903 ± 0.0054 |
| cat | 0.6465 ± 0.0177 | truck | 0.8786 ± 0.0100 |
| deer | 0.7908 ± 0.0051 | dog | 0.7386 ± 0.0138 |

Notable: cat (0.6465) and dog (0.7386) are the weakest classes visually similar and the hardest to separate. Automobile (0.9055) is the strongest.

### Phase 2 - Random Subset *(results to be added)*

### Phase 3 - Stratified Subset *(results to be added)*

### Phase 4 - Intelligent Curation *(results to be added)*

### Phase 5 - Hard-Example Mining *(results to be added)*

---

## Repository Structure

```
dissertation-few-shot-classification/
│
├── README.md
│
├── phase1-baseline/
│   └── phase1-baseline-final.ipynb
│
├── phase2-random-subset/
│   └── phase2-random-subset.ipynb
│
├── phase3-stratified-subset/
│   └── phase3-stratified-subset.ipynb
│
├── phase4-curation-strategies/
│   └── phase4-curation.ipynb
│
├── phase5-hard-example-mining/
│   └── phase5-hard-example-mining.ipynb
│
├── results/
│   └── baseline_results.md
│
└── .gitignore
```

---

## How to Run

All notebooks are designed to run on **Kaggle with GPU T4 accelerator**.

1. Upload the notebook to Kaggle
2. Set accelerator: **Settings → Accelerator → GPU T4 x2**
3. Run All cells top to bottom
4. Save Version once complete

Each phase notebook is self-contained — it includes all imports, the full SimpleCNN definition, and all helper functions. No external dependencies beyond standard PyTorch + torchvision + sklearn.

---

## Key Findings *(updated as phases complete)*

- Phase 1 establishes that SimpleCNN reaches **98.98%** on MNIST and **80.93%** on CIFAR-10 with full data
- Cat/dog visual similarity is the primary weakness of the baseline model (gap of ~0.26 F1 vs best class)

---
