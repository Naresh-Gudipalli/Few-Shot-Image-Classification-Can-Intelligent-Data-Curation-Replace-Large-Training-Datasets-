# Few-Shot Image Classification: Can Intelligent Data Curation Replace Large Training Datasets?

---
 
## Overview
 
Training deep neural networks on large labelled datasets is expensive in time, money and compute. But not all training examples are equally useful. This project investigates whether intelligently choosing a small subset of training data can approach the performance of training on the full dataset.
 
The experiment is fully controlled: one architecture (SimpleCNN), one optimiser, five fixed random seeds, two datasets (MNIST and CIFAR-10) and five subset sizes from 0.2% to 5%. The only variable across phases is the selection strategy:- Random, class-balanced, diversity-based or importance-based. Because everything else is locked, Any difference in accuracy is attributable to the selection method alone.
 
---
 
## Repository Structure
 
```
├── phase1-baseline/
│   └── phase1-baseline-final.ipynb        # Full-dataset ceiling (COMPLETE ✅)
├── phase2-random-subsampling/
│   └── phase2_random_subsampling.py       # Random subset floor (COMPLETE ✅)
├── .gitignore
└── README.md
```
 
> Phases 3–5 will be added as each phase completes.
 
---
 
## Experimental Setup (Fixed Across All Phases)
 
| Component | Value |
|-----------|-------|
| Architecture | SimpleCNN 3 conv blocks (32→64→128), AdaptiveAvgPool2d(2,2), FC 512→256→10 |
| Optimiser | Adam lr=1e-3, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau (min, factor=0.5, patience=3) steps on training loss |
| Seeds | [42, 123, 456, 789, 1011] 5 seeds for mean ± std |
| Subset sizes | 0.2%, 0.5%, 1%, 2%, 5% of training set |
| Batch size | 128 |
| Epochs | MNIST: 20 · CIFAR-10: 30 |
| MNIST | Normalise only  mean 0.1307, std 0.3081 |
| CIFAR-10 | RandomHorizontalFlip + RandomCrop(32, padding=4) + normalise (train only) |
| Platform | Kaggle GPU T4 x2 |
 
---
 
## Phase Status & Results
 
### Phase 1: Full Dataset Ceiling ✅ COMPLETE
 
Trains SimpleCNN on the complete training set to establish the performance ceiling.
Every later phase is reported as a distance from these numbers.
 
| Dataset | Top-1 Accuracy | Macro-F1 |
|---------|---------------|----------|
| MNIST | **98.98% ± 0.43** | 0.9898 ± 0.0042 |
| CIFAR-10 | **80.93% ± 0.33** | 0.8084 ± 0.0028 |
 
CIFAR-10 weakest classes: cat = 0.6465 · bird = 0.7262 · dog = 0.7386
CIFAR-10 strongest class: automobile = 0.9055
 
---
 
### Phase 2: Random Subsampling Floor ✅ COMPLETE
 
Trains SimpleCNN on randomly sampled tiny subsets. This is the naive baseline the number every intelligent curation strategy in Phases 3–5 must achieve.
 
**MNIST** (ceiling: 98.98%)
 
| Subset | Images | Top-1 Mean | ± Std | Gap to ceiling |
|--------|--------|-----------|-------|----------------|
| 0.2% | 120 | 24.31% | 10.76% | 74.7 pp |
| 0.5% | 300 | 82.03% | 10.29% | 16.9 pp |
| 1% | 600 | 94.54% | 0.82% | 4.4 pp |
| 2% | 1200 | 95.68% | 1.22% | 3.3 pp |
| 5% | 3000 | **96.87%** | 0.46% | 2.1 pp |
 
**CIFAR-10** (ceiling: 80.93%)
 
| Subset | Images | Top-1 Mean | ± Std | Gap to ceiling |
|--------|--------|-----------|-------|----------------|
| 0.2% | 100 | 23.29% | 2.98% | 57.6 pp |
| 0.5% | 250 | 33.12% | 3.06% | 47.8 pp |
| 1% | 500 | 40.08% | 0.25% | 40.9 pp |
| 2% | 1000 | 48.00% | 2.25% | 32.9 pp |
| 5% | 2500 | **58.05%** | 1.33% | 22.9 pp |
 
> CIFAR-10 has a **22.9 pp gap** at 5% between random selection and the full-dataset ceiling. That gap is what Phases 3–5 are designed to close.
 
---
 
### Phase 3: Class-Balanced Selection 🔲 PENDING
 
Fixes random sampling's key weakness accidental class imbalance at tiny sizes
by drawing an equal number of examples from every class. Isolates how much of
the random baseline's weakness is purely due to imbalance vs which examples are chosen.
 
---
 
### Phase 4: K-Means / Coverage Selection 🔲 PENDING
 
Selects examples that maximise diversity across the feature space, reducing
redundancy within each subset. Tests the coreset hypothesis: a well-spread
subset outperforms a random or merely balanced one.
 
---
 
### Phase 5: Hard-Example Mining 🔲 PENDING
 
Scores examples by difficulty and selects the most informative ones.
Reuses Phase 1 seed-42 checkpoints for cheap scoring.
Includes a guard against the over-pruning failure mode identified by Paul et al. (2021).
 
---
 
### Phase 6: Analysis & Synthesis 🔲 PENDING
 
Cross-phase comparison, per-class analysis, cost–benefit discussion
and a direct evidenced answer to the dissertation's title question.
 
---
 
## Key Papers
 
| Paper | Summary |
|-------|---------|
| [Toneva et al. (2019)](https://arxiv.org/abs/1812.05159) | Forgetting events — not all training examples are equal |
| [Coleman et al. (2020)](https://arxiv.org/abs/1906.11829) | Selection via Proxy — cheap proxy models can rank example importance |
| [Paul et al. (2021)](https://arxiv.org/abs/2107.07075) | EL2N scoring — importance-based pruning with an over-pruning warning |
 
---
 
## Declared Limitations
 
1. **No validation split**: scheduler watches training loss; consistent across all phases so cross-phase comparisons remain fair, but absolute numbers are affected.
2. **CIFAR augmentation confound**: augmentation inflates effective training data at tiny sizes; kept identical across all phases so comparisons are fair.
3. **AdaptiveAvgPool2d(2,2)**: retains 512 spatial features rather than collapsing to 128; deliberate choice to preserve coarse spatial layout.
---
