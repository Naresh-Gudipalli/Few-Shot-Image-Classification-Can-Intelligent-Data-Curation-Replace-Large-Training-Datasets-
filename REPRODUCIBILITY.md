# Reproducibility

How to re-run this project and obtain the results reported in the README and the dissertation.

---

## 1. Environment

Developed and run entirely on **Kaggle notebooks with a T4x2 GPU accelerator**.

```
Python 3.11
torch, torchvision      (CUDA build; CPU works but is substantially slower)
numpy
scikit-learn            KMeans, NearestNeighbors, F1 / confusion metrics
scipy                   spearmanr, ttest_rel, pearsonr
matplotlib
```

```bash
pip install torch torchvision numpy scikit-learn scipy matplotlib
```

No other dependencies. No code from third parties is included in this repository the
forgetting-event and coreset methods were implemented from the descriptions in their respective
papers (see the Key Papers table in the README), not copied from the authors' repositories.

---

## 2. Datasets

Both datasets are standard public benchmarks and download automatically on first run no
manual step is required.

| Dataset | Source | Obtained by |
|---|---|---|
| MNIST | LeCun et al. | `torchvision.datasets.MNIST(root=..., download=True)` |
| CIFAR-10 | Krizhevsky | `torchvision.datasets.CIFAR10(root=..., download=True)` |

Raw dataset files are **not** committed to this repository (`.gitignore` excludes them). On
Kaggle, CIFAR-10 downloads slowly; staging it by symlink from an existing notebook output is
faster than re-downloading, and torchvision MD5-verifies the archive either way, so the worst
case is a slow download rather than corrupted data.

---

## 3. Run order

Later phases consume earlier phases outputs, so order matters.

| Step | Phase | Depends on | Produces |
|---|---|---|---|
| 1 | `phase1-baseline` | Raw datasets, random seeds, training hyperparameters | Five per-seed checkpoints (one per seed, per dataset) + the ceiling results |
| 2 | `phase2-random-subsampling` | Nothing from Phase 1 | The random-selection floor |
| 3 | `phase3-class-balanced` | Nothing from Phase 1 | Class-balanced results, and the per-class quota used by Phases 4–5 |
| 4 | `phase4-k-means` | Phase 1 checkpoints (for embeddings) | K-means coreset results, selection indices |
| 5 | `phase-5-hard-examples-mining` | Phase 1 checkpoints (for forgetting scores) | Hardest-first results, plus the Phase 5b difficulty-axis arms |

**Phase 1 must run first.** Its checkpoints are read-only inputs to Phases 4 and 5; those phases
will not run without them.

---

## 4. Checkpoints

Phase 1 writes five checkpoints per dataset, named:

```
baseline_mnist_seed{42,123,456,789,1011}_model.pth
baseline_cifar10_seed{42,123,456,789,1011}_model.pth
```

Note `cifar10`, not `cifar`. Later phases locate these with a recursive `glob()` search rather
than a hardcoded path, so they can be attached from any Kaggle input directory without editing
code. If running locally, place them anywhere under the search root and the glob will find them.

Checkpoints are **not committed to this repository** (`.gitignore` excludes `*.pth`) they are
binaries, and regenerating them is a documented, deterministic step. To recreate them, run
Phase 1's training path in full.

---

## 5. Determinism and what actually reproduces

Every source of randomness is seeded:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# plus seed_worker() for DataLoader worker processes, which are separate
# processes with independent RNG state
```

**What reproduces exactly:** all CIFAR-10 results, across every phase. Re-running the Phase 5
`hard` arm reproduces its locked values (10.11 / 9.89 / 15.01 / 22.42 / 38.06) byte-identically.

**What does not:** MNIST results for the difficulty-based phases (5 and 5b). Three independent
runs of identical code produced three different result sets the largest discrepancy being
17.92 pp at the 0.5% subset. The cause is structural rather than a coding error: MNIST converges
so completely that almost all forgetting counts collapse to 0 or 1, so the difficulty ranking is
dominated by random tie-breaking rather than genuine signal. Small platform-level
non-determinism (already documented in Phase 1, where two runs of the ceiling gave 98.98% vs
99.16%) is amplified by this into large swings in which examples fall into the thin never-learned
tier.


---

## 6. Configuration integrity

Every phase serialises its fixed constants, hashes them, and embeds the first ten hex characters
of that hash in the filename of any cached statistics it writes:

```python
CONFIG_HASH = hashlib.md5(
    json.dumps(CONFIG_FIELDS, sort_keys=True).encode()).hexdigest()[:10]
```

`sort_keys=True` makes the hash independent of dictionary insertion order. If any constant were
edited, the hash would change, every cache lookup would miss, and the change would surface
immediately as a re-run rather than silently as a contaminated comparison.

**The hash `db554d58b3` has held unchanged from Phase 1 through Phase 5b.** This is the strongest
single piece of evidence that no experimental constant drifted across the project, which matters
because every finding here is a comparison *between* phases.

---

## 7. Fixed constants

Identical in every phase. Changing any of these invalidates cross-phase comparison.

| | |
|---|---|
| Architecture | SimpleCNN 3 conv blocks (32→64→128), AdaptiveAvgPool2d(2,2), FC 512→256→10 |
| Parameters | 227,018 (MNIST, 1 channel) · 227,594 (CIFAR-10, 3 channels) |
| Optimiser | Adam, lr=1e-3, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau(mode='min', factor=0.5, patience=3), steps on **training** loss |
| Loss | CrossEntropyLoss (no explicit softmax in the model the loss applies it) |
| Dropout | 0.5 |
| Seeds | [42, 123, 456, 789, 1011] |
| Subset sizes | 0.2%, 0.5%, 1%, 2%, 5% |
| Epochs | MNIST 20 · CIFAR-10 30 |
| Batch size | 128 (`min(BATCH_SIZE, len(subset))` for very small subsets) |
| Workers | 2, with `seed_worker` |
| MNIST normalisation | mean 0.1307, std 0.3081 no augmentation |
| CIFAR-10 normalisation | mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010) |
| CIFAR-10 augmentation | RandomHorizontalFlip + RandomCrop(32, padding=4) **train split only** |

---
