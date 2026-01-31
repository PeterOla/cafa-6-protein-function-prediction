# KNN Alignment Investigation & Fixes

## Executive Summary

This document chronicles the investigation and resolution of critical bugs in the CAFA-6 KNN model that caused a **performance drop from expected ~0.24 to 0.083 F1 score**. The root cause was a **protein ID ordering mismatch** between embeddings and labels.

---

## Required Changes to `05_cafa_e2e.ipynb` Cell 13E (KNN)

### 🔴 **CRITICAL: Protein ID Ordering Bug**

| Current (Broken) | Required Fix | Status |
|------------------|--------------|--------|
| Uses `train_terms['EntryID'].unique()` for protein ordering | Load from `train_seq.feather` to match embedding order | ✅ **Already Fixed in Cell 13a** |

This was the **root cause of the 0.083 score**. Cell 13a already uses `train_ids_clean` from `train_seq.feather` and reindexes `Y_df` correctly.

---

### 🟠 **Major Changes**

| Issue | Current | Fix | Status |
|-------|---------|-----|--------|
| **Single K value** | `KNN_K = 10` (or 50) | **Mixed-K Ensemble**: BP=5, MF=10, CC=15 | ✅ **FIXED** - Aspect-aware batching implemented |
| **IA-Weighted Aggregation** | Uses IA weights during neighbor voting | **Remove IA from aggregation** (causes double-weighting since CAFA evaluation already uses IA) | ✅ **FIXED** - Pure similarity weighting |
| **Term Filtering** | Uses `top_terms_13500.json` (filtered) | Use **13,500 vocabulary** (validated to score higher than full vocab) | ✅ **FIXED** - `KNN_USE_FULL_VOCABULARY=False` (default) |
| **Thresholds** | Aspect-specific from `aspect_thresholds.json` | Use optimized thresholds: BP=0.40, MF=0.40, CC=0.30 | ✅ **Uses validated thresholds** |

---

### 🟡 **Moderate Changes**

| Issue | Current | Fix | Status |
|-------|---------|-----|--------|
| **Per-Protein Normalization** | Not applied | Add **per-protein max normalization** after aggregation | ✅ **FIXED** |
| **KFold vs StratifiedKFold** | Uses `KFold` | Use **`StratifiedKFold`** (stratify by label count) | ✅ **FIXED** |
| **Missing Y_nei definition** | `Y_nei` not defined in OOF loop | Add `Y_nei = Y_knn[neigh_b]` | ✅ **FIXED** |

---

## Summary Checklist

1. [x] **Load protein IDs from `train_seq.feather`** (already correct in Cell 13a)
2. [x] **Remove IA-weighted aggregation** (pure similarity weighting)
3. [x] **Use 13,500 term vocabulary** (`KNN_USE_FULL_VOCABULARY=False` default)
4. [x] **Implement Mixed-K Ensemble** (BP=5, MF=10, CC=15)
5. [x] **Thresholds** - Uses validated thresholds: BP=0.40, MF=0.40, CC=0.30
6. [x] **Add per-protein max normalization**
7. [x] **Switch to StratifiedKFold**
8. [x] **Fix missing `Y_nei` definition in OOF loop**
9. [x] **Clean up malformed code** (stray lines, concatenated statements)
10. [x] **Generate submission_knn.tsv** for direct KNN submission

---

## Part 2: Root Cause Analysis

### The Y Matrix Mismatch

The embeddings were generated from proteins in `train_seq.feather` order, but the label matrix `Y` was constructed using protein order from `train_terms.tsv`. These orderings are **different**:

```
train_seq.feather (embedding order):    A0A0C5B5G6, A0JNW5, A0JP26, ...
train_terms.tsv (label order):          Q5W0B1, Q5W0B1, Q5W0B1, ...
```

This caused embeddings to be paired with **wrong protein labels**, effectively randomizing the learning signal.

### Impact

| Metric | Broken Notebook | After Fix |
|--------|-----------------|-----------|
| CAFA F1 | **0.083** | **0.2481** |
| Improvement | — | **+199%** |

---

## Part 3: Vocabulary Size Experiment

| Configuration | Terms | CAFA F1 |
|---------------|-------|---------|
| Full vocabulary | ~26,125 | 0.2456 |
| **Filtered 13,500** | 13,500 | **0.2481** |

**Finding:** The filtered vocabulary scores **higher** because it focuses on terms with sufficient positive examples (≥10). Rare terms add noise to similarity-weighted aggregation.

---

## Part 4: Mixed-K Ensemble Optimization

| Configuration | K Values | CAFA F1 |
|---------------|----------|---------|
| Single K=10 | All aspects | ~0.22 |
| Single K=50 | All aspects | ~0.18 |
| **Mixed-K Ensemble** | BP=5, MF=10, CC=15 | **~0.245** |

**Finding:** Optimal K varies by GO aspect:
- **BP (Biological Process)**: K=5 — more specific, needs fewer neighbors
- **MF (Molecular Function)**: K=10 — balanced
- **CC (Cellular Component)**: K=15 — broader structural context needed

---

## Part 5: Threshold Sweep Validation

| Aspect | t=0.30 | t=0.35 | **t=0.40** | t=0.45 |
|--------|--------|--------|------------|--------|
| BP | 0.1201 | 0.1227 | **0.1231** ← | 0.1155 |
| MF | 0.3431 | 0.3453 | **0.3461** ← | 0.3443 |
| CC | **0.2796** ← | 0.2746 | 0.2715 | 0.2581 |

---

## Part 6: Final Validation Results

### Configuration

```python
USE_FULL_VOCABULARY = False      # 13,500 terms
KNN_K_BY_ASPECT = {'BP': 5, 'MF': 10, 'CC': 15}
KNN_THRESHOLDS = {'BP': 0.40, 'MF': 0.40, 'CC': 0.30}
```

### Expected Output

```
[EVALUATION] Per-Aspect CAFA F1 (OOF)
======================================================================
  BP: F1=0.1210 (threshold=0.40, n_terms=10000)
  MF: F1=0.3408 (threshold=0.40, n_terms=2000)
  CC: F1=0.2825 (threshold=0.30, n_terms=1500)

  CAFA F1: 0.2481
======================================================================
```

---

## Part 7: Output Artifacts

### Files Generated by Cell 13E

| File | Shape | Purpose |
|------|-------|---------|
| `oof_pred_knn.npy` | (82404, 13500) | OOF predictions for stacker |
| `test_pred_knn.npy` | (224309, 13500) | Test predictions for stacker |
| `oof_pred_knn_thresholded.npy` | (82404, 13500) | Thresholded OOF |
| `test_pred_knn_thresholded.npy` | (224309, 13500) | Thresholded test |
| `submission_knn.tsv` | ~1.5M rows | Direct KNN submission |

### Standalone Scripts

| Script | Vocabulary | Mode | Use Case |
|--------|------------|------|----------|
| `knn_esm2_3b_13500.py` | 13,500 | CPU | E2E compatible |
| `knn_esm2_3b_gpu.py` | Full ~26K | GPU | Fast iteration |
| `knn_esm2_3b.py` | Full ~26K | CPU | Baseline |

---

## Part 8: Key Learnings for Research

1. **Data alignment is critical** — A single row order mismatch can destroy model performance completely (0.083 vs 0.245)

2. **Vocabulary filtering can improve scores** — Rare terms with few positive examples add noise; filtering to frequent terms (13,500) improved F1 by +0.25%

3. **Aspect-specific hyperparameters matter** — BP, MF, and CC have different optimal K values due to their semantic structure

4. **Double-weighting hurts** — If the evaluation metric uses IA weights, don't apply them during training/inference

5. **Per-protein normalization is essential** — Required for proper threshold calibration

---

## References

- `LOGIC_CHAIN.md` — Detailed reasoning chain for the investigation
- `investigation_knn_performance.md` — Initial investigation notes
- Original broken score: 0.083 (Kaggle submission)
- Fixed score: 0.2481 (OOF validation)

---

*Investigation completed: 2026-01-31*