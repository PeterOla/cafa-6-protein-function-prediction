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

### Bug #1: Protein ID Ordering Mismatch (Early Fix)

The embeddings were generated from proteins in `train_seq.feather` order, but the label matrix `Y` was constructed using protein order from `train_terms.tsv`. These orderings are **different**:

```
train_seq.feather (embedding order):    A0A0C5B5G6, A0JNW5, A0JP26, ...
train_terms.tsv (label order):          Q5W0B1, Q5W0B1, Q5W0B1, ...
```

This caused embeddings to be paired with **wrong protein labels**, effectively randomizing the learning signal.

**Fix:** Cell 13a loads protein IDs from `train_seq.feather` and reindexes `Y_df` to match embedding order.

---

### Bug #2: Y Matrix Column Ordering (2026-01-31) — THE CRITICAL BUG

Even after fixing protein row order, results were **inverted**:

| Aspect | Observed | Expected |
|--------|----------|----------|
| BP | 0.2985 | ~0.12 |
| MF | 0.0215 | ~0.34 |
| CC | 0.0120 | ~0.28 |

**Root Cause:** `pd.pivot_table()` creates columns in **alphabetical order**, not the expected **BP→MF→CC** order.

#### The Problem

```python
# Cell 13a built Y_df like this:
Y_df = train_terms.pivot_table(
    index='protein', columns='term', values='label', fill_value=0
)
# Columns are sorted ALPHABETICALLY: GO:0000001, GO:0000002, ...
```

But `top_terms` (the vocabulary list) is ordered **BP first, then MF, then CC**:

```python
top_terms = bp_terms[:10000] + mf_terms[:2000] + cc_terms[:1500]
# Order: [BP terms at idx 0-9999] [MF terms at idx 10000-11999] [CC terms at idx 12000-13499]
```

#### Why This Broke Everything

The aspect indices assumed `top_terms` order:

```python
bp_term_idx = np.where([term_aspects.get(t) == 'BP' for t in top_terms])[0]  # Expected: [0, 1, ..., 9999]
mf_term_idx = np.where([term_aspects.get(t) == 'MF' for t in top_terms])[0]  # Expected: [10000, ..., 11999]
cc_term_idx = np.where([term_aspects.get(t) == 'CC' for t in top_terms])[0]  # Expected: [12000, ..., 13499]
```

But `Y_df` columns were **alphabetical**, so `Y_knn[:, bp_term_idx]` fetched the **wrong columns**:

| Index | Expected (BP→MF→CC) | Actual (Alphabetical) |
|-------|---------------------|----------------------|
| 0 | GO:0045944 (BP) | GO:0000001 (CC) |
| 5000 | GO:0014883 (BP) | GO:0005515 (MF) |
| 10000 | GO:0005515 (MF) | GO:0045944 (BP) |

GO terms starting with `GO:00...` span all three aspects, so alphabetical sorting **scrambled** the aspect boundaries.

#### The Fix

Add explicit column reindexing after pivot:

```python
# Reorder columns to match top_terms order (BP→MF→CC)
Y_df = Y_df.reindex(columns=top_terms, fill_value=0)
```

This single line ensures:
- Y columns are in **exact same order** as `top_terms`
- Aspect indices select **correct** column ranges
- BP labels → BP predictions, MF labels → MF predictions, CC labels → CC predictions

#### Verification Added

Cell 13a now includes 7 alignment checks:

```python
# 1. Column match
assert list(Y_df.columns) == top_terms, "[FATAL] Y_df columns != top_terms"

# 2. Aspect distribution
bp_count = sum(1 for t in top_terms if term_aspects.get(t) == 'BP')  # Must be 10000

# 3. Boundary checks
# idx 0-9999 must be BP, idx 10000-11999 must be MF, idx 12000-13499 must be CC

# 4-7. Sample term verification at specific indices
```

Cell 28 (KNN) also verifies alignment before evaluation.

### Impact

| Metric | Bug #1 Only Fixed | Bug #2 Fixed |
|--------|-------------------|--------------|
| BP | 0.2985 (wrong) | **0.1210** ✓ |
| MF | 0.0215 (wrong) | **0.3408** ✓ |
| CC | 0.0120 (wrong) | **0.2825** ✓ |
| CAFA F1 | 0.1107 | **0.2481** ✓ |

---

### Why This Bug Was Hard to Find

1. **No errors thrown** — NumPy happily slices wrong columns
2. **Plausible-looking scores** — F1 values were in valid range (0-1)
3. **Inverted pattern was subtle** — BP was ~3x expected, MF/CC were ~10x lower
4. **No obvious red flags** — Shape checks all passed (82404 × 13500)

### Key Insight

> When using `pivot_table()` to build a label matrix, **always** call `.reindex(columns=vocabulary)` to enforce the expected column order. Pandas does not preserve insertion order in pivot operations.

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

2. **Column ordering in pivot tables is dangerous** — `pd.pivot_table()` returns alphabetically sorted columns, not insertion order. Always use `.reindex(columns=vocabulary)` to enforce expected order.

3. **Vocabulary filtering can improve scores** — Rare terms with few positive examples add noise; filtering to frequent terms (13,500) improved F1 by +0.25%

4. **Aspect-specific hyperparameters matter** — BP, MF, and CC have different optimal K values due to their semantic structure

5. **Double-weighting hurts** — If the evaluation metric uses IA weights, don't apply them during training/inference

6. **Per-protein normalization is essential** — Required for proper threshold calibration

7. **Add fail-fast verification** — Every alignment point should have explicit checks that raise errors on mismatch. Silent failures are debugging nightmares.

8. **Inverted metrics are a red flag** — If one aspect is too high while others are crushed, suspect column/row ordering bugs, not hyperparameter issues.

---

## Part 9: Defensive Coding Patterns

### Pattern 1: Explicit Column Reindexing

```python
# BAD: Trust pivot_table ordering
Y_df = df.pivot_table(index='protein', columns='term', values='label')

# GOOD: Force explicit ordering
Y_df = df.pivot_table(index='protein', columns='term', values='label')
Y_df = Y_df.reindex(columns=vocabulary, fill_value=0)  # ← CRITICAL
```

### Pattern 2: Fail-Fast Alignment Checks

```python
# After building Y matrix
if list(Y_df.columns) != top_terms:
    raise RuntimeError(f"[FATAL] Column mismatch: Y_df has {len(Y_df.columns)} cols, expected {len(top_terms)}")

# Verify aspect boundaries
for idx, expected_aspect in [(0, 'BP'), (10000, 'MF'), (12000, 'CC')]:
    actual = term_aspects.get(top_terms[idx])
    if actual != expected_aspect:
        raise RuntimeError(f"[FATAL] top_terms[{idx}]={top_terms[idx]} is {actual}, expected {expected_aspect}")
```

### Pattern 3: Sample Spot Checks

```python
# Verify random indices across aspect boundaries
spot_checks = [0, 5000, 9999, 10000, 11000, 12000, 13499]
for idx in spot_checks:
    term = top_terms[idx]
    expected = 'BP' if idx < 10000 else ('MF' if idx < 12000 else 'CC')
    actual = term_aspects.get(term)
    print(f"  idx={idx}: {term}, expected={expected}, actual={actual} {'[OK]' if actual == expected else '[FAIL]'}")
```

---

## References

- `LOGIC_CHAIN.md` — Detailed reasoning chain for the investigation
- `investigation_knn_performance.md` — Initial investigation notes
- Original broken score: 0.083 (Kaggle submission, row mismatch)
- Intermediate broken score: 0.1107 (column ordering bug)
- Fixed score: 0.2481 (OOF validation)

---

## Timeline

| Date | Issue | Score |
|------|-------|-------|
| Initial | Row ordering mismatch (protein IDs) | 0.083 |
| Fix #1 | Reindex Y_df rows to match embeddings | ~0.11 |
| 2026-01-31 | **Column ordering bug discovered** | 0.1107 |
| 2026-01-31 | Added `Y_df.reindex(columns=top_terms)` | **0.2481** ✓ |

---

*Investigation completed: 2026-01-31*