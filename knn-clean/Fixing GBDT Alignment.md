# GBDT Performance Mystery: A Complete Explanation

## The Goal

Train a Gradient Boosted Decision Tree (GBDT) model to predict GO terms for proteins.
The approach: "Build an ensemble of decision trees that sequentially correct each other's errors."

---

## Part 1: How GBDT Works (The Concept)

GBDT is an ensemble method:
1. **Tree 1**: Fits to the target labels
2. **Tree 2**: Fits to the **residual errors** of Tree 1
3. **Tree 3**: Fits to the residual errors of Trees 1+2
4. ... and so on for N trees

Each tree is weak, but together they become strong.

For **multi-output** (1,585 GO terms), py-boost trains a single model that predicts ALL targets simultaneously using `target_splitter='OneVsAll'`.

---

## Part 2: The Data Alignment Contract

Same contract as KNN and LogReg:

```
X (features)    → Ordered by train_seq.feather
Y (targets)     → MUST also be ordered by train_seq.feather
```

**Row i of X MUST correspond to Row i of Y.**

---

## Part 3: What the Current Code Does

### Data Loading (Cell 13a)
```python
train_ids = pd.read_feather(WORK_ROOT / "parsed" / "train_seq.feather")["id"].astype(str)
train_ids_clean = train_ids.str.extract(r"\|(.*?)\|")[0]
train_ids_clean = train_ids_clean.fillna(train_ids)

train_terms_top = train_terms[train_terms["term"].isin(top_terms)]
Y_df = train_terms_top.pivot_table(index="EntryID", columns="term", aggfunc="size", fill_value=0)
Y_df = Y_df.reindex(train_ids_clean, fill_value=0)  # ← Alignment step
Y = Y_df.values.astype(np.float32)
```

### GBDT Training (Cell 13b)
```python
# Uses elite 1,585 terms (≥50 positives)
elite_cols = np.asarray(stable_idx, dtype=np.int64)
Y_elite = Y[:, elite_cols].astype(np.float32, copy=False)

# KFold splitting
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
    X_tr = np.ascontiguousarray(X[tr_idx])
    Y_tr = np.ascontiguousarray(Y_elite[tr_idx])
    
    model = GradientBoosting(**base_params, target_splitter='OneVsAll')
    model.fit(X_tr, Y_tr, eval_sets=[{'X': X_va, 'y': Y_va}])
```

---

## Part 4: The Alignment Bug (Same Root Cause)

### Bug 1: ID Cleaning Mismatch

The regex `r"\|(.*?)\|"` assumes UniProt FASTA format.

```python
train_ids_clean = train_ids.str.extract(r"\|(.*?)\|")[0]
train_ids_clean = train_ids_clean.fillna(train_ids)
```

**Problem**: If IDs in `train_seq.feather` don't match the pattern, the regex returns NaN, and `fillna` uses the original ID.

But if `train_terms.tsv` uses a **different format**, `Y_df.reindex(train_ids_clean)` will:
1. **Not find** some proteins → all-zero rows
2. **Misorder** proteins → wrong labels attached

### Bug 2: Y Inherited from Cell 13a

GBDT uses `Y` from Cell 13a:
```python
Y_elite = Y[:, elite_cols].astype(np.float32, copy=False)
```

If Cell 13a's alignment is broken, GBDT inherits the broken labels.

---

## Part 5: GBDT-Specific Issues

### Issue 1: Elite 1,585 vs Full 13,500

**Current Design:**
```python
elite_cols = np.asarray(stable_idx, dtype=np.int64)  # 1,585 terms
Y_elite = Y[:, elite_cols]  # Train only on elite terms

# But outputs are written to FULL 13,500-wide matrix:
oof_pred_gbdt[np.ix_(va_idx, elite_cols)] = va
test_pred_gbdt[:, elite_cols] += te / float(n_splits)
```

**Problem**: 
- Non-elite columns (13,500 - 1,585 = 11,915 terms) remain **all zeros**
- Downstream stacker expects predictions for ALL 13,500 terms
- Zero predictions = zero contribution for 88% of terms

**Question**: Is this intentional? If so, the stacker must handle sparse GBDT contributions.

### Issue 2: KFold vs StratifiedKFold

**Current Code:**
```python
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
```

**Problem**: Even with elite terms (≥50 positives), some folds may have very few positives for certain terms, leading to unstable tree splits.

### Issue 3: Test Prediction Averaging with Early Stopping

**Current Code:**
```python
model.fit(X_tr, Y_tr, eval_sets=[{'X': X_va, 'y': Y_va}])  # Early stopping enabled
# ...
test_pred_gbdt[:, elite_cols] += te / float(n_splits)  # Averaged
```

**Observation**: Each fold's model uses early stopping, potentially stopping at different iterations (best_iter varies per fold).

**Is this a bug?** No, this is standard practice. Different folds should be allowed to stop at different iterations.

### Issue 4: No IA Weighting

Unlike LogReg (which attempted class_weight), GBDT has **no IA weighting**.

**Current:** Equal weight for all 1,585 elite terms.

**Potential Fix:**
```python
# Option 1: sample_weight (weight each protein)
# Option 2: Post-hoc scaling (multiply predictions by IA)
# Option 3: Custom loss function (py-boost supports this)
```

For CAFA evaluation, IA weighting happens at **metric computation**, not training. So this may not be a bug, just an optimisation opportunity.

---

## Part 6: Visual Diagnosis (What to Check)

### Check 1: Y Matrix Sanity
```python
# Run AFTER Cell 13a, BEFORE Cell 13b:
print(f"Y shape: {Y.shape}")  # Expected: (n_train, 13500)
print(f"Y dtype: {Y.dtype}")  # Expected: float32

# Row sums: each protein should have ≥1 GO term
row_sums = Y.sum(axis=1)
print(f"Row sums: min={row_sums.min()}, max={row_sums.max()}, mean={row_sums.mean():.1f}")
# WARNING if min == 0: some proteins have NO labels

# Elite subset
Y_elite = Y[:, stable_idx]
elite_row_sums = Y_elite.sum(axis=1)
print(f"Elite row sums: min={elite_row_sums.min()}, mean={elite_row_sums.mean():.1f}")
```

### Check 2: Elite Column Validity
```python
# Verify stable_idx maps to correct terms
print(f"stable_idx: min={stable_idx.min()}, max={stable_idx.max()}, len={len(stable_idx)}")
print(f"Y.shape[1]={Y.shape[1]}")  # stable_idx.max() must be < Y.shape[1]

# Check term counts for elite terms
col_sums = Y[:, stable_idx].sum(axis=0)
print(f"Elite column sums: min={col_sums.min()}, mean={col_sums.mean():.1f}")
# Expected: min ≥ 50 (by definition of "elite")
```

### Check 3: OOF Prediction Sanity
```python
# Run AFTER GBDT training:
oof = np.load(PRED_DIR / 'oof_pred_gbdt.npy')

# Elite columns should have predictions
elite_slice = oof[:, stable_idx]
print(f"Elite OOF: min={elite_slice.min():.4f}, max={elite_slice.max():.4f}")

# Non-elite columns should be zero
non_elite_mask = np.ones(oof.shape[1], dtype=bool)
non_elite_mask[stable_idx] = False
non_elite_slice = oof[:, non_elite_mask]
print(f"Non-elite OOF: max={non_elite_slice.max():.4f}")  # Expected: 0.0
```

---

## Part 7: Required Changes Summary

### 🔴 **CRITICAL: Verify ID Alignment**

| Check | Action |
|-------|--------|
| Run diagnostic from Part 6 | Confirm Y has correct shape and non-zero rows |
| If `row_sums.min() == 0` | ID alignment is broken; fix in Cell 13a |

### 🟠 **Major Considerations**

| Issue | Current | Recommendation |
|-------|---------|----------------|
| **Elite-only training** | 1,585 / 13,500 terms | Verify stacker handles sparse GBDT contribution |
| **KFold** | Regular KFold | Consider StratifiedKFold for rare terms |
| **No IA weighting** | Equal weight | Post-hoc IA scaling or custom loss (optional) |

### 🟡 **Moderate Checks**

| Issue | Current | Fix |
|-------|---------|-----|
| **X/Y from globals** | Depends on Cell 13a | Add explicit sanity assertions |
| **Y_elite indexing** | `Y[:, elite_cols]` | Verify `elite_cols` matches `stable_terms_1585.json` |

---

## Part 8: The Elite Term Strategy

GBDT trains only on **1,585 "elite" terms** (≥50 positives).

**Rationale:**
- GBDT struggles with extreme class imbalance (rare terms with <50 positives)
- Elite terms have enough signal for decision trees to learn meaningful splits
- Rare terms are better handled by LogReg (linear) or KNN (neighbour-based)

**Consequence for Stacker:**
- GBDT contributes predictions for 1,585 / 13,500 = 11.7% of terms
- For other 88.3% of terms, stacker must rely on LogReg, KNN, DNN

**Is this a bug?** No, it's an intentional design choice. But the stacker must be aware of this sparsity.

---

## Part 9: Diagnostic Checklist

```python
# Run BEFORE GBDT training:

# 1. X/Y Shape Check
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
assert X.shape[0] == Y.shape[0], "Row mismatch!"

# 2. Elite Index Validity
print(f"stable_idx: len={len(stable_idx)}, max={stable_idx.max()}")
assert stable_idx.max() < Y.shape[1], "stable_idx out of bounds!"
assert len(stable_idx) == 1585, "Expected 1585 elite terms"

# 3. Y Row Sum Check
row_sums = Y.sum(axis=1)
zero_rows = (row_sums == 0).sum()
print(f"Y rows with zero labels: {zero_rows}")
if zero_rows > 0:
    print("WARNING: Some proteins have NO GO terms. Check ID alignment!")

# 4. Elite Column Sum Check
col_sums = Y[:, stable_idx].sum(axis=0)
low_cols = (col_sums < 50).sum()
print(f"Elite columns with <50 positives: {low_cols}")
# Expected: 0 (by definition)
```

---

## Part 10: Implementation Plan

1. [ ] **Run Y matrix diagnostic** (Part 6, Check 1)
2. [ ] **Verify stable_idx validity** (Part 6, Check 2)
3. [ ] **If issues found → fix Cell 13a alignment**
4. [ ] **After training → verify OOF predictions** (Part 6, Check 3)
5. [ ] **Optional: add IA weighting via post-hoc scaling**
6. [ ] **Optional: switch to StratifiedKFold**

---

## Part 11: Key Difference from LogReg

| Aspect | LogReg | GBDT |
|--------|--------|------|
| **Terms trained** | 13,500 (per-aspect split) | 1,585 (elite only) |
| **IA weighting** | Attempted (incorrectly) | None |
| **Multi-output** | OVR (13,500 binary) | Single model (1,585 outputs) |
| **Output shape** | Per-aspect files | Single 13,500-wide matrix |

GBDT is **simpler** in some ways (single multi-output model) but **sparser** (elite-only).

---

## User Input

Does this analysis make sense? Would you like me to:
1. Run the diagnostics to confirm alignment?
2. Proceed to DNN documentation?
