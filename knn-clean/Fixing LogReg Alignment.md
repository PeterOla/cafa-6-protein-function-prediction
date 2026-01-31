# LogReg Performance Mystery: A Complete Explanation

## The Goal

Train a Logistic Regression model to predict GO terms for proteins.
The approach: "Learn linear decision boundaries from protein embeddings."

---

## Part 1: How LogReg Works (The Concept)

Imagine you have 82,000 proteins with known GO functions.
Logistic Regression learns a **weighted sum** of embedding features for each GO term:

```
P(protein X has function GO:0001) = sigmoid(w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b)
```

Where `x` is the protein embedding and `w` are learned weights.

For **multi-label** classification (13,500 GO terms), we use **One-vs-Rest (OVR)**:
- Train 13,500 independent binary classifiers
- Each classifier answers: "Does this protein have THIS specific GO term?"

---

## Part 2: The Data Alignment Contract

The same alignment problem from KNN applies here. We have:

```
train_embeds_*.npy   → Ordered by train_seq.feather
Y_target_13500.npy   → Must ALSO be ordered by train_seq.feather
```

**Critical**: Row i of embeddings MUST correspond to Row i of targets.

---

## Part 3: What the Current Code Does

### ID Loading (Cell 13a)
```python
train_ids = pd.read_feather(WORK_ROOT / "parsed" / "train_seq.feather")["id"].astype(str)

# FIX: Clean IDs to match EntryID format
train_ids_clean = train_ids.str.extract(r"\|(.*?)\|")[0]
train_ids_clean = train_ids_clean.fillna(train_ids)
```

### Target Matrix Construction (Cell 13a)
```python
train_terms_top = train_terms[train_terms["term"].isin(top_terms)]
Y_df = train_terms_top.pivot_table(index="EntryID", columns="term", aggfunc="size", fill_value=0)
Y_df = Y_df.reindex(train_ids_clean, fill_value=0)  # ← KEY LINE
Y = Y_df.values.astype(np.float32)
```

The `reindex(train_ids_clean)` is **supposed** to align Y to the embedding order.

---

## Part 4: The Alignment Bug

### Bug 1: ID Cleaning Mismatch

The ID cleaning regex assumes UniProt format: `sp|P12345|NAME_HUMAN`

```python
train_ids_clean = train_ids.str.extract(r"\|(.*?)\|")[0]
```

**Problem**: If IDs in `train_seq.feather` are already clean (e.g., just `P12345`), the regex **returns NaN** and then `fillna(train_ids)` uses the original.

But what if `train_terms.tsv` uses a **different format**?

| `train_seq.feather` | `train_terms.tsv` | Match? |
|---------------------|-------------------|--------|
| `sp|P12345|NAME`    | `P12345`          | ✗ (after cleaning: P12345 ✓) |
| `P12345`            | `P12345`          | ✓ |
| `P12345`            | `sp|P12345|NAME`  | ✗ |

**Diagnosis Needed**: Verify the actual ID formats in both files.

### Bug 2: Y Reconstruction in Cell 13c

When `Y_target_13500.npy` doesn't exist, Cell 13c **rebuilds it**:

```python
train_ids = pd.read_feather(WORK_ROOT / 'parsed' / 'train_seq.feather')['id'].astype(str)
train_ids_clean = train_ids.str.extract(r"\|(.*?)\|")[0].fillna(train_ids)

train_terms_top = train_terms[train_terms["term"].isin(top_terms)]
Y_df = train_terms_top.pivot_table(index="EntryID", columns="term", aggfunc="size", fill_value=0)
Y_df = Y_df.reindex(train_ids_clean, fill_value=0)
```

**Problem**: If `Y_target_13500.npy` was created with a DIFFERENT ID cleaning method or order, you get silent misalignment.

---

## Part 5: Visual Diagnosis (What to Check)

### Check 1: ID Format Consistency
```python
# Run this diagnostic:
train_seq_ids = pd.read_feather(WORK_ROOT / "parsed" / "train_seq.feather")["id"]
train_term_ids = pd.read_parquet(WORK_ROOT / "parsed" / "train_terms.parquet")["EntryID"].unique()

print("train_seq sample:", train_seq_ids.head(5).tolist())
print("train_terms sample:", train_term_ids[:5].tolist())

# Check overlap
seq_set = set(train_seq_ids.astype(str))
term_set = set(train_term_ids.astype(str))
print(f"Overlap: {len(seq_set & term_set)} / {len(seq_set)} seq, {len(term_set)} terms")
```

### Check 2: Order Preservation
```python
# Verify Y is aligned with embeddings
Y_full = np.load(FEAT_DIR / 'Y_target_13500.npy', mmap_mode='r')
print(f"Y shape: {Y_full.shape}")  # Should be (n_train, 13500)

# Check a known protein
test_idx = 0
test_id = train_ids_clean.iloc[test_idx]
expected_terms = set(train_terms[train_terms['EntryID'] == test_id]['term'].tolist())
predicted_terms = set(top_terms[i] for i in np.where(Y_full[test_idx] > 0)[0])
print(f"Protein {test_id}:")
print(f"  Expected terms (from TSV): {len(expected_terms)}")
print(f"  Y matrix terms: {len(predicted_terms)}")
print(f"  Overlap: {len(expected_terms & predicted_terms)}")
```

---

## Part 6: Additional Issues

### Issue 1: IA-Weighted class_weight Misuse

**Current Code:**
```python
weights_chunk = weights_full[cols]
cw_dict = {i: float(weights_chunk[i]) for i in range(chunk_width)}
clf_chunk = cuOVR(cuLogReg(
    solver='qn', 
    max_iter=1000, 
    tol=1e-2, 
    class_weight=cw_dict  # ← This is WRONG
))
```

**Problem**: `class_weight` in sklearn/cuML is for **balancing classes within a single binary classifier**, not for weighting different GO terms.

- `class_weight={0: 1.0, 1: 5.0}` means "weight positive samples 5× more than negatives"
- `class_weight={0: 0.5, 1: 0.8, 2: 1.2, ...}` is meaningless for OVR

**Current code passes IA values as if they were class indices**, which is incorrect.

**Fix Options:**
1. **Remove class_weight entirely** (simplest)
2. **Use sample_weight** to weight each protein differently
3. **Post-hoc IA weighting** (multiply predictions by IA after inference)

### Issue 2: KFold vs StratifiedKFold

**Current Code:**
```python
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
```

**Problem**: Regular KFold doesn't account for label imbalance. Rare GO terms might have 0 positives in some folds.

**Fix:**
```python
from sklearn.model_selection import StratifiedKFold

# For multi-label, stratify by label count (a proxy)
label_counts = Y_full.sum(axis=1)
label_bins = pd.qcut(label_counts, q=5, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (idx_tr, idx_val) in enumerate(skf.split(np.arange(X.shape[0]), label_bins)):
    ...
```

### Issue 3: Test Prediction Averaging

**Current Code:**
```python
test_pred[:, start:end] += (test_gpu_buffer.get() / n_splits)
```

**Problem**: Test predictions are averaged across folds. This is correct for ensemble diversity, but the scaler is **re-fit per fold**. Different scalers → different feature scales → averaged predictions from incompatible models.

**Fix**: Either:
1. Fit scaler ONCE on all training data (outside fold loop)
2. Or, save per-fold models and do proper TTA (Test-Time Augmentation)

---

## Part 7: Required Changes Summary

### 🔴 **CRITICAL: Verify ID Alignment**

| Check | Action |
|-------|--------|
| Run diagnostic from Part 5 | Confirm IDs match between files |
| If mismatch | Fix ID cleaning to match formats |

### 🟠 **Major Changes**

| Issue | Current | Fix |
|-------|---------|-----|
| **class_weight misuse** | Passes IA as class indices | Remove or use post-hoc IA weighting |
| **KFold** | Regular KFold | StratifiedKFold by label count |
| **Scaler per fold** | Refit scaler each fold | Fit once outside loop |

### 🟡 **Moderate Changes**

| Issue | Current | Fix |
|-------|---------|-----|
| **Y_target_13500.npy rebuild** | Inline in Cell 13c | Use Cell 13a version only |
| **Degenerate labels** | Handled with -10 bias | Verify handling is correct |

---

## Part 8: Diagnostic Checklist

```python
# Run BEFORE training LogReg:

# 1. ID Format Check
train_seq_ids = pd.read_feather(WORK_ROOT / "parsed" / "train_seq.feather")["id"]
train_term_ids = pd.read_parquet(WORK_ROOT / "parsed" / "train_terms.parquet")["EntryID"].unique()
print("Seq ID sample:", train_seq_ids.head(3).tolist())
print("Term ID sample:", list(train_term_ids[:3]))

# 2. Alignment Sanity Check
Y_full = np.load(FEAT_DIR / 'Y_target_13500.npy', mmap_mode='r')
row_sums = Y_full.sum(axis=1)
print(f"Y row sums: min={row_sums.min()}, max={row_sums.max()}, mean={row_sums.mean():.1f}")
# Expect: min > 0 (every protein has at least 1 GO term)

# 3. Column Check
col_sums = Y_full.sum(axis=0)
zero_cols = (col_sums == 0).sum()
print(f"Zero columns in Y: {zero_cols} / {Y_full.shape[1]}")
# Expect: 0 (every term has at least 1 positive)
```

---

## Part 9: Implementation Plan

1. [ ] **Run ID diagnostic** (Part 5, Check 1)
2. [ ] **Run alignment diagnostic** (Part 5, Check 2)
3. [ ] **Fix ID cleaning if needed**
4. [ ] **Remove class_weight or fix IA application**
5. [ ] **Switch to StratifiedKFold**
6. [ ] **Fit scaler once outside fold loop**
7. [ ] **Re-run training and compare F1**

---

## User Input

Does this analysis make sense? Would you like me to:
1. Run the diagnostics first to confirm the bug exists?
2. Jump to implementing fixes?
