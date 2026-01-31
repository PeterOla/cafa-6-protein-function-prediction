# DNN Performance Mystery: A Complete Explanation

## The Goal

Train a Deep Neural Network to predict GO terms for proteins.
The approach: "Learn complex non-linear feature interactions from multimodal protein embeddings."

---

## Part 1: How the Multi-Branch DNN Works (The Concept)

Unlike a simple MLP, this DNN has **one branch per modality**:

```
ProtT5 (1024D) ──→ [Head] ──┐
ESM2-650M (1280D) ──→ [Head] ──┤
ESM2-3B (2560D) ──→ [Head] ──┤
Ankh (1536D) ──→ [Head] ──├──→ [Fusion] ──→ [Trunk] ──→ 13,500 outputs
Text TF-IDF (10279D) ──→ [Head] ──┤
Taxa OHE (~100D) ──→ [Head] ──┤
(Optional) GBDT probs ──→ [Head] ──┘
```

Each **Head** is:
```
Linear(in_dim → 1024) → BatchNorm → ReLU → Dropout
Linear(1024 → 512) → BatchNorm → ReLU → Dropout
```

The **Trunk** is:
```
Concat all head outputs → Linear(fused_dim → 2048) → BatchNorm → ReLU → Dropout → Linear(2048 → 13500)
```

**Loss**: IA-weighted Binary Cross-Entropy (each term weighted by its Information Accretion).

---

## Part 2: The Data Alignment Contract

Same contract as KNN, LogReg, GBDT:

```
features_train[modality]  → Ordered by train_seq.feather
Y (targets)               → MUST also be ordered by train_seq.feather
```

**Row i of features MUST correspond to Row i of Y.**

---

## Part 3: What the Current Code Does

### Data Loading (from Cell 13a globals)
```python
# DNN uses features_train and features_test from Cell 13a
if 'features_train' not in globals() or 'features_test' not in globals():
    raise RuntimeError('Missing `features_train`/`features_test`. Run Cell 13a first.')
if 'Y' not in globals():
    raise RuntimeError('Missing Y. Run Cell 13a first (targets).')
```

### Multi-Modal Dataset
```python
class MultiModalDataset(Dataset):
    def __init__(self, X_dict, y, keys, idx):
        self.X_dict = X_dict
        self.y = y
        self.keys = keys
        self.idx = np.asarray(idx, dtype=np.int64)
    
    def __getitem__(self, i):
        j = int(self.idx[i])
        xs = [np.asarray(self.X_dict[k][j], dtype=np.float32) for k in self.keys]
        yy = np.asarray(self.y[j], dtype=np.float32)
        return xs, yy
```

### Training Loop
```python
n_splits = 5
n_seeds = 5  # Extreme ensembling: 5 × 5 = 25 models

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(train_n)), start=1):
    for seed in range(n_seeds):
        model = train_one_seed_fold(tr_idx, va_idx, seed=seed, ...)
        
        # OOF preds accumulate across seeds
        oof_pred_dnn[va_idx] += preds_va
        counts[va_idx] += 1.0
        
        # Test preds: averaged within fold, then across folds
        fold_test += preds_te
    
    test_pred_dnn += (fold_test / float(n_seeds))

# Final averaging
oof_pred_dnn = oof_pred_dnn / np.maximum(counts, 1.0)
test_pred_dnn = test_pred_dnn / float(n_splits)
```

---

## Part 4: The Alignment Bug (Same Root Cause)

### Bug 1: Inherited from Cell 13a

DNN uses `Y` and `features_train` directly from Cell 13a:

```python
# Cell 13a creates Y with:
train_ids_clean = train_ids.str.extract(r"\|(.*?)\|")[0].fillna(train_ids)
Y_df = Y_df.reindex(train_ids_clean, fill_value=0)
Y = Y_df.values.astype(np.float32)
```

If Cell 13a's alignment is broken, DNN inherits broken labels.

### Bug 2: Modality Arrays from load_features_dict

```python
# Cell 13a's load_features_dict:
ft_train[key] = np.load(tr_path, mmap_mode="r")
```

These .npy files were created by earlier cells (e.g., embeddings generator).
**All must be ordered by `train_seq.feather`.**

**Risk**: If ANY modality was generated with a different ordering, fusion is corrupted.

---

## Part 5: DNN-Specific Issues

### Issue 1: IA-Weighted BCE — Correct Implementation

**Current Code:**
```python
ia_map = dict(zip(ia_df[term_col].astype(str).values, ia_df[ia_col].astype(np.float32).values))
weights = np.asarray([ia_map.get(t, np.float32(1.0)) for t in top_terms], dtype=np.float32)
w_t = torch.from_numpy(weights).view(1, -1)

# In training loop:
loss_per = F.binary_cross_entropy_with_logits(logits, yb, reduction='none')
loss = (loss_per * w).mean()  # ← IA weighting per term
```

**Verdict**: ✅ Correct. Unlike LogReg's broken `class_weight` usage, DNN applies IA weights **post-loss** as a per-term multiplier. This is the right approach.

### Issue 2: KFold vs StratifiedKFold

**Current Code:**
```python
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
```

**Same issue as LogReg/GBDT**: Regular KFold doesn't stratify by label count.

**Impact for DNN**: With 25 models (5×5), rare term variance is somewhat smoothed by extreme ensembling. Less critical than single-model approaches, but still suboptimal.

### Issue 3: GBDT as 7th Modality (Teacher Features)

**Current Code:**
```python
gbdt_oof_path = PRED_DIR / 'oof_pred_gbdt.npy'
gbdt_test_path = PRED_DIR / 'test_pred_gbdt.npy'
use_pb = bool(gbdt_oof_path.exists() and gbdt_test_path.exists())

if use_pb:
    pb_oof = np.load(gbdt_oof_path, mmap_mode='r')
    dnn_train['pb'] = pb_oof  # GBDT OOF as training features
    dnn_test['pb'] = pb_test   # GBDT test as test features
```

**Leakage Analysis:**
- Train: Uses **OOF** GBDT predictions → No leakage (OOF is held-out per fold)
- Test: Uses **test** GBDT predictions → No leakage (test is always held out)

**Verdict**: ✅ Leakage-safe. This is knowledge distillation, not target leakage.

**Subtle Issue**: If GBDT alignment is broken, its OOF predictions are random noise as features.

### Issue 4: Extreme Ensembling (5×5 = 25 Models)

**Current Code:**
```python
n_splits = 5
n_seeds = 5

for fold, (tr_idx, va_idx) in enumerate(kf.split(...)):
    for seed in range(n_seeds):
        torch.manual_seed(42 + seed)
        model = train_one_seed_fold(...)
        oof_pred_dnn[va_idx] += preds_va  # Accumulate
        fold_test += preds_te
```

**Observation**: OOF predictions are averaged across 5 seeds; test predictions are averaged across 5 seeds × 5 folds.

**Is this correct?**
- OOF: Each validation sample is predicted by 5 seeds → averaged → good
- Test: Each test sample is predicted by 25 models → averaged → good

**Verdict**: ✅ Correct ensembling logic.

### Issue 5: Output Contract (13,500 Columns)

**Current Code:**
```python
out_dim = int(Y.shape[1])
if out_dim != 13500:
    raise RuntimeError(f'DNN expects 13,500 labels; got out_dim={out_dim}')

# ... training ...

if int(oof_pred_dnn.shape[1]) != 13500 or int(test_pred_dnn.shape[1]) != 13500:
    raise RuntimeError(f'DNN output contract violated')
```

**Verdict**: ✅ Contract is explicitly enforced.

---

## Part 6: Visual Diagnosis (What to Check)

### Check 1: Y Matrix Sanity (Before DNN Training)
```python
# Same as GBDT check
print(f"Y shape: {Y.shape}")  # Expected: (n_train, 13500)
row_sums = Y.sum(axis=1)
print(f"Row sums: min={row_sums.min()}, max={row_sums.max()}")
# WARNING if min == 0
```

### Check 2: Modality Alignment
```python
# Verify all modalities have same number of rows
for k in dnn_keys:
    print(f"{k}: {dnn_train[k].shape}")
# All should have shape (n_train, dim_k)

# Spot-check: Compare embedding norms at known positions
import numpy as np
idx = 0
print(f"Protein at index {idx}:")
print(f"  T5 norm: {np.linalg.norm(features_train['t5'][idx]):.4f}")
print(f"  ESM2-3B norm: {np.linalg.norm(features_train['esm2_3b'][idx]):.4f}")
```

### Check 3: OOF Prediction Sanity (After Training)
```python
oof = np.load(dnn_oof_path)
print(f"OOF shape: {oof.shape}")  # Expected: (n_train, 13500)
print(f"OOF range: [{oof.min():.4f}, {oof.max():.4f}]")  # Expected: [0, 1]
print(f"OOF mean: {oof.mean():.4f}")  # Typical: 0.001–0.05

# Check for NaN/Inf
print(f"NaN: {np.isnan(oof).sum()}, Inf: {np.isinf(oof).sum()}")
```

### Check 4: IA Weights Mapping
```python
# Verify IA weights match term order
print(f"weights shape: {weights.shape}")  # Expected: (13500,)
print(f"weights range: [{weights.min():.4f}, {weights.max():.4f}]")

# Spot-check term-to-IA mapping
sample_terms = top_terms[:5]
for t in sample_terms:
    print(f"  {t}: IA={ia_map.get(t, 'MISSING')}")
```

---

## Part 7: Required Changes Summary

### 🔴 **CRITICAL: Verify Inherited Alignment**

| Check | Action |
|-------|--------|
| Run Y matrix diagnostic | Confirm Y rows match embedding rows |
| Run modality alignment check | All modalities must have same row count |
| If misaligned | Fix Cell 13a (single source of truth) |

### 🟠 **Major Considerations**

| Issue | Current | Recommendation |
|-------|---------|----------------|
| **IA-weighted BCE** | ✅ Correct | Keep as-is |
| **KFold** | Regular KFold | Consider StratifiedKFold (lower priority due to 25-model smoothing) |
| **GBDT as modality** | ✅ Leakage-safe | Ensure GBDT alignment is correct first |

### 🟡 **Moderate Checks**

| Issue | Current | Fix |
|-------|---------|-----|
| **Output contract** | ✅ Enforced | Keep guardrails |
| **Extreme ensembling** | ✅ Correct | Keep 5×5 strategy |

---

## Part 8: Comparison with Other Models

| Aspect | LogReg | GBDT | DNN |
|--------|--------|------|-----|
| **Terms trained** | 13,500 (per-aspect) | 1,585 (elite) | 13,500 (all) |
| **IA weighting** | ❌ Broken | ❌ None | ✅ Correct (BCE) |
| **Models per run** | 1 per aspect | 1 | 25 (5×5) |
| **Modalities** | Flat concat | Flat concat | Per-modality heads |
| **Teacher features** | No | No | Yes (GBDT OOF optional) |

DNN is the **most complex** Level-1 model:
- Multi-branch architecture
- Correct IA weighting
- Optional teacher features from GBDT
- Extreme ensembling (25 models)

**If alignment is correct**, DNN should be the strongest single Level-1 contributor.

---

## Part 9: Diagnostic Checklist

```python
# Run BEFORE DNN training:

# 1. Globals Check
assert 'features_train' in globals(), "Missing features_train"
assert 'features_test' in globals(), "Missing features_test"
assert 'Y' in globals(), "Missing Y"

# 2. Shape Consistency
print(f"Y shape: {Y.shape}")
for k in ['t5', 'esm2_650m', 'esm2_3b', 'ankh', 'text', 'taxa']:
    print(f"  {k}: {features_train[k].shape}")
    assert features_train[k].shape[0] == Y.shape[0], f"{k} row mismatch!"

# 3. Y Row Sum Check
row_sums = Y.sum(axis=1)
zero_rows = (row_sums == 0).sum()
print(f"Y rows with zero labels: {zero_rows}")
if zero_rows > 0:
    print("WARNING: Some proteins have NO GO terms. Check ID alignment!")

# 4. IA Weights Check
ia_path = WORK_ROOT / 'IA.tsv'
assert ia_path.exists(), "Missing IA.tsv"
ia_df = pd.read_csv(ia_path, sep='\t')
print(f"IA.tsv terms: {len(ia_df)}")
```

---

## Part 10: Implementation Plan

1. [ ] **Verify Cell 13a alignment** (single source of truth)
2. [ ] **Run modality shape checks**
3. [ ] **Run Y row sum diagnostic**
4. [ ] **If issues found → fix Cell 13a first** (all Level-1 models depend on it)
5. [ ] **After training → verify OOF predictions are finite and sensible**
6. [ ] **Optional: switch to StratifiedKFold** (lower priority)

---

## Part 11: The IA Weighting is Correct (Unlike LogReg)

**LogReg (BROKEN):**
```python
cw_dict = {i: float(weights_chunk[i]) for i in range(chunk_width)}
clf_chunk = cuOVR(cuLogReg(class_weight=cw_dict))  # ← Wrong semantic!
```

**DNN (CORRECT):**
```python
w_t = torch.from_numpy(weights).view(1, -1)  # Shape: (1, 13500)

# In loss computation:
loss_per = F.binary_cross_entropy_with_logits(logits, yb, reduction='none')  # (batch, 13500)
loss = (loss_per * w).mean()  # Element-wise multiply, then mean
```

DNN applies IA as a **per-term loss weight**, which is mathematically correct.

---

## User Input

Does this analysis make sense? Would you like me to:
1. Run the diagnostics to confirm alignment?
2. Proceed to GCN Stacker documentation?
3. Create a unified fix plan for all Level-1 models?
