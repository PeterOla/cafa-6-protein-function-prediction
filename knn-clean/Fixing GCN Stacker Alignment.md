# GCN Stacker Performance Mystery: A Complete Explanation

## The Goal

Train a Graph Convolutional Network (GCN) to optimally combine Level-1 model predictions.
The approach: "Learn to mix predictions while respecting the GO term hierarchy."

---

## Part 1: How the GCN Stacker Works (The Concept)

The GCN Stacker is **not** a typical neural network. It operates on a **graph** where:
- **Nodes** = GO terms (13,500 nodes)
- **Edges** = Parent-child relationships from the Gene Ontology
- **Node features** = Level-1 model predictions (DNN, GBDT, KNN, LogReg)

```
Input per protein:
┌─────────────────────────────────────────────────────────────────┐
│ For each GO term (node), we have:                               │
│   - DNN prediction (probability)                                │
│   - GBDT prediction (probability)                               │
│   - KNN prediction (probability)                                │
│   - LogReg prediction (probability)                             │
│   → [p_dnn, p_gbdt, p_knn, p_logreg] = 4 channels               │
└─────────────────────────────────────────────────────────────────┘

GCN Operations:
1. Graph Convolution: Aggregate neighbour information
   h_i = σ(W · Σⱼ (Aᵢⱼ · hⱼ))
   
2. This propagates information UP/DOWN the GO hierarchy:
   - If a protein likely has GO:0001 (parent), its children get a boost
   - If a protein likely has GO:0002 (child), its parent gets a boost

3. Final layer: Linear(hidden → 1) → logits per term
```

**Key Insight**: The GCN enforces **hierarchy consistency** — if you predict a child term, the parent should also be predicted.

---

## Part 2: The Data Alignment Contract (Phase 3)

GCN Stacker has a **different** alignment contract than Level-1 models:

```
Level-1 Contract:
  X (embeddings) aligned with Y (targets) via train_seq.feather

Phase 3 Contract:
  OOF predictions aligned with Y (targets)
  OOF predictions aligned with each other
```

**Critical**: All Level-1 OOF predictions must use the **same row ordering**.

If KNN uses one ordering, DNN uses another, and GBDT uses a third, the GCN receives:
```
Row 0: [DNN(protein_A), GBDT(protein_B), KNN(protein_C), LogReg(protein_D)]
                                    ↑
                         GARBAGE FEATURES!
```

---

## Part 3: What the Current Code Does

### Loading Level-1 Predictions
```python
model_paths = {
    'dnn': ('oof_pred_dnn.npy', 'test_pred_dnn.npy'),
    'gbdt': ('oof_pred_gbdt.npy', 'test_pred_gbdt.npy'),
    'knn': ('oof_pred_knn.npy', 'test_pred_knn.npy'),
    'logreg': ('oof_pred_logreg.npy', 'test_pred_logreg.npy')
}

for m, (oof_name, test_name) in model_paths.items():
    oof_p = PRED_DIR / oof_name
    test_p = PRED_DIR / test_name
    oof_by_model[m] = np.load(oof_p, mmap_mode='r')
    test_by_model[m] = np.load(test_p, mmap_mode='r')
```

### Building Input Features
```python
def _batch_stack(arrs: list, rows: np.ndarray, cols: list[int], prior_np) -> torch.Tensor:
    feats = []
    for a in arrs:
        x = np.asarray(a[rows][:, cols], dtype=np.float32)
        feats.append(x)
    x = np.stack(feats, axis=2)  # (B, N, C) where C = n_models
    return torch.from_numpy(x).to(device)
```

### Graph Construction
```python
def build_adjacency(terms_list, parents_dict):
    # Build edges from GO hierarchy
    for child in terms_list:
        parents = parents_dict.get(child, set())
        for parent in parents:
            # Bidirectional edges
            src.append(child_idx); dst.append(parent_idx)
            src.append(parent_idx); dst.append(child_idx)
    
    # Self-loops for all nodes
    src.extend(range(n_terms))
    dst.extend(range(n_terms))
    
    # D^{-1/2} A D^{-1/2} normalization
    ...
```

### Training Loop
```python
for aspect_name, idx_cols in [('MF', mf_idx), ('BP', bp_idx), ('CC', cc_idx)]:
    model = HierarchyAwareGCN(n_channels, hidden_dims, adj_matrix)
    
    for epoch in range(EPOCHS):
        for i in range(0, n_samples, BS):
            rows = perm[i:i + BS]
            xb = _batch_stack([oof_by_model[m] for m in models_used], rows, idx_cols, prior_train_np)
            yb = _batch_y(Y_full, rows, idx_cols)
            
            logits = model(xb, training=True)
            loss = criterion(logits, yb)  # IA-weighted BCE
            loss.backward()
```

---

## Part 4: The Alignment Bug (Compounded)

### Bug 1: Level-1 Models Already Misaligned

If Level-1 models have different row orderings:
```
oof_pred_dnn.npy:    [protein_A, protein_B, protein_C, ...]
oof_pred_gbdt.npy:   [protein_B, protein_C, protein_A, ...]  ← DIFFERENT ORDER!
oof_pred_knn.npy:    [protein_C, protein_A, protein_B, ...]  ← DIFFERENT ORDER!
oof_pred_logreg.npy: [protein_A, protein_B, protein_C, ...]  ← Same as DNN (maybe)
```

Then stacking them:
```python
xb = _batch_stack([oof_by_model[m] for m in models_used], rows, idx_cols, prior_np)
# Row 0: [DNN(A), GBDT(B), KNN(C), LogReg(A)]
# This is nonsense — mixing predictions from different proteins!
```

### Bug 2: Y Still Misaligned

Even if all Level-1 OOF predictions are aligned with each other, they must also align with `Y`:
```python
yb = _batch_y(Y_full, rows, idx_cols)
# If Y is misaligned, we're training on wrong targets
```

### Bug 3: No Explicit Alignment Check

The code **assumes** all .npy files have the same row ordering. No validation:
```python
# Missing:
assert oof_by_model['dnn'].shape[0] == oof_by_model['gbdt'].shape[0]
# But even if shapes match, ORDER might differ!
```

---

## Part 5: GCN-Specific Issues

### Issue 1: Aspect-Specific Training (Not Stacking Issue)

**Current Code:**
```python
for aspect_name, idx_cols in [('MF', mf_idx), ('BP', bp_idx), ('CC', cc_idx)]:
    model = HierarchyAwareGCN(...)
    # Train separate GCN per aspect
```

**Observation**: Each aspect gets its own GCN. This is **correct** because:
- GO hierarchy is aspect-specific (BP terms don't have MF parents)
- Aspect-specific thresholds apply differently

### Issue 2: TTA (Test-Time Augmentation) via Dropout

**Current Code:**
```python
TTA_VARIANTS = 5

def _predict_aspect_tta(model, idx_cols, tta_variant, bs=64):
    torch.manual_seed(42 + tta_variant)
    model.train()  # Enable dropout for TTA
    
    with torch.no_grad():
        logits = model(xb, training=True)  # Dropout active
```

**Observation**: Using dropout during inference creates prediction diversity. The 5 TTA variants are then averaged.

**Verdict**: ✅ Correct TTA implementation.

### Issue 3: IA-Weighted Loss

**Current Code:**
```python
class IAWeightedBCELoss(nn.Module):
    def __init__(self, ia_weights):
        self.ia_weights = ia_weights  # (N,)
    
    def forward(self, logits, targets):
        loss_per = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_loss = loss_per * self.ia_weights.unsqueeze(0)
        return weighted_loss.mean()
```

**Verdict**: ✅ Correct. Same approach as DNN — IA as per-term loss weight.

### Issue 4: External Priors (UniProt IEA)

**Current Code:**
```python
EXTERNAL_PRIOR_WEIGHT = 0.15

if prior_train_path.exists() and prior_test_path.exists():
    prior_train_np = np.load(prior_train_path, mmap_mode='r')
    prior_test_np = np.load(prior_test_path, mmap_mode='r')

# In _batch_stack:
if prior_np is not None:
    p = np.asarray(prior_np[rows][:, cols], dtype=np.float32)
    x = np.maximum(x, (EXTERNAL_PRIOR_WEIGHT * p)[:, :, None])
```

**Observation**: External priors are injected as a floor on input features.

**Risk**: If `prior_train_np` has different row ordering, priors are misaligned too!

### Issue 5: Missing OOF for GCN

**Current Code**: The GCN does NOT produce OOF predictions for stacking.

```python
# Only test predictions are saved:
final_gcn = np.mean(tta_outputs, axis=0)
np.save(out_path, final_gcn)  # test_pred_gcn.npy
```

**Observation**: GCN is the **final stacker** — there's no Level-2 stacker that would need GCN OOF.

**Verdict**: ✅ Acceptable for current architecture.

---

## Part 6: Visual Diagnosis (What to Check)

### Check 1: Level-1 OOF Shapes
```python
for m in models_used:
    print(f"{m}: {oof_by_model[m].shape}")

# All should be (n_train, 13500)
# But same shape doesn't guarantee same ordering!
```

### Check 2: Cross-Model Correlation (Alignment Proxy)
```python
# If models are aligned, their predictions should be correlated
# (all trying to predict same targets from same proteins)

dnn_oof = oof_by_model['dnn']
gbdt_oof = oof_by_model['gbdt']

# Sample correlation
sample_idx = np.random.choice(dnn_oof.shape[0], size=1000, replace=False)
sample_cols = np.random.choice(dnn_oof.shape[1], size=100, replace=False)

dnn_sample = dnn_oof[np.ix_(sample_idx, sample_cols)].flatten()
gbdt_sample = gbdt_oof[np.ix_(sample_idx, sample_cols)].flatten()

corr = np.corrcoef(dnn_sample, gbdt_sample)[0, 1]
print(f"DNN-GBDT correlation: {corr:.4f}")
# Expect: 0.3–0.8 if aligned
# If ~0: likely misaligned
```

### Check 3: OOF vs Y Agreement
```python
# For a correctly aligned model, OOF predictions should have some predictive power
from sklearn.metrics import roc_auc_score

sample_cols = np.random.choice(Y_full.shape[1], size=50, replace=False)
sample_cols = sample_cols[Y_full[:, sample_cols].sum(axis=0) > 100]  # Filter for enough positives

for m in models_used:
    oof = oof_by_model[m]
    aucs = []
    for c in sample_cols:
        if Y_full[:, c].sum() > 0 and Y_full[:, c].sum() < Y_full.shape[0]:
            auc = roc_auc_score(Y_full[:, c], oof[:, c])
            aucs.append(auc)
    print(f"{m}: mean AUC = {np.mean(aucs):.4f}")
# Expect: 0.6–0.9 if aligned
# If ~0.5 (random): misaligned
```

---

## Part 7: Required Changes Summary

### 🔴 **CRITICAL: All Level-1 Models Must Use Same Row Order**

| Check | Action |
|-------|--------|
| Verify all OOF files have same row count | Shape check |
| Verify cross-model correlation | Part 6, Check 2 |
| Verify OOF vs Y agreement | Part 6, Check 3 |
| If misaligned | **Fix ALL Level-1 models first** |

### 🟠 **Major Considerations**

| Issue | Current | Recommendation |
|-------|---------|----------------|
| **IA-weighted loss** | ✅ Correct | Keep as-is |
| **TTA variants** | ✅ Correct | Keep 5 variants |
| **Aspect-specific GCN** | ✅ Correct | Keep separate models |

### 🟡 **Validation to Add**

| Issue | Current | Fix |
|-------|---------|-----|
| **No alignment check** | Implicit assumption | Add explicit shape + correlation checks |
| **External priors** | Loaded if exists | Verify alignment with train_seq.feather |

---

## Part 8: The Cascade Effect

GCN Stacker is **downstream** of all Level-1 models:

```
train_seq.feather (ordering truth)
        │
        ├──→ Embeddings (.npy files)
        │           │
        ├──→ Y (targets)
        │           │
        │    ┌──────┴──────┐
        │    │   Level-1   │
        │    │  KNN, LogReg│
        │    │  GBDT, DNN  │
        │    └──────┬──────┘
        │           │
        │    ┌──────┴──────┐
        │    │   OOF Preds │ ← If ANY Level-1 is misaligned,
        │    └──────┬──────┘    GCN receives garbage features
        │           │
        │    ┌──────┴──────┐
        │    │ GCN Stacker │
        │    └─────────────┘
```

**Key Insight**: Fixing GCN alone is useless. You must fix Cell 13a first, then re-run ALL Level-1 models, then re-run GCN.

---

## Part 9: Diagnostic Checklist

```python
# Run BEFORE GCN training:

# 1. OOF Shape Check
for m in models_used:
    shape = oof_by_model[m].shape
    print(f"{m}: {shape}")
    assert shape[0] == Y_full.shape[0], f"{m} row mismatch!"
    assert shape[1] == 13500, f"{m} col mismatch!"

# 2. Cross-Model Correlation (Quick Sanity)
import numpy as np
dnn = oof_by_model.get('dnn')
knn = oof_by_model.get('knn')
if dnn is not None and knn is not None:
    sample = np.random.choice(dnn.shape[0], size=500, replace=False)
    corr = np.corrcoef(dnn[sample].flatten()[:10000], 
                       knn[sample].flatten()[:10000])[0, 1]
    print(f"DNN-KNN correlation: {corr:.4f}")
    if abs(corr) < 0.1:
        print("WARNING: Very low correlation. Possible misalignment!")

# 3. Y Row Sum Check (inherited from Level-1)
row_sums = Y_full.sum(axis=1)
zero_rows = (row_sums == 0).sum()
print(f"Y rows with zero labels: {zero_rows}")
if zero_rows > 0:
    print("WARNING: ID alignment broken upstream!")
```

---

## Part 10: Implementation Plan

1. [ ] **Fix Cell 13a alignment** (single source of truth)
2. [ ] **Re-run ALL Level-1 models** (KNN, LogReg, GBDT, DNN)
3. [ ] **Verify OOF alignment** (shape + correlation checks)
4. [ ] **Only then run GCN Stacker**
5. [ ] **Verify test predictions are sensible**

---

## Part 11: What GCN Does Correctly

Despite upstream issues, the GCN implementation itself is solid:

| Component | Status | Notes |
|-----------|--------|-------|
| **Adjacency matrix** | ✅ Correct | Builds from GO OBO file |
| **Graph convolution** | ✅ Correct | Sparse matrix multiplication |
| **IA-weighted loss** | ✅ Correct | Same as DNN |
| **Aspect-specific training** | ✅ Correct | Separate GCN per BP/MF/CC |
| **TTA** | ✅ Correct | Dropout-based diversity |
| **Output contract** | ✅ Enforced | (n_test, 13500) |

**The GCN code is fine. The inputs (Level-1 OOF predictions) are the problem.**

---

## User Input

Does this analysis make sense? 

**Summary of All Alignment Issues:**

| Model | Root Cause | Fix Location |
|-------|------------|--------------|
| KNN | Uses `train_terms.unique()` for row order | Cell 13E |
| LogReg | Inherits Y from Cell 13a | Cell 13a |
| GBDT | Inherits Y from Cell 13a | Cell 13a |
| DNN | Inherits Y and features from Cell 13a | Cell 13a |
| GCN | Consumes all Level-1 OOF predictions | Fix Level-1 first |

**The fix must be applied to Cell 13a (single source of truth), then propagated to all Level-1 models.**

Would you like me to create a **unified fix plan** that addresses all models in sequence?
