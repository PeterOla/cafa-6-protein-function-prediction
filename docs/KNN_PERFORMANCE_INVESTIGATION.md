# KNN Performance Investigation: Why 8M ESM Outperforms 3B ESM + Stacked Models

**Investigation Date:** 2 January 2026  
**Status:** CRITICAL BUGS IDENTIFIED  
**Impact:** Performance regression from F1 0.216 → significantly lower

---

## Executive Summary

The baseline KNN (02_baseline_knn.ipynb) achieving F1 > 0.216 used a **tiny 8M-parameter ESM-2 model** with simple cosine similarity KNN. The new e2e pipeline (05_cafa_e2e.ipynb) using a **3B-parameter ESM-2 model** with GPU-accelerated KNN, IA weighting, and multi-model stacking is underperforming.

**Root Cause Identified:** The e2e pipeline KNN is **NOT actually being evaluated** in the standard way. It's being used as a Level-1 model feeding into a GCN stacker, which introduces multiple layers of transformation, thresholding, and potential error accumulation.

---

## Critical Differences Matrix

| **Component** | **02_baseline_knn.ipynb (F1 > 0.216)** | **05_cafa_e2e.ipynb (Underperforming)** |
|---------------|----------------------------------------|------------------------------------------|
| **Embedding Model** | `facebook/esm2_t6_8M_UR50D` (8M params) | `esm2_3b` (3B params) |
| **Embedding Pooling** | Mean pooling over `last_hidden_state` (excluding padding) | Pre-computed embeddings (source unknown) |
| **Normalization** | **NONE** (raw embeddings) | **L2 normalization** applied |
| **Distance Metric** | **Cosine similarity** (`metric='cosine'`) | **Euclidean** after L2-norm (claims cosine equivalence) |
| **K neighbors** | k=10 | k=50 (5× more neighbors) |
| **Backend** | sklearn `NearestNeighbors` (CPU) | cuML `NearestNeighbors` (GPU) or sklearn fallback |
| **Weighting Scheme** | Similarity-weighted vote: `similarity × term` | **IA-weighted** vote: `similarity × term × IA_weight` |
| **Aggregation** | `term_scores[term] += similarity` → normalize by max | `weighted_votes.sum(axis=1) / similarity.sum()` |
| **Train/Val Split** | 80/20 random split (single fold) | **5-fold CV** with OOF predictions |
| **Evaluation** | Direct predictions → threshold scan → F1 | OOF → feed to **GCN stacker** → complex post-processing |
| **Threshold** | Scanned [0.1-0.8], per-aspect optimization | Loaded from `aspect_thresholds.json` (external dependency) |
| **Post-processing** | None (direct KNN output) | **Hierarchy propagation** (12 iterations max/min) |

---

## Detailed Analysis

### 1. **Embedding Quality Mismatch**

**Baseline (Working):**
```python
model_name = "facebook/esm2_t6_8M_UR50D"
# Mean pooling with attention masking
attention_mask = inputs['attention_mask']
token_embeddings = outputs.last_hidden_state
input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
pooled = (sum_embeddings / sum_mask).cpu().numpy()
```

**E2E Pipeline (Underperforming):**
```python
# Uses pre-computed 'esm2_3b' embeddings loaded from disk
X_knn = features_train['esm2_3b'].astype(np.float32)
# L2 normalization applied
X_knn = _l2_norm(X_knn)
```

**CRITICAL ISSUE #1: Unknown Embedding Source**  
The e2e pipeline loads pre-computed `esm2_3b` embeddings from `features/train_embeds_esm2_3b.npy`. The notebook does **NOT show how these were generated**. Potential issues:

- Wrong pooling strategy (CLS token vs mean pooling)
- Different sequence truncation (max_length)
- Missing attention mask handling
- Cached embeddings from different data split
- Corrupted or outdated artefacts

**Evidence:** Baseline explicitly shows embedding generation logic. E2e pipeline assumes embeddings exist and are correct.

---

### 2. **Distance Metric Bug**

**Baseline (Working):**
```python
knn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
```

Uses true **cosine similarity**: `1 - cosine_distance = dot(A, B) / (||A|| × ||B||)`

**E2E Pipeline (Broken):**
```python
# L2 normalize first
X_knn = _l2_norm(X_knn)

# Then use Euclidean metric
knn = cuNearestNeighbors(n_neighbors=KNN_K, metric='euclidean')

# Convert "distances" to similarities
sims_va = np.clip((1.0 - dists_va).astype(np.float32), 0.0, 1.0)
```

**CRITICAL ISSUE #2: Mathematical Error**

While it's true that **Euclidean distance on L2-normalized vectors is related to cosine**, the conversion `similarity = 1 - euclidean_distance` is **WRONG**.

Correct relationship:
```
euclidean²(u, v) = 2 - 2·dot(u, v)    [for L2-normalized u, v]
cosine_similarity = dot(u, v)
cosine_similarity = 1 - euclidean²/2
```

The code does: `sim = 1 - euclidean` (not `1 - euclidean²/2`)

**Impact:** Similarity scores are systematically distorted, affecting neighbor weighting.

---

### 3. **IA Weighting Interference**

**Baseline (Working):**
```python
# Simple weighted voting
for term in nei_terms:
    term_scores[term] += similarity

# Normalize to probabilities
for term, score in term_scores.items():
    probability = score / max_score
```

**E2E Pipeline (Complex):**
```python
# IA-weighted voting
weighted_votes = (sims_b[:, :, np.newaxis] * Y_nei * w_ia_broadcast).sum(axis=1)
denom = np.maximum(sims_b.sum(axis=1, keepdims=True), 1e-8)
scores = (weighted_votes / denom).astype(np.float32)
```

**CRITICAL ISSUE #3: IA Weights May Harm KNN**

IA weights prioritize **rare, high-information terms**. However, KNN works by **local smoothing** — it should predict what **similar proteins** have, not what's **globally rare**.

For a protein with k=50 neighbors:
- If 40 neighbors have common term "GO:0005737" (cytoplasm, low IA)
- And 2 neighbors have rare term "GO:0098765" (exotic process, high IA)

IA weighting will **boost the rare term** despite weak evidence, potentially **adding noise**.

**Baseline doesn't use IA in aggregation** — it weights by similarity only, which is semantically correct for KNN.

---

### 4. **Evaluation Pipeline Complexity**

**Baseline (Direct):**
```python
predictions_df = [predictions for all val proteins]
evaluate_predictions(predictions_df, val_data, ia_weights, threshold=0.01)
```

Simple flow: KNN → threshold → evaluate

**E2E Pipeline (Multi-stage):**
```
KNN → OOF predictions (5-fold) 
    → Stack with LogReg/GBDT/DNN predictions 
    → Feed to GCN stacker (IA-weighted loss, 10 epochs, hierarchy-aware)
    → Hierarchy propagation (12 iters max/min)
    → Per-aspect thresholding
    → Evaluate
```

**CRITICAL ISSUE #4: Accumulated Error**

Each stage introduces potential issues:
- **5-fold CV:** More robust but also more complex; fold imbalance
- **GCN stacker:** Trains on KNN outputs, can amplify errors
- **Hierarchy propagation:** Can blur predictions
- **Threshold dependency:** Requires external calibration

The baseline **directly evaluates KNN**, while e2e evaluates a **heavily transformed version** of KNN outputs.

---

### 5. **Neighbor Count Impact**

**Baseline:** k=10  
**E2E Pipeline:** k=50

**CRITICAL ISSUE #5: Noise Amplification**

More neighbors = more smoothing, but also:
- More irrelevant proteins included
- Dilution of strong similarity signal
- Higher computational cost

For protein function prediction, **k=10-20 is typical**. k=50 may include too many distant neighbors that add noise.

---

### 6. **Train/Val Split Methodology**

**Baseline:**
```python
train_proteins, val_proteins = train_test_split(
    all_proteins, test_size=0.2, random_state=42
)
```

Single 80/20 split, fixed seed.

**E2E Pipeline:**
```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_knn), start=1):
    # Train KNN on tr_idx
    # Predict on va_idx
    # Aggregate OOF predictions
```

5-fold CV with OOF aggregation.

**CRITICAL ISSUE #6: Fold-Specific Overfitting**

While 5-fold CV is generally better, the e2e pipeline:
- Fits 5 separate KNN models
- Each fold sees only 80% of training data
- OOF predictions are aggregated (averaged)

If folds are imbalanced or contain protein families, KNN performance can degrade.

---

## Why Small ESM Beats Large ESM + Stacking

**Paradox Explanation:**

1. **Embedding Quality:** The tiny 8M ESM with **proper mean pooling** may produce **cleaner, more discriminative** embeddings than the pre-computed 3B embeddings (if those used wrong pooling or are corrupted).

2. **Direct Evaluation:** Baseline directly evaluates KNN performance. E2e evaluates after multiple transformation layers that can degrade signal.

3. **Simpler is Better:** Baseline uses standard cosine KNN with similarity weighting — a well-understood, robust method. E2e adds IA weighting, L2+Euclidean conversion, and complex aggregation that introduce bugs.

4. **No Interference:** Baseline predictions are **pure KNN output**. E2e KNN outputs are **inputs to a GCN** that may distort them (especially with IA-weighted loss that prioritizes rare terms).

5. **Calibration:** Baseline scans thresholds on actual KNN outputs. E2e loads pre-computed thresholds that may not match the KNN distribution.

---

## Root Cause Summary

| **Bug ID** | **Issue** | **Impact** | **Severity** |
|------------|-----------|-----------|--------------|
| **BUG-1** | Unknown embedding source/generation method | Cannot verify correctness | **CRITICAL** |
| **BUG-2** | Incorrect Euclidean→similarity conversion | Systematic similarity distortion | **CRITICAL** |
| **BUG-3** | IA weighting in KNN aggregation | Rare term over-weighting, noise | **HIGH** |
| **BUG-4** | Multi-stage evaluation pipeline | Error accumulation, hard to debug | **HIGH** |
| **BUG-5** | k=50 too large | Noise amplification | **MEDIUM** |
| **BUG-6** | 5-fold CV complexity | Potential fold imbalance | **LOW** |

---

## Recommended Fixes

### **Fix #1: Verify/Regenerate ESM-3B Embeddings**

Add cell to 05_cafa_e2e.ipynb to **explicitly generate** `esm2_3b` embeddings using the **same pooling strategy** as baseline:

```python
# CELL: Generate ESM-3B embeddings (mean pooling, attention-masked)
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "facebook/esm2_t33_650M_UR50D"  # or esm2_t36_3B_UR50D
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

def embed_sequences_proper(sequences, batch_size=8):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024  # ESM-3B can handle longer sequences
            ).to(device)
            
            outputs = model(**inputs)
            
            # CRITICAL: Use attention-masked mean pooling (same as baseline)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = (sum_embeddings / sum_mask).cpu().numpy()
            
            embeddings.append(pooled)
    
    return np.vstack(embeddings)
```

### **Fix #2: Correct Distance Metric**

**Option A (Recommended):** Use native cosine similarity:
```python
if USE_CUML:
    # cuML supports cosine metric
    knn = cuNearestNeighbors(n_neighbors=KNN_K, metric='cosine')
else:
    knn = NearestNeighbors(n_neighbors=KNN_K, metric='cosine', n_jobs=-1)

# Remove L2 normalization (not needed for cosine)
# knn.fit(X_knn)  # Use raw embeddings

# Distances are already cosine distances; convert to similarity
sims_va = 1.0 - dists_va
```

**Option B (Fix Current Approach):** Correct the Euclidean→cosine conversion:
```python
# Keep L2 normalization
X_knn = _l2_norm(X_knn)

# Use Euclidean metric
knn = cuNearestNeighbors(n_neighbors=KNN_K, metric='euclidean')

# CORRECT conversion: euclidean²(u,v) = 2 - 2·cos_sim
# Therefore: cos_sim = 1 - euclidean²/2
sims_va = np.clip(1.0 - (dists_va**2) / 2.0, 0.0, 1.0).astype(np.float32)
```

### **Fix #3: Remove IA Weighting from KNN Aggregation**

IA weights should **only** be used in **evaluation**, not in neighbor voting:

```python
# BEFORE (IA-weighted aggregation):
weighted_votes = (sims_b[:, :, np.newaxis] * Y_nei * w_ia_broadcast).sum(axis=1)

# AFTER (similarity-only aggregation):
weighted_votes = (sims_b[:, :, np.newaxis] * Y_nei).sum(axis=1)
# IA weights applied later during evaluation/thresholding
```

### **Fix #4: Reduce k to 10-20**

Match baseline or use standard KNN best practices:

```python
KNN_K = 10  # or 15, 20 — test empirically
```

### **Fix #5: Add Direct KNN Evaluation Cell**

Before feeding to GCN stacker, evaluate **raw KNN OOF predictions**:

```python
# CELL: Evaluate KNN directly (before stacking)
from sklearn.metrics import f1_score

# Load KNN OOF predictions
knn_oof = np.load(PRED_DIR / 'oof_pred_knn.npy')

# Evaluate with aspect-specific thresholds
for aspect in ['MF', 'BP', 'CC']:
    aspect_terms = [t for t in top_terms if go_namespaces.get(t) == aspect_map[aspect]]
    aspect_idx = [top_terms.index(t) for t in aspect_terms]
    
    y_true_asp = Y[:, aspect_idx]
    y_pred_asp = knn_oof[:, aspect_idx]
    
    thr = ASPECT_THRESHOLDS[aspect]
    y_pred_bin = (y_pred_asp >= thr).astype(int)
    
    f1 = f1_score(y_true_asp.flatten(), y_pred_bin.flatten(), average='binary')
    print(f"{aspect} F1 @ {thr:.2f}: {f1:.4f}")
```

---

## Testing Protocol

1. **Baseline Reproduction:** Re-run 02_baseline_knn.ipynb to confirm F1 > 0.216
2. **Fix Application:** Apply fixes #1-4 to 05_cafa_e2e.ipynb
3. **Isolated KNN Test:** Evaluate KNN alone (fix #5) before GCN stacking
4. **Ablation Study:** Compare:
   - KNN alone (fixed)
   - KNN + GCN stacker
   - Full pipeline (KNN + LogReg + GBDT + DNN + GCN)

Expected outcome: **Fixed KNN should match or exceed baseline 0.216 F1**

---

## Conclusion

The e2e pipeline KNN underperforms due to a **cascade of implementation bugs**:

1. **Unverified embedding quality** (unknown source/pooling)
2. **Mathematical error** in distance-to-similarity conversion
3. **IA weighting applied incorrectly** (should be evaluation-only)
4. **Over-smoothing** with k=50 neighbors
5. **Evaluation through multiple transformation layers** that obscure KNN performance

The baseline works because it:
- Uses **verified, attention-masked mean pooling**
- Uses **standard cosine similarity**
- Uses **simple similarity weighting**
- **Directly evaluates** KNN output

**Bottom line:** Bigger model ≠ better performance when implementation has bugs. The 3B ESM KNN can and should outperform 8M ESM KNN, but only after fixing the above issues.
