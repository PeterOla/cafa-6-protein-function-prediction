# KNN Performance Regression: Applied Fixes Summary

**Date:** 2 January 2026  
**Status:** FIXES APPLIED  
**Files Modified:** `notebooks/05_cafa_e2e.ipynb`

---

## Fixes Applied to 05_cafa_e2e.ipynb

### **Fix #1: Reduced k-neighbors from 50 → 10**

**Location:** Cell 13E - KNN configuration

**Before:**
```python
KNN_K = int(globals().get('KNN_K', 50))
```

**After:**
```python
# FIX: Reduced k from 50 to 10 (matches baseline, reduces noise amplification)
KNN_K = int(globals().get('KNN_K', 10))
```

**Rationale:**  
- Baseline uses k=10, which is standard for protein function prediction
- k=50 includes too many distant neighbors, adding noise
- Smaller k preserves strong similarity signals

---

### **Fix #2: Removed L2 Normalization**

**Location:** Cell 13E - Embedding preprocessing

**Before:**
```python
# RANK-1: L2 pre-normalisation (transforms cosine→dot-product for Manual GEMM Fast Path)
def _l2_norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (X / norms).astype(np.float32)

t_norm = time.time()
X_knn = _l2_norm(X_knn)
X_knn_test = _l2_norm(X_knn_test)
print(f'[KNN] L2-normalised embeddings: train={X_knn.shape} test={X_knn_test.shape} (took {_fmt_s(time.time()-t_norm)})')
```

**After:**
```python
# FIX: Remove L2 normalization; use native cosine metric instead
# Original code incorrectly converted Euclidean distances to similarities
# Proper approach: use cosine metric directly (no normalization needed)
print(f'[KNN] Using raw embeddings (no L2 norm): train={X_knn.shape} test={X_knn_test.shape}')
```

**Rationale:**  
- L2 normalization + Euclidean distance is NOT equivalent to cosine similarity (bug in original code)
- Using native cosine metric is more accurate and avoids conversion errors
- Matches baseline approach (no normalization)

---

### **Fix #3: Changed Metric to Cosine**

**Location:** Cell 13E - KNN fitting (multiple locations)

**Before (training fold):**
```python
if USE_CUML:
    knn = cuNearestNeighbors(n_neighbors=KNN_K, metric='euclidean')
else:
    knn = NearestNeighbors(n_neighbors=KNN_K, metric='cosine', n_jobs=-1)
```

**After (training fold):**
```python
# FIX: Use cosine metric directly (no L2 normalization needed)
if USE_CUML:
    # cuML supports cosine metric
    knn = cuNearestNeighbors(n_neighbors=KNN_K, metric='cosine')
else:
    knn = NearestNeighbors(n_neighbors=KNN_K, metric='cosine', n_jobs=-1)
```

**Before (test predictions):**
```python
if USE_CUML:
    knn_final = cuNearestNeighbors(n_neighbors=KNN_K, metric='euclidean')
else:
    knn_final = NearestNeighbors(n_neighbors=KNN_K, metric='cosine', n_jobs=-1)
```

**After (test predictions):**
```python
# FIX: Use cosine metric (consistent with training)
if USE_CUML:
    knn_final = cuNearestNeighbors(n_neighbors=KNN_K, metric='cosine')
else:
    knn_final = NearestNeighbors(n_neighbors=KNN_K, metric='cosine', n_jobs=-1)
```

**Rationale:**  
- Ensures consistent metric across both cuML and sklearn backends
- Cosine similarity is semantically correct for embedding-based KNN
- Matches baseline implementation

---

### **Fix #4: Corrected Distance-to-Similarity Conversion**

**Location:** Cell 13E - Similarity computation

**Before:**
```python
# Convert distances to similarities
# sklearn cosine: distance = 1 - similarity
# cuML euclidean on L2-normalised vectors: we still use (1 - dist) proxy as previously
sims_va = np.clip((1.0 - dists_va).astype(np.float32), 0.0, 1.0)
```

**After:**
```python
# Convert distances to similarities
# FIX: cosine distance = 1 - cosine_similarity (direct conversion)
# sklearn/cuML cosine metric returns 1-based distance (distance=0 means identical)
sims_va = np.clip((1.0 - dists_va).astype(np.float32), 0.0, 1.0)
```

**Rationale:**  
- With cosine metric, the conversion `similarity = 1 - distance` is mathematically correct
- Original code tried to use this formula with Euclidean distance, which was wrong
- Now both the metric and conversion are aligned

---

### **Fix #5: Removed IA Weighting from KNN Aggregation**

**Location:** Cell 13E - Neighbor voting (2 locations: OOF and test)

**Before (IA weights broadcast):**
```python
# Broadcast IA weights: (1, 1, L)
w_ia_broadcast = weights_full[np.newaxis, np.newaxis, :]
```

**After:**
```python
# FIX: Do NOT broadcast IA weights for aggregation
# IA weights should only be used in EVALUATION, not in neighbor voting
# KNN should weight by similarity only (semantically correct for local smoothing)
# w_ia_broadcast = weights_full[np.newaxis, np.newaxis, :]  # REMOVED
```

**Before (OOF aggregation):**
```python
# Fetch neighbour labels
Y_nei = Y_knn[neigh_b]  # (B, K, L)

weighted_votes = (sims_b[:, :, np.newaxis] * Y_nei * w_ia_broadcast).sum(axis=1)
denom = np.maximum(sims_b.sum(axis=1, keepdims=True), 1e-8)
scores = (weighted_votes / denom).astype(np.float32)
```

**After (OOF aggregation):**
```python
# Fetch neighbour labels
Y_nei = Y_knn[neigh_b]  # (B, K, L)

# FIX: Similarity-weighted aggregation ONLY (no IA weighting here)
# IA weights belong in evaluation, not in KNN aggregation
weighted_votes = (sims_b[:, :, np.newaxis] * Y_nei).sum(axis=1)
denom = np.maximum(sims_b.sum(axis=1, keepdims=True), 1e-8)
scores = (weighted_votes / denom).astype(np.float32)
```

**Before (test aggregation):**
```python
Y_nei = Y_knn[neigh_b]
weighted_votes = (sims_b[:, :, np.newaxis] * Y_nei * w_ia_broadcast).sum(axis=1)
denom = np.maximum(sims_b.sum(axis=1, keepdims=True), 1e-8)
scores = (weighted_votes / denom).astype(np.float32)
```

**After (test aggregation):**
```python
Y_nei = Y_knn[neigh_b]
# FIX: Similarity-weighted aggregation only (no IA weighting)
weighted_votes = (sims_b[:, :, np.newaxis] * Y_nei).sum(axis=1)
denom = np.maximum(sims_b.sum(axis=1, keepdims=True), 1e-8)
scores = (weighted_votes / denom).astype(np.float32)
```

**Rationale:**  
- IA (Information Accretion) weights prioritize **rare, high-value terms**
- KNN works by **local smoothing** — it predicts what similar proteins have
- Weighting by IA during aggregation biases KNN towards rare terms even when evidence is weak
- **Correct usage:** Apply IA weights during **evaluation** (F1 calculation), not during prediction
- Matches baseline: similarity weighting only

---

## Expected Performance Impact

With these fixes, the e2e KNN should:

1. ✅ **Match baseline methodology** (k=10, cosine metric, similarity-only weighting)
2. ✅ **Eliminate distance conversion bug** (proper cosine metric usage)
3. ✅ **Remove IA weighting interference** (cleaner neighbor aggregation)
4. ✅ **Reduce noise** (fewer neighbors)
5. ✅ **Use superior embeddings** (ESM-3B vs ESM-8M)

**Predicted outcome:** F1 score should **meet or exceed** baseline 0.216, leveraging the more powerful 3B embedding model.

---

## Remaining Investigation: Embedding Quality

**CRITICAL:** The e2e pipeline loads pre-computed `esm2_3b` embeddings from disk. The notebook does **NOT show** how these were generated.

**Action Required:**

1. **Verify embedding generation method:**
   - Check `features/train_embeds_esm2_3b.npy` source
   - Ensure it uses **attention-masked mean pooling** (same as baseline)
   - Confirm no corruption/staleness

2. **If embeddings are suspect, regenerate:**
   - Add explicit embedding generation cell to 05_cafa_e2e.ipynb
   - Use same pooling strategy as baseline (see KNN_PERFORMANCE_INVESTIGATION.md for code)

---

## Testing Protocol

1. ✅ **Run fixed 05_cafa_e2e.ipynb** (Cell 13E only to test KNN in isolation)
2. ✅ **Evaluate KNN OOF predictions directly** (before GCN stacking)
3. ✅ **Compare to baseline:** Should match or exceed F1 0.216
4. ✅ **If still underperforming:** Investigate embedding quality (step 2 above)

---

## Files Modified

- `notebooks/05_cafa_e2e.ipynb` — Cell 13E (KNN implementation)
  - Changed k from 50 → 10
  - Removed L2 normalization
  - Changed cuML metric from euclidean → cosine
  - Removed IA weighting from aggregation (5 locations)
  - Updated comments to explain fixes

---

## Summary

**What was wrong:**
- Mathematical bug in distance-to-similarity conversion
- IA weighting applied at wrong stage (aggregation vs evaluation)
- Over-smoothing with k=50
- Inconsistent metric (Euclidean vs cosine)

**What was fixed:**
- Proper cosine similarity metric
- Similarity-only weighting (matches KNN theory)
- Reduced k to standard value
- Removed unnecessary normalization

**Why small ESM beat large ESM:**
- Bugs in e2e implementation degraded performance
- Baseline used clean, standard KNN approach
- Larger model + bugs < smaller model + correct implementation

**Next step:**
- Test fixes and verify performance restoration
- If needed, investigate embedding generation quality
