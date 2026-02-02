# LogReg Alignment & Logic Fixes

## Executive Summary

The Logistic Regression model was performing poorly (F1 ≈ 0.0003 → 0.0026) due to **data alignment issues** with the multi-modal `X_train_mmap.npy` embeddings. The fix was to use the **exact same data loading approach as the working KNN cell**.

---

## Part 4: KNN-Aligned Data Loading Fix (2026-02-01)

### 🔴 **ROOT CAUSE: X_train_mmap vs Direct Embedding Alignment**

| Component | Old (Broken) | New (Fixed) |
|-----------|--------------|-------------|
| **X embeddings** | `X_train_mmap.npy` (multi-modal) | `esm2_3b_train.npy` (direct) |
| **Y matrix source** | `train_terms.parquet` | `train_terms.tsv` |
| **Y building method** | Vectorised pandas | Row-by-row iteration |
| **ID cleaning** | `clean_id_vec()` | `_clean_id()` (KNN-identical) |

### Why This Bug Occurred

The `X_train_mmap.npy` file is a concatenation of multiple embedding modalities built in Cell 13a. Its protein ordering may differ from the `train_seq.feather` ordering used to build the label matrix. KNN avoids this by loading embeddings **directly** from the per-modality `.npy` files.

### The Fix

1. **Load ESM2-3B embeddings directly**: `esm2_3b_train.npy` and `esm2_3b_test.npy`
2. **Use identical data loading as KNN**: Same `_clean_id()` function, same file sources
3. **Build Y from TSV**: Row-by-row iteration matching KNN exactly

### Expected Impact

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| BP F1 | 0.0011 | ~0.12 (KNN-like) |
| MF F1 | 0.0027 | ~0.34 (KNN-like) |
| CC F1 | 0.0039 | ~0.30 (KNN-like) |

---

## Part 1: Historical Context (Early Fixes)

### 1. 🔴 `class_weight` Misuse in OVR
**Issue:** Code passed a dictionary of weights to OVR, corrupting the solver.
**Fix:** Switched to standard `class_weight='balanced'`, which automatically downweights frequent negatives and upweights rare positives.

### 2. 🟠 Scaler Leakage
**Issue:** Scaler refit per-fold caused test data drift.
**Fix:** Implemented **incremental global scaling** (`partial_fit`) for consistent features.

### 3. 🟡 Y Matrix Alignment (Initial)
**Issue:** Potential ID scrambling.
**Fix:** Implemented **Vectorized ID Cleaning** separately, ensuring parity with KNN logic while keeping speed.

### 4. 🔵 Convergence Warnings
**Issue:** `max_iter=100` was too low.
**Fix:** Increased to **1000**.

---

## Part 2: Validation Results (Top 50 Terms - Historical)

| Run Configuration | best F1 | improvement |
| :--- | :--- | :--- |
| **Baseline (Broken)** | 0.0003 | - |
| No Class Weight | 0.0194 | 64x |
| **Balanced Class Weight** | **0.0525** | **175x** |

**Note:** These results were for Top 50 Terms only, not all 13,500.

---

## Part 3: Deployment

The verified logic is now in `05_cafa_e2e.ipynb` Cell 13c (LogReg).
Key changes:
1. Uses `esm2_3b_train.npy` directly (not `X_train_mmap.npy`)
2. Uses `train_terms.tsv` (not parquet)
3. Uses identical `_clean_id()` function as KNN cell
4. PyTorch GPU linear layer for parallel training

*Last updated: 2026-02-01*
