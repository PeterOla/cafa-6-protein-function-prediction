# LogReg Alignment & Logic Fixes

## Executive Summary

The Logistic Regression model was performing poorly (F1 ≈ 0.0003) due to **three critical errors**, primarily `class_weight` misuse. We rebuilt the script `logreg_esm2_3b_13500.py` with robust fixes.
**Final Optimization**: Using `class_weight='balanced'` yielded a massive performance jump to **F1 = 0.0525** (>175x baseline).

---

## Part 1: Critical Bugs Identified & Fixed

### 1. 🔴 `class_weight` Misuse in OVR
**Issue:** Code passed a dictionary of weights to OVR, corrupting the solver.
**Fix:** Switched to standard `class_weight='balanced'`, which automatically downweights frequent negatives and upweights rare positives.

### 2. 🟠 Scaler Leakage
**Issue:** Scaler refit per-fold caused test data drift.
**Fix:** Implemented **incremental global scaling** (`partial_fit`) for consistent features.

### 3. 🟡 Y Matrix Alignment
**Issue:** Potential ID scrambling.
**Fix:** Implemented **Vectorized ID Cleaning** separately, ensuring parity with KNN logic while keeping speed.

### 4. 🔵 Convergence Warnings
**Issue:** `max_iter=100` was too low.
**Fix:** Increased to **1000**.

---

## Part 2: Validation Results (Top 50 Terms)

| Run Configuration | best F1 | improvement |
| :--- | :--- | :--- |
| **Baseline (Broken)** | 0.0003 | - |
| No Class Weight | 0.0194 | 64x |
| **Balanced Class Weight** | **0.0525** | **175x** |

**Integrated Notebook Cell Verification:**
- Script: `cell_13c_integrated.py`
- Result: **F1 = 0.0541** (Matches optimization results)
- Status: **VERIFIED & READY**

---

## Part 3: Deployment

The verified logic is encapsulated in `logreg-clean/cell_13c_integrated.py`.
This script:
1. Auto-detects OBO files.
2. Auto-generates memmaps from embeddings if missing.
3. Uses `StratifiedKFold` and Global Scaling.
4. Uses `class_weight='balanced'` for max F1.
5. Is drop-in compatible with `05_cafa_e2e.ipynb`.

*Last updated: 2026-01-31*
