# CAFA-6 E2E Pipeline: Comprehensive Multi-Model Forensic Analysis
**Investigation Date:** 2026-01-03  
**Scope:** LogReg, GBDT, DNN, KNN  
**Objective:** Identify root causes of model underperformance vs. baseline

---

## Executive Summary

### Critical Bugs Identified

| Model | Bug Type | Severity | Impact | Status |
|-------|----------|----------|--------|--------|
| **GBDT** | Elite-Subset Training | **CRITICAL** | Predicts zero for 88.3% of target space | **Verified** |
| **KNN** | 5× systematic errors | **HIGH** | Over-smoothing + metric mismatch | **FIXED** |
| DNN | (Under investigation) | TBD | TBD | Pending |
| LogReg | (Under investigation) | TBD | TBD | Pending |

---

## CRITICAL BUG #1: GBDT "Elite Subset" Strategy

### Impact Assessment
- **Training Scope**: 1,585 terms (≥50 positives)
- **Zero-Prediction Terms**: **11,915 terms** (<50 positives)
- **Target Coverage**: Only **11.7%** of 13,500-term contract
- **Performance Impact**: **MASSIVE** — stacker receives mostly-zero predictions

### Evidence

**Target Distribution:**
```
Percentile | Positives
-----------|----------
10th       | 4
25th       | 7
50th       | 12  ← MEDIAN
75th       | 24
90th       | 56
95th       | 99
99th       | 368
```

**Rarest 20 Terms (all have only 3 positives):**
```
GO:0000041, GO:0000052, GO:0000053, GO:0000098, GO:0000393,
GO:0000435, GO:0000708, GO:0000751, GO:0000914, GO:0000959,
GO:0000962, GO:0001112, GO:0001188, GO:0001543, GO:0001545,
GO:0001546, GO:0001573, GO:0001748, GO:0001805, GO:0001827
```

**Code Location:**  
[05_cafa_e2e.ipynb](05_cafa_e2e.ipynb#L3954-3960)

```python
# Line 3954: Y_elite = Y[:, stable_idx]  # Only 1,585 columns
# ...
# Line 4188: oof_pred_gbdt[np.ix_(va_idx, elite_cols)] = va  
# Line 4189: # Zeros for remaining 11,915 terms!
```

### Root Cause
**Design Decision:** GBDT was intentionally configured to train only on "stable" terms (≥50 positives) to avoid noisy/rare-term overfitting.

**Why This Fails:**
1. **CAFA-6 evaluation penalizes missing predictions** — predicting zero for 88% of terms guarantees poor recall
2. **Stacker degradation** — GCN receives mostly-zero features from GBDT, reducing ensemble diversity
3. **Rare-term bias** — The baseline likely trains on ALL 13,500 terms, giving it a huge advantage on the long tail

### Recommended Fix

**Option A: Train on All 13,500 Terms (Immediate)**
- Remove elite subset logic entirely
- Use full `Y` matrix (shape: `(82404, 13500)`)
- Accept longer training time (~3-5× slower)
- **Pros**: Direct fix, matches baseline strategy
- **Cons**: Higher RAM usage, slower training

**Option B: Rare-Term Imputation Strategy**
- Train GBDT on elite 1,585 terms (fast)
- For unseen 11,915 terms: impute from KNN or LogReg predictions (weighted average)
- **Pros**: Keeps fast training, better than zeros
- **Cons**: Adds complexity, still suboptimal vs full training

**Recommendation**: **Option A** — train on all 13,500 terms. The speed/RAM cost is acceptable for a single GPU run.

---

## FIXED: KNN Bugs (5 Critical Issues)

| Bug | Before | After | Impact |
|-----|--------|-------|--------|
| Over-smoothing | k=50 | k=10 | Reduced label dilution |
| L2 norm + Euclidean | `sim = 1 - dist` | Removed L2, use cosine | Fixed metric |
| cuML metric mismatch | `euclidean` | `cosine` | Consistent similarity |
| IA weight in aggregation | Applied during voting | Removed (eval-only) | Correct protocol |
| Unknown embedding quality | Pre-computed ESM2-3B | (Cannot verify) | Potential data issue |

**Documentation**: 
- [docs/KNN_PERFORMANCE_INVESTIGATION.md](../docs/KNN_PERFORMANCE_INVESTIGATION.md)
- [docs/KNN_FIXES_APPLIED.md](../docs/KNN_FIXES_APPLIED.md)

---

## DNN Analysis (Partial)

### ✅ Verified Correct Behaviors
1. **Full Target Training**: DNN trains on all 13,500 terms (verified line 4927-4930)
2. **IA-Weighted Loss**: Uses IA weights for BCE loss (not aggregation)
3. **Multi-Branch Architecture**: 6-7 modality heads (t5, esm2_650m, esm2_3b, ankh, text, taxa, optional pb)
4. **Aspect-Specific Thresholds**: Loads calibrated thresholds (BP: 0.25, MF: 0.35, CC: 0.35)

### 🔍 Pending Investigation
- [ ] Verify normalization consistency across modalities
- [ ] Check for data leakage in PB (GBDT teacher) features
- [ ] Confirm batch size / learning rate are optimal
- [ ] Validate multi-seed ensembling (5 seeds × 5 folds = 25 models)

---

## LogReg Analysis (Pending)

### Status
- **Code Location**: Not yet located (cuML LogisticRegression imported at line 5053)
- **Training Strategy**: Unknown (needs cell identification)

### Next Steps
1. Find LogReg training cell
2. Verify it trains on full 13,500 terms (not elite subset)
3. Check regularization (C parameter)
4. Validate aspect-wise training (BP/MF/CC splits seen in output filenames)

---

## Data Integrity Audit

### ✅ VERIFIED: No ID Mismatch

**Test**: Checked overlap between `train_seq.feather` IDs and `train_terms.tsv` EntryIDs

```
Total train_ids:     82,404
Total EntryIDs:      82,404
Overlap:             82,404 (100%)
Missing from terms:  0
Extra in terms:      0
```

**ID Extraction Pattern**: `r"\|(.*?)\|"` correctly extracts UniProt accessions from FASTA headers  
**Example**: `sp|A0A0C5B5G6|MOTSC_HUMAN` → `A0A0C5B5G6`

### ✅ VERIFIED: No Duplicate Terms

**Test**: Checked `train_terms.tsv` for duplicate (EntryID, term) pairs

```
Total rows:                537,027
Unique EntryID-term pairs: 537,027
Duplicates:                0
```

**Pivot Aggregation**: `aggfunc='size'` produces correct binary labels (each pair appears exactly once)

### ✅ VERIFIED: Aspect Filtering Correct

**Official Ground Truth** (`Train/train_terms.tsv`):
- Columns: `EntryID`, `term`, `aspect`
- Aspect values: **P**, **F**, **C** (single-letter codes)

**Notebook Mapping**:
```python
ns_map = {
    "biological_process": "BP",  # Correct
    "molecular_function": "MF",  # Correct
    "cellular_component": "CC",  # Correct
}
```

Aspect normalization handles both full namespace strings and {P,F,C} codes correctly (line 3422-3434).

---

## Evaluation Metrics Analysis

### ⚠️ WARNING: Metric Definition Not Visible

**Observation**: `ia_weighted_f1()` function is **referenced** but **not defined** in visible notebook cells.

**Hypotheses**:
1. Imported from external module (e.g., `src/cafa6/metrics.py`)
2. Defined in an earlier cell (not yet examined)
3. Loaded from utils/checkpoint

**Risk**: Cannot verify:
- Precision/recall calculation correctness
- IA weight application (should be per-class, not per-sample)
- Threshold handling (aspect-specific vs global)

**Action Required**: Locate metric definition and audit against CAFA-6 official evaluation protocol.

---

## Performance Bottleneck Hypotheses

### 1. **GBDT Zero-Predictions** (CONFIRMED CRITICAL)
- **Magnitude**: 88.3% of target space
- **Mechanism**: Elite subset (1,585 / 13,500 terms)
- **Fix**: Train on full 13,500 terms

### 2. **KNN Parameter Tuning** (FIXED)
- Over-smoothing (k=50 → 10)
- Metric mismatch (Euclidean → cosine)
- IA weight misuse (removed from aggregation)

### 3. **Embedding Quality** (UNVERIFIED)
- ESM2-3B embeddings are pre-computed
- Cannot verify generation method or quality
- Potential corruption/truncation during caching

### 4. **Stacker Degradation** (HYPOTHESIS)
- GCN receives mostly-zero GBDT features
- Reduced ensemble diversity
- Hierarchy propagation cannot recover from weak base predictions

### 5. **Threshold Miscalibration** (PENDING)
- Aspect-specific thresholds: BP=0.25, MF=0.35, CC=0.35
- May not be optimal for post-stacker predictions
- Need to verify calibration was done on OOF predictions (not raw model outputs)

---

## Comparison to Baseline (02_baseline_knn.ipynb)

### Baseline Configuration
- **Model**: KNN only (no stacking)
- **Embeddings**: ESM2-8M (attention-masked mean pooling)
- **k**: 10 neighbors
- **Metric**: Cosine similarity
- **Evaluation**: Direct (no multi-stage pipeline)
- **Performance**: F1 > 0.216

### E2E Pipeline Differences
| Component | Baseline | E2E Pipeline |
|-----------|----------|--------------|
| Embeddings | ESM2-8M | ESM2-3B (400× larger) |
| Models | KNN only | LogReg + GBDT + DNN + KNN |
| Stacking | None | GCN (hierarchy-aware) |
| Post-processing | Basic | Hierarchy propagation |
| **Complexity** | Low | Very high |

### Why Baseline Outperforms (Hypothesis)
1. **Simpler is Better**: No compounding errors from multi-stage pipeline
2. **Full Target Coverage**: KNN predicts all terms (no elite subset)
3. **Cleaner Embeddings**: ESM2-8M may have better quality control than pre-computed 3B
4. **No Stacker Degradation**: Direct evaluation avoids GCN's dependency on weak GBDT features

---

## Next Steps (Priority Order)

1. **[URGENT]** Fix GBDT to train on all 13,500 terms
2. **[URGENT]** Locate and audit `ia_weighted_f1()` metric definition
3. **[HIGH]** Complete LogReg analysis (find training cell, verify term coverage)
4. **[HIGH]** Verify DNN normalization and batch size
5. **[MEDIUM]** Audit embedding quality (compare ESM2-8M vs ESM2-3B on sample proteins)
6. **[MEDIUM]** Investigate GCN stacker for propagation errors
7. **[LOW]** Profile full pipeline for RAM/VRAM bottlenecks

---

## Appendix: File References

- **Notebook**: [notebooks/05_cafa_e2e.ipynb](../notebooks/05_cafa_e2e.ipynb)
- **Official Data**: 
  - `Train/train_terms.tsv` (537,027 annotations, 82,404 proteins)
  - `Train/go-basic.obo` (ontology graph)
  - `IA.tsv` (40,123 term weights)
- **Artefacts**:
  - `cafa6_data/features/top_terms_13500.json` (full target contract)
  - `cafa6_data/features/stable_terms_1585.json` (GBDT elite subset)
  - `cafa6_data/features/stable_terms_1585_meta.json` (subset metadata)
- **Diagnostic Scripts**:
  - `scripts/_debug_id_overlap.py`
  - `scripts/_debug_gbdt_elite_strategy.py`

---

**Report Generated**: 2026-01-03  
**Investigation Status**: **Phase 1 Complete** (GBDT bug confirmed, KNN fixed)  
**Next Phase**: LogReg + DNN analysis, metric audit, embedding quality check
