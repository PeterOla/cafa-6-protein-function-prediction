# CAFA-6 Data Alignment Contract

> **Root Cause Document** — All model failures traced to X/Y row misalignment  
> Created: 2026-02-01  
> Status: **ACTIVE — All code must comply**

---

## Executive Summary

Multiple models (LogReg, potentially others) produced near-zero F1 scores (~0.003 instead of ~0.15-0.25) due to **row misalignment between X (embeddings) and Y (labels)**. This document establishes the canonical alignment contract that ALL pipeline components must follow.

---

## 1. The Failure Pattern

### Symptoms Observed
| Model | Expected F1 | Actual F1 | Root Cause |
|-------|-------------|-----------|------------|
| LogReg | ~0.15-0.25 | 0.0026 | X/Y row mismatch |
| KNN (before fix) | ~0.20 | ~0.003 | X/Y row mismatch |
| GBDT | TBD | TBD | Uses Cell 13a globals (likely safe) |

### Why Near-Zero F1?
When X row `i` contains embeddings for protein A but Y row `i` contains labels for protein B:
- Model learns **noise** (random embedding → random label mapping)
- Predictions are essentially random
- F1 collapses to near-zero

---

## 2. Evidence Chain

### 2.1 The Single Source of Truth

**`train_seq.feather`** defines the canonical protein ordering:
```
Row 0: sp|A0A009IHW8|EFTU2_ACIBA
Row 1: sp|A0A010P8A5|Y1354_PSEAI
Row 2: sp|A0A010QSW1|FOLD2_PSEAI
...
Row 82403: tr|A0A0G2K627|A0A0G2K627_9GAMM
```

### 2.2 What Must Align to train_seq.feather

| Artefact | Row Index Source | Alignment Status |
|----------|------------------|------------------|
| `esm2_3b_train.npy` | Pre-computed in train_seq order | ✅ Aligned |
| `esm2_650m_train.npy` | Pre-computed in train_seq order | ✅ Aligned |
| `t5_train.npy` | Pre-computed in train_seq order | ✅ Aligned |
| `ankh_train.npy` | Pre-computed in train_seq order | ✅ Aligned |
| `text_train.npy` | Pre-computed in train_seq order | ✅ Aligned |
| `X_train_mmap.npy` | **Must be rebuilt** | ⚠️ Needs verification |
| `Y` (label matrix) | Built from train_terms + train_seq order | ✅ If done correctly |

### 2.3 The Dangerous Pattern (AVOID)

```python
# ❌ WRONG: Iterating train_terms without explicit ordering
train_terms_df = pd.read_parquet('train_terms.parquet')
for _, row in train_terms_df.iterrows():
    pid = row['EntryID']
    row_idx = pid_to_row[pid]  # What order is pid_to_row built in?
    Y[row_idx, col] = 1
```

**Problem**: If `pid_to_row` is built from a different source (e.g., dict iteration, different file), the row indices won't match the embedding order.

### 2.4 The Safe Pattern (USE THIS)

```python
# ✅ CORRECT: Explicit ordering from train_seq.feather
train_seq = pd.read_feather('train_seq.feather')
train_ids = train_seq['id'].tolist()
train_ids_clean = [_clean_id(x) for x in train_ids]

# Build row index from THE SAME ordering
pid_to_row = {pid: i for i, pid in enumerate(train_ids_clean)}

# Now iterate labels (order doesn't matter - we look up the correct row)
train_terms_df = pd.read_csv('train_terms.tsv', sep='\t')
for _, row in train_terms_df.iterrows():
    pid = _clean_id(row['EntryID'])
    if pid in pid_to_row:
        row_idx = pid_to_row[pid]  # Always maps to train_seq order
        Y[row_idx, col] = 1
```

---

## 3. Cell-by-Cell Alignment Audit

### Cell 19 (Phase 2 Setup / Cell 13a) — ✅ CANONICAL

This cell establishes the global alignment. All downstream cells inherit from it.

**What it does correctly:**
1. Loads `train_seq.feather` → `train_ids_clean`
2. Builds `Y` with `.reindex(train_ids_clean)` — rows aligned to train_seq
3. Loads embedding `.npy` files (already in train_seq order)
4. Builds `X_train_mmap.npy` by concatenating in `train_ids` order
5. Exports globals: `X`, `Y`, `X_test`, `train_ids_clean`, `test_ids_clean`, `features_train`, `features_test`

**Contract**: Any cell using these globals gets aligned data.

### Cell 23 (GBDT) — ✅ SAFE

```python
# Uses globals from Cell 13a
X_tr = X[tr_idx]  # X is from Cell 13a (X_train_mmap.npy)
Y_elite = Y[:, elite_cols]  # Y is from Cell 13a
```

**Status**: Safe because it uses Cell 13a globals directly.

### Cell 25 (KNN / Cell 13E) — ✅ FIXED

**Original bug**: Built its own Y from TSV with different iteration order.

**Fix applied**: Now uses identical data loading pattern with explicit `train_seq.feather` ordering.

### Cell 26 (DNN / Cell 13D) — ✅ SAFE

```python
# Uses globals from Cell 13a
if 'features_train' not in globals() or 'features_test' not in globals():
    raise RuntimeError('Missing `features_train`/`features_test`. Run Cell 13a first.')
if 'Y' not in globals():
    raise RuntimeError('Missing Y. Run Cell 13a first.')

# Uses features_train dict (aligned to train_seq)
dnn_train = dict(features_train)
dnn_test = dict(features_test)
```

**Status**: Safe because it uses Cell 13a globals directly.

### Cell 27 (LogReg / Cell 13C) — ✅ FIXED (2026-02-01)

**Original bug**: 
- Loaded `esm2_3b_train.npy` only (2560 dims)
- Built Y from `train_terms.tsv` with row-by-row iteration
- Missed all other modalities (T5, ESM2-650M, Ankh, Text, Taxa)

**Fix applied**: Now uses Cell 13a globals (`X`, `Y`, `X_test`) directly:
- Uses full 7168-dim X_train_mmap (all modalities)
- Uses verified Y matrix from Cell 13a (aligned to train_seq.feather)
- Consistent with GBDT and DNN approach

### Cell 29 (GCN Stacker / Cell 15) — ✅ CONDITIONALLY SAFE

```python
# Uses global Y from Cell 13a
if 'Y' not in globals():
    raise RuntimeError("Missing Y (targets). Run Cell 13a first.")

# Loads Level-1 OOF predictions from disk
for m, (oof_name, test_name) in model_paths.items():
    oof = np.load(oof_p, mmap_mode='r')  # Must be in train_seq order!
```

**Status**: Safe **IF** all Level-1 models produced correctly aligned OOF predictions.
- Uses `Y` from Cell 13a (correct)
- Loads `oof_pred_*.npy` files which **must** be in train_seq order
- **Critical dependency**: If any Level-1 model had alignment bugs, GCN inherits them

### Summary Table

| Cell | Model | Data Source | Alignment Status |
|------|-------|-------------|------------------|
| 19 | Setup (13a) | Defines alignment | ✅ CANONICAL |
| 23 | GBDT | Cell 13a globals | ✅ Safe |
| 25 | KNN | Fixed (train_seq order) | ✅ Fixed |
| 26 | DNN | Cell 13a globals | ✅ Safe |
| 27 | LogReg | Fixed (esm2_3b + train_seq) | ✅ Fixed |
| 29 | GCN Stacker | Cell 13a Y + Level-1 OOF | ✅ Conditional |

---

## 4. Rebuilding X_train_mmap.npy Correctly

### 4.1 Current Issues

The current `X_train_mmap.npy` build in Cell 13a **should** be correct because:
- It uses `features_train` dict which loads from `.npy` files
- Those files are pre-aligned to train_seq order
- It concatenates in a consistent order

**However**, we should add explicit verification.

### 4.2 Rebuild Protocol

```python
# === CANONICAL X_train_mmap.npy BUILD ===
import numpy as np
import pandas as pd
from pathlib import Path

WORK_ROOT = Path('cafa6_data')
FEAT_DIR = WORK_ROOT / 'features'

# Step 1: Load canonical protein ordering
train_seq = pd.read_feather(WORK_ROOT / 'parsed' / 'train_seq.feather')
train_ids = train_seq['id'].tolist()
n_train = len(train_ids)

def _clean_id(raw: str) -> str:
    """Extract UniProt accession from FASTA header."""
    if '|' in raw:
        parts = raw.split('|')
        return parts[1] if len(parts) >= 2 else raw
    return raw

train_ids_clean = [_clean_id(x) for x in train_ids]

# Step 2: Define embedding sources and their dimensions
EMBED_SOURCES = [
    ('t5', 'train_embeds_t5.npy', 1024),
    ('esm2_650m', 'train_embeds_esm2.npy', 1280),
    ('esm2_3b', 'train_embeds_esm2_3b.npy', 2560),
    ('ankh', 'train_embeds_ankh.npy', 1536),
    ('text', 'train_embeds_text.npy', 768),
]

# Step 3: Verify all sources have correct row count
for name, filename, expected_dim in EMBED_SOURCES:
    path = FEAT_DIR / filename
    arr = np.load(path, mmap_mode='r')
    assert arr.shape[0] == n_train, f"{name}: expected {n_train} rows, got {arr.shape[0]}"
    assert arr.shape[1] == expected_dim, f"{name}: expected {expected_dim} cols, got {arr.shape[1]}"
    print(f"✓ {name}: {arr.shape}")

# Step 4: Build concatenated X_train_mmap.npy
total_dim = sum(d for _, _, d in EMBED_SOURCES)
X_path = FEAT_DIR / 'X_train_mmap.npy'

X_mm = np.lib.format.open_memmap(
    str(X_path), mode='w+', dtype=np.float32, shape=(n_train, total_dim)
)

col = 0
for name, filename, dim in EMBED_SOURCES:
    print(f"Writing {name} to cols {col}:{col+dim}")
    arr = np.load(FEAT_DIR / filename, mmap_mode='r')
    # Stream in chunks to avoid RAM spike
    chunk = 5000
    for i in range(0, n_train, chunk):
        j = min(i + chunk, n_train)
        X_mm[i:j, col:col+dim] = arr[i:j]
    col += dim

X_mm.flush()
del X_mm

print(f"✓ Saved: {X_path} shape=({n_train}, {total_dim})")

# Step 5: Verification — sample rows should match source embeddings
X_verify = np.load(X_path, mmap_mode='r')
for name, filename, dim in EMBED_SOURCES[:2]:  # Check first two sources
    arr = np.load(FEAT_DIR / filename, mmap_mode='r')
    # Check first row
    col_start = sum(d for _, _, d in EMBED_SOURCES[:EMBED_SOURCES.index((name, filename, dim))])
    assert np.allclose(X_verify[0, col_start:col_start+dim], arr[0]), f"{name} row 0 mismatch!"
    # Check last row
    assert np.allclose(X_verify[-1, col_start:col_start+dim], arr[-1]), f"{name} row -1 mismatch!"
print("✓ Verification passed")
```

### 4.3 Column Layout Contract

After rebuild, `X_train_mmap.npy` has this column layout:

| Columns | Embedding | Dimensions |
|---------|-----------|------------|
| 0:1024 | T5 | 1024 |
| 1024:2304 | ESM2-650M | 1280 |
| 2304:4864 | ESM2-3B | 2560 |
| 4864:6400 | Ankh | 1536 |
| 6400:7168 | Text | 768 |
| **Total** | | **7168** |

**To use a subset:**
```python
# ESM2-3B only (for LogReg)
X_esm2_3b = X[:, 2304:4864]

# All embeddings (for GBDT)
X_full = X[:, :]
```

---

## 5. Y Matrix Alignment Contract

### 5.1 Building Y Correctly

```python
# === CANONICAL Y BUILD ===
import numpy as np
import pandas as pd
from pathlib import Path

WORK_ROOT = Path('cafa6_data')

# Step 1: Load canonical protein ordering (SAME as X build)
train_seq = pd.read_feather(WORK_ROOT / 'parsed' / 'train_seq.feather')
train_ids_clean = [_clean_id(x) for x in train_seq['id'].tolist()]
n_train = len(train_ids_clean)
pid_to_row = {pid: i for i, pid in enumerate(train_ids_clean)}

# Step 2: Load term vocabulary (defines column ordering)
import json
top_terms = json.loads((WORK_ROOT / 'features' / 'top_terms_13500.json').read_text())
n_terms = len(top_terms)
term_to_col = {t: i for i, t in enumerate(top_terms)}

# Step 3: Build sparse Y from train_terms.tsv
train_terms_df = pd.read_csv(WORK_ROOT / 'Train' / 'train_terms.tsv', sep='\t')

rows, cols = [], []
for _, row in train_terms_df.iterrows():
    pid = _clean_id(str(row['EntryID']))
    term = str(row['term'])
    if pid in pid_to_row and term in term_to_col:
        rows.append(pid_to_row[pid])
        cols.append(term_to_col[term])

from scipy import sparse
Y_sparse = sparse.csr_matrix(
    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
    shape=(n_train, n_terms)
)
Y = Y_sparse.toarray()

print(f"✓ Y shape: {Y.shape}")
print(f"  Positive entries: {Y.sum():,.0f}")
```

### 5.2 Aspect Column Ranges

The `top_terms_13500.json` is ordered BP → MF → CC:

| Aspect | Column Range | Count |
|--------|--------------|-------|
| BP (Biological Process) | 0:10000 | 10,000 |
| MF (Molecular Function) | 10000:12000 | 2,000 |
| CC (Cellular Component) | 12000:13500 | 1,500 |

**To evaluate per-aspect:**
```python
Y_bp = Y[:, 0:10000]
Y_mf = Y[:, 10000:12000]
Y_cc = Y[:, 12000:13500]
```

---

## 6. Fix Checklist by Component

### 6.1 Models — Status After Audit

| Component | Cell | Data Source | Status | Notes |
|-----------|------|-------------|--------|-------|
| KNN | 25 (13E) | Direct embeddings + train_seq order | ✅ Fixed | Uses safe pattern |
| LogReg | 27 (13C) | esm2_3b_train.npy + train_seq order | ✅ Fixed | KNN-aligned loading |
| GBDT | 23 (13B) | Cell 13a globals (X_train_mmap + Y) | ✅ Safe | Inherits alignment |
| DNN | 26 (13D) | Cell 13a globals (features_train + Y) | ✅ Safe | Inherits alignment |
| GCN Stacker | 29 (15) | Cell 13a Y + Level-1 OOF files | ✅ Conditional | Depends on L1 alignment |

### 6.2 Level-1 Predictions

All OOF predictions must have shape `(82404, 13500)` with rows aligned to `train_seq.feather`:

| Artefact | Expected Shape | Row Alignment |
|----------|----------------|---------------|
| `oof_pred_knn.npy` | (82404, 13500) | train_seq order |
| `oof_pred_logreg.npy` | (82404, 13500) | train_seq order |
| `oof_pred_gbdt.npy` | (82404, 13500) | train_seq order |

### 6.3 Stacker

The stacker receives Level-1 predictions as features. It must:
1. Stack predictions: `X_stack = np.concatenate([oof_knn, oof_logreg, oof_gbdt], axis=1)`
2. Use the **same Y** as Level-1 models (Cell 13a global)
3. Maintain row alignment throughout

---

## 7. Verification Protocol

### Before Training Any Model

Run this verification:

```python
def verify_alignment(X, Y, name="Model"):
    """Verify X and Y are aligned before training."""
    assert X.shape[0] == Y.shape[0], f"{name}: X rows ({X.shape[0]}) != Y rows ({Y.shape[0]})"
    assert X.shape[0] == 82404, f"{name}: Expected 82404 rows, got {X.shape[0]}"
    assert Y.shape[1] == 13500, f"{name}: Expected 13500 cols, got {Y.shape[1]}"
    
    # Check Y has labels in all aspects
    y_bp = (Y[:, 0:10000] > 0).sum()
    y_mf = (Y[:, 10000:12000] > 0).sum()
    y_cc = (Y[:, 12000:13500] > 0).sum()
    
    assert y_bp > 0, f"{name}: No BP labels!"
    assert y_mf > 0, f"{name}: No MF labels!"
    assert y_cc > 0, f"{name}: No CC labels!"
    
    print(f"✓ {name} alignment verified: X={X.shape}, Y={Y.shape}")
    print(f"  Labels: BP={y_bp:,}, MF={y_mf:,}, CC={y_cc:,}")
```

### After Training — F1 Sanity Check

```python
def f1_sanity_check(f1_scores, name="Model"):
    """Fail fast if F1 is suspiciously low."""
    min_expected = 0.05  # Anything below this suggests alignment issues
    
    for aspect, score in f1_scores.items():
        if score < min_expected:
            raise RuntimeError(
                f"{name} {aspect} F1 = {score:.4f} — suspiciously low! "
                "Check X/Y alignment."
            )
    print(f"✓ {name} F1 sanity check passed")
```

---

## 8. Golden Rules

1. **Single Source of Truth**: `train_seq.feather` defines row ordering
2. **Never Iterate Without Index**: Always build `pid_to_row` from train_seq first
3. **Verify Before Train**: Run `verify_alignment()` before any model.fit()
4. **F1 Sanity Check**: If F1 < 0.05, assume alignment bug until proven otherwise
5. **Use Cell 13a Globals**: When possible, use `X`, `Y` from Cell 13a instead of loading separately
6. **Document Column Ranges**: Always document which columns correspond to which embeddings/aspects

---

## 9. Quick Reference

### Canonical File Paths
```
cafa6_data/
├── parsed/
│   ├── train_seq.feather      # Canonical protein ordering (82,404 rows)
│   └── test_seq.feather       # Test protein ordering (224,309 rows)
├── Train/
│   └── train_terms.tsv        # Ground truth labels
├── features/
│   ├── top_terms_13500.json   # Term vocabulary (column ordering)
│   ├── esm2_3b_train.npy      # (82404, 2560) - aligned to train_seq
│   ├── esm2_3b_test.npy       # (224309, 2560) - aligned to test_seq
│   ├── X_train_mmap.npy       # (82404, 7168) - concatenated embeddings
│   └── X_test_mmap.npy        # (224309, 7168) - concatenated embeddings
└── features/level1_preds/
    ├── oof_pred_knn.npy       # (82404, 13500) - aligned to train_seq
    ├── oof_pred_logreg.npy    # (82404, 13500) - aligned to train_seq
    └── oof_pred_gbdt.npy      # (82404, 13500) - aligned to train_seq
```

### Shape Contracts
| Artefact | Train Shape | Test Shape |
|----------|-------------|------------|
| Sequences | 82,404 | 224,309 |
| ESM2-3B | (82404, 2560) | (224309, 2560) |
| X_mmap (full) | (82404, 7168) | (224309, 7168) |
| Y (labels) | (82404, 13500) | N/A |
| OOF predictions | (82404, 13500) | (224309, 13500) |

---

## Changelog

- **2026-02-01**: Initial document created after LogReg F1=0.0026 failure traced to X/Y misalignment
- **2026-02-01**: LogReg cell fixed to use KNN-aligned data loading
- **2026-02-01**: Established canonical alignment contract
- **2026-02-01**: Full cell-by-cell audit completed:
  - DNN (Cell 26): Confirmed ✅ Safe — uses Cell 13a globals
  - GCN Stacker (Cell 29): Confirmed ✅ Conditional — depends on Level-1 alignment
  - All Level-1 models now verified as aligned or fixed
