import numpy as np
import pandas as pd
from scipy import sparse
import argparse
from pathlib import Path
import json
import gc
import time
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Try to import cuML, fallback to sklearn SGD
try:
    import cupy as cp
    from cuml.linear_model import LogisticRegression as cuLogReg
    from cuml.multiclass import OneVsRestClassifier as cuOVR
    HAS_GPU = True
    print("Using cuML (GPU acceleration enabled)")
except ImportError:
    import psutil
    from sklearn.linear_model import SGDClassifier
    from sklearn.multiclass import OneVsRestClassifier
    HAS_GPU = False
    print("WARNING: cuML not found. Using sklearn SGDClassifier (CPU). This will be SLOW.")

# -------------------------------------------------------------------------
# CONSTANTS & CONFIG
# -------------------------------------------------------------------------
DEFAULT_N_FOLDS = 5
SEED = 42

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------
def extract_accession(protein_id: str) -> str:
    """Extracts UniProt accession from a raw ID (e.g. 'sp|P12345|NAME' -> 'P12345')."""
    protein_id = str(protein_id)
    if '|' in protein_id:
        match = re.search(r'\|(.+?)\|', protein_id)
        return match.group(1) if match else protein_id
    return protein_id

def load_data(work_dir: Path):
    print(f"Loading data from {work_dir}...")
    
    # 1. Load Embeddings (X)
    train_emb_path = work_dir / 'features' / 'train_embeds_esm2_3b.npy'
    test_emb_path = work_dir / 'features' / 'test_embeds_esm2_3b.npy'
    
    if not train_emb_path.exists():
        raise FileNotFoundError(f"Missing train embeddings: {train_emb_path}")
        
    print(f"  Loading embeddings...")
    # Load full if possible, but keep raw float32
    train_emb = np.load(train_emb_path).astype(np.float32)
    test_emb = np.load(test_emb_path).astype(np.float32)
    print(f"    Train X: {train_emb.shape}")
    print(f"    Test X:  {test_emb.shape}")

    # 2. Load 13,500 Vocabulary
    terms_path = work_dir / 'features' / 'top_terms_13500.json'
    if not terms_path.exists():
        raise FileNotFoundError(f"Missing vocabulary: {terms_path}")
    
    with open(terms_path) as f:
        top_terms = json.load(f)
    print(f"  Loaded {len(top_terms)} terms")

    # 3. Load Protein IDs
    train_seq_path = work_dir / 'parsed' / 'train_seq.feather'
    train_seq = pd.read_feather(train_seq_path)
    train_ids = [extract_accession(pid) for pid in train_seq['id'].tolist()]
    
    test_seq_path = work_dir / 'parsed' / 'test_seq.feather'
    test_seq = pd.read_feather(test_seq_path)
    test_ids = [extract_accession(pid) for pid in test_seq['id'].tolist()]
    
    print(f"  Loaded {len(train_ids)} train IDs, {len(test_ids)} test IDs")
    return train_emb, test_emb, train_ids, test_ids, top_terms

def build_y_matrix(train_ids, top_terms, work_dir: Path):
    print("Building aligned Y matrix (Vectorized + Safe)...")
    train_terms_path = work_dir / 'Train' / 'train_terms.tsv'
    train_terms_df = pd.read_csv(train_terms_path, sep='\t')
    
    # Clean IDs vectorically (Fast & Safe)
    print("  Cleaning annotation IDs...")
    train_terms_df['EntryID_Clean'] = train_terms_df['EntryID'].apply(extract_accession)
    
    term_to_idx = {t: i for i, t in enumerate(top_terms)}
    protein_to_idx = {pid: i for i, pid in enumerate(train_ids)}
    n_terms = len(top_terms)
    
    rows, cols = [], []
    
    # Filter using the CLEANED ID
    valid_df = train_terms_df[
        train_terms_df['EntryID_Clean'].isin(protein_to_idx) & 
        train_terms_df['term'].isin(term_to_idx)
    ]
    
    print(f"  Mapping {len(valid_df)} annotations to matrix...")
    for pid, term in zip(valid_df['EntryID_Clean'], valid_df['term']):
        rows.append(protein_to_idx[pid])
        cols.append(term_to_idx[term])
        
    Y = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(train_ids), n_terms)
    )
    print(f"  Y matrix shape: {Y.shape}, nnz: {Y.nnz:,}")
    return Y

def get_aspect_masks(top_terms, work_dir: Path):
    print("Parsing GO aspects...")
    obo_path = work_dir / 'Train' / 'go-basic.obo'
    aspects = {}
    with open(obo_path) as f:
        cur_term = None
        for line in f:
            line = line.strip()
            if line.startswith('id: GO:'): cur_term = line[4:]
            elif line.startswith('namespace:') and cur_term:
                ns = line.split(': ')[1]
                if ns == 'biological_process': aspects[cur_term] = 'BP'
                elif ns == 'molecular_function': aspects[cur_term] = 'MF'
                elif ns == 'cellular_component': aspects[cur_term] = 'CC'
    
    masks = {'BP': [], 'MF': [], 'CC': []}
    for i, term in enumerate(top_terms):
        asp = aspects.get(term, 'UNK')
        if asp in masks: masks[asp].append(i)
    return masks

def train_predict_logreg(X, Y, X_test, aspect_indices, aspect_name, work_dir, n_folds=5):
    print(f"\nTraining LogReg for Aspect: {aspect_name} ({len(aspect_indices)} terms)")
    
    Y_aspect = Y[:, aspect_indices].toarray() 
    n_train = X.shape[0]
    n_test = X_test.shape[0]
    n_classes = len(aspect_indices)
    
    pred_dir = work_dir / 'features' / 'level1_preds'
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    oof_path = pred_dir / f'oof_pred_logreg_{aspect_name}.npy'
    test_path = pred_dir / f'test_pred_logreg_{aspect_name}.npy'
    
    oof_preds = np.lib.format.open_memmap(str(oof_path), mode='w+', dtype=np.float32, shape=(n_train, n_classes))
    test_preds = np.lib.format.open_memmap(str(test_path), mode='w+', dtype=np.float32, shape=(n_test, n_classes))
    test_fold_accum = np.zeros((n_test, n_classes), dtype=np.float32)

    # Scaling - MEMORY EFFICIENT
    print("  Scaling features (incremental)...")
    scaler = StandardScaler()
    chunk_size = 4096
    for i in range(0, n_train, chunk_size):
        end = min(i + chunk_size, n_train)
        scaler.partial_fit(X[i:end])

    # Stratified CV
    label_counts = Y_aspect.sum(axis=1)
    strat_labels = np.clip(label_counts, 0, 10)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    fold = 0
    for train_idx, val_idx in skf.split(X, strat_labels):
        fold += 1
        print(f"    Fold {fold}/{n_folds}...")
        
        # Helper to get scaled subset efficiently
        def get_scaled_subset(indices):
            subset = np.zeros((len(indices), X.shape[1]), dtype=np.float32)
            for i in range(0, len(indices), chunk_size):
                b_start = i
                b_end = min(i + chunk_size, len(indices))
                b_idx = indices[b_start:b_end]
                # Force float32 output
                subset[b_start:b_end] = scaler.transform(X[b_idx]).astype(np.float32)
            return subset

        X_tr_fold = get_scaled_subset(train_idx)
        X_va_fold = get_scaled_subset(val_idx)
        
        if HAS_GPU:
             X_tr_fold = cp.asarray(X_tr_fold)
             X_va_fold = cp.asarray(X_va_fold)
             Y_tr_fold = cp.asarray(Y_aspect[train_idx], dtype=np.float32)
        else:
             Y_tr_fold = Y_aspect[train_idx]
        
        # Model
        if HAS_GPU:
            # class_weight='balanced'
            model = cuOVR(cuLogReg(penalty='l2', C=1.0, solver='qn', max_iter=1000, class_weight='balanced', output_type='cupy', verbose=False))
            model.fit(X_tr_fold, Y_tr_fold)
            probas_val = model.predict_proba(X_va_fold)
            if isinstance(probas_val, cp.ndarray): probas_val = probas_val.get()
        else:
            # class_weight='balanced'
            model = OneVsRestClassifier(SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=1000, class_weight='balanced', n_jobs=-1, random_state=SEED), n_jobs=-1)
            model.fit(X_tr_fold, Y_tr_fold)
            probas_val = model.predict_proba(X_va_fold)

        oof_preds[val_idx] = probas_val
        
        # Predict Test (Batched)
        for i in range(0, n_test, chunk_size):
            end = min(i + chunk_size, n_test)
            X_test_batch = scaler.transform(X_test[i:end]).astype(np.float32)
            if HAS_GPU: X_test_batch = cp.asarray(X_test_batch)
            p_batch = model.predict_proba(X_test_batch)
            if HAS_GPU and isinstance(p_batch, cp.ndarray): p_batch = p_batch.get()
            test_fold_accum[i:end] += p_batch
            
        del model, X_tr_fold, X_va_fold, Y_tr_fold
        if HAS_GPU: cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    test_preds[:] = test_fold_accum / n_folds
    oof_preds.flush()
    test_preds.flush()
    print(f"  Saved {oof_path}")
    return oof_preds, test_preds

def evaluate_predictions(oof_preds, Y, aspect_indices, aspect_name):
    Y_true = Y[:, aspect_indices].toarray()
    n_sample = 20000
    if Y_true.shape[0] > n_sample:
        idx = np.random.choice(Y_true.shape[0], n_sample, replace=False)
        y_t = Y_true[idx]
        y_p = np.array(oof_preds[idx])
    else:
        y_t = Y_true
        y_p = np.array(oof_preds)

    print(f"\nEvaluation (Sample {len(y_t)}) - {aspect_name}:")
    best_f1, best_thr = 0, 0
    for thr in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        pred_bin = (y_p >= thr).astype(np.int8)
        tp = (pred_bin * y_t).sum(axis=1)
        denom = pred_bin.sum(axis=1) + y_t.sum(axis=1)
        f1_scores = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom!=0)
        mean_f1 = f1_scores.mean()
        print(f"  Thr={thr:.2f}: F1={mean_f1:.4f}")
        if mean_f1 > best_f1: best_f1, best_thr = mean_f1, thr
            
    print(f"  >> Best {aspect_name}: F1={best_f1:.4f} @ {best_thr:.2f}")
    return best_f1, best_thr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, default=Path('cafa6_data'))
    parser.add_argument('--skip-test', action='store_true')
    args = parser.parse_args()
    
    train_emb, test_emb, train_ids, test_ids, top_terms = load_data(args.work_dir)
    Y = build_y_matrix(train_ids, top_terms, args.work_dir)
    aspect_masks = get_aspect_masks(top_terms, args.work_dir)
    results = {}
    
    for aspect in ['BP', 'MF', 'CC']:
        indices = aspect_masks[aspect]
        if not indices: continue
        oof, test = train_predict_logreg(train_emb, Y, test_emb, indices, aspect, args.work_dir)
        f1, thr = evaluate_predictions(oof, Y, indices, aspect)
        results[aspect] = {'f1': f1, 'thr': thr}

    print("\n========================================")
    print(results)

if __name__ == '__main__':
    main()
