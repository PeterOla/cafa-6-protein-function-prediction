#!/usr/bin/env python3
"""
================================================================================
CAFA-6 Super-KNN Late Fusion - Multi-Model Ensemble (13,500 Terms)
================================================================================

Late Fusion: Train separate KNN per embedding model, then combine predictions
with optimized weights. This preserves each model's similarity space.

MODELS:
    - ESM2-3B (2560 dims)
    - Ankh (1536 dims)  
    - T5 (1024 dims)

OPTIMIZATION:
    1. Train KNN for each model separately
    2. Grid search optimal weights on OOF
    3. Optionally retune K per aspect

USAGE:
    python knn_late_fusion.py --work-dir /path/to/cafa6_data

Author: Late fusion optimization for multi-modal KNN
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import time
import json
import gc
import re
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# CONFIGURATION
# =============================================================================
# Default K values (will be retuned)
DEFAULT_ASPECT_K = {'BP': 5, 'MF': 10, 'CC': 15}

OPTIMAL_THRESHOLDS = {
    'BP': 0.40,
    'MF': 0.40,
    'CC': 0.30,
    'ALL': 0.30
}

DEFAULT_N_FOLDS = 5
DEFAULT_BATCH_SIZE = 250

# Models to ensemble
EMBEDDING_MODELS = ['esm2_3b', 'ankh', 't5']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_uniprot_accession(protein_id: str) -> str:
    """Extract UniProt accession from FASTA-style protein ID."""
    protein_id = str(protein_id)
    if '|' in protein_id:
        match = re.search(r'\|(.+?)\|', protein_id)
        return match.group(1) if match else protein_id
    return protein_id


def parse_go_aspects(obo_path: Path) -> dict:
    """Parse GO ontology file to get term -> aspect mapping."""
    aspects = {}
    current_term = None
    
    namespace_map = {
        'biological_process': 'BP',
        'molecular_function': 'MF',
        'cellular_component': 'CC'
    }
    
    with open(obo_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[Term]'):
                current_term = None
            elif line.startswith('id: GO:'):
                current_term = line[4:]
            elif line.startswith('namespace:') and current_term:
                ns = line.split(': ', 1)[1]
                aspects[current_term] = namespace_map.get(ns, 'UNK')
    
    return aspects


def load_13500_vocabulary(work_dir: Path) -> list:
    """Load the 13,500 term vocabulary."""
    vocab_path = work_dir / 'features' / 'top_terms_13500.json'
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_cafa_f1(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """Compute per-protein F1 averaged across samples."""
    f1_sum = 0
    n_samples = y_true.shape[0]
    
    for i in range(n_samples):
        tp = ((y_pred_binary[i] == 1) & (y_true[i] == 1)).sum()
        fp = ((y_pred_binary[i] == 1) & (y_true[i] == 0)).sum()
        fn = ((y_pred_binary[i] == 0) & (y_true[i] == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_sum += f1
    
    return f1_sum / n_samples


def evaluate_predictions(oof_preds: np.ndarray, Y_train: sparse.csr_matrix, 
                         aspect_masks: dict, thresholds: dict = None) -> dict:
    """Evaluate predictions with given thresholds."""
    if thresholds is None:
        thresholds = OPTIMAL_THRESHOLDS
    
    results = {}
    for aspect in ['BP', 'MF', 'CC']:
        col_indices = aspect_masks.get(aspect, [])
        if not col_indices:
            continue
        
        threshold = thresholds.get(aspect, 0.3)
        y_true = Y_train[:, col_indices].toarray()
        y_pred = (oof_preds[:, col_indices] >= threshold).astype(np.int8)
        
        f1 = compute_cafa_f1(y_true, y_pred)
        results[aspect] = f1
    
    results['CAFA'] = sum(results.values()) / len(results)
    return results


# =============================================================================
# SINGLE MODEL KNN TRAINING
# =============================================================================

def train_single_knn(
    model_name: str,
    train_emb: np.ndarray,
    Y_train: sparse.csr_matrix,
    aspect_masks: dict,
    aspect_k: dict,
    n_folds: int = 5,
    batch_size: int = 250
) -> np.ndarray:
    """
    Train KNN for a single embedding model and return OOF predictions.
    """
    print(f"\n{'='*60}")
    print(f"[KNN] Training: {model_name} ({train_emb.shape[1]} dims)")
    print(f"{'='*60}")
    
    k_max = max(aspect_k.values())
    n_train, n_terms = Y_train.shape
    
    # Convert to float16 for memory
    train_emb = train_emb.astype(np.float16)
    
    # Initialize OOF predictions
    oof_preds = np.zeros((n_train, n_terms), dtype=np.float32)
    
    # Stratification
    stratify_target = Y_train.sum(axis=1).A1
    stratify_target = np.minimum(stratify_target, 10).astype(int)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(train_emb, stratify_target), 1):
        fold_start = time.time()
        print(f"  Fold {fold_num}/{n_folds} (train={len(train_idx):,}, val={len(val_idx):,})")
        
        X_train_fold = train_emb[train_idx].astype(np.float32)
        X_val_fold = train_emb[val_idx].astype(np.float32)
        Y_train_fold = Y_train[train_idx]
        
        # Build KNN
        knn = NearestNeighbors(n_neighbors=k_max, metric='cosine', n_jobs=2)
        knn.fit(X_train_fold)
        
        # Query validation
        distances, indices = knn.kneighbors(X_val_fold)
        similarities = 1 - distances
        
        # Predict with Mixed-K
        for batch_start in range(0, len(val_idx), batch_size):
            batch_end = min(batch_start + batch_size, len(val_idx))
            batch_size_actual = batch_end - batch_start
            
            batch_indices = indices[batch_start:batch_end]
            batch_sims = similarities[batch_start:batch_end]
            
            flat_indices = batch_indices.flatten()
            neighbor_labels = Y_train_fold[flat_indices].toarray()
            neighbor_labels = neighbor_labels.reshape(batch_size_actual, k_max, -1)
            
            combined_scores = np.zeros((batch_size_actual, n_terms), dtype=np.float32)
            
            for aspect, k_val in aspect_k.items():
                col_indices = aspect_masks.get(aspect, [])
                if not col_indices:
                    continue
                
                a_sims = batch_sims[:, :k_val]
                a_labels = neighbor_labels[:, :k_val, :][:, :, col_indices]
                
                sims_expanded = a_sims[:, :, np.newaxis]
                a_scores = (sims_expanded * a_labels).sum(axis=1)
                a_scores /= (a_sims.sum(axis=1, keepdims=True) + 1e-9)
                
                combined_scores[:, col_indices] = a_scores
            
            oof_preds[val_idx[batch_start:batch_end]] = combined_scores
        
        print(f"    Time: {time.time() - fold_start:.1f}s")
        
        del knn, X_train_fold
        gc.collect()
    
    # Per-protein max normalization
    row_max = oof_preds.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    oof_preds /= row_max
    
    return oof_preds


# =============================================================================
# K TUNING
# =============================================================================

def tune_k_for_model(
    model_name: str,
    train_emb: np.ndarray,
    Y_train: sparse.csr_matrix,
    aspect_masks: dict,
    n_folds: int = 3,  # Fewer folds for speed
    batch_size: int = 250
) -> dict:
    """
    Tune K values per aspect for a single model.
    Tests K in [3, 5, 7, 10, 15, 20].
    """
    print(f"\n{'='*60}")
    print(f"[K-TUNING] {model_name}")
    print(f"{'='*60}")
    
    k_candidates = [3, 5, 7, 10, 15, 20]
    k_max = max(k_candidates)
    
    n_train, n_terms = Y_train.shape
    train_emb = train_emb.astype(np.float16)
    
    # Stratification
    stratify_target = Y_train.sum(axis=1).A1
    stratify_target = np.minimum(stratify_target, 10).astype(int)
    
    # Collect predictions for different K values
    # Shape: (n_train, n_terms, len(k_candidates))
    oof_by_k = {k: np.zeros((n_train, n_terms), dtype=np.float32) for k in k_candidates}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(train_emb, stratify_target), 1):
        print(f"  Fold {fold_num}/{n_folds}")
        
        X_train_fold = train_emb[train_idx].astype(np.float32)
        X_val_fold = train_emb[val_idx].astype(np.float32)
        Y_train_fold = Y_train[train_idx]
        
        knn = NearestNeighbors(n_neighbors=k_max, metric='cosine', n_jobs=2)
        knn.fit(X_train_fold)
        
        distances, indices = knn.kneighbors(X_val_fold)
        similarities = 1 - distances
        
        for batch_start in range(0, len(val_idx), batch_size):
            batch_end = min(batch_start + batch_size, len(val_idx))
            batch_size_actual = batch_end - batch_start
            
            batch_indices = indices[batch_start:batch_end]
            batch_sims = similarities[batch_start:batch_end]
            
            flat_indices = batch_indices.flatten()
            neighbor_labels = Y_train_fold[flat_indices].toarray()
            neighbor_labels = neighbor_labels.reshape(batch_size_actual, k_max, -1)
            
            # Compute for each K
            for k in k_candidates:
                k_sims = batch_sims[:, :k]
                k_labels = neighbor_labels[:, :k, :]
                
                sims_expanded = k_sims[:, :, np.newaxis]
                scores = (sims_expanded * k_labels).sum(axis=1)
                scores /= (k_sims.sum(axis=1, keepdims=True) + 1e-9)
                
                oof_by_k[k][val_idx[batch_start:batch_end]] = scores
        
        del knn, X_train_fold
        gc.collect()
    
    # Normalize each
    for k in k_candidates:
        row_max = oof_by_k[k].max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        oof_by_k[k] /= row_max
    
    # Find best K per aspect
    best_k = {}
    for aspect in ['BP', 'MF', 'CC']:
        col_indices = aspect_masks.get(aspect, [])
        if not col_indices:
            continue
        
        threshold = OPTIMAL_THRESHOLDS.get(aspect, 0.3)
        y_true = Y_train[:, col_indices].toarray()
        
        best_f1 = -1
        best_k_val = 5
        
        for k in k_candidates:
            y_pred = (oof_by_k[k][:, col_indices] >= threshold).astype(np.int8)
            f1 = compute_cafa_f1(y_true, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_k_val = k
        
        best_k[aspect] = best_k_val
        print(f"  {aspect}: Best K={best_k_val} (F1={best_f1:.4f})")
    
    return best_k


# =============================================================================
# WEIGHT OPTIMIZATION
# =============================================================================

def grid_search_weights(
    model_oofs: dict,
    Y_train: sparse.csr_matrix,
    aspect_masks: dict,
    weight_steps: int = 11
) -> tuple:
    """
    Grid search optimal weights for combining model predictions.
    
    Returns:
        (best_weights, best_f1, all_results)
    """
    print(f"\n{'='*60}")
    print("[WEIGHT OPTIMIZATION] Grid Search")
    print(f"{'='*60}")
    
    models = list(model_oofs.keys())
    n_models = len(models)
    
    # Generate weight combinations (sum to 1)
    weights_range = np.linspace(0, 1, weight_steps)
    
    best_f1 = -1
    best_weights = None
    all_results = []
    
    # For 3 models, grid search with constraint sum=1
    if n_models == 3:
        for w1 in weights_range:
            for w2 in weights_range:
                w3 = 1.0 - w1 - w2
                if w3 < 0 or w3 > 1:
                    continue
                
                weights = {models[0]: w1, models[1]: w2, models[2]: w3}
                
                # Combine predictions
                combined = np.zeros_like(model_oofs[models[0]])
                for m, w in weights.items():
                    combined += w * model_oofs[m]
                
                # Evaluate
                results = evaluate_predictions(combined, Y_train, aspect_masks)
                cafa_f1 = results['CAFA']
                
                all_results.append({'weights': weights.copy(), 'f1': cafa_f1, 'results': results})
                
                if cafa_f1 > best_f1:
                    best_f1 = cafa_f1
                    best_weights = weights.copy()
    
    elif n_models == 2:
        for w1 in weights_range:
            w2 = 1.0 - w1
            weights = {models[0]: w1, models[1]: w2}
            
            combined = np.zeros_like(model_oofs[models[0]])
            for m, w in weights.items():
                combined += w * model_oofs[m]
            
            results = evaluate_predictions(combined, Y_train, aspect_masks)
            cafa_f1 = results['CAFA']
            
            all_results.append({'weights': weights.copy(), 'f1': cafa_f1})
            
            if cafa_f1 > best_f1:
                best_f1 = cafa_f1
                best_weights = weights.copy()
    
    print(f"\n  Best weights: {best_weights}")
    print(f"  Best CAFA F1: {best_f1:.4f}")
    
    # Show top 5
    all_results.sort(key=lambda x: x['f1'], reverse=True)
    print(f"\n  Top 5 configurations:")
    for i, r in enumerate(all_results[:5]):
        w_str = ', '.join([f"{k}={v:.2f}" for k, v in r['weights'].items()])
        print(f"    {i+1}. F1={r['f1']:.4f} | {w_str}")
    
    return best_weights, best_f1, all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CAFA-6 KNN Late Fusion')
    parser.add_argument('--work-dir', type=str, required=True)
    parser.add_argument('--skip-k-tuning', action='store_true',
                        help='Skip K tuning, use defaults')
    parser.add_argument('--models', type=str, default='esm2_3b,ankh,t5',
                        help='Comma-separated list of models')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    models = [m.strip() for m in args.models.split(',')]
    
    print("="*70)
    print("CAFA-6 Super-KNN Late Fusion")
    print("="*70)
    print(f"Work directory: {work_dir}")
    print(f"Models: {models}")
    
    # -------------------------------------------------------------------------
    # Load vocabulary and data
    # -------------------------------------------------------------------------
    top_terms = load_13500_vocabulary(work_dir)
    term_to_idx = {t: i for i, t in enumerate(top_terms)}
    n_terms = len(top_terms)
    
    # Load protein IDs
    train_seq = pd.read_feather(work_dir / 'parsed' / 'train_seq.feather')
    id_col = 'id' if 'id' in train_seq.columns else 'EntryID'
    train_ids = [extract_uniprot_accession(x) for x in train_seq[id_col].tolist()]
    del train_seq
    
    # Build label matrix
    train_terms_df = pd.read_csv(work_dir / 'Train' / 'train_terms.tsv', sep='\t')
    protein_to_idx = {pid: i for i, pid in enumerate(train_ids)}
    
    rows, cols = [], []
    for _, row in train_terms_df.iterrows():
        pid_clean = extract_uniprot_accession(row['EntryID'])
        if pid_clean in protein_to_idx and row['term'] in term_to_idx:
            rows.append(protein_to_idx[pid_clean])
            cols.append(term_to_idx[row['term']])
    
    Y_train = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(train_ids), n_terms)
    )
    print(f"Label matrix: {Y_train.shape}, nnz: {Y_train.nnz:,}")
    del rows, cols, train_terms_df
    
    # Load aspects
    term_aspects = parse_go_aspects(work_dir / 'Train' / 'go-basic.obo')
    aspect_masks = {'BP': [], 'MF': [], 'CC': []}
    for i, term in enumerate(top_terms):
        aspect = term_aspects.get(term, 'UNK')
        if aspect in aspect_masks:
            aspect_masks[aspect].append(i)
    
    print(f"Aspects: BP={len(aspect_masks['BP'])}, MF={len(aspect_masks['MF'])}, CC={len(aspect_masks['CC'])}")
    
    # -------------------------------------------------------------------------
    # Load embeddings
    # -------------------------------------------------------------------------
    feat_dir = work_dir / 'features'
    embeddings = {}
    for model in models:
        path = feat_dir / f'train_embeds_{model}.npy'
        if path.exists():
            embeddings[model] = np.load(path, mmap_mode='r')
            print(f"Loaded {model}: {embeddings[model].shape}")
        else:
            print(f"WARNING: {model} not found, skipping")
    
    models = list(embeddings.keys())  # Update to only available models
    
    # -------------------------------------------------------------------------
    # Step 1: K Tuning (optional)
    # -------------------------------------------------------------------------
    model_k = {}
    if not args.skip_k_tuning:
        for model in models:
            model_k[model] = tune_k_for_model(
                model, embeddings[model], Y_train, aspect_masks,
                n_folds=3, batch_size=args.batch_size
            )
    else:
        for model in models:
            model_k[model] = DEFAULT_ASPECT_K.copy()
    
    print(f"\nK values per model:")
    for model, k_dict in model_k.items():
        print(f"  {model}: {k_dict}")
    
    # -------------------------------------------------------------------------
    # Step 2: Train each model with tuned K
    # -------------------------------------------------------------------------
    model_oofs = {}
    for model in models:
        model_oofs[model] = train_single_knn(
            model, embeddings[model], Y_train, aspect_masks,
            aspect_k=model_k[model],
            n_folds=DEFAULT_N_FOLDS,
            batch_size=args.batch_size
        )
        
        # Evaluate individually
        results = evaluate_predictions(model_oofs[model], Y_train, aspect_masks)
        print(f"  {model} standalone: CAFA F1={results['CAFA']:.4f} (BP={results['BP']:.4f}, MF={results['MF']:.4f}, CC={results['CC']:.4f})")
    
    # -------------------------------------------------------------------------
    # Step 3: Grid search weights
    # -------------------------------------------------------------------------
    best_weights, best_f1, all_results = grid_search_weights(
        model_oofs, Y_train, aspect_masks, weight_steps=11
    )
    
    # -------------------------------------------------------------------------
    # Step 4: Final combined predictions
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[FINAL] Combined Predictions")
    print(f"{'='*60}")
    
    combined_oof = np.zeros_like(model_oofs[models[0]])
    for model, weight in best_weights.items():
        combined_oof += weight * model_oofs[model]
    
    final_results = evaluate_predictions(combined_oof, Y_train, aspect_masks)
    
    print(f"\n  FINAL CAFA F1: {final_results['CAFA']:.4f}")
    print(f"    BP: {final_results['BP']:.4f}")
    print(f"    MF: {final_results['MF']:.4f}")
    print(f"    CC: {final_results['CC']:.4f}")
    
    # Save combined OOF
    pred_dir = work_dir / 'features' / 'level1_preds'
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(pred_dir / 'oof_pred_knn_fusion.npy', combined_oof)
    print(f"\nSaved: {pred_dir / 'oof_pred_knn_fusion.npy'}")
    
    # Save config
    config = {
        'models': models,
        'weights': best_weights,
        'k_values': model_k,
        'cafa_f1': final_results['CAFA'],
        'aspect_f1': {k: v for k, v in final_results.items() if k != 'CAFA'}
    }
    config_path = pred_dir / 'knn_fusion_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {config_path}")
    
    print("\n" + "="*70)
    print("[COMPLETE]")
    print("="*70)


if __name__ == '__main__':
    main()
