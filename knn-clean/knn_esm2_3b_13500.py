#!/usr/bin/env python3
"""
================================================================================
CAFA-6 Super-KNN Ensemble - 13,500 Term Vocabulary (E2E Compatible)
================================================================================

This script implements the "Super-KNN" Mixed-K Ensemble using the FILTERED
13,500 term vocabulary (matching the e2e notebook's term contract).

PURPOSE:
    - Compare performance between full vocabulary (~26K) and filtered (13.5K)
    - Enable ensembling with other e2e models (LogReg, GBDT, DNN)
    - Generate predictions compatible with the stacking pipeline

SUPER-KNN ENSEMBLE:
    BP (Biological Process): K=5  at threshold 0.40
    MF (Molecular Function): K=10 at threshold 0.40
    CC (Cellular Component): K=15 at threshold 0.30

USAGE:
    python knn_esm2_3b_13500.py --work-dir /path/to/cafa6_data

Author: Based on knn_esm2_3b.py, modified for 13,500 term vocabulary
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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# CONFIGURATION - SUPER-KNN ENSEMBLE WINNERS
# =============================================================================
ASPECT_K = {'BP': 5, 'MF': 10, 'CC': 15}
K_MAX = max(ASPECT_K.values())  # 15

OPTIMAL_THRESHOLDS = {
    'BP': 0.40,
    'MF': 0.40,
    'CC': 0.30,
    'ALL': 0.30
}

DEFAULT_N_FOLDS = 5
DEFAULT_BATCH_SIZE = 250  # Memory-safe default


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
    """
    Load the 13,500 term vocabulary from top_terms_13500.json.
    This matches the e2e notebook's term contract.
    """
    vocab_path = work_dir / 'features' / 'top_terms_13500.json'
    
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Missing top_terms_13500.json at {vocab_path}\n"
            "This file is created by the e2e notebook (Cell 13a).\n"
            "Run the notebook first or use knn_esm2_3b.py for full vocabulary."
        )
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        top_terms = json.load(f)
    
    print(f"[VOCAB] Loaded {len(top_terms)} terms from top_terms_13500.json")
    return top_terms


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_knn_13500(
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    Y_train: sparse.csr_matrix,
    aspect_masks: dict,
    work_dir: Path,
    n_folds: int = 5,
    batch_size: int = 250,
    skip_test: bool = False
) -> tuple:
    """
    Train the Super-KNN Ensemble on 13,500 filtered terms.
    
    Uses aspect-specific K values: BP=5, MF=10, CC=15
    Uses memory-mapped files for large arrays to avoid OOM.
    """
    print(f"\n{'='*70}")
    print("[KNN TRAINING] Super-KNN Ensemble (13,500 Terms)")
    print(f"{'='*70}")
    print(f"  Aspect K values: BP={ASPECT_K['BP']}, MF={ASPECT_K['MF']}, CC={ASPECT_K['CC']}")
    print(f"  Max K for lookup: {K_MAX}")
    print(f"  Folds: {n_folds}")
    
    n_train, n_terms = Y_train.shape
    n_test = test_emb.shape[0] if test_emb is not None else 0
    
    # Convert to float16 for memory efficiency
    train_emb = train_emb.astype(np.float16)
    if test_emb is not None:
        test_emb = test_emb.astype(np.float16)
    
    # Use memory-mapped files for large arrays to avoid OOM
    pred_dir = work_dir / 'features' / 'level1_preds'
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    oof_path = pred_dir / 'oof_pred_knn.npy'
    test_path = pred_dir / 'test_pred_knn.npy'
    
    print(f"  Using memory-mapped arrays to avoid OOM...")
    print(f"  OOF: {oof_path}")
    
    # Create memory-mapped arrays
    oof_preds = np.lib.format.open_memmap(
        str(oof_path), mode='w+', dtype=np.float32, shape=(n_train, n_terms)
    )
    
    if not skip_test:
        print(f"  Test: {test_path}")
        test_preds = np.lib.format.open_memmap(
            str(test_path), mode='w+', dtype=np.float32, shape=(n_test, n_terms)
        )
    else:
        test_preds = None
    
    # Stratification target
    stratify_target = Y_train.sum(axis=1).A1
    stratify_target = np.minimum(stratify_target, 10).astype(int)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    total_start = time.time()
    
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(train_emb, stratify_target), 1):
        fold_start = time.time()
        print(f"\n  Fold {fold_num}/{n_folds}")
        print(f"    Train: {len(train_idx):,} | Val: {len(val_idx):,}")
        
        X_train_fold = train_emb[train_idx].astype(np.float32)
        X_val_fold = train_emb[val_idx].astype(np.float32)
        Y_train_fold = Y_train[train_idx]
        
        # Build KNN index
        knn = NearestNeighbors(n_neighbors=K_MAX, metric='cosine', n_jobs=2)
        knn.fit(X_train_fold)
        
        # Query validation set
        distances, indices = knn.kneighbors(X_val_fold)
        similarities = 1 - distances
        
        # Predict on validation set with Mixed-K
        print("    Predicting (Mixed-K)...")
        for batch_start in range(0, len(val_idx), batch_size):
            batch_end = min(batch_start + batch_size, len(val_idx))
            batch_size_actual = batch_end - batch_start
            
            batch_indices = indices[batch_start:batch_end]
            batch_sims = similarities[batch_start:batch_end]
            
            flat_indices = batch_indices.flatten()
            neighbor_labels = Y_train_fold[flat_indices].toarray()
            neighbor_labels = neighbor_labels.reshape(batch_size_actual, K_MAX, -1)
            
            combined_scores = np.zeros((batch_size_actual, n_terms), dtype=np.float32)
            
            for aspect, k_val in ASPECT_K.items():
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
        
        # Predict on test set
        if not skip_test:
            print("    Predicting test set...")
            test_distances, test_indices = knn.kneighbors(test_emb.astype(np.float32))
            test_similarities = 1 - test_distances
            
            for batch_start in range(0, n_test, batch_size):
                batch_end = min(batch_start + batch_size, n_test)
                batch_size_actual = batch_end - batch_start
                
                batch_indices = test_indices[batch_start:batch_end]
                batch_sims = test_similarities[batch_start:batch_end]
                
                flat_indices = batch_indices.flatten()
                neighbor_labels = Y_train_fold[flat_indices].toarray()
                neighbor_labels = neighbor_labels.reshape(batch_size_actual, K_MAX, -1)
                
                for aspect, k_val in ASPECT_K.items():
                    col_indices = aspect_masks.get(aspect, [])
                    if not col_indices:
                        continue
                    
                    a_sims = batch_sims[:, :k_val]
                    a_labels = neighbor_labels[:, :k_val, :][:, :, col_indices]
                    
                    sims_expanded = a_sims[:, :, np.newaxis]
                    a_scores = (sims_expanded * a_labels).sum(axis=1)
                    a_scores /= (a_sims.sum(axis=1, keepdims=True) + 1e-9)
                    
                    test_preds[batch_start:batch_end, col_indices] += a_scores
        
        fold_time = time.time() - fold_start
        print(f"    Fold time: {fold_time:.1f}s")
        
        # Cleanup
        del knn, X_train_fold
        gc.collect()
    
    # Average test predictions
    if not skip_test:
        test_preds /= n_folds
    
    # Per-protein max normalization
    print("\n  Applying per-protein max normalization...")
    row_max = oof_preds.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    oof_preds /= row_max
    
    if not skip_test:
        test_row_max = test_preds.max(axis=1, keepdims=True)
        test_row_max[test_row_max == 0] = 1.0
        test_preds /= test_row_max
    
    total_time = time.time() - total_start
    print(f"\n  Total training time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    
    return oof_preds, test_preds, OPTIMAL_THRESHOLDS


def evaluate_oof(oof_preds: np.ndarray, Y_train: sparse.csr_matrix, 
                 aspect_masks: dict, top_terms: list) -> dict:
    """Evaluate OOF predictions using CAFA-style per-aspect F1."""
    print(f"\n{'='*70}")
    print("[EVALUATION] Per-Aspect CAFA F1")
    print(f"{'='*70}")
    
    results = {}
    
    for aspect in ['BP', 'MF', 'CC']:
        col_indices = aspect_masks.get(aspect, [])
        if not col_indices:
            continue
        
        threshold = OPTIMAL_THRESHOLDS.get(aspect, 0.3)
        
        y_true = Y_train[:, col_indices].toarray()
        y_pred = (oof_preds[:, col_indices] >= threshold).astype(np.int8)
        
        # Calculate per-protein F1
        f1_sum = 0
        n_samples = y_true.shape[0]
        
        for i in range(n_samples):
            tp = ((y_pred[i] == 1) & (y_true[i] == 1)).sum()
            fp = ((y_pred[i] == 1) & (y_true[i] == 0)).sum()
            fn = ((y_pred[i] == 0) & (y_true[i] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_sum += f1
        
        f1_avg = f1_sum / n_samples
        results[aspect] = {'f1': f1_avg, 'threshold': threshold}
        print(f"  {aspect}: F1={f1_avg:.4f} (threshold={threshold})")
    
    # CAFA F1 = average of aspect F1s
    cafa_f1 = sum(r['f1'] for r in results.values()) / len(results)
    results['CAFA'] = cafa_f1
    print(f"\n  CAFA F1: {cafa_f1:.4f}")
    
    return results


def generate_submission(
    test_preds: np.ndarray,
    test_ids: list,
    top_terms: list,
    term_aspects: dict,
    output_path: Path,
    thresholds: dict = None
):
    """Generate CAFA-format submission file."""
    print(f"\n{'='*70}")
    print("[GENERATING SUBMISSION]")
    print(f"{'='*70}")
    
    if thresholds is None:
        thresholds = OPTIMAL_THRESHOLDS
    
    threshold_vec = np.array([
        thresholds.get(term_aspects.get(term, 'ALL'), thresholds.get('ALL', 0.3))
        for term in top_terms
    ], dtype=np.float32)
    
    above_threshold = test_preds >= threshold_vec
    
    rows = []
    for i, protein_id in enumerate(test_ids):
        term_indices = np.where(above_threshold[i])[0]
        if len(term_indices) == 0:
            continue
        
        scores = test_preds[i, term_indices]
        sorted_order = np.argsort(scores)[::-1]
        term_indices = term_indices[sorted_order][:1500]
        scores = scores[sorted_order][:1500]
        
        for term_idx, score in zip(term_indices, scores):
            score = min(max(score, 0.001), 1.0)
            rows.append({
                'protein_id': protein_id,
                'go_term': top_terms[term_idx],
                'score': f"{score:.3f}"
            })
    
    print(f"  Total predictions: {len(rows):,}")
    
    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(output_path, sep='\t', header=False, index=False)
    print(f"  Saved: {output_path}")
    
    if len(rows) > 0:
        print(f"  Proteins with predictions: {submission_df['protein_id'].nunique():,}")
        print(f"  Unique GO terms: {submission_df['go_term'].nunique():,}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CAFA-6 Super-KNN Ensemble (13,500 Terms)')
    parser.add_argument('--work-dir', type=str, required=True)
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip test predictions (validation only)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    
    print("="*70)
    print("CAFA-6 Super-KNN Ensemble (13,500 Term Vocabulary)")
    print("="*70)
    print(f"Work directory: {work_dir}")
    
    # -------------------------------------------------------------------------
    # Load 13,500 term vocabulary (from e2e notebook)
    # -------------------------------------------------------------------------
    top_terms = load_13500_vocabulary(work_dir)
    term_to_idx = {t: i for i, t in enumerate(top_terms)}
    n_terms = len(top_terms)
    
    # -------------------------------------------------------------------------
    # Load embeddings
    # -------------------------------------------------------------------------
    print("\n[DATA LOADING]")
    feat_dir = work_dir / 'features'
    
    train_emb = np.load(feat_dir / 'train_embeds_esm2_3b.npy')
    test_emb = np.load(feat_dir / 'test_embeds_esm2_3b.npy') if not args.skip_test else None
    print(f"  Train: {train_emb.shape}, Test: {test_emb.shape if test_emb is not None else 'skipped'}")
    
    # -------------------------------------------------------------------------
    # Load protein IDs from train_seq.feather (CRITICAL for alignment!)
    # -------------------------------------------------------------------------
    train_seq = pd.read_feather(work_dir / 'parsed' / 'train_seq.feather')
    
    # Detect ID column
    id_col = 'id' if 'id' in train_seq.columns else 'EntryID'
    train_ids = [extract_uniprot_accession(x) for x in train_seq[id_col].tolist()]
    
    if not args.skip_test:
        test_seq = pd.read_feather(work_dir / 'parsed' / 'test_seq.feather')
        test_ids = [extract_uniprot_accession(x) for x in test_seq[id_col].tolist()]
    else:
        test_ids = []
    
    del train_seq
    gc.collect()
    
    # -------------------------------------------------------------------------
    # Load annotations and build label matrix
    # -------------------------------------------------------------------------
    train_terms_df = pd.read_csv(work_dir / 'Train' / 'train_terms.tsv', sep='\t')
    
    # Create protein -> row index mapping
    protein_to_idx = {pid: i for i, pid in enumerate(train_ids)}
    
    # Build sparse label matrix (only for terms in top_terms!)
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
    print(f"  Label matrix: {Y_train.shape}, nnz: {Y_train.nnz:,}")
    
    del rows, cols, train_terms_df
    gc.collect()
    
    # -------------------------------------------------------------------------
    # Load GO aspects
    # -------------------------------------------------------------------------
    term_aspects = parse_go_aspects(work_dir / 'Train' / 'go-basic.obo')
    
    aspect_masks = {'BP': [], 'MF': [], 'CC': []}
    for i, term in enumerate(top_terms):
        aspect = term_aspects.get(term, 'UNK')
        if aspect in aspect_masks:
            aspect_masks[aspect].append(i)
    
    print(f"  BP: {len(aspect_masks['BP'])}, MF: {len(aspect_masks['MF'])}, CC: {len(aspect_masks['CC'])}")
    
    # -------------------------------------------------------------------------
    # Train Super-KNN
    # -------------------------------------------------------------------------
    oof_preds, test_preds, thresholds = train_knn_13500(
        train_emb, test_emb, Y_train, aspect_masks,
        work_dir=work_dir,
        n_folds=DEFAULT_N_FOLDS,
        batch_size=args.batch_size,
        skip_test=args.skip_test
    )
    
    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    results = evaluate_oof(oof_preds, Y_train, aspect_masks, top_terms)
    
    # -------------------------------------------------------------------------
    # Finalize predictions (already saved via mmap, just flush)
    # -------------------------------------------------------------------------
    pred_dir = work_dir / 'features' / 'level1_preds'
    
    # Flush mmap to disk
    if hasattr(oof_preds, 'flush'):
        oof_preds.flush()
    print(f"\nSaved: {pred_dir / 'oof_pred_knn.npy'}")
    
    if not args.skip_test and test_preds is not None:
        if hasattr(test_preds, 'flush'):
            test_preds.flush()
        print(f"Saved: {pred_dir / 'test_pred_knn.npy'}")
        
        # Generate submission
        generate_submission(
            test_preds, test_ids, top_terms, term_aspects,
            work_dir / 'submission_knn_13500.tsv', thresholds
        )
    
    print("\n[COMPLETE]")
    print(f"  CAFA F1: {results['CAFA']:.4f}")


if __name__ == '__main__':
    main()
