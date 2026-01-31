#!/usr/bin/env python3
"""
================================================================================
CAFA-6 Super-KNN Ensemble - GPU-Accelerated Version (cuML/RAPIDS)
================================================================================

This script implements the "Super-KNN" Mixed-K Ensemble using NVIDIA RAPIDS cuML
for GPU-accelerated nearest neighbor lookups.

SUPER-KNN ENSEMBLE:
    BP (Biological Process): K=5  at threshold 0.40
    MF (Molecular Function): K=10 at threshold 0.40
    CC (Cellular Component): K=15 at threshold 0.30

PERFORMANCE:
    Record CAFA F1: 0.2456 (validated on 5-fold CV)

REQUIREMENTS:
    - CUDA-capable GPU
    - RAPIDS cuML (pip install cuml-cu11 or conda install cuml)
    - Pre-computed ESM2-3B embeddings

USAGE:
    python knn_esm2_3b_gpu.py --work-dir /path/to/cafa6_data

Author: Optimized for GPU from CPU-validated knn_esm2_3b.py
================================================================================
"""

import argparse
import time
import json
import gc
import re
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# GPU backend - REQUIRED (no fallback)
try:
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    print("[GPU] cuML/RAPIDS detected - using GPU acceleration")
except ImportError as e:
    raise RuntimeError(
        "ERROR: cuML/RAPIDS not found. This script requires GPU acceleration.\n"
        "Install with: conda install -c rapidsai cuml\n"
        f"Original error: {e}"
    ) from e

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

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
DEFAULT_BATCH_SIZE = 1000  # Larger batches for GPU


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


# =============================================================================
# GPU-ACCELERATED KNN TRAINING
# =============================================================================

def train_knn_gpu(
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    Y_train: sparse.csr_matrix,
    aspect_masks: dict,
    n_folds: int = 5,
    batch_size: int = 1000,
    skip_test: bool = False
) -> tuple:
    """
    Train the Super-KNN Ensemble using GPU acceleration.
    
    Uses cuML NearestNeighbors for GPU-accelerated KNN lookups.
    """
    print(f"\n{'='*70}")
    print("[KNN TRAINING] Super-KNN Ensemble (GPU Mode)")
    print(f"{'='*70}")
    print(f"  Aspect K values: BP={ASPECT_K['BP']}, MF={ASPECT_K['MF']}, CC={ASPECT_K['CC']}")
    print(f"  Max K for lookup: {K_MAX}")
    print(f"  Folds: {n_folds}")
    
    n_train, n_terms = Y_train.shape
    n_test = test_emb.shape[0] if test_emb is not None else 0
    
    # Allocate prediction matrices
    oof_preds = np.zeros((n_train, n_terms), dtype=np.float32)
    test_preds = np.zeros((n_test, n_terms), dtype=np.float32) if not skip_test else None
    
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
        
        # Build KNN index (GPU)
        knn = cuNearestNeighbors(n_neighbors=K_MAX, metric='cosine')
        knn.fit(cp.asarray(X_train_fold))
        
        # Query validation set
        distances, indices = knn.kneighbors(cp.asarray(X_val_fold))
        distances = cp.asnumpy(distances)
        indices = cp.asnumpy(indices)
        
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
                col_indices = aspect_masks[aspect]
                if not col_indices: continue
                
                a_sims = batch_sims[:, :k_val]
                a_labels = neighbor_labels[:, :k_val, col_indices]
                
                sims_expanded = a_sims[:, :, np.newaxis]
                a_scores = (sims_expanded * a_labels).sum(axis=1)
                a_scores /= (a_sims.sum(axis=1, keepdims=True) + 1e-9)
                
                combined_scores[:, col_indices] = a_scores
            
            oof_preds[val_idx[batch_start:batch_end]] = combined_scores
        
        # Predict on test set
        if not skip_test:
            print("    Predicting test set...")
            test_distances, test_indices = knn.kneighbors(cp.asarray(test_emb.astype(np.float32)))
            test_distances = cp.asnumpy(test_distances)
            test_indices = cp.asnumpy(test_indices)
            
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
                    col_indices = aspect_masks[aspect]
                    if not col_indices: continue
                    
                    a_sims = batch_sims[:, :k_val]
                    a_labels = neighbor_labels[:, :k_val, col_indices]
                    
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


# =============================================================================
# SUBMISSION GENERATION
# =============================================================================

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
# KAGGLE SUBMISSION
# =============================================================================

def _load_kaggle_credentials(work_dir: Path) -> bool:
    """Load Kaggle credentials from .env file or environment variables."""
    # Try to load from .env file (check work_dir and project root)
    env_paths = [
        work_dir / '.env',
        work_dir.parent / '.env',
        Path(__file__).parent.parent / '.env',  # Project root
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            print(f'[KAGGLE] Loading credentials from {env_path}...')
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key in ['KAGGLE_USERNAME', 'KAGGLE_KEY'] and value:
                                os.environ[key] = value
                break
            except Exception as e:
                print(f'  [WARN] Failed to load .env: {e}')
    
    # Check if credentials are set
    username = os.environ.get('KAGGLE_USERNAME', '').strip()
    key = os.environ.get('KAGGLE_KEY', '').strip()
    
    if username and key:
        print(f'  ✓ Kaggle credentials loaded: {username}')
        return True
    else:
        print('  ✗ Kaggle credentials not found in environment or .env')
        return False


def submit_to_kaggle(submission_path: Path, work_dir: Path) -> bool:
    """Submit the generated file to Kaggle competition."""
    print(f"\n{'='*70}")
    print("[KAGGLE SUBMISSION]")
    print(f"{'='*70}")
    
    COMPETITION_NAME = 'cafa-6-protein-function-prediction'
    SUBMISSION_MESSAGE = 'Super-KNN Ensemble (ESM2-3B, Mixed-K, GPU)'
    
    # Check if submission file exists
    if not submission_path.exists():
        print(f'  ✗ Submission file not found: {submission_path}')
        return False
    
    print(f'  Submission file: {submission_path}')
    
    # Load credentials
    if not _load_kaggle_credentials(work_dir):
        print('\n  ERROR: Kaggle credentials not configured')
        print('  Please set KAGGLE_USERNAME and KAGGLE_KEY in .env file')
        print('  Get your API key from: https://www.kaggle.com/settings')
        return False
    
    # Check if Kaggle CLI is installed
    try:
        result = subprocess.run(
            ['kaggle', '--version'],
            capture_output=True,
            text=True
        )
        print(f'  ✓ Kaggle CLI version: {result.stdout.strip()}')
    except FileNotFoundError:
        print('\n  ERROR: Kaggle CLI not installed')
        print('  Install with: pip install kaggle')
        return False
    
    # Build and execute the kaggle command
    print(f'\n  Submitting to {COMPETITION_NAME}...')
    print(f'  Message: "{SUBMISSION_MESSAGE}"')
    
    cmd = [
        'kaggle',
        'competitions',
        'submit',
        '-c', COMPETITION_NAME,
        '-f', str(submission_path),
        '-m', SUBMISSION_MESSAGE
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        print('\n' + '='*60)
        print('✅ SUBMISSION SUCCESSFUL!')
        print('='*60)
        if result.stdout:
            print('\nKaggle output:')
            print(result.stdout)
        if result.stderr:
            print('\nAdditional info:')
            print(result.stderr)
        print('\nCheck your submission at:')
        print(f'https://www.kaggle.com/competitions/{COMPETITION_NAME}/submissions')
        print('='*60)
        return True
    except subprocess.CalledProcessError as e:
        print('\n' + '='*60)
        print('❌ SUBMISSION FAILED')
        print('='*60)
        if e.stderr:
            print('\nError output:')
            print(e.stderr)
        if e.stdout:
            print('\nStdout:')
            print(e.stdout)
        print('\nCommon issues:')
        print('  - Competition rules not accepted')
        print('  - Invalid submission format')
        print('  - Daily submission limit reached')
        print('  - Incorrect credentials')
        print('='*60)
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CAFA-6 Super-KNN Ensemble (GPU)')
    parser.add_argument('--work-dir', type=str, required=True)
    parser.add_argument('--skip-test', action='store_true')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    
    print("="*70)
    print("CAFA-6 Super-KNN Ensemble (GPU-Accelerated)")
    print("="*70)
    print(f"Work directory: {work_dir}")
    
    # Load data
    print("\n[DATA LOADING]")
    feat_dir = work_dir / 'features'
    
    train_emb = np.load(feat_dir / 'train_embeds_esm2_3b.npy')
    test_emb = np.load(feat_dir / 'test_embeds_esm2_3b.npy') if not args.skip_test else None
    print(f"  Train: {train_emb.shape}, Test: {test_emb.shape if test_emb is not None else 'skipped'}")
    
    # Load protein IDs
    train_seq = pd.read_feather(work_dir / 'parsed' / 'train_seq.feather')
    train_ids = [extract_uniprot_accession(x) for x in train_seq['EntryID'].tolist()]
    
    if not args.skip_test:
        test_seq = pd.read_feather(work_dir / 'parsed' / 'test_seq.feather')
        test_ids = [extract_uniprot_accession(x) for x in test_seq['EntryID'].tolist()]
    else:
        test_ids = []
    
    # Load annotations
    train_terms = pd.read_csv(work_dir / 'Train' / 'train_terms.tsv', sep='\t')
    top_terms = train_terms['term'].value_counts().index.tolist()
    term_to_idx = {t: i for i, t in enumerate(top_terms)}
    protein_to_idx = {pid: i for i, pid in enumerate(train_ids)}
    
    # Build label matrix
    rows, cols = [], []
    for _, row in train_terms.iterrows():
        pid_clean = extract_uniprot_accession(row['EntryID'])
        if pid_clean in protein_to_idx and row['term'] in term_to_idx:
            rows.append(protein_to_idx[pid_clean])
            cols.append(term_to_idx[row['term']])
    
    Y_train = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(train_ids), len(top_terms))
    )
    print(f"  Label matrix: {Y_train.shape}, nnz: {Y_train.nnz:,}")
    
    # Load GO aspects
    term_aspects = parse_go_aspects(work_dir / 'Train' / 'go-basic.obo')
    
    aspect_masks = {'BP': [], 'MF': [], 'CC': []}
    for i, term in enumerate(top_terms):
        aspect = term_aspects.get(term, 'UNK')
        if aspect in aspect_masks:
            aspect_masks[aspect].append(i)
    
    print(f"  BP: {len(aspect_masks['BP'])}, MF: {len(aspect_masks['MF'])}, CC: {len(aspect_masks['CC'])}")
    
    # Train
    oof_preds, test_preds, thresholds = train_knn_gpu(
        train_emb, test_emb, Y_train, aspect_masks,
        n_folds=DEFAULT_N_FOLDS,
        batch_size=args.batch_size,
        skip_test=args.skip_test
    )
    
    # Save predictions
    np.save(work_dir / 'oof_knn_superknn.npy', oof_preds)
    if not args.skip_test:
        np.save(work_dir / 'test_knn_superknn.npy', test_preds)
        submission_path = work_dir / 'submission_superknn.tsv'
        generate_submission(test_preds, test_ids, top_terms, term_aspects,
                          submission_path, thresholds)
        
        # Auto-submit to Kaggle
        submit_to_kaggle(submission_path, work_dir)
    
    print("\n[COMPLETE]")


if __name__ == '__main__':
    main()
