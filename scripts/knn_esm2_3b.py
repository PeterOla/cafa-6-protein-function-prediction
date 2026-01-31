#!/usr/bin/env python3
"""
================================================================================
CAFA-6 Protein Function Prediction - KNN Baseline Model
================================================================================

This script implements a K-Nearest Neighbors (KNN) model for predicting
protein Gene Ontology (GO) term annotations using ESM2-3B embeddings.

CRITICAL FIX APPLIED:
    The original notebook had a PROTEIN ORDERING BUG that caused embeddings
    to be matched to WRONG protein labels, resulting in a 0.083 score.
    
    The fix: Load protein IDs from train_seq.feather (same source as embeddings)
    instead of from train_terms.tsv (which has different ordering).

HOW IT WORKS:
    1. Load pre-computed ESM2-3B protein embeddings (2560-dimensional vectors)
    2. Build a label matrix Y where Y[i,j] = 1 if protein i has GO term j
    3. For each test protein, find k nearest training proteins (by cosine similarity)
    4. Aggregate neighbor labels weighted by similarity to predict GO terms
    5. Apply per-protein max normalization to scale scores to [0, 1]
    6. Threshold predictions and format for CAFA submission

USAGE:
    python knn_esm2_3b.py --work-dir /path/to/cafa6_data --k 10

EXPECTED OUTPUTS:
    - oof_knn.npy:      Out-of-fold predictions for validation
    - test_knn.npy:     Test set predictions  
    - submission.tsv:   CAFA-formatted submission file

Author: Converted from knn_standalone.ipynb with critical bug fix
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, precision_score, recall_score


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default hyperparameters (can be overridden via command line)
DEFAULT_K = 10                    # Number of nearest neighbors
DEFAULT_N_FOLDS = 5               # Number of cross-validation folds
DEFAULT_BATCH_SIZE = 500          # Batch size for memory-safe processing

# Thresholds for converting probabilities to binary predictions
# These are aspect-specific because different GO aspects have different
# prediction characteristics (MF is easier than BP, CC is in between)
DEFAULT_THRESHOLDS = {
    'BP': 0.25,   # Biological Process - hardest to predict, lower threshold
    'MF': 0.50,   # Molecular Function - easier, higher threshold  
    'CC': 0.35,   # Cellular Component - medium difficulty
    'ALL': 0.30   # Fallback for unknown aspects
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_uniprot_accession(protein_id: str) -> str:
    """
    Extract UniProt accession from FASTA-style protein ID.
    
    Examples:
        "sp|P12345|HUMAN" -> "P12345"  (Swiss-Prot entry)
        "tr|A0A123|MOUSE" -> "A0A123"  (TrEMBL entry)
        "P12345"          -> "P12345"  (already clean)
    
    The embeddings are stored indexed by these clean accession numbers,
    so we need consistent ID extraction across all data sources.
    """
    protein_id = str(protein_id)
    if '|' in protein_id:
        # FASTA format: prefix|accession|name
        match = re.search(r'\|(.+?)\|', protein_id)
        return match.group(1) if match else protein_id
    return protein_id


def parse_go_aspects(obo_path: Path) -> dict:
    """
    Parse GO ontology file to extract aspect (namespace) for each GO term.
    
    GO has 3 aspects/namespaces:
        - BP (biological_process): e.g., cell division, metabolism
        - MF (molecular_function): e.g., enzyme activity, binding
        - CC (cellular_component): e.g., nucleus, membrane
    
    This is needed for:
        1. Aspect-specific thresholding
        2. Per-aspect evaluation (CAFA metric averages F1 across aspects)
    
    Returns:
        dict mapping GO term ID (e.g., "GO:0003674") to aspect ("BP"/"MF"/"CC")
    """
    aspect_map = {
        'biological_process': 'BP',
        'molecular_function': 'MF',
        'cellular_component': 'CC'
    }
    
    go_aspects = {}
    current_term = None
    
    with open(obo_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('id: GO:'):
                # Start of new term definition
                current_term = line[4:]  # e.g., "GO:0003674"
            elif line.startswith('namespace:') and current_term:
                # Found the aspect for current term
                namespace = line.split(': ')[1]
                go_aspects[current_term] = aspect_map.get(namespace, 'UNK')
    
    return go_aspects


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """
    L2-normalize rows of X to unit vectors.
    
    Why do this?
        After L2 normalization, cosine similarity = dot product.
        This enables faster computation, especially on GPU.
        
        cos(a, b) = (a · b) / (||a|| * ||b||)
        
        If ||a|| = ||b|| = 1, then cos(a, b) = a · b
    
    Args:
        X: (n_samples, n_features) array
        
    Returns:
        Normalized X where each row has unit length
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    return X / norms


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(work_dir: Path) -> dict:
    """
    Load all required data for KNN training.
    
    CRITICAL: This function implements the FIX for the protein ordering bug.
    
    The embeddings are ordered by train_seq.feather (the sequence file).
    The labels come from train_terms.tsv (the annotation file).
    These have DIFFERENT protein orderings!
    
    The fix: Load protein IDs from train_seq.feather and build the label
    matrix in that same order, ensuring alignment with embeddings.
    
    Returns dict with:
        - train_emb: (n_train, 2560) training protein embeddings
        - test_emb: (n_test, 2560) test protein embeddings
        - Y_train: (n_train, n_terms) sparse label matrix
        - protein_ids: list of training protein IDs (in embedding order!)
        - test_ids: list of test protein IDs (in embedding order!)
        - top_terms: list of GO term IDs
        - term_aspects: dict mapping term -> aspect
    """
    print("\n" + "="*70)
    print("[DATA LOADING]")
    print("="*70)
    
    feat_dir = work_dir / 'features'
    parsed_dir = work_dir / 'parsed'
    train_dir = work_dir / 'Train'
    
    # -------------------------------------------------------------------------
    # 1. Load embeddings (these are ordered by train_seq.feather)
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading ESM2-3B embeddings...")
    train_emb = np.load(feat_dir / 'train_embeds_esm2_3b.npy')
    test_emb = np.load(feat_dir / 'test_embeds_esm2_3b.npy')
    print(f"      Train embeddings: {train_emb.shape}")
    print(f"      Test embeddings:  {test_emb.shape}")
    print(f"      Embedding dim:    {train_emb.shape[1]} (ESM2-3B)")
    
    # -------------------------------------------------------------------------
    # 2. Load protein IDs from SAME source as embeddings
    #    THIS IS THE CRITICAL FIX!
    # -------------------------------------------------------------------------
    print("\n[2/6] Loading protein IDs from train_seq.feather...")
    print("      (CRITICAL: This ensures alignment with embeddings)")
    
    train_seq_df = pd.read_feather(parsed_dir / 'train_seq.feather')
    test_seq_df = pd.read_feather(parsed_dir / 'test_seq.feather')
    
    # Extract UniProt accessions in same order as embeddings
    protein_ids = [extract_uniprot_accession(pid) for pid in train_seq_df['id'].tolist()]
    test_ids = [extract_uniprot_accession(pid) for pid in test_seq_df['id'].tolist()]
    
    print(f"      Training proteins: {len(protein_ids)}")
    print(f"      Test proteins:     {len(test_ids)}")
    
    # Verify counts match embeddings (sanity check)
    assert len(protein_ids) == train_emb.shape[0], \
        f"MISMATCH: {len(protein_ids)} proteins but {train_emb.shape[0]} embeddings!"
    assert len(test_ids) == test_emb.shape[0], \
        f"MISMATCH: {len(test_ids)} test proteins but {test_emb.shape[0]} embeddings!"
    print("      ✓ Protein counts match embedding counts")
    
    # -------------------------------------------------------------------------
    # 3. Load training annotations (GO terms for each protein)
    # -------------------------------------------------------------------------
    print("\n[3/6] Loading training annotations...")
    train_terms_df = pd.read_csv(train_dir / 'train_terms.tsv', sep='\t')
    print(f"      Total annotations: {len(train_terms_df)}")
    print(f"      Unique proteins:   {train_terms_df['EntryID'].nunique()}")
    print(f"      Unique GO terms:   {train_terms_df['term'].nunique()}")
    
    # -------------------------------------------------------------------------
    # 4. Build term vocabulary (all unique GO terms in training data)
    #    Using FULL vocabulary - no filtering to top N terms
    # -------------------------------------------------------------------------
    print("\n[4/6] Building term vocabulary...")
    
    # Sort by frequency (most common first) for better sparse matrix layout
    top_terms = train_terms_df["term"].value_counts().index.tolist()
    term_to_idx = {term: i for i, term in enumerate(top_terms)}
    
    print(f"      Vocabulary size: {len(top_terms)} terms (FULL vocabulary)")
    print(f"      [BASELINE MODE] No term filtering - this preserves recall")
    
    # -------------------------------------------------------------------------
    # 5. Build label matrix Y with CORRECT protein ordering
    #    Row order must match embedding order (from train_seq.feather)
    # -------------------------------------------------------------------------
    print("\n[5/6] Building label matrix (embedding-aligned protein order)...")
    
    n_proteins = len(protein_ids)
    n_terms = len(top_terms)
    
    # Create protein -> row index mapping (using train_seq.feather order!)
    protein_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    
    # Build sparse matrix coordinate lists
    rows = []
    cols = []
    skipped = 0
    
    for _, row in train_terms_df.iterrows():
        protein_idx = protein_to_idx.get(row['EntryID'])
        if protein_idx is None:
            # This protein is in annotations but not in sequences - skip
            skipped += 1
            continue
        
        term_idx = term_to_idx.get(row['term'])
        if term_idx is not None:
            rows.append(protein_idx)
            cols.append(term_idx)
    
    if skipped > 0:
        print(f"      WARNING: Skipped {skipped} annotations (protein not in train_seq.feather)")
    
    # Create sparse CSR matrix (efficient for row slicing in KNN)
    Y_train = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_proteins, n_terms)
    )
    
    print(f"      Label matrix shape: {Y_train.shape}")
    print(f"      Non-zero entries:   {Y_train.nnz}")
    print(f"      Density:            {Y_train.nnz / (n_proteins * n_terms) * 100:.4f}%")
    
    # -------------------------------------------------------------------------
    # 6. Load GO aspects for each term
    # -------------------------------------------------------------------------
    print("\n[6/6] Loading GO term aspects...")
    
    obo_path = train_dir / 'go-basic.obo'
    if obo_path.exists():
        go_aspects = parse_go_aspects(obo_path)
        term_aspects = {term: go_aspects.get(term, 'UNK') for term in top_terms}
        
        # Count aspects
        aspect_counts = {'BP': 0, 'MF': 0, 'CC': 0, 'UNK': 0}
        for aspect in term_aspects.values():
            aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
        
        print(f"      Loaded aspects for {len(go_aspects)} GO terms")
        print(f"      BP (Biological Process): {aspect_counts['BP']}")
        print(f"      MF (Molecular Function): {aspect_counts['MF']}")
        print(f"      CC (Cellular Component): {aspect_counts['CC']}")
        if aspect_counts.get('UNK', 0) > 0:
            print(f"      UNK (Unknown):           {aspect_counts['UNK']}")
    else:
        print(f"      WARNING: {obo_path} not found - using 'ALL' for all terms")
        term_aspects = {term: 'ALL' for term in top_terms}
    
    print("\n" + "="*70)
    print("[DATA LOADING COMPLETE]")
    print("="*70)
    
    return {
        'train_emb': train_emb,
        'test_emb': test_emb,
        'Y_train': Y_train,
        'protein_ids': protein_ids,
        'test_ids': test_ids,
        'top_terms': top_terms,
        'term_aspects': term_aspects
    }


# =============================================================================
# KNN TRAINING
# =============================================================================

def train_knn(
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    Y_train: sparse.csr_matrix,
    k: int = 10,
    n_folds: int = 5,
    batch_size: int = 500
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Train KNN model with cross-validation.
    
    The KNN approach for protein function prediction:
    
    1. For each query protein, find k most similar training proteins
       (using cosine similarity on ESM2 embeddings)
    
    2. Aggregate the GO annotations of those k neighbors:
       score(term) = Σ(similarity[i] * has_term[i]) / Σ(similarity[i])
       
       This gives a weighted vote - closer neighbors have more influence.
    
    3. Normalize scores per-protein to [0, 1] range:
       normalized = score / max(all_scores_for_this_protein)
       
       This allows comparable thresholds across proteins.
    
    Args:
        train_emb: (n_train, dim) training protein embeddings
        test_emb: (n_test, dim) test protein embeddings
        Y_train: (n_train, n_terms) sparse label matrix
        k: number of nearest neighbors
        n_folds: number of cross-validation folds
        batch_size: batch size for memory-efficient processing
    
    Returns:
        oof_preds: (n_train, n_terms) out-of-fold predictions
        test_preds: (n_test, n_terms) test predictions
        best_threshold: optimal threshold from validation
    """
    print("\n" + "="*70)
    print(f"[KNN TRAINING] k={k}, folds={n_folds}")
    print("="*70)
    
    n_train = len(train_emb)
    n_test = len(test_emb)
    n_terms = Y_train.shape[1]
    
    # Initialize prediction arrays
    oof_preds = np.zeros((n_train, n_terms), dtype=np.float32)
    test_preds = np.zeros((n_test, n_terms), dtype=np.float32)
    
    # Stratification target: number of labels per protein (capped at 10)
    # This helps ensure each fold has similar label distributions
    stratify_target = Y_train.sum(axis=1).A1  # .A1 converts to 1D array
    stratify_target = np.minimum(stratify_target, 10).astype(int)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(train_emb, stratify_target), 1):
        print(f"\n  Fold {fold_num}/{n_folds}")
        print(f"    Train: {len(train_idx):,} proteins")
        print(f"    Val:   {len(val_idx):,} proteins")
        
        # Split embeddings and labels
        X_train_fold = train_emb[train_idx]
        X_val_fold = train_emb[val_idx]
        Y_train_fold = Y_train[train_idx]
        
        # Build KNN index
        # Using cosine metric - appropriate for comparing embedding vectors
        print(f"    Building KNN index (k={k}, metric=cosine)...")
        knn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
        knn.fit(X_train_fold)
        
        # ---------------------------------------------------------------------
        # Predict on validation fold (for OOF scores)
        # ---------------------------------------------------------------------
        print("    Predicting on validation set...")
        
        # Get k nearest neighbors for each validation protein
        distances, indices = knn.kneighbors(X_val_fold)
        
        # Convert distances to similarities
        # cosine distance = 1 - cosine_similarity
        similarities = 1 - distances
        
        # Process in batches to avoid memory issues
        for batch_start in range(0, len(val_idx), batch_size):
            batch_end = min(batch_start + batch_size, len(val_idx))
            batch_size_actual = batch_end - batch_start
            
            batch_indices = indices[batch_start:batch_end]  # (batch, k)
            batch_sims = similarities[batch_start:batch_end]  # (batch, k)
            
            # Fetch neighbor labels
            # Need to flatten, index, then reshape because sparse matrices
            # don't support 2D fancy indexing
            flat_indices = batch_indices.flatten()
            neighbor_labels = Y_train_fold[flat_indices].toarray()
            neighbor_labels = neighbor_labels.reshape(batch_size_actual, k, -1)
            # Now neighbor_labels is (batch, k, n_terms)
            
            # Weighted aggregation: score = Σ(sim * label) / Σ(sim)
            # This gives similarity-weighted voting
            sims_expanded = batch_sims[:, :, np.newaxis]  # (batch, k, 1)
            raw_scores = (sims_expanded * neighbor_labels).sum(axis=1)  # (batch, n_terms)
            raw_scores /= batch_sims.sum(axis=1, keepdims=True)
            
            oof_preds[val_idx[batch_start:batch_end]] = raw_scores.astype(np.float32)
        
        # ---------------------------------------------------------------------
        # Predict on test set (accumulate across folds, will average later)
        # ---------------------------------------------------------------------
        print("    Predicting on test set...")
        
        test_distances, test_indices = knn.kneighbors(test_emb)
        test_similarities = 1 - test_distances
        
        for batch_start in range(0, n_test, batch_size):
            batch_end = min(batch_start + batch_size, n_test)
            batch_size_actual = batch_end - batch_start
            
            batch_indices = test_indices[batch_start:batch_end]
            batch_sims = test_similarities[batch_start:batch_end]
            
            flat_indices = batch_indices.flatten()
            neighbor_labels = Y_train_fold[flat_indices].toarray()
            neighbor_labels = neighbor_labels.reshape(batch_size_actual, k, -1)
            
            sims_expanded = batch_sims[:, :, np.newaxis]
            test_raw_scores = (sims_expanded * neighbor_labels).sum(axis=1)
            test_raw_scores /= batch_sims.sum(axis=1, keepdims=True)
            
            # Accumulate (will divide by n_folds later)
            test_preds[batch_start:batch_end] += test_raw_scores.astype(np.float32)
    
    # Average test predictions across folds
    test_preds /= n_folds
    
    # -------------------------------------------------------------------------
    # Per-protein max normalization
    # -------------------------------------------------------------------------
    # Normalize scores to [0, 1] range by dividing by max score per protein
    # This ensures:
    #   - Most confident prediction for each protein = 1.0
    #   - Threshold of 0.5 means "at least half as confident as top prediction"
    #   - Comparable thresholds across proteins
    print("\n  Applying per-protein max normalization...")
    
    # OOF predictions
    row_max = oof_preds.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0  # Avoid division by zero
    oof_preds /= row_max
    
    # Test predictions
    test_max = test_preds.max(axis=1, keepdims=True)
    test_max[test_max == 0] = 1.0
    test_preds /= test_max
    
    # -------------------------------------------------------------------------
    # Find optimal threshold on OOF predictions
    # -------------------------------------------------------------------------
    print("\n  Finding optimal threshold on OOF predictions...")
    
    Y_true = Y_train.toarray()
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_f1 = 0
    best_threshold = 0.3
    
    print("\n  Threshold   F1       Precision  Recall")
    print("  " + "-"*45)
    
    for threshold in thresholds:
        Y_pred = (oof_preds >= threshold).astype(int)
        f1 = f1_score(Y_true, Y_pred, average='samples', zero_division=0)
        precision = precision_score(Y_true, Y_pred, average='samples', zero_division=0)
        recall = recall_score(Y_true, Y_pred, average='samples', zero_division=0)
        
        marker = "  <-- BEST" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
        print(f"  {threshold:.2f}        {f1:.4f}   {precision:.4f}     {recall:.4f}{marker}")
    
    print(f"\n  Best threshold: {best_threshold:.2f} (F1 = {best_f1:.4f})")
    
    print("\n" + "="*70)
    print("[KNN TRAINING COMPLETE]")
    print("="*70)
    
    return oof_preds, test_preds, best_threshold


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
    """
    Generate CAFA-formatted submission file.
    
    CAFA submission format (TSV):
        Column 1: Protein ID (UniProt accession)
        Column 2: GO term ID 
        Column 3: Confidence score (0-1, max 3 decimal places)
    
    Rules:
        - Max 1500 predictions per protein
        - Scores must be in (0, 1] range (exclusive 0, inclusive 1)
        - 3 significant figures for scores
    
    Args:
        test_preds: (n_test, n_terms) prediction matrix
        test_ids: list of test protein IDs
        top_terms: list of GO term IDs (column order)
        term_aspects: dict mapping term -> aspect
        output_path: where to save submission.tsv
        thresholds: dict of aspect-specific thresholds
    """
    print("\n" + "="*70)
    print("[GENERATING SUBMISSION]")
    print("="*70)
    
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    
    print(f"  Using thresholds: BP={thresholds.get('BP', 0.3):.2f}, "
          f"MF={thresholds.get('MF', 0.3):.2f}, "
          f"CC={thresholds.get('CC', 0.3):.2f}")
    
    # Build threshold vector (one threshold per term based on its aspect)
    threshold_vec = np.array([
        thresholds.get(term_aspects.get(term, 'ALL'), thresholds.get('ALL', 0.3))
        for term in top_terms
    ], dtype=np.float32)
    
    # Apply thresholds
    above_threshold = test_preds >= threshold_vec
    
    # Generate submission rows
    rows = []
    
    for i, protein_id in enumerate(test_ids):
        # Get indices of terms above threshold
        term_indices = np.where(above_threshold[i])[0]
        
        if len(term_indices) == 0:
            continue
        
        # Get scores for these terms
        scores = test_preds[i, term_indices]
        
        # Sort by score descending
        sorted_order = np.argsort(scores)[::-1]
        term_indices = term_indices[sorted_order]
        scores = scores[sorted_order]
        
        # Limit to top 1500 per protein (CAFA rule)
        if len(term_indices) > 1500:
            term_indices = term_indices[:1500]
            scores = scores[:1500]
        
        # Add rows
        for term_idx, score in zip(term_indices, scores):
            # Clamp score to (0, 1] and format to 3 significant figures
            score = min(max(score, 0.001), 1.0)  # Ensure > 0
            rows.append({
                'protein_id': protein_id,
                'go_term': top_terms[term_idx],
                'score': f"{score:.3f}"
            })
    
    print(f"  Total prediction rows: {len(rows):,}")
    
    # Create DataFrame and save
    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(output_path, sep='\t', header=False, index=False)
    
    print(f"  Saved to: {output_path}")
    
    # Show statistics
    if len(rows) > 0:
        proteins_with_preds = submission_df['protein_id'].nunique()
        terms_predicted = submission_df['go_term'].nunique()
        avg_preds = len(rows) / proteins_with_preds
        
        print(f"\n  Submission statistics:")
        print(f"    Proteins with predictions: {proteins_with_preds:,}")
        print(f"    Unique GO terms predicted: {terms_predicted:,}")
        print(f"    Avg predictions/protein:   {avg_preds:.1f}")
    
    print("\n" + "="*70)
    print("[SUBMISSION COMPLETE]")
    print("="*70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CAFA-6 KNN Model with ESM2-3B Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--work-dir', '-w',
        type=Path,
        required=True,
        help='Path to cafa6_data directory (contains features/, parsed/, Train/)'
    )
    
    parser.add_argument(
        '--k', '-k',
        type=int,
        default=DEFAULT_K,
        help=f'Number of nearest neighbors (default: {DEFAULT_K})'
    )
    
    parser.add_argument(
        '--n-folds', '-f',
        type=int,
        default=DEFAULT_N_FOLDS,
        help=f'Number of cross-validation folds (default: {DEFAULT_N_FOLDS})'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size for processing (default: {DEFAULT_BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--skip-submission',
        action='store_true',
        help='Skip submission file generation'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CAFA-6 KNN Model with ESM2-3B Embeddings")
    print("="*70)
    print(f"Work directory: {args.work_dir}")
    print(f"k (neighbors):  {args.k}")
    print(f"n_folds:        {args.n_folds}")
    print(f"batch_size:     {args.batch_size}")
    
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    data = load_data(args.work_dir)
    
    # -------------------------------------------------------------------------
    # 2. Train KNN
    # -------------------------------------------------------------------------
    oof_preds, test_preds, best_threshold = train_knn(
        train_emb=data['train_emb'],
        test_emb=data['test_emb'],
        Y_train=data['Y_train'],
        k=args.k,
        n_folds=args.n_folds,
        batch_size=args.batch_size
    )
    
    # -------------------------------------------------------------------------
    # 3. Save predictions
    # -------------------------------------------------------------------------
    print("\n[SAVING PREDICTIONS]")
    
    oof_path = args.work_dir / 'oof_knn.npy'
    test_path = args.work_dir / 'test_knn.npy'
    terms_path = args.work_dir / 'features' / 'top_terms_full.json'
    
    np.save(oof_path, oof_preds)
    np.save(test_path, test_preds)
    
    with open(terms_path, 'w', encoding='utf-8') as f:
        json.dump(data['top_terms'], f)
    
    print(f"  OOF predictions:  {oof_path}")
    print(f"  Test predictions: {test_path}")
    print(f"  Term list:        {terms_path}")
    
    # -------------------------------------------------------------------------
    # 4. Generate submission
    # -------------------------------------------------------------------------
    if not args.skip_submission:
        generate_submission(
            test_preds=test_preds,
            test_ids=data['test_ids'],
            top_terms=data['top_terms'],
            term_aspects=data['term_aspects'],
            output_path=args.work_dir / 'submission.tsv',
            thresholds=DEFAULT_THRESHOLDS
        )
    
    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"[COMPLETE] Total time: {elapsed/60:.1f} minutes")
    print("="*70)


if __name__ == '__main__':
    main()
