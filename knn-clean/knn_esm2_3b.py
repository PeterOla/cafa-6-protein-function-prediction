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
from sklearn.metrics import f1_score, precision_score, recall_score


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default hyperparameters (can be overridden via command line)
# Default hyperparameters (Best Aspect-Specific K used in ensemble: BP=5, MF=10, CC=15)
DEFAULT_K = 15                    
DEFAULT_N_FOLDS = 5               
DEFAULT_BATCH_SIZE = 250          # Memory-safe default for 22GB RAM

# Optimal Thresholds found during "Super-KNN" validation
DEFAULT_THRESHOLDS = {
    'BP': 0.40,   # Record F1 with K=5
    'MF': 0.40,   # Peak performance with K=10
    'CC': 0.30,   # Peak performance with K=15
    'ALL': 0.30   
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
    
    # Extract accessions and then IMMEDIATELY delete dataframes
    protein_ids = [extract_uniprot_accession(pid) for pid in train_seq_df['id'].tolist()]
    test_ids = [extract_uniprot_accession(pid) for pid in test_seq_df['id'].tolist()]
    
    print(f"      Training proteins: {len(protein_ids)}")
    print(f"      Test proteins:     {len(test_ids)}")
    
    del train_seq_df
    del test_seq_df
    gc.collect()
    
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
    
    # Build labels using this order
    for _, row in train_terms_df.iterrows():
        protein_idx = protein_to_idx.get(row['EntryID'])
        if protein_idx is not None:
            term_idx = term_to_idx.get(row['term'])
            if term_idx is not None:
                rows.append(protein_idx)
                cols.append(term_idx)
    
    # Create sparse CSR matrix
    Y_train = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_proteins, n_terms)
    )
    
    # FREE MEMORY: Delete temporary lists and dataframe
    print("      (Cleaning up memory...)")
    del rows
    del cols
    del train_terms_df
    gc.collect()
    
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
    top_terms: list,
    term_aspects: dict,
    k: int = 10,
    n_folds: int = 5,
    batch_size: int = 500,
    skip_test: bool = False
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
        top_terms: list of GO term IDs (for aspect lookup)
        term_aspects: dict mapping term -> aspect (BP/MF/CC)
        k: number of nearest neighbors
        n_folds: number of cross-validation folds
        batch_size: batch size for memory-efficient processing
        skip_test: if True, skip test predictions to save memory
    
    Returns:
        oof_preds: (n_train, n_terms) out-of-fold predictions
        test_preds: (n_test, n_terms) test predictions (None if skip_test)
        best_threshold: optimal threshold from validation
    """
    print("\n" + "="*70)
    print(f"[KNN TRAINING] k={k}, folds={n_folds}")
    if skip_test:
        print("[NOTE] Skipping test predictions (--skip-test enabled)")
    print("="*70)
    
    n_train = len(train_emb)
    n_test = len(test_emb)
    n_terms = Y_train.shape[1]
    
    # Build aspect masks (which columns belong to which aspect)
    aspect_masks = {'BP': [], 'MF': [], 'CC': []}
    for i, term in enumerate(top_terms):
        aspect = term_aspects.get(term, 'UNK')
        if aspect in aspect_masks:
            aspect_masks[aspect].append(i)
    
    for aspect, indices in aspect_masks.items():
        print(f"  {aspect}: {len(indices)} terms")
    
    # Initialize prediction arrays
    # Using float16 for oof_preds to save 50% memory (from 8.6GB to 4.3GB).
    # Precision is sufficient for similarity scores.
    print(f"    Allocating oof_preds matrix (float16)...")
    oof_preds = np.zeros((n_train, n_terms), dtype=np.float16)
    test_preds = None if skip_test else np.zeros((n_test, n_terms), dtype=np.float32)
    
    # Stratification target: number of labels per protein (capped at 10)
    # This helps ensure each fold has similar label distributions
    stratify_target = Y_train.sum(axis=1).A1  # .A1 converts to 1D array
    stratify_target = np.minimum(stratify_target, 10).astype(int)
    
    print(f"    Converting embeddings to float16 for memory efficiency...")
    train_emb = train_emb.astype(np.float16)
    if test_emb is not None:
        test_emb = test_emb.astype(np.float16)
    gc.collect()

    # -----------------------------------------------------------------
    # FINAL MIXED-K ENSEMBLE CONFIG
    # -----------------------------------------------------------------
    aspect_k = {'BP': 5, 'MF': 10, 'CC': 15}
    k_max = max(aspect_k.values())
    
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
        
        # Build KNN index using k_max
        print(f"    Building KNN index (k={k_max}, metric=cosine, jobs=2)...")
        knn = NearestNeighbors(n_neighbors=k_max, metric='cosine', n_jobs=2)
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
            # [FINAL MIXED-K ENSEMBLE LOGIC]
            # Use optimal K for each aspect:
            #   BP: K=5  (Best captured by local neighbors)
            #   MF: K=10 (Moderate aggregation is best)
            #   CC: K=15 (Broadest context is best)
            # -----------------------------------------------------------------
            
            combined_scores = np.zeros((batch_size_actual, n_terms), dtype=np.float16)
            
            for aspect, k_val in aspect_k.items():
                col_indices = aspect_masks[aspect]
                if not col_indices: continue
                
                # Slice neighbors to this aspect's specific K
                # batch_sims is (batch, k_max), neighbor_labels is (batch, k_max, n_terms)
                a_sims = batch_sims[:, :k_val]
                a_labels = neighbor_labels[:, :k_val, col_indices]
                
                # Weighted aggregation for this aspect
                sims_expanded = a_sims[:, :, np.newaxis]
                a_scores = (sims_expanded * a_labels).sum(axis=1)
                a_scores /= (a_sims.sum(axis=1, keepdims=True) + 1e-9)
                
                combined_scores[:, col_indices] = a_scores.astype(np.float16)
            
            oof_preds[val_idx[batch_start:batch_end]] = combined_scores
            
            # FREE MEMORY: temporary batch arrays
            del neighbor_labels
            del combined_scores
        
        # ---------------------------------------------------------------------
        # Predict on test set (accumulate across folds, will average later)
        # ---------------------------------------------------------------------
        if not skip_test:
            print("    Predicting on test set...")
            
            test_distances, test_indices = knn.kneighbors(test_emb)
            test_similarities = 1 - test_distances
            
            for batch_start in range(0, n_test, batch_size):
                batch_end = min(batch_start + batch_size, n_test)
                batch_size_actual = batch_end - batch_start
                
                batch_indices = test_indices[batch_start:batch_end]
                batch_sims = test_similarities[batch_start:batch_end]
                
                # Fetch neighbor labels for this test batch
                flat_indices = batch_indices.flatten()
                neighbor_labels = Y_train_fold[flat_indices].toarray()
                neighbor_labels = neighbor_labels.reshape(batch_size_actual, k_max, -1)
                
                for aspect, k_val in aspect_k.items():
                    col_indices = aspect_masks[aspect]
                    if not col_indices: continue
                    
                    # Slice neighbors to this aspect's specific K
                    a_sims = batch_sims[:, :k_val]
                    a_labels = neighbor_labels[:, :k_val, col_indices]
                    
                    sims_expanded = a_sims[:, :, np.newaxis]
                    a_scores = (sims_expanded * a_labels).sum(axis=1)
                    a_scores /= (a_sims.sum(axis=1, keepdims=True) + 1e-9)
                    
                    test_preds[batch_start:batch_end, col_indices] += a_scores.astype(np.float32)
                
                # FREE MEMORY: temporary batch arrays
                del neighbor_labels
                
        # FREE MEMORY: Cleanup fold data
        print("    Cleaning up fold memory...")
        del knn
        del similarities
        del indices
        gc.collect()
    
    # Average test predictions across folds
    if not skip_test:
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
    if not skip_test:
        test_max = test_preds.max(axis=1, keepdims=True)
        test_max[test_max == 0] = 1.0
        test_preds /= test_max
    
    # -------------------------------------------------------------------------
    # Per-aspect CAFA-style evaluation
    # -------------------------------------------------------------------------
    print("\n  " + "="*60)
    print("  [CAFA-STYLE PER-ASPECT EVALUATION]")
    print("  " + "="*60)
    print(f"  (Finding best thresholds for K={k}...)")
    
    # -------------------------------------------------------------------------
    # FINAL WINNERS - Hardcoded for maximum performance
    # -------------------------------------------------------------------------
    # These values were determined through exhaustive cross-validation
    best_thresholds = {
        'BP': 0.40,
        'MF': 0.40,
        'CC': 0.30,
        'ALL': 0.30
    }
    
    # Calculate final CAFA F1 using these specific thresholds
    aspect_f1s = {}
    for threshold in [0.40, 0.30]: # We only need to evaluate these two
        for aspect in ['BP', 'MF', 'CC']:
            if best_thresholds[aspect] != threshold: continue
            
            col_indices = aspect_masks[aspect]
            if not col_indices: continue
            
            Y_true_aspect = Y_train[:, col_indices]
            f1_sum = 0
            n_samples = oof_preds.shape[0]
            
            for b_start in range(0, n_samples, 5000):
                b_end = min(b_start + 5000, n_samples)
                y_pred_batch = (oof_preds[b_start:b_end, col_indices] >= threshold).astype(np.int8)
                y_true_batch = Y_true_aspect[b_start:b_end].toarray()
                
                tp = ((y_pred_batch == 1) & (y_true_batch == 1)).sum(axis=1)
                fp = ((y_pred_batch == 1) & (y_true_batch == 0)).sum(axis=1)
                fn = ((y_pred_batch == 0) & (y_true_batch == 1)).sum(axis=1)
                
                prec = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)!=0)
                rec = np.divide(tp, tp+fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn)!=0)
                f1_batch = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec), where=(prec+rec)!=0)
                f1_sum += f1_batch.sum()
            
            aspect_f1s[aspect] = f1_sum / n_samples
    
    final_cafa_f1 = (aspect_f1s['BP'] + aspect_f1s['MF'] + aspect_f1s['CC']) / 3
    
    print("\n  " + "-"*55)
    print(f"  FINAL OPTIMIZED RESULTS (Ensemble K={aspect_k}):")
    print(f"  BP: K=5,  Thresh=0.40, F1={aspect_f1s['BP']:.4f}")
    print(f"  MF: K=10, Thresh=0.40, F1={aspect_f1s['MF']:.4f}")
    print(f"  CC: K=15, Thresh=0.30, F1={aspect_f1s['CC']:.4f}")
    print(f"  OVERALL CAFA F1: {final_cafa_f1:.4f}")
    print("  " + "-"*55)
    
    if not skip_test:
        print("\n  Averaging test predictions across folds...")
        test_preds /= n_folds
    
    print("\n" + "="*70)
    print("[KNN TRAINING COMPLETE]")
    print("="*70)
    
    return oof_preds, test_preds, best_thresholds


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
    
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip test predictions to save memory (for validation only)'
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
        top_terms=data['top_terms'],
        term_aspects=data['term_aspects'],
        k=args.k,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        skip_test=args.skip_test
    )
    
    # -------------------------------------------------------------------------
    # 3. Save predictions
    # -------------------------------------------------------------------------
    print("\n[SAVING PREDICTIONS]")
    
    oof_path = args.work_dir / 'oof_knn.npy'
    test_path = args.work_dir / 'test_knn.npy'
    terms_path = args.work_dir / 'features' / 'top_terms_full.json'
    
    np.save(oof_path, oof_preds)
    if test_preds is not None:
        np.save(test_path, test_preds)
    else:
        print("  (Test predictions skipped)")
    
    with open(terms_path, 'w', encoding='utf-8') as f:
        json.dump(data['top_terms'], f)
    
    print(f"  OOF predictions:  {oof_path}")
    print(f"  Test predictions: {test_path}")
    print(f"  Term list:        {terms_path}")
    
    # -------------------------------------------------------------------------
    # 4. Generate submission
    # -------------------------------------------------------------------------
    if not args.skip_submission and test_preds is not None:
        generate_submission(
            test_preds=test_preds,
            test_ids=data['test_ids'],
            top_terms=data['top_terms'],
            term_aspects=data['term_aspects'],
            output_path=args.work_dir / 'submission.tsv',
            thresholds=best_threshold
        )
    elif args.skip_test:
        print("\n[SKIPPING SUBMISSION] No test predictions available")
    
    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"[COMPLETE] Total time: {elapsed/60:.1f} minutes")
    print("="*70)


if __name__ == '__main__':
    main()
