"""Evaluation metrics"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Set


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate precision, recall, F1"""
    
    # Binarize predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_weighted_f1(y_true, y_pred, ia_weights, threshold=0.5):
    """Calculate IA-weighted F1 score"""
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate TP, FP, FN per term
    tp = ((y_true == 1) & (y_pred_binary == 1)).sum(axis=0)
    fp = ((y_true == 0) & (y_pred_binary == 1)).sum(axis=0)
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum(axis=0)
    
    # Precision and recall per term
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # Weight by IA
    weighted_precision = (precision * ia_weights).sum() / ia_weights.sum()
    weighted_recall = (recall * ia_weights).sum() / ia_weights.sum()
    
    # F1
    if weighted_precision + weighted_recall == 0:
        return 0.0
    
    f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    
    return f1


def calculate_f1_score(true_terms: List[Set[str]], pred_terms: List[Set[str]]) -> float:
    """
    Calculate average F1 score across all samples using sets of terms.
    
    Args:
        true_terms: List of sets, where each set contains ground truth GO terms for a protein.
        pred_terms: List of sets, where each set contains predicted GO terms for a protein.
        
    Returns:
        Average F1 score.
    """
    if len(true_terms) != len(pred_terms):
        raise ValueError("Length of true_terms and pred_terms must match")
    
    f1_scores = []
    
    for true_set, pred_set in zip(true_terms, pred_terms):
        if len(true_set) == 0:
            # If no ground truth, F1 is 0 unless prediction is also empty (perfect match)
            f1_scores.append(1.0 if len(pred_set) == 0 else 0.0)
            continue
            
        # Intersection
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set) - tp
        fn = len(true_set) - tp
        
        if tp == 0:
            f1_scores.append(0.0)
            continue
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
        
    return np.mean(f1_scores)


def calculate_precision_recall(true_terms: List[Set[str]], pred_terms: List[Set[str]]) -> tuple:
    """
    Calculate average Precision and Recall using sets of terms.
    """
    precisions = []
    recalls = []
    
    for true_set, pred_set in zip(true_terms, pred_terms):
        if len(true_set) == 0:
            continue
            
        tp = len(true_set.intersection(pred_set))
        fp = len(pred_set) - tp
        fn = len(true_set) - tp
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        
    return np.mean(precisions), np.mean(recalls)
