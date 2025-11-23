import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Add src to path to import modules
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR / 'src'))

from data.loaders import LabelLoader, SequenceLoader
from evaluation.metrics import calculate_f1_score, calculate_precision_recall

def run_frequency_baseline():
    print("=== Running Frequency Baseline Model ===\n")
    
    # Paths
    TRAIN_TERMS = BASE_DIR / 'Train/train_terms.tsv'
    TRAIN_SEQ = BASE_DIR / 'Train/train_sequences.fasta'
    
    # 1. Load Data
    print("1. Loading Data...")
    label_loader = LabelLoader(TRAIN_TERMS)
    
    # Get all protein IDs
    all_proteins = list(label_loader.protein_to_terms.keys())
    print(f"Total proteins with labels: {len(all_proteins)}")
    
    # 2. Split Data (Train/Validation)
    print("\n2. Splitting Data (80/20)...")
    train_ids, val_ids = train_test_split(all_proteins, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_ids)}")
    print(f"Val size:   {len(val_ids)}")
    
    # 3. Train (Calculate Frequencies)
    print("\n3. Training (Counting Term Frequencies)...")
    # Filter labels to only include training proteins
    train_df = label_loader.df[label_loader.df['EntryID'].isin(train_ids)]
    
    # Count terms
    term_counts = train_df['term'].value_counts()
    # Calculate probability (frequency / num_training_proteins)
    term_probs = term_counts / len(train_ids)
    
    print("Top 10 Most Frequent Terms:")
    print(term_probs.head(10))
    
    # 4. Predict
    print("\n4. Predicting on Validation Set...")
    # Strategy: Predict the top N terms for EVERY protein
    # We'll try a few different N values
    
    top_terms_list = term_probs.index.tolist()
    
    # Prepare ground truth for validation
    val_true_terms = [label_loader.get_terms(pid) for pid in val_ids]
    
    results = []
    
    for n in [10, 20, 50, 100]:
        print(f"\nEvaluating Top-{n} predictions...")
        
        # Prediction is the same set for all proteins
        pred_set = set(top_terms_list[:n])
        val_pred_terms = [pred_set for _ in val_ids]
        
        # Evaluate
        f1 = calculate_f1_score(val_true_terms, val_pred_terms)
        precision, recall = calculate_precision_recall(val_true_terms, val_pred_terms)
        
        print(f"Top-{n} -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        results.append({
            'N': n,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
        
    # Find best N
    best_result = max(results, key=lambda x: x['F1'])
    print(f"\nBest Result: Top-{best_result['N']} terms with F1 = {best_result['F1']:.4f}")
    
    return best_result

if __name__ == "__main__":
    run_frequency_baseline()
