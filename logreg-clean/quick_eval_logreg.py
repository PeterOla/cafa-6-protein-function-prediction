
import sys
from pathlib import Path
import numpy as np
import argparse

# Add logreg-clean to path
sys.path.append(str(Path('logreg-clean').resolve()))

import logreg_esm2_3b_13500 as lr_mod

def main():
    print("Running Quick Eval for LogReg (Top 50 terms)...")
    work_dir = Path('cafa6_data')
    
    # 1. Load Data using the FIXED logic
    train_emb, test_emb, train_ids, test_ids, top_terms = lr_mod.load_data(work_dir)
    
    # 2. Build Y using the FIXED logic (KNN parity)
    Y = lr_mod.build_y_matrix(train_ids, top_terms, work_dir)
    
    # 3. Select subset (Top 50 terms)
    # These are usually BP terms, but let's check
    subset_indices = list(range(50))
    subset_terms = [top_terms[i] for i in subset_indices]
    
    print(f"Validating on {len(subset_indices)} terms.")
    
    # 4. Train/Predict
    # We use the actual function from the module to verify the code path
    # n_folds=3 for speed
    oof, test = lr_mod.train_predict_logreg(
        train_emb, Y, test_emb, subset_indices, "QUICK_VAL_50", work_dir, n_folds=3
    )
    
    # 5. Evaluate
    f1, thr = lr_mod.evaluate_predictions(oof, Y, subset_indices, "QUICK_VAL_50")
    print(f"\nFinal Validation Result: F1={f1:.4f} @ Thr={thr:.2f}")

if __name__ == "__main__":
    main()
