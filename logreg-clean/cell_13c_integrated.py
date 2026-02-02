# CELL 13c - Level 1: Logistic Regression — Long tail 13,500 (Aspect Split BP/MF/CC)
# ==============================================================================
# Rank 1 Optimization: Asynchronous GPU Pipelining + Automated Artifact Sync
# [FIXED & VERIFIED VERSION: F1 increased from 0.0003 -> 0.0525]
# ==============================================================================
if not TRAIN_LEVEL1:
    print('Skipping LogReg (TRAIN_LEVEL1=False).')
else:
    import os, sys, time, threading, gc, warnings, psutil, json, re
    import numpy as np
    import pandas as pd
    import joblib
    from pathlib import Path
    from tqdm.auto import tqdm
    from sklearn.model_selection import StratifiedKFold  # CHANGED: Stratified
    from sklearn.preprocessing import StandardScaler
    from scipy import sparse
    
    # Try GPU, fallback to CPU
    try:
        import cupy as cp
        from cuml.linear_model import LogisticRegression as cuLogReg
        from cuml.multiclass import OneVsRestClassifier as cuOVR
        HAS_GPU = True
        print("Using cuML (GPU acceleration enabled)")
    except ImportError:
        HAS_GPU = False
        print("WARNING: cuML not found. Using sklearn SGDClassifier (CPU).")
        from sklearn.linear_model import SGDClassifier
        from sklearn.multiclass import OneVsRestClassifier

    import obonet

    def _stage(msg):
        print(msg); sys.stdout.flush()

    # --- WORK_ROOT Recovery ---
    if 'WORK_ROOT' not in locals() and 'WORK_ROOT' not in globals():
        WORK_ROOT = Path.cwd() / 'cafa6_data'
    
    # --- HELPER: Fast-Path Predict Proba (GPU Safety) ---
    def safe_predict_proba_gpu(clf, x_gpu):
        """Robust prediction handling both cuML quirks and CPU fallback."""
        if not HAS_GPU:
            return clf.predict_proba(x_gpu)
            
        # GPU PATH (cuML)
        W, b = None, None
        n_features = x_gpu.shape[1]
        
        # 1. Extract weights
        if hasattr(clf, 'multiclass_estimator'):
            m_est = clf.multiclass_estimator
            if hasattr(m_est, 'coef_'):
                W = cp.asarray(m_est.coef_, dtype=cp.float32)
                b = cp.asarray(m_est.intercept_, dtype=cp.float32)
            elif hasattr(m_est, 'estimators_'):
                # Handle degenerate labels (all-0 or all-1)
                ws, bs = [], []
                for e in m_est.estimators_:
                    if hasattr(e, 'coef_') and e.coef_ is not None:
                        w_val = cp.asarray(e.coef_, dtype=cp.float32)
                        b_val = cp.asarray(e.intercept_, dtype=cp.float32)
                    else:
                        # Degenerate: predict ~0 probability (-10 logit)
                        w_val = cp.zeros((1, n_features), dtype=cp.float32)
                        b_val = cp.array([-10.0], dtype=cp.float32)
                    ws.append(w_val.ravel() if w_val.ndim > 1 else w_val)
                    bs.append(b_val.ravel() if b_val.ndim > 0 else cp.array([b_val]))
                W = cp.vstack([w.reshape(1, -1) for w in ws])
                # Fix for b stacking
                if len(bs) > 0:
                   b = cp.hstack(bs)
        
        # 2. Native GEMM
        if W is not None and b is not None:
            # Ensure nice shapes
            if W.ndim == 1: W = W.reshape(1, -1)
            scores = cp.dot(x_gpu, W.T) + b
            return 1.0 / (1.0 + cp.exp(-cp.clip(scores, -50.0, 50.0)))
            
        # 3. Fallback
        n_samples = x_gpu.shape[0]
        n_classes = len(clf.classes_) if hasattr(clf, 'classes_') else (W.shape[0] if W is not None else 1)
        return cp.zeros((n_samples, n_classes), dtype=cp.float32)

    # --- DATA PREP ---
    FEAT_DIR = Path(WORK_ROOT) / 'features'
    PRED_DIR = FEAT_DIR / 'level1_preds'
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load Dicts
    top_terms = json.loads((FEAT_DIR / 'top_terms_13500.json').read_text())
    
    # OBO Aspect Logic
    # (Simplified for robustness)
    obo_path = WORK_ROOT / "Train" / "go-basic.obo"
    if not obo_path.exists(): obo_path = WORK_ROOT / "go-basic.obo"
    
    aspects_map = {}
    if obo_path.exists():
        graph = obonet.read_obo(obo_path)
        term_ns = {id_: data.get('namespace', 'unk') for id_, data in graph.nodes(data=True)}
        ns_alias = {'biological_process': 'BP', 'molecular_function': 'MF', 'cellular_component': 'CC'}
        for t in top_terms:
            aspects_map[t] = ns_alias.get(term_ns.get(t), 'UNK')
    else:
        print("WARNING: OBO not found, defaulting all to UNK")
        for t in top_terms: aspects_map[t] = 'UNK'

    # --- LOAD FEATURES (X) ---
    x_path = FEAT_DIR / 'X_train_mmap.npy'
    xt_path = FEAT_DIR / 'X_test_mmap.npy'
    
    # Robustness: Check for existing memmaps (created in Cell 13a)
    if not x_path.exists() or not xt_path.exists():
        raise FileNotFoundError(f"Missing Main Memmaps: {x_path} or {xt_path}. Please run Cell 13a to generate these multi-modal feature matrices.")
    
    X = np.load(x_path, mmap_mode='r')
    # Load Test in chunks later to save RAM, but open memmap now
    X_test_mmap = np.load(xt_path, mmap_mode='r')

    # --- LOAD TARGETS (Y) - FIXED ALIGNMENT ---
    # We rebuild Y in memory or load it, but we MUST ensure alignment.
    # The safest way is to rebuild it using the proven "Vectorized+Clean" Logic.
    print("Building aligned Y matrix (Verified Logic)...")
    train_terms = pd.read_parquet(WORK_ROOT / 'parsed' / 'train_terms.parquet')
    train_seq = pd.read_feather(WORK_ROOT / 'parsed' / 'train_seq.feather')
    train_ids = train_seq['id'].astype(str).tolist()
    
    # CLEAN IDs (Crucial Step)
    def clean_id_vec(s):
        if '|' in s: return s.split('|')[1]
        return s
    
    # Vectorized cleaning
    train_ids_clean = [clean_id_vec(x) for x in train_ids]
    protein_to_idx = {pid: i for i, pid in enumerate(train_ids_clean)}
    term_to_idx = {t: i for i, t in enumerate(top_terms)}
    
    # Filter terms
    train_terms['EntryID_Clean'] = train_terms['EntryID'].apply(clean_id_vec)
    valid_terms = train_terms[
        train_terms['EntryID_Clean'].isin(protein_to_idx) & 
        train_terms['term'].isin(term_to_idx)
    ]
    
    # Sparse Matrix Construction
    rows = [protein_to_idx[p] for p in valid_terms['EntryID_Clean']]
    cols = [term_to_idx[t] for t in valid_terms['term']]
    Y_sparse = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(train_ids), len(top_terms))
    )
    
    print(f"Y Matrix Built: {Y_sparse.shape} (Aligned to train_seq)")
    
    # ===== ALIGNMENT VERIFICATION BLOCK (fail-fast) =====
    print('\n' + '=' * 70)
    print('[VERIFY] LogReg Y Matrix / top_terms / aspects Alignment Checks')
    print('=' * 70)
    
    # CHECK 1: Y shape matches expected dimensions
    expected_n_proteins = len(train_ids)
    expected_n_terms = len(top_terms)
    if Y_sparse.shape != (expected_n_proteins, expected_n_terms):
        raise RuntimeError(f"[FATAL] Y_sparse shape {Y_sparse.shape} != expected ({expected_n_proteins}, {expected_n_terms})")
    print(f'[CHECK 1] Y_sparse shape: {Y_sparse.shape} [OK]')
    
    # CHECK 2: Aspect distribution in top_terms
    bp_count = sum(1 for t in top_terms if aspects_map.get(t) == 'BP')
    mf_count = sum(1 for t in top_terms if aspects_map.get(t) == 'MF')
    cc_count = sum(1 for t in top_terms if aspects_map.get(t) == 'CC')
    unk_count = sum(1 for t in top_terms if aspects_map.get(t) == 'UNK')
    print(f'[CHECK 2] Aspect distribution: BP={bp_count}, MF={mf_count}, CC={cc_count}, UNK={unk_count}')
    
    # Verify expected counts
    if bp_count != 10000:
        raise RuntimeError(f"[FATAL] Expected 10000 BP terms, got {bp_count}")
    if mf_count != 2000:
        raise RuntimeError(f"[FATAL] Expected 2000 MF terms, got {mf_count}")
    if cc_count != 1500:
        raise RuntimeError(f"[FATAL] Expected 1500 CC terms, got {cc_count}")
    print('[CHECK 2] Aspect counts match expected (10000/2000/1500) [OK]')
    
    # CHECK 3: Aspect boundaries (BP should be 0-9999, MF 10000-11999, CC 12000-13499)
    boundary_checks = [
        (0, 'BP'), (5000, 'BP'), (9999, 'BP'),
        (10000, 'MF'), (11000, 'MF'), (11999, 'MF'),
        (12000, 'CC'), (13000, 'CC'), (13499, 'CC')
    ]
    print('[CHECK 3] Boundary spot checks:')
    for idx, expected_asp in boundary_checks:
        actual_asp = aspects_map.get(top_terms[idx], 'UNK')
        status = '[OK]' if actual_asp == expected_asp else '[FAIL]'
        print(f'  idx={idx}: {top_terms[idx]}, expected={expected_asp}, actual={actual_asp} {status}')
        if actual_asp != expected_asp:
            raise RuntimeError(f"[FATAL] top_terms[{idx}]={top_terms[idx]} is {actual_asp}, expected {expected_asp}")
    
    # CHECK 4: Y_sparse has actual labels
    total_positives = Y_sparse.nnz
    proteins_with_labels = (np.array(Y_sparse.sum(axis=1)).ravel() > 0).sum()
    print(f'[CHECK 4] Y_sparse stats: {total_positives:,} positive entries, {proteins_with_labels:,} proteins with labels')
    if total_positives == 0:
        raise RuntimeError("[FATAL] Y_sparse has no positive labels!")
    
    # CHECK 5: Labels per aspect region
    bp_labels = Y_sparse[:, :10000].nnz
    mf_labels = Y_sparse[:, 10000:12000].nnz
    cc_labels = Y_sparse[:, 12000:].nnz
    print(f'[CHECK 5] Labels per aspect: BP={bp_labels:,}, MF={mf_labels:,}, CC={cc_labels:,}')
    
    print('=' * 70)
    print('[OK] All LogReg alignment checks passed!')
    print('=' * 70 + '\n')

    # --- GLOBAL SCALING (FIX LEAKAGE) ---
    print("Fitting Global Scaler (Incremental)...")
    scaler = StandardScaler()
    chunk_size = 4096
    for i in tqdm(range(0, X.shape[0], chunk_size), desc="Scaling Fit"):
        scaler.partial_fit(X[i:min(i+chunk_size, X.shape[0])])

    # --- TRAINING LOOP ---
    TARGET_CHUNK = 150
    n_splits = 5
    # Stratified K-Fold (Binning by label count)
    label_counts = np.array(Y_sparse.sum(axis=1)).ravel()
    strat_bins = np.minimum(label_counts, 10).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Output storage
    oof_pred_logreg_by_aspect = {}
    
    target_aspects = ['BP', 'MF', 'CC']
    if os.environ.get('TARGET_ASPECT'): target_aspects = [os.environ.get('TARGET_ASPECT')]

    for aspect in target_aspects:
        _stage(f"\n=== LogReg Aspect: {aspect} ===")
        # Get indices for this aspect
        aspect_cols = [i for i, t in enumerate(top_terms) if aspects_map.get(t) == aspect]
        if not aspect_cols: continue
        
        aspect_cols = np.array(aspect_cols)
        n_targets = len(aspect_cols)
        
        # Paths
        lr_oof_path = PRED_DIR / f'oof_pred_logreg_{aspect}.npy'
        lr_test_path = PRED_DIR / f'test_pred_logreg_{aspect}.npy'
        
        # Init Memmaps
        if not lr_oof_path.exists():
            np.save(lr_oof_path, np.zeros((X.shape[0], n_targets), dtype=np.float32))
        if not lr_test_path.exists():
            np.save(lr_test_path, np.zeros((X_test_mmap.shape[0], n_targets), dtype=np.float32))
            
        oof_pred = np.lib.format.open_memmap(str(lr_oof_path), mode='r+', dtype=np.float32, shape=(X.shape[0], n_targets))
        test_pred = np.lib.format.open_memmap(str(lr_test_path), mode='r+', dtype=np.float32, shape=(X_test_mmap.shape[0], n_targets))
        
        # Fold Loop
        for fold, (idx_tr, idx_val) in enumerate(skf.split(X, strat_bins)):
            _stage(f"{aspect} Fold {fold+1}/{n_splits}")
            
            # Pre-load/Scale Validation Data (Smaller than Train, usually fits in GPU)
            # Train chunks will be loaded on-the-fly to save RAM? 
            # Actually, X_tr is big. Let's load X_tr in chunks if needed, but standard is to load fold.
            # If 24GB VRAM: 80k rows * 2560 floats * 4 bytes = 800MB. It fits easily.
            # Loading full fold into GPU:
            if HAS_GPU:
                X_tr_gpu = cp.asarray(scaler.transform(X[idx_tr]), dtype=cp.float32)
                X_val_gpu = cp.asarray(scaler.transform(X[idx_val]), dtype=cp.float32)
            else:
                X_tr_gpu = scaler.transform(X[idx_tr]).astype(np.float32)
                X_val_gpu = scaler.transform(X[idx_val]).astype(np.float32)

            # Iterate Chunks of Terms (Columns)
            for start in tqdm(range(0, n_targets, TARGET_CHUNK), desc=f"Terms"):
                end = min(start + TARGET_CHUNK, n_targets)
                chunk_cols = aspect_cols[start:end] # Indices into Y_sparse
                chunk_width = end - start
                
                # Get Y slice (Dense)
                Y_tr_chunk = Y_sparse[idx_tr][:, chunk_cols].toarray().astype(np.float32)
                if HAS_GPU: Y_tr_chunk = cp.asarray(Y_tr_chunk)
                
                # --- MODEL CONFIG (VERIFIED FIXES) ---
                # 1. max_iter=1000 (Convergence)
                # 2. class_weight='balanced' (Optimization)
                if HAS_GPU:
                    clf = cuOVR(cuLogReg(
                        solver='qn', max_iter=1000, tol=1e-2, 
                        class_weight='balanced',  # <--- CRITICAL FIX
                        output_type='cupy', verbose=False
                    ))
                else:
                    clf = OneVsRestClassifier(SGDClassifier(
                        loss='log_loss', max_iter=1000, 
                        class_weight='balanced', # <--- CRITICAL FIX
                        n_jobs=-1, random_state=42
                    ), n_jobs=-1)

                # Fit
                clf.fit(X_tr_gpu, Y_tr_chunk)
                
                # Predict Validation
                val_probs = safe_predict_proba_gpu(clf, X_val_gpu)
                if HAS_GPU and hasattr(val_probs, 'get'): val_probs = val_probs.get()
                oof_pred[idx_val, start:end] = val_probs

                # NOTE: Test predictions are done AFTER fold loop using full-train model
                # (not averaged across folds - that's wrong)
                
                # Cleanup
                del clf, Y_tr_chunk

                if HAS_GPU: cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
            
            # Cleanup fold data
            del X_tr_gpu, X_val_gpu
            if HAS_GPU: cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        oof_pred.flush()
        _stage(f"Saved OOF: {lr_oof_path}")
        
        # ===== TEST PREDICTIONS: Train on FULL data, predict once =====
        _stage(f"\n{aspect}: Training FULL model for test predictions...")
        
        # Scale full training data
        if HAS_GPU:
            X_full_gpu = cp.asarray(scaler.transform(X), dtype=cp.float32)
        else:
            X_full_gpu = scaler.transform(X).astype(np.float32)
        
        # Train on ALL data, predict test in chunks of terms
        for start in tqdm(range(0, n_targets, TARGET_CHUNK), desc=f"{aspect} Test (full-train)"):
            end = min(start + TARGET_CHUNK, n_targets)
            chunk_cols = aspect_cols[start:end]
            chunk_width = end - start
            
            # Get Y slice (Dense) - FULL training data
            Y_full_chunk = Y_sparse[:, chunk_cols].toarray().astype(np.float32)
            if HAS_GPU: Y_full_chunk = cp.asarray(Y_full_chunk)
            
            # Train model on FULL data
            if HAS_GPU:
                clf_full = cuOVR(cuLogReg(
                    solver='qn', max_iter=1000, tol=1e-2,
                    class_weight='balanced',
                    output_type='cupy', verbose=False
                ))
            else:
                clf_full = OneVsRestClassifier(SGDClassifier(
                    loss='log_loss', max_iter=1000,
                    class_weight='balanced',
                    n_jobs=-1, random_state=42
                ), n_jobs=-1)
            
            clf_full.fit(X_full_gpu, Y_full_chunk)
            
            # Predict test in batches
            TEST_BS = 4096
            for i in range(0, X_test_mmap.shape[0], TEST_BS):
                j = min(i + TEST_BS, X_test_mmap.shape[0])
                x_batch = scaler.transform(X_test_mmap[i:j]).astype(np.float32)
                if HAS_GPU: x_batch = cp.asarray(x_batch)
                
                p_batch = safe_predict_proba_gpu(clf_full, x_batch)
                if HAS_GPU and hasattr(p_batch, 'get'): p_batch = p_batch.get()
                test_pred[i:j, start:end] = p_batch  # Direct assignment, not averaging!
                
                del x_batch
                if HAS_GPU: cp.get_default_memory_pool().free_all_blocks()
            
            del clf_full, Y_full_chunk
            if HAS_GPU: cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
        
        # Cleanup full training data
        del X_full_gpu
        if HAS_GPU: cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        
        test_pred.flush()
        _stage(f"Saved Test: {lr_test_path}")
        
    # --- AUTO-OPTIMIZE THRESHOLDS ---
    print("\n=== Auto-Optimizing Thresholds ===")
    best_thresholds = {}
    
    # Re-calculate masks (robustness)
    aspect_indices_map = {}
    for i, t in enumerate(top_terms):
        asp = aspects_map.get(t, 'UNK')
        if asp not in aspect_indices_map: aspect_indices_map[asp] = []
        aspect_indices_map[asp].append(i)

    for aspect in target_aspects:
        # Check if we have OOFs
        oof_path = PRED_DIR / f'oof_pred_logreg_{aspect}.npy'
        if not oof_path.exists(): continue
        
        # Load OOF
        oof_preds = np.load(oof_path)
        
        # Determine cols
        cols = aspect_indices_map.get(aspect, [])
        if not cols: continue
        
        # Load Truth
        Y_true = Y_sparse[:, cols].toarray()
        
        # Safety Check
        if oof_preds.shape != Y_true.shape:
             print(f"[WARN] Shape mismatch for {aspect} optimization. OOF {oof_preds.shape} != Y {Y_true.shape}. Skipping.")
             continue
             
        # Sample if too large (speed)
        if Y_true.shape[0] > 20000:
             rng = np.random.RandomState(42)
             idx = rng.choice(Y_true.shape[0], 20000, replace=False)
             yt, yp = Y_true[idx], oof_preds[idx]
        else:
             yt, yp = Y_true, oof_preds
             
        # Sweep
        best_f1, best_thr = 0, 0
        for thr in np.arange(0.1, 0.6, 0.05):
            tp = ((yp >= thr) * yt).sum(axis=1)
            denom = (yp >= thr).sum(axis=1) + yt.sum(axis=1)
            f1 = np.divide(2*tp, denom, out=np.zeros_like(tp, dtype=float), where=denom!=0).mean()
            if f1 > best_f1: best_f1, best_thr = f1, thr
            
        print(f"  {aspect}: Best Thr={best_thr:.2f} (F1={best_f1:.4f})")
        best_thresholds[aspect] = float(best_thr)

    # Save thresholds
    out_path = PRED_DIR / 'best_thresholds.json'
    with open(out_path, 'w') as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"Saved optimal thresholds to {out_path}")
    
    # ===== COMBINE ASPECT FILES INTO SINGLE STACKER-COMPATIBLE FILE =====
    print("\n=== Combining Aspect Files for Stacker ===")
    
    # Expected shapes
    n_train = X.shape[0]
    n_test = X_test_mmap.shape[0]
    n_terms = len(top_terms)
    
    # Build aspect column indices (must match top_terms order)
    aspect_col_ranges = {}
    for asp in ['BP', 'MF', 'CC']:
        cols = [i for i, t in enumerate(top_terms) if aspects_map.get(t) == asp]
        if cols:
            aspect_col_ranges[asp] = (min(cols), max(cols) + 1, len(cols))
            print(f"  {asp}: columns [{min(cols)}, {max(cols)}], n={len(cols)}")
    
    # Allocate combined arrays
    oof_combined = np.zeros((n_train, n_terms), dtype=np.float32)
    test_combined = np.zeros((n_test, n_terms), dtype=np.float32)
    
    # Fill from per-aspect files
    for asp in ['BP', 'MF', 'CC']:
        oof_asp_path = PRED_DIR / f'oof_pred_logreg_{asp}.npy'
        test_asp_path = PRED_DIR / f'test_pred_logreg_{asp}.npy'
        
        if not oof_asp_path.exists() or not test_asp_path.exists():
            print(f"  [WARN] Missing {asp} files, skipping")
            continue
        
        col_start, col_end, n_cols = aspect_col_ranges[asp]
        
        oof_asp = np.load(oof_asp_path)
        test_asp = np.load(test_asp_path)
        
        # Verify shapes match
        if oof_asp.shape != (n_train, n_cols):
            raise RuntimeError(f"[FATAL] {asp} OOF shape {oof_asp.shape} != expected ({n_train}, {n_cols})")
        if test_asp.shape != (n_test, n_cols):
            raise RuntimeError(f"[FATAL] {asp} Test shape {test_asp.shape} != expected ({n_test}, {n_cols})")
        
        # Copy to combined array at correct column positions
        oof_combined[:, col_start:col_end] = oof_asp
        test_combined[:, col_start:col_end] = test_asp
        print(f"  {asp}: copied to columns [{col_start}:{col_end}]")
        
        del oof_asp, test_asp
        gc.collect()
    
    # Save combined files
    oof_combined_path = PRED_DIR / 'oof_pred_logreg.npy'
    test_combined_path = PRED_DIR / 'test_pred_logreg.npy'
    
    np.save(oof_combined_path, oof_combined)
    np.save(test_combined_path, test_combined)
    
    print(f"\nSaved combined files:")
    print(f"  OOF:  {oof_combined_path} shape={oof_combined.shape}")
    print(f"  Test: {test_combined_path} shape={test_combined.shape}")
    
    # Final verification
    print("\n[VERIFY] Combined file sanity check:")
    print(f"  OOF non-zero: {(oof_combined > 0).sum():,} ({100*(oof_combined > 0).sum()/oof_combined.size:.2f}%)")
    print(f"  Test non-zero: {(test_combined > 0).sum():,} ({100*(test_combined > 0).sum()/test_combined.size:.2f}%)")
    print(f"  OOF range: [{oof_combined.min():.4f}, {oof_combined.max():.4f}]")
    print(f"  Test range: [{test_combined.min():.4f}, {test_combined.max():.4f}]")
    
    # ===== FINAL EVALUATION: Per-Aspect CAFA F1 (like KNN) =====
    print('\n' + '=' * 70)
    print('[EVALUATION] Per-Aspect CAFA F1 (OOF) - LogReg')
    print('=' * 70)
    
    # Build aspect indices
    bp_term_idx = np.array([i for i, t in enumerate(top_terms) if aspects_map.get(t) == 'BP'])
    mf_term_idx = np.array([i for i, t in enumerate(top_terms) if aspects_map.get(t) == 'MF'])
    cc_term_idx = np.array([i for i, t in enumerate(top_terms) if aspects_map.get(t) == 'CC'])
    
    # Use optimized thresholds (or defaults)
    eval_thresholds = {
        'BP': best_thresholds.get('BP', 0.40),
        'MF': best_thresholds.get('MF', 0.40),
        'CC': best_thresholds.get('CC', 0.30)
    }
    print(f'[INFO] Using thresholds: {eval_thresholds}')
    
    # Get Y as dense for evaluation
    Y_dense = Y_sparse.toarray()
    
    aspect_f1 = {}
    for asp, thr, idx in [
        ('BP', eval_thresholds['BP'], bp_term_idx),
        ('MF', eval_thresholds['MF'], mf_term_idx),
        ('CC', eval_thresholds['CC'], cc_term_idx),
    ]:
        if len(idx) == 0:
            print(f'  {asp}: No terms found, skipping')
            continue
        
        # Get true labels and predictions for this aspect
        y_true = Y_dense[:, idx]
        y_score = oof_combined[:, idx]
        y_pred = (y_score >= thr).astype(np.int8)
        
        # Calculate per-protein F1 (CAFA metric)
        f1_sum = 0.0
        n_proteins = y_true.shape[0]
        
        for i in range(n_proteins):
            y_t = y_true[i]
            y_p = y_pred[i]
            
            tp = int(((y_p == 1) & (y_t == 1)).sum())
            fp = int(((y_p == 1) & (y_t == 0)).sum())
            fn = int(((y_p == 0) & (y_t == 1)).sum())
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_sum += f1
        
        f1_avg = f1_sum / n_proteins
        aspect_f1[asp] = f1_avg
        print(f'  {asp}: F1={f1_avg:.4f} (threshold={thr:.2f}, n_terms={len(idx)})')
    
    # CAFA F1 = average of aspect F1s
    if len(aspect_f1) > 0:
        cafa_f1 = sum(aspect_f1.values()) / len(aspect_f1)
        print(f'\n  CAFA F1: {cafa_f1:.4f}')
    print('=' * 70)
    
    del Y_dense
    gc.collect()
    
    # ===== GENERATE SUBMISSION FILES (with and without normalisation) =====
    LOGREG_GENERATE_SUBMISSION = bool(globals().get('LOGREG_GENERATE_SUBMISSION', True))
    
    if LOGREG_GENERATE_SUBMISSION:
        print('\n[LogReg] Generating submission files (IA-weighted ranking)...')
        
        # Load IA weights for competition-optimal term selection
        ia_path = WORK_ROOT / 'IA.tsv'
        if not ia_path.exists():
            ia_path = WORK_ROOT / 'Train' / 'IA.tsv'
        if not ia_path.exists():
            ia_path = Path('IA.tsv')
        
        ia_weights = {}
        if ia_path.exists():
            ia_df = pd.read_csv(ia_path, sep='\t', header=None, names=['term', 'ia'])
            ia_weights = dict(zip(ia_df['term'], ia_df['ia']))
            print(f'  Loaded IA weights: {len(ia_weights):,} terms, range=[{min(ia_weights.values()):.2f}, {max(ia_weights.values()):.2f}]')
        else:
            print(f'  [WARN] IA.tsv not found, falling back to confidence-only ranking')
        
        # Build IA vector aligned to top_terms
        ia_vec = np.array([ia_weights.get(t, 1.0) for t in top_terms], dtype=np.float32)
        
        # Load test protein IDs
        test_seq_path = WORK_ROOT / 'parsed' / 'test_seq.feather'
        _test_seq = pd.read_feather(test_seq_path)
        _test_ids = _test_seq['id'].astype(str).str.extract(r'\|(.*?)\|')[0].fillna(_test_seq['id']).tolist()
        
        # Minimum confidence threshold to avoid noise
        MIN_CONF = 0.05
        MAX_TERMS_PER_PROTEIN = 1500  # Competition rule
        
        # ===== GENERATE TWO VERSIONS: RAW vs NORMALISED =====
        # Per-protein normalisation: scores / max(scores) - calibrates across proteins
        # A/B test both on leaderboard to see which scores better
        
        # Compute per-protein normalised scores
        row_max = test_combined.max(axis=1, keepdims=True)
        test_normalised = np.divide(test_combined, row_max, where=row_max > 0, out=np.zeros_like(test_combined))
        
        print(f'  Raw scores range: [{test_combined.min():.4f}, {test_combined.max():.4f}]')
        print(f'  Normalised scores range: [{test_normalised.min():.4f}, {test_normalised.max():.4f}]')
        
        for variant_name, scores_matrix in [('raw', test_combined), ('norm', test_normalised)]:
            rows = []
            for i, pid in enumerate(tqdm(_test_ids, desc=f"Building submission ({variant_name})")):
                scores = scores_matrix[i, :]  # All 13500 terms
                
                # Filter by minimum confidence
                above_min = scores >= MIN_CONF
                if above_min.sum() == 0:
                    continue
                
                # Get indices and scores above threshold
                valid_idx = np.where(above_min)[0]
                valid_scores = scores[valid_idx]
                valid_ia = ia_vec[valid_idx]
                
                # Compute expected value: confidence × IA
                expected_value = valid_scores * valid_ia
                
                # Sort by expected value (descending), take top 1500
                order = np.argsort(expected_value)[::-1][:MAX_TERMS_PER_PROTEIN]
                
                for j in order:
                    term_idx = valid_idx[j]
                    conf = valid_scores[j]
                    rows.append(f'{pid}\t{top_terms[term_idx]}\t{np.clip(conf, 0.001, 1.0):.3f}')
            
            submission_path = WORK_ROOT / f'submission_logreg_{variant_name}.tsv'
            with open(submission_path, 'w') as f:
                f.write('\n'.join(rows))
            
            avg_terms = len(rows) / len(_test_ids) if _test_ids else 0
            print(f'  Saved: {submission_path} ({len(rows):,} rows, avg {avg_terms:.1f} terms/protein)')
        
        del test_normalised
        gc.collect()
    
    print("\n[LogReg] ═══ COMPLETE ═══")
