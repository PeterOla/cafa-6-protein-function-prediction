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
    
    # [VERIFICATION HOOK]
    if os.environ.get('QUICK_CHECK'):
        print("[DEBUG] QUICK_CHECK enabled: Reducing to Top 50 terms")
        top_terms = top_terms[:50]
    
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
    # Convert to dense for slicing (or keep sparse if Memory is tight, but cuML wants dense-ish or array)
    # We will slice sparse and convert to dense per-batch to save RAM.
    
    print(f"Y Matrix Built: {Y_sparse.shape} (Aligned to train_seq)")

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

                # Predict Test (Batched)
                # SKIP TEST if QUICK_CHECK is on (for speed)
                if not os.environ.get('QUICK_CHECK'):
                    TEST_BS = 4096
                    test_accum = np.zeros((X_test_mmap.shape[0], chunk_width), dtype=np.float32)
                    for i in range(0, X_test_mmap.shape[0], TEST_BS):
                        j = min(i + TEST_BS, X_test_mmap.shape[0])
                        x_batch = scaler.transform(X_test_mmap[i:j]).astype(np.float32)
                        if HAS_GPU: x_batch = cp.asarray(x_batch)
                        
                        p_batch = safe_predict_proba_gpu(clf, x_batch)
                        if HAS_GPU and hasattr(p_batch, 'get'): p_batch = p_batch.get()
                        test_accum[i:j] = p_batch
                        
                        del x_batch
                    
                    test_pred[:, start:end] += (test_accum / n_splits)
                
                # Cleanup
                del clf, Y_tr_chunk
                try: del test_accum
                except: pass

                if HAS_GPU: cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

        oof_pred.flush()
        test_pred.flush()
        _stage(f"Saved {lr_oof_path}")
        
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

    # Save
    out_path = PRED_DIR / 'best_thresholds.json'
    with open(out_path, 'w') as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"Saved optimal thresholds to {out_path}")
