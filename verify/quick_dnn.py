#!/usr/bin/env python3
"""
Quick DNN alignment verification using Cell 13a data loading approach.
Uses X_train_mmap.npy (full 7168 dims) + Y built from train_seq.feather ordering.

Expected F1: ~0.10-0.25 if aligned, ~0.003 if broken.
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import sparse

WORK_ROOT = Path('cafa6_data')
FEAT_DIR = WORK_ROOT / 'features'

# === PARAMETERS (small for quick local run) ===
N_PROTEINS = 5000      # Subset of proteins
N_TERMS = 500          # Subset of terms
TEST_SIZE = 0.2
EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3

print("=" * 60)
print("QUICK DNN ALIGNMENT VERIFICATION")
print("=" * 60)

# === Check for PyTorch ===
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    print(f"✓ PyTorch {torch.__version__}")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {DEVICE}")
except ImportError:
    print("ERROR: PyTorch not found!")
    sys.exit(1)

# === 1. Load canonical protein ordering (Cell 13a approach) ===
print("\n[1] Loading canonical protein ordering...")
train_seq = pd.read_feather(WORK_ROOT / 'parsed' / 'train_seq.feather')
id_col = 'id' if 'id' in train_seq.columns else 'EntryID'

def _clean_id(x):
    x = str(x)
    if '|' in x:
        parts = x.split('|')
        return parts[1] if len(parts) > 1 else x
    return x

train_ids = [_clean_id(x) for x in train_seq[id_col].tolist()]
n_train_full = len(train_ids)
print(f"  Full train set: {n_train_full:,} proteins")

# === 2. Load X_train_mmap (Cell 13a approach) ===
print("\n[2] Loading X_train_mmap.npy...")
X_path = FEAT_DIR / 'X_train_mmap.npy'
if not X_path.exists():
    print(f"  ERROR: {X_path} not found!")
    sys.exit(1)

X_full = np.load(X_path, mmap_mode='r')
print(f"  X shape: {X_full.shape}")
assert X_full.shape[0] == n_train_full, f"X rows ({X_full.shape[0]}) != n_train ({n_train_full})"

# === 3. Load term vocabulary ===
print("\n[3] Loading term vocabulary...")
import json
top_terms_path = FEAT_DIR / 'top_terms_13500.json'
top_terms = json.loads(top_terms_path.read_text())
term_to_idx = {t: i for i, t in enumerate(top_terms)}
print(f"  Vocabulary: {len(top_terms)} terms")

# === 4. Build Y matrix (Cell 13a approach - train_seq ordering) ===
print("\n[4] Building Y matrix (Cell 13a approach)...")
protein_to_idx = {pid: i for i, pid in enumerate(train_ids)}

train_terms_df = pd.read_csv(WORK_ROOT / 'Train' / 'train_terms.tsv', sep='\t')
print(f"  Train terms rows: {len(train_terms_df):,}")

rows, cols = [], []
for _, row in train_terms_df.iterrows():
    pid = _clean_id(str(row['EntryID']))
    term = str(row['term'])
    if pid in protein_to_idx and term in term_to_idx:
        rows.append(protein_to_idx[pid])
        cols.append(term_to_idx[term])

Y_sparse = sparse.csr_matrix(
    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
    shape=(n_train_full, len(top_terms))
)
Y_full = Y_sparse.toarray()
print(f"  Y shape: {Y_full.shape}")
print(f"  Positive labels: {Y_full.sum():,.0f}")

# === 5. Subset for quick run ===
print(f"\n[5] Subsetting to {N_PROTEINS} proteins, {N_TERMS} terms...")
np.random.seed(42)
prot_idx = np.random.choice(n_train_full, size=min(N_PROTEINS, n_train_full), replace=False)
prot_idx.sort()

# Take top N_TERMS by frequency
term_freq = Y_full.sum(axis=0)
term_idx = np.argsort(term_freq)[::-1][:N_TERMS]

X = X_full[prot_idx].copy().astype(np.float32)
Y = Y_full[np.ix_(prot_idx, term_idx)].astype(np.float32)
print(f"  X subset: {X.shape}")
print(f"  Y subset: {Y.shape}")
print(f"  Y positives: {Y.sum():,.0f}")

# === 6. Train/val split ===
print(f"\n[6] Train/val split ({1-TEST_SIZE:.0%}/{TEST_SIZE:.0%})...")
X_tr, X_val, Y_tr, Y_val = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=42
)
print(f"  Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}")

# === 7. Scale ===
print("\n[7] Scaling features...")
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)

# === 8. Define DNN ===
class SimpleDNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)

# === 9. Train DNN ===
print(f"\n[8] Training DNN ({EPOCHS} epochs)...")
t0 = time.time()

in_dim = X_tr_scaled.shape[1]
out_dim = Y_tr.shape[1]

model = SimpleDNN(in_dim, out_dim).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Create DataLoader
train_ds = TensorDataset(
    torch.from_numpy(X_tr_scaled),
    torch.from_numpy(Y_tr)
)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    
    for xb, yb in train_dl:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={epoch_loss/n_batches:.4f}")

train_time = time.time() - t0
print(f"  Training complete: {train_time:.1f}s")

# === 10. Predict on validation ===
print("\n[9] Predicting on validation set...")
model.eval()
with torch.no_grad():
    X_val_t = torch.from_numpy(X_val_scaled).to(DEVICE)
    logits = model(X_val_t)
    val_preds = torch.sigmoid(logits).cpu().numpy()

print(f"  Predictions shape: {val_preds.shape}")

# === 11. Evaluate F1 ===
print("\n[10] Evaluating F1 at various thresholds...")

def compute_f1(y_true, y_pred, threshold):
    y_bin = (y_pred >= threshold).astype(int)
    tp = ((y_bin == 1) & (y_true == 1)).sum()
    fp = ((y_bin == 1) & (y_true == 0)).sum()
    fn = ((y_bin == 0) & (y_true == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
print(f"\n{'Threshold':<12} {'F1':<10} {'Precision':<12} {'Recall':<10}")
print("-" * 44)

best_f1 = 0
best_thr = 0
for thr in thresholds:
    f1, prec, rec = compute_f1(Y_val, val_preds, thr)
    print(f"{thr:<12.2f} {f1:<10.4f} {prec:<12.4f} {rec:<10.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

# === 12. Verdict ===
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
print(f"  Best F1: {best_f1:.4f} (at threshold {best_thr})")

if best_f1 > 0.05:
    print(f"  ✅ ALIGNMENT LOOKS CORRECT")
    print(f"  F1 > 0.05 indicates X and Y rows correspond to same proteins")
else:
    print(f"  ❌ ALIGNMENT LIKELY BROKEN")
    print(f"  F1 < 0.05 suggests X/Y row mismatch (random predictions)")

print(f"\n  Prediction stats:")
print(f"    Mean: {val_preds.mean():.4f}")
print(f"    Max:  {val_preds.max():.4f}")
print(f"    >0.5: {(val_preds > 0.5).sum():,}")
