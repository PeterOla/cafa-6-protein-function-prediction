import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

# Add src to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR / 'src'))

from data.loaders import LabelLoader, SequenceLoader
from data.datasets import ProteinEmbeddingDataset
from models.architecture import EmbeddingMLP
from training.loss import AsymmetricLoss
from evaluation.metrics import calculate_f1_score

def train_mlp():
    # Config
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 20
    HIDDEN_DIMS = [512, 256]
    DROPOUT = 0.3
    TOP_N_TERMS = 5000 # Increased to 5000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    TRAIN_TERMS = BASE_DIR / 'Train/train_terms.tsv'
    EMBED_CACHE_DIR = BASE_DIR / 'data/cache'
    MODELS_DIR = BASE_DIR / 'models'
    MODELS_DIR.mkdir(exist_ok=True)
    
    # 1. Load Data & Embeddings
    print("Loading Data...")
    label_loader = LabelLoader(TRAIN_TERMS)
    
    TRAIN_SEQ = BASE_DIR / 'Train/train_sequences.fasta'
    seq_loader = SequenceLoader(TRAIN_SEQ)
    common_ids = list(set(label_loader.protein_to_terms.keys()) & set(seq_loader.get_all_ids()))
    common_ids.sort()
    
    # Split IDs same as before
    train_ids, val_ids = train_test_split(common_ids, test_size=0.2, random_state=42)
    
    # Load embeddings
    print("Loading embeddings from cache...")
    train_embeds = np.load(EMBED_CACHE_DIR / "train_embeddings.npy")
    val_embeds = np.load(EMBED_CACHE_DIR / "val_embeddings.npy")
    
    print(f"Train embeddings: {train_embeds.shape}")
    print(f"Val embeddings: {val_embeds.shape}")
    
    # 2. Define Vocabulary (Terms)
    print(f"Selecting top {TOP_N_TERMS} terms...")
    all_train_terms = []
    for pid in train_ids:
        all_train_terms.extend(label_loader.get_terms(pid))
        
    term_counts = Counter(all_train_terms)
    most_common = term_counts.most_common(TOP_N_TERMS)
    terms_list = [t for t, c in most_common]
    print(f"Selected {len(terms_list)} target terms.")
    
    # Save terms list for inference later
    np.save(MODELS_DIR / "terms_list_expert.npy", terms_list)
    
    # 3. Create Datasets & Loaders
    train_dataset = ProteinEmbeddingDataset(train_embeds, train_ids, label_loader, terms_list)
    val_dataset = ProteinEmbeddingDataset(val_embeds, val_ids, label_loader, terms_list)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Model Setup
    input_dim = train_embeds.shape[1]
    model = EmbeddingMLP(input_dim, len(terms_list), hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    model = model.to(device)
    
    # Use Asymmetric Loss
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 5. Training Loop
    best_val_loss = float('inf')
    
    print("Starting training (Expert Mode)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for embeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODELS_DIR / "mlp_expert.pth")
            print("Saved best model.")
            
    print("Training complete.")

if __name__ == "__main__":
    train_mlp()
