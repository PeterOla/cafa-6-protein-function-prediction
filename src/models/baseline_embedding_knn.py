import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Add src to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR / 'src'))

from data.loaders import LabelLoader, SequenceLoader
from evaluation.metrics import calculate_f1_score, calculate_precision_recall

# Check for transformers
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Installing transformers...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import AutoTokenizer, AutoModel

class EmbeddingGenerator:
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", batch_size: int = 32, device: str = None):
        self.model_name = model_name
        self.batch_size = batch_size
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def generate(self, sequences: List[str], desc: str = "Generating embeddings") -> np.ndarray:
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), self.batch_size), desc=desc):
            batch_seqs = sequences[i : i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_seqs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Mean pooling (excluding padding)
            # attention_mask: (batch, seq_len) - 1 for token, 0 for pad
            # last_hidden_state: (batch, seq_len, hidden_dim)
            
            mask = inputs.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            masked_embeddings = outputs.last_hidden_state * mask
            
            # Sum over sequence length and divide by valid token count
            summed = torch.sum(masked_embeddings, dim=1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            
            mean_pooled = summed / counts
            embeddings.append(mean_pooled.cpu().numpy())
            
        return np.vstack(embeddings)

def run_embedding_knn():
    print("=== Running Embedding KNN Baseline (Neural BLAST) ===\n")
    
    # Paths
    TRAIN_TERMS = BASE_DIR / 'Train/train_terms.tsv'
    TRAIN_SEQ = BASE_DIR / 'Train/train_sequences.fasta'
    EMBED_CACHE_DIR = BASE_DIR / 'data/cache'
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("1. Loading Data...")
    label_loader = LabelLoader(TRAIN_TERMS)
    seq_loader = SequenceLoader(TRAIN_SEQ)
    
    # Filter proteins that have both sequence and labels
    common_ids = list(set(label_loader.protein_to_terms.keys()) & set(seq_loader.get_all_ids()))
    common_ids.sort() # Ensure deterministic order
    print(f"Proteins with both sequence and labels: {len(common_ids)}")
    
    # 2. Split Data
    print("\n2. Splitting Data (80/20)...")
    train_ids, val_ids = train_test_split(common_ids, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_ids)}")
    print(f"Val size:   {len(val_ids)}")
    
    # 3. Generate/Load Embeddings
    print("\n3. Handling Embeddings...")
    embed_gen = EmbeddingGenerator()
    
    # Helper to get/cache embeddings
    def get_embeddings(ids, name):
        cache_path = EMBED_CACHE_DIR / f"{name}_embeddings.npy"
        if cache_path.exists():
            print(f"Loading {name} embeddings from cache...")
            return np.load(cache_path)
        
        print(f"Generating {name} embeddings...")
        sequences = [seq_loader.get_sequence(pid) for pid in ids]
        embeddings = embed_gen.generate(sequences, desc=f"Embedding {name}")
        np.save(cache_path, embeddings)
        return embeddings

    train_embeddings = get_embeddings(train_ids, "train")
    val_embeddings = get_embeddings(val_ids, "val")
    
    # 4. Train KNN
    print("\n4. Training KNN (Building Index)...")
    # We use cosine distance for embeddings usually, but Euclidean is standard for KDTree.
    # Normalized embeddings + Euclidean is equivalent to Cosine.
    # Let's normalize first.
    
    from sklearn.preprocessing import normalize
    train_embeddings_norm = normalize(train_embeddings)
    val_embeddings_norm = normalize(val_embeddings)
    
    # K=5 is a reasonable starting point for "transfer from similar proteins"
    K = 10
    knn = NearestNeighbors(n_neighbors=K, metric='euclidean', n_jobs=-1)
    knn.fit(train_embeddings_norm)
    
    # 5. Predict
    print(f"\n5. Predicting (Finding {K} nearest neighbors)...")
    distances, indices = knn.kneighbors(val_embeddings_norm)
    
    print("Transferring labels...")
    val_pred_terms = []
    val_true_terms = [label_loader.get_terms(pid) for pid in val_ids]
    
    # For each validation protein
    for i in tqdm(range(len(val_ids)), desc="Predicting"):
        neighbor_indices = indices[i]
        neighbor_dists = distances[i]
        
        # Collect terms from neighbors
        term_scores = {}
        
        for neighbor_idx, dist in zip(neighbor_indices, neighbor_dists):
            neighbor_id = train_ids[neighbor_idx]
            neighbor_terms = label_loader.get_terms(neighbor_id)
            
            # Weight: Simple frequency (1.0) or distance-weighted (1 / (1 + dist))
            # Let's use distance weight
            weight = 1.0 / (1.0 + dist)
            
            for term in neighbor_terms:
                term_scores[term] = term_scores.get(term, 0.0) + weight
                
        # Select top terms
        # Strategy: Normalize scores by sum of weights (soft voting)
        # Or just take top N highest scoring terms
        
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Dynamic thresholding is hard, let's pick top 20 (similar to frequency baseline)
        # Or we can pick all terms that appear in at least > 30% of neighbors (weighted)
        
        # Let's stick to Top-N for direct comparison with Frequency Baseline
        top_n = 20
        pred_set = set([t for t, s in sorted_terms[:top_n]])
        val_pred_terms.append(pred_set)
        
    # 6. Evaluate
    print("\n6. Evaluating...")
    f1 = calculate_f1_score(val_true_terms, val_pred_terms)
    precision, recall = calculate_precision_recall(val_true_terms, val_pred_terms)
    
    print(f"KNN (K={K}, Top-20) -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return f1

if __name__ == "__main__":
    run_embedding_knn()
