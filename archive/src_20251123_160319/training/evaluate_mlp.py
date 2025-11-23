import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add src to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR / 'src'))

from data.loaders import LabelLoader, SequenceLoader
from models.architecture import EmbeddingMLP
from evaluation.metrics import calculate_f1_score, calculate_precision_recall

def evaluate_mlp():
    print("=== Evaluating MLP Model ===\n")
    
    # Config
    BATCH_SIZE = 32
    HIDDEN_DIMS = [512, 256]
    DROPOUT = 0.3
    THRESHOLD = 0.3 # Confidence threshold
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    TRAIN_TERMS = BASE_DIR / 'Train/train_terms.tsv'
    EMBED_CACHE_DIR = BASE_DIR / 'data/cache'
    MODELS_DIR = BASE_DIR / 'models'
    
    # 1. Load Data & Embeddings
    print("Loading Data...")
    label_loader = LabelLoader(TRAIN_TERMS)
    
    TRAIN_SEQ = BASE_DIR / 'Train/train_sequences.fasta'
    seq_loader = SequenceLoader(TRAIN_SEQ)
    common_ids = list(set(label_loader.protein_to_terms.keys()) & set(seq_loader.get_all_ids()))
    common_ids.sort()
    
    # Split IDs same as before
    _, val_ids = train_test_split(common_ids, test_size=0.2, random_state=42)
    
    # Load embeddings
    print("Loading embeddings from cache...")
    val_embeds = np.load(EMBED_CACHE_DIR / "val_embeddings.npy")
    
    # Load Terms List
    terms_list = np.load(MODELS_DIR / "terms_list_expert.npy")
    num_classes = len(terms_list)
    print(f"Model has {num_classes} output classes.")
    
    # 2. Load Model
    print("Loading Model...")
    input_dim = val_embeds.shape[1]
    model = EmbeddingMLP(input_dim, num_classes, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    model.load_state_dict(torch.load(MODELS_DIR / "mlp_expert.pth"))
    model = model.to(device)
    model.eval()
    
    # 3. Predict
    print("Predicting...")
    val_true_terms = [label_loader.get_terms(pid) for pid in val_ids]
    num_samples = len(val_ids)
    
    # 4. Evaluate with multiple thresholds
    print("\nEvaluating with multiple thresholds...")
    
    # Store all probs to avoid re-predicting
    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, BATCH_SIZE), desc="Collecting Probs"):
            batch_embeds = torch.from_numpy(val_embeds[i : i + BATCH_SIZE]).float().to(device)
            outputs = model(batch_embeds)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            
    all_probs = np.vstack(all_probs)
    
    best_f1 = 0.0
    best_t = 0.0
    
    for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        val_pred_terms = []
        for j in range(len(all_probs)):
            sample_probs = all_probs[j]
            indices = np.where(sample_probs > t)[0]
            pred_set = set([terms_list[idx] for idx in indices])
            val_pred_terms.append(pred_set)
            
        f1 = calculate_f1_score(val_true_terms, val_pred_terms)
        precision, recall = calculate_precision_recall(val_true_terms, val_pred_terms)
        print(f"Threshold {t:.2f} -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    print(f"\nBest Threshold: {best_t} with F1: {best_f1:.4f}")
    
    # 5. Evaluate Top-N (Fair comparison with Baselines)
    print("\nEvaluating Top-N (Ranking)...")
    for n in [10, 20, 50]:
        val_pred_terms = []
        for j in range(len(all_probs)):
            sample_probs = all_probs[j]
            # Get top N indices
            top_indices = np.argsort(sample_probs)[-n:]
            pred_set = set([terms_list[idx] for idx in top_indices])
            val_pred_terms.append(pred_set)
            
        f1 = calculate_f1_score(val_true_terms, val_pred_terms)
        print(f"Top-{n} -> F1: {f1:.4f}")

    return best_f1

if __name__ == "__main__":
    evaluate_mlp()
