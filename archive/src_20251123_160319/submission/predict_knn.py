import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import argparse

# Add src to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR / 'src'))

from data.loaders import LabelLoader, SequenceLoader
from models.baseline_embedding_knn import EmbeddingGenerator

def generate_submission(limit: int = None):
    print("=== Generating Submission with Embedding KNN ===\n")
    
    # Paths
    TRAIN_TERMS = BASE_DIR / 'Train/train_terms.tsv'
    TRAIN_SEQ = BASE_DIR / 'Train/train_sequences.fasta'
    TEST_SEQ = BASE_DIR / 'Test/testsuperset.fasta'
    EMBED_CACHE_DIR = BASE_DIR / 'data/cache'
    SUBMISSION_DIR = BASE_DIR / 'submissions'
    SUBMISSION_DIR.mkdir(exist_ok=True)
    
    # 1. Load Data
    print("1. Loading Data...")
    label_loader = LabelLoader(TRAIN_TERMS)
    train_seq_loader = SequenceLoader(TRAIN_SEQ)
    test_seq_loader = SequenceLoader(TEST_SEQ)
    
    # Get IDs
    train_ids = list(set(label_loader.protein_to_terms.keys()) & set(train_seq_loader.get_all_ids()))
    train_ids.sort()
    
    test_ids = test_seq_loader.get_all_ids()
    test_ids.sort()
    
    if limit:
        print(f"LIMITING to {limit} sequences for testing...")
        test_ids = test_ids[:limit]
        # We still need enough training data to find neighbors
    
    print(f"Training sequences: {len(train_ids)}")
    print(f"Test sequences: {len(test_ids)}")
    
    # 2. Embeddings
    print("\n2. Handling Embeddings...")
    embed_gen = EmbeddingGenerator()
    
    def get_embeddings(ids, loader, name):
        cache_path = EMBED_CACHE_DIR / f"{name}_embeddings.npy"
        
        # If we are limiting, we might not want to load the full cache or save partial cache
        # But for simplicity, if full cache exists, load it and slice
        if cache_path.exists():
            print(f"Loading {name} embeddings from cache...")
            full_embeds = np.load(cache_path)
            if len(full_embeds) == len(ids):
                return full_embeds
            else:
                print(f"Cache size mismatch ({len(full_embeds)} vs {len(ids)}). Regenerating/Slicing...")
                # If cache is larger (e.g. full test set) and we want subset
                # This is tricky without an ID map. 
                # For now, let's just regenerate if mismatch, or assume the user manages cache.
                # actually, for the test set, if we limit, we should just generate on the fly
                pass

        print(f"Generating {name} embeddings...")
        sequences = [loader.get_sequence(pid) for pid in ids]
        embeddings = embed_gen.generate(sequences, desc=f"Embedding {name}")
        
        if not limit or name == 'train': # Only save if full set
            np.save(cache_path, embeddings)
            
        return embeddings

    # Train embeddings (Always full)
    train_embeddings = get_embeddings(train_ids, train_seq_loader, "train")
    
    # Test embeddings (Full or Limited)
    # Note: We name the cache 'test' only if it's the full set
    test_cache_name = "test" if not limit else f"test_limit_{limit}"
    test_embeddings = get_embeddings(test_ids, test_seq_loader, test_cache_name)
    
    # 3. Train KNN
    print("\n3. Training KNN...")
    train_embeddings_norm = normalize(train_embeddings)
    test_embeddings_norm = normalize(test_embeddings)
    
    K = 10
    knn = NearestNeighbors(n_neighbors=K, metric='euclidean', n_jobs=-1)
    knn.fit(train_embeddings_norm)
    
    # 4. Predict
    print(f"\n4. Predicting...")
    distances, indices = knn.kneighbors(test_embeddings_norm)
    
    submission_rows = []
    
    for i in tqdm(range(len(test_ids)), desc="Generating Predictions"):
        test_id = test_ids[i]
        neighbor_indices = indices[i]
        neighbor_dists = distances[i]
        
        term_scores = {}
        
        for neighbor_idx, dist in zip(neighbor_indices, neighbor_dists):
            neighbor_id = train_ids[neighbor_idx]
            neighbor_terms = label_loader.get_terms(neighbor_id)
            
            # Weighting
            weight = 1.0 / (1.0 + dist)
            
            for term in neighbor_terms:
                term_scores[term] = term_scores.get(term, 0.0) + weight
        
        # Normalize scores to 0-1 range (roughly)
        # A simple way is to divide by K (sum of max weights approx K)
        # Or just output raw scores? CAFA expects probabilities.
        # Let's normalize by sum of weights of all neighbors
        total_weight = sum([1.0 / (1.0 + d) for d in neighbor_dists])
        
        # Top 50 terms
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:50]
        
        for term, score in sorted_terms:
            prob = score / total_weight
            # Round to 3 decimals
            prob = round(prob, 3)
            submission_rows.append({
                'EntryID': test_id,
                'term': term,
                'score': prob
            })
            
    # 5. Save
    print("\n5. Saving Submission...")
    df = pd.DataFrame(submission_rows)
    out_path = SUBMISSION_DIR / 'submission_knn.tsv'
    
    # CAFA format: No header? The sample has no header.
    # Columns: EntryID, term, score
    df.to_csv(out_path, sep='\t', index=False, header=False)
    print(f"Saved to {out_path}")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test sequences")
    args = parser.parse_args()
    
    generate_submission(limit=args.limit)
