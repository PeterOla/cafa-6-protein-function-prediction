import numpy as np
import sys
from pathlib import Path

def verify_embeddings(path):
    print(f"Loading {path}...")
    try:
        data = np.load(path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")

    if np.isnan(data).any():
        print("❌ FAILURE: NaNs detected in embeddings!")
        nan_count = np.isnan(data).sum()
        print(f"Count of NaNs: {nan_count}")
    else:
        print("✅ SUCCESS: No NaNs detected.")

    if np.isinf(data).any():
        print("❌ FAILURE: Infinite values detected in embeddings!")
        inf_count = np.isinf(data).sum()
        print(f"Count of Infs: {inf_count}")
    else:
        print("✅ SUCCESS: No Infinite values detected.")

    print(f"Min: {np.min(data)}")
    print(f"Max: {np.max(data)}")
    print(f"Mean: {np.mean(data)}")
    print(f"Std: {np.std(data)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_ankh_nans.py <path_to_npy>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    verify_embeddings(file_path)
