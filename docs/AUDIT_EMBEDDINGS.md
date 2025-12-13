# Auditor Note — Embeddings (End-to-End Progress)

**Date:** 13 Dec 2025  
**Repo:** `cafa-6-protein-function-prediction`

## Summary (what exists today)

- **Optional embedding generator implemented**: `scripts/02_generate_optional_embeddings.py`
  - Reads exact row order from:
    - `artefacts_local/artefacts/parsed/train_seq.feather`
    - `artefacts_local/artefacts/parsed/test_seq.feather`
  - Writes `.npy` artefacts under: `artefacts_local/artefacts/features/`

### Modalities supported
- **ESM2‑3B** (`--mode esm2_3b`, model `facebook/esm2_t36_3B_UR50D`)
  - Outputs:
    - `train_embeds_esm2_3b.npy`
    - `test_embeds_esm2_3b.npy`

- **Ankh** (`--mode ankh`, model `ElnaggarLab/ankh-large`)
  - Outputs:
    - `train_embeds_ankh.npy`
    - `test_embeds_ankh.npy`

- **Text (10279D)** (`--mode text`) implemented as **TF‑IDF**
  - Requires an external text corpus file: `EntryID -> text` (TSV/CSV).
  - Outputs:
    - `train_embeds_text.npy`
    - `test_embeds_text.npy`
    - `text_vectorizer.joblib` (for reproducibility)
  - Engineering note:
    - We guarantee fixed width `--text-dim` by zero-padding when TF‑IDF vocabulary < target dimension.
    - We normalise UniProt IDs (e.g., `sp|ACC|NAME` ↔ `ACC`) when joining text to proteins.

## Why text helps (strategic rationale)

- Sequence embeddings capture patterns in amino-acid space; text captures *human-curated knowledge* (function phrases, processes, compartments, pathway vocabulary).
- This can help long-tail GO terms and disambiguate near-homologous proteins.

## Practical constraints

- Dense TF‑IDF at 10279D is large:
  - Train: `82404 × 10279 × 2 bytes ≈ 1.7 GB` (float16)
  - Test: `224309 × 10279 × 2 bytes ≈ 4.6 GB` (float16)
- Therefore embeddings are generated **offline (Colab/local)** and uploaded to Kaggle as a Dataset.

## End-to-end generation (copy-paste)

### Prerequisite
Run Phase 1 parsing first so these exist:
- `artefacts_local/artefacts/parsed/train_seq.feather`
- `artefacts_local/artefacts/parsed/test_seq.feather`

### Colab (recommended)
```python
%cd /content/<your_repo>
!pip -q install -r requirements.txt

# (Optional) Build EntryID -> text corpus (UniProt + PubMed abstracts)
# This writes: artefacts_local/artefacts/external/entryid_text.tsv
!python scripts/03_build_entryid_text_from_uniprot_pubmed.py \
  --email your_email@example.com \
  --strip-go

# ESM2-3B
!python scripts/02_generate_optional_embeddings.py --mode esm2_3b --batch-size 1 --max-len 1024

# Ankh
!python scripts/02_generate_optional_embeddings.py --mode ankh --trust-remote-code --batch-size 2 --max-len 1024

# Text TF-IDF (requires entryid_text.tsv)
!python scripts/02_generate_optional_embeddings.py --mode text --device cpu --text-path /content/entryid_text.tsv --text-dim 10279 --text-dtype float16
```

### Kaggle consumption snippet
```python
import shutil
from pathlib import Path

ARTEFACTS_DIR = Path('artefacts')  # adjust to your notebook config
FEATURES = ARTEFACTS_DIR / 'features'
FEATURES.mkdir(parents=True, exist_ok=True)

src = Path('/kaggle/input/<your-embeddings-dataset>')
for name in [
    'train_embeds_esm2_3b.npy','test_embeds_esm2_3b.npy',
    'train_embeds_ankh.npy','test_embeds_ankh.npy',
    'train_embeds_text.npy','test_embeds_text.npy','text_vectorizer.joblib'
]:
    p = src / name
    if p.exists():
        shutil.copy2(p, FEATURES / name)
```

## UniProt + PubMed corpus builder (implemented)

Script: `scripts/03_build_entryid_text_from_uniprot_pubmed.py`

What it does:
- Reads CAFA protein IDs from `artefacts_local/artefacts/parsed/*_seq.feather`.
- Normalises UniProt IDs (`sp|ACC|NAME` → `ACC`).
- Fetches UniProt text fields + PubMed IDs via UniProt REST (`lit_pubmed_id`, `cc_function`, etc.).
- Fetches PubMed abstracts via NCBI E-utilities (`efetch`).
- Writes `artefacts_local/artefacts/external/entryid_text.tsv` for `--mode text`.

Operational notes:
- Uses on-disk caches under `artefacts_local/artefacts/external/uniprot_pubmed_cache/` so runs can resume.
- Use `--max-ids` for a fast dry run; expect full runs to be slow at CAFA scale.
