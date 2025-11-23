# CAFA-6 Notebooks

Progressive series of notebooks for protein function prediction on Kaggle.

## 📚 Notebook Sequence

Run notebooks in order for complete pipeline:

### 01_baseline_frequency.ipynb
**Frequency Baseline** - Simplest baseline predicting GO terms by training frequency
- No sequence information used
- Predicts most common terms for all proteins
- **Expected F1:** ~0.14
- **Runtime:** 2-3 minutes

### 02_baseline_knn.ipynb
**K-Nearest Neighbours** - ESM-2 embeddings + similarity-based annotation transfer
- Generates embeddings for all sequences
- Finds k=10 nearest neighbours by cosine similarity
- Transfers annotations weighted by similarity
- **Expected F1:** ~0.18
- **Runtime:** 15-20 minutes (with GPU)

### 03_model_esm_finetuned.ipynb
**ESM-2 Fine-Tuned Classifier** - Train neural network on embeddings
- Fine-tunes ESM-2 8M with classification head
- Multi-label classification for 5000 GO terms
- Asymmetric loss and early stopping
- **Expected F1:** ~0.23
- **Runtime:** 2-3 hours (with GPU, 10 epochs)

### 04_label_propagation.ipynb
**Label Propagation** - Graph-based consistency improvement
- Uses GO ontology DAG structure
- Propagates predictions to ancestor terms
- Tests multiple propagation strategies
- **Expected F1:** ~0.25-0.27 (base + propagation)
- **Runtime:** 10-15 minutes
- **Prerequisites:** Trained model from notebook 03

## 🔧 Environment Configuration

Each notebook has an `ENVIRONMENT` variable at the top:

```python
ENVIRONMENT = 'local'  # Change to 'kaggle' when running on Kaggle
```

**Local:** Uses `Path.cwd().parent` (project root)  
**Kaggle:** Uses `/kaggle/input/cafa-6-dataset`

## 📊 Expected Performance Progression

| Model | F1 Score | Improvement |
|-------|----------|-------------|
| Frequency Baseline | 0.1412 | baseline |
| KNN (k=10) | 0.1776 | +0.0364 |
| ESM-2 Fine-tuned | 0.2331 | +0.0555 |
| + Label Propagation | 0.25-0.27 | +0.02-0.04 |

## 🎯 Target Metric

**Goal:** F1 ≥ 0.25 (IA-weighted F1 score)

## 📦 Required Data Files

Notebooks expect this data structure:

```
base_dir/
├── Train/
│   ├── train_sequences.fasta      # 82k protein sequences
│   ├── train_terms.tsv            # GO annotations
│   ├── train_taxonomy.tsv         # Taxonomy info
│   └── go-basic.obo               # GO ontology graph
├── Test/
│   ├── testsuperset.fasta         # Test sequences
│   └── testsuperset-taxon-list.tsv
└── IA.tsv                         # Information Accretion weights
```

## 🚀 Quick Start

### Run Locally
1. Ensure Python environment has required packages
2. Keep `ENVIRONMENT = 'local'`
3. Run notebooks sequentially (01 → 02 → 03 → 04)

### Run on Kaggle
1. Upload dataset as Kaggle dataset
2. Change each notebook: `ENVIRONMENT = 'kaggle'`
3. Run notebooks sequentially
4. Note: Notebook 04 requires saved model from notebook 03

## 📝 Notes

- **GPU recommended** for notebooks 02 and 03 (ESM-2 embeddings + training)
- **Notebook 03** saves model to `models/esm_finetuned/best_model/`
- **Notebook 04** loads model from that path (or uses dummy predictions as fallback)
- All notebooks are **self-contained** (no `src` imports needed)
- Results saved to CSV files in notebook directory

## 🔄 Re-running Notebooks

Each notebook can be re-run independently:
- **01-02:** Always safe to re-run (no dependencies)
- **03:** Overwrites previous model checkpoints
- **04:** Can run standalone even without trained model (uses fallback)

## 🐛 Troubleshooting

**"Module not found" error:**  
- Run first cell (`%pip install ...`) before other cells

**"File not found" error:**  
- Check `ENVIRONMENT` variable matches your setup
- Verify data files exist at expected paths

**"CUDA out of memory":**  
- Reduce `BATCH_SIZE` in notebooks 02-03
- Use CPU (slower but functional)

**Model F1 lower than expected:**  
- Check validation split matches (80/20, seed=42)
- Verify vocabulary size (5000 terms, min_count=10)
- Ensure sufficient training epochs (monitor early stopping)
