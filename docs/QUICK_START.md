# 🚀 Quick Start Guide

## 📖 Files You Should Read (In Order)

1. **EXPLAINER.md** ← Start here! (5 min read)
   - Simple explanation of the project
   - What you're building
   - Why it matters

2. **PLAN.md** ← Your roadmap (15 min read)
   - Detailed step-by-step plan
   - Code templates
   - Checkboxes to track progress

3. **docs/overview.md** ← Competition details
   - Official problem description
   - Evaluation metrics

4. **docs/dataset_description.md** ← Data details
   - What each file contains
   - File formats

---

## ⚡ Quick Setup (Copy-Paste)

### 1. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn
pip install biopython networkx obonet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install jupyter notebook
```

### 3. Create Project Structure
```powershell
mkdir notebooks, src, experiments, submissions, logs
mkdir src\data, src\models, src\evaluation, src\utils
```

### 4. Launch Jupyter
```powershell
jupyter notebook
```

---

## 🧩 Optional: Generate additional multimodal embeddings (Local / Colab Pro)

`notebooks/CAFA6_Rank1_Solution.ipynb` (Phase 2) will automatically use any of these files if present under `artefacts_local/artefacts/features/`:
- `train_embeds_esm2_3b.npy` / `test_embeds_esm2_3b.npy`
- `train_embeds_ankh.npy` / `test_embeds_ankh.npy`
- `train_embeds_text.npy` / `test_embeds_text.npy` (requires your own `EntryID -> text` source)

### Generate on Colab (recommended hardware: A100)
Pre-req: run Phase 1 parsing first so you have `artefacts_local/artefacts/parsed/train_seq.feather` and `test_seq.feather`.

```powershell
pip install -r requirements.txt

# ESM2-3B (very large; start with batch size 1)
python scripts\02_generate_optional_embeddings.py --mode esm2_3b --batch-size 1 --max-len 1024

# Ankh (often needs trust_remote_code)
python scripts\02_generate_optional_embeddings.py --mode ankh --trust-remote-code --batch-size 2 --max-len 1024

# Text (TF-IDF; requires your own EntryID -> text TSV/CSV)
# Expected columns by default: EntryID, text
python scripts\02_generate_optional_embeddings.py --mode text --device cpu --text-path path\to\entryid_text.tsv --text-dim 10279 --text-dtype float16
```

### Bring back into Kaggle
Upload the resulting `.npy` files as a Kaggle Dataset, mount it, and copy them into `ARTEFACTS_DIR / 'features'`.

---

## 📂 File Structure

```
project/
├── EXPLAINER.md           ← Read first!
├── PLAN.md                ← Your detailed roadmap
├── QUICK_START.md         ← This file
│
├── data/                  ← Put raw data here
│   ├── Train/
│   ├── Test/
│   └── ...
│
├── notebooks/             ← Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_deep_learning.ipynb
│
├── src/                   ← Python code
│   ├── data/              ← Data loading & processing
│   ├── models/            ← Model definitions
│   ├── evaluation/        ← Metrics & evaluation
│   └── utils/             ← Helper functions
│
├── experiments/           ← Saved models & logs
└── submissions/           ← Final predictions
```

---

## 🎯 First Steps (Next 30 Minutes)

### Step 1: Verify Data (5 min)
```powershell
# Check if all files exist
ls Train/
ls Test/
```

**Expected files:**
- `train_sequences.fasta`
- `train_terms.tsv`
- `train_taxonomy.tsv`
- `go-basic.obo`
- `testsuperset.fasta`
- `IA.tsv`
- `sample_submission.tsv`

### Step 2: Create First Notebook (10 min)
```python
# In Jupyter: notebooks/01_eda.ipynb

import pandas as pd
from Bio import SeqIO

# Load training labels
labels = pd.read_csv('../Train/train_terms.tsv', sep='\t')
print(f"Total annotations: {len(labels)}")
print(f"Unique proteins: {labels['EntryID'].nunique()}")
print(f"Unique GO terms: {labels['term'].nunique()}")

# Load sequences
sequences = []
for record in SeqIO.parse('../Train/train_sequences.fasta', 'fasta'):
    sequences.append({
        'id': record.id,
        'length': len(record.seq)
    })

seq_df = pd.DataFrame(sequences)
print(f"\nTotal sequences: {len(seq_df)}")
print(f"Avg length: {seq_df['length'].mean():.0f}")
```

### Step 3: Visualize Data (15 min)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot sequence length distribution
plt.figure(figsize=(10, 5))
plt.hist(seq_df['length'], bins=50)
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.title('Distribution of Protein Sequence Lengths')
plt.show()

# Plot terms per protein
terms_per_protein = labels.groupby('EntryID').size()
plt.figure(figsize=(10, 5))
plt.hist(terms_per_protein, bins=50)
plt.xlabel('Number of GO Terms')
plt.ylabel('Number of Proteins')
plt.title('GO Terms per Protein')
plt.show()

# Ontology distribution
ontology_counts = labels['aspect'].value_counts()
print("\nOntology distribution:")
print(ontology_counts)
```

---

## 🔑 Key Concepts (Quick Reference)

### What You're Predicting
For each test protein → Predict GO terms with confidence scores

**Example:**
```
Protein_XYZ    GO:0003677    0.95    (DNA binding)
Protein_XYZ    GO:0005634    0.87    (Located in nucleus)
Protein_XYZ    GO:0006281    0.73    (DNA repair process)
```

### The 3 Ontologies
- **MFO** (Molecular Function): What it does (e.g., "binds DNA")
- **BPO** (Biological Process): Which process (e.g., "DNA repair")
- **CCO** (Cellular Component): Where it is (e.g., "nucleus")

### Success Metric
**Weighted F1 Score** = Balance between precision & recall, weighted by term rarity

---

## 📊 Expected Performance

| Stage | Model | Expected F1 |
|-------|-------|------------|
| Week 1 | Frequency baseline | 0.15 - 0.20 |
| Week 2 | BLAST baseline | 0.30 - 0.35 |
| Week 3 | K-mer + ML | 0.25 - 0.30 |
| Week 4 | Deep Learning (CNN) | 0.40 - 0.45 |
| Week 5 | Pre-trained (ProtBERT) | 0.50 - 0.60 |
| Week 6 | Ensemble | 0.55 - 0.65+ |

---

## 🆘 Need Help?

### Stuck on Setup?
- Check Python version (need 3.8+)
- Make sure pip is updated: `python -m pip install --upgrade pip`
- Try conda instead of venv: `conda create -n cafa python=3.9`

### Can't Load Data?
- Verify file paths are correct
- Check file encoding (should be UTF-8)
- Try reading first few lines manually

### Model Not Training?
- Check GPU availability: `torch.cuda.is_available()`
- Reduce batch size if OOM error
- Lower learning rate if loss explodes

### Low Scores?
- Check label propagation (add parent terms)
- Try different thresholds
- Ensemble multiple models

---

## 🎮 Challenge Mode (For Experienced Users)

Skip baselines and go straight to:
1. **Load ProtBERT** (Week 4)
2. **Fine-tune on CAFA data** (Week 5)
3. **Submit & iterate** (Week 6)

---

## 📅 Recommended Timeline

| Day | Task | Hours |
|-----|------|-------|
| Day 1 | Setup + EDA | 2-3 |
| Day 2-3 | Data processing | 3-4 |
| Day 4-5 | Baseline models | 4-6 |
| Day 6-10 | Deep learning | 10-15 |
| Day 11-12 | Optimization | 4-6 |
| Day 13-14 | Submission | 2-3 |

**Total: 25-40 hours over 2 weeks**

---

## ✅ Daily Checklist Template

Use this for each day:

```
[ ] Morning: Review PLAN.md for today's tasks
[ ] Work: Complete 2-3 checkboxes from PLAN.md
[ ] Afternoon: Test code & commit to git
[ ] Evening: Note tomorrow's priorities
```

---

## 🎯 Your First Goal

**By End of Day 1:**
- [ ] Environment setup complete
- [ ] Data loaded successfully
- [ ] Created 3 visualizations
- [ ] Understood problem clearly

**Action:** Open `notebooks/01_eda.ipynb` and start exploring! 🚀

---

**Good luck! You've got this! 💪**
