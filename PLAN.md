# 🎯 CAFA-6 Protein Function Prediction - DETAILED BUILD PLAN

## 📋 Table of Contents
1. [Project Setup](#1-project-setup)
2. [Data Exploration & Understanding](#2-data-exploration--understanding)
3. [Data Processing Pipeline](#3-data-processing-pipeline)
4. [Baseline Model](#4-baseline-model)
5. [Advanced Models](#5-advanced-models)
6. [Evaluation System](#6-evaluation-system)
7. [Submission Pipeline](#7-submission-pipeline)
8. [Optimization & Iteration](#8-optimization--iteration)

---

## 1. Project Setup

### ✅ Environment Setup
- [ ] Create Python virtual environment
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- [ ] Install core dependencies
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn
  pip install biopython networkx obonet
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install transformers datasets
  pip install jupyter notebook
  ```

- [ ] Create project folder structure
  ```
  project/
  ├── Train/             # Training data (sequences, labels, ontology)
  ├── Test/              # Test sequences and taxon list
  ├── notebooks/         # Jupyter notebooks for exploration
  ├── src/               # Source code
  │   ├── data/          # Data processing scripts
  │   ├── models/        # Model definitions
  │   ├── evaluation/    # Evaluation scripts
  │   └── utils/         # Helper functions
  ├── experiments/       # Model experiments
  ├── submissions/       # Submission files
  └── logs/              # Training logs
  ```

- [ ] Initialize git repository
  ```bash
  git init
  git add .
  git commit -m "Initial project setup"
  ```

### ✅ Quick Data Check
- [ ] Verify all data files are present
- [ ] Check file sizes (make sure nothing corrupted)
- [ ] Count lines in key files
  ```powershell
  Get-Content Train/train_sequences.fasta | Measure-Object -Line
  Get-Content Train/train_terms.tsv | Measure-Object -Line
  ```

---

## 2. Data Exploration & Understanding

### 📊 Exploratory Data Analysis (EDA)

#### Step 2.1: Understand GO Ontology
- [ ] Parse `go-basic.obo` file
  - Load with `obonet` library
  - Identify root nodes for BPO, MFO, CCO
  - Visualize small subgraph (first 50 nodes)
  
- [ ] Analyze GO hierarchy structure
  - Count total GO terms
  - Calculate depth distribution
  - Find most common parent-child relationships

**Code Template:**
```python
import obonet
import networkx as nx

# Load ontology
graph = obonet.read_obo('Train/go-basic.obo')

# Print basic stats
print(f"Total GO terms: {len(graph)}")
print(f"Total relationships: {graph.number_of_edges()}")

# Root nodes
roots = {'BPO': 'GO:0008150', 'CCO': 'GO:0005575', 'MFO': 'GO:0003674'}
```

#### Step 2.2: Analyze Training Sequences
- [ ] Load `train_sequences.fasta`
  - Count total proteins
  - Calculate sequence length distribution
  - Find min/max/average length
  - Plot histogram of lengths

- [ ] Check sequence composition
  - Count amino acid frequency
  - Identify unusual characters (should be 20 standard amino acids)

**Code Template:**
```python
from Bio import SeqIO

# Parse FASTA
sequences = []
for record in SeqIO.parse('Train/train_sequences.fasta', 'fasta'):
    sequences.append({
        'id': record.id,
        'length': len(record.seq),
        'sequence': str(record.seq)
    })

print(f"Total proteins: {len(sequences)}")
```

#### Step 2.3: Analyze Training Labels
- [ ] Load `train_terms.tsv`
  - Count unique proteins
  - Count unique GO terms
  - Count total annotations

- [ ] Label distribution analysis
  - Terms per protein (how many functions each has)
  - Proteins per term (how common each function is)
  - Distribution across BPO/MFO/CCO

- [ ] Create visualizations
  - Bar chart: Top 20 most common GO terms
  - Histogram: Number of annotations per protein
  - Pie chart: Distribution across 3 subontologies

**Code Template:**
```python
import pandas as pd

# Load labels
labels_df = pd.read_csv('Train/train_terms.tsv', sep='\t')

print(f"Unique proteins: {labels_df['EntryID'].nunique()}")
print(f"Unique GO terms: {labels_df['term'].nunique()}")
print(f"Total annotations: {len(labels_df)}")

# Terms per protein
terms_per_protein = labels_df.groupby('EntryID').size()
print(f"Avg terms per protein: {terms_per_protein.mean():.2f}")
```

#### Step 2.4: Information Accretion Analysis
- [ ] Load `IA.tsv`
  - Understand weight distribution
  - Find highest/lowest IA values
  - Correlate IA with term frequency

- [ ] Visualize IA distribution
  - Histogram of IA values
  - Scatter: IA vs. term frequency

#### Step 2.5: Test Data Analysis
- [ ] Load `testsuperset.fasta`
  - Count test proteins
  - Compare length distribution with training
  - Check for overlapping IDs

**Key Questions to Answer:**
- How many proteins? How many GO terms?
- Are labels balanced or imbalanced?
- What's the average protein length?
- Which functions are most common?

---

## 3. Data Processing Pipeline

### 🔧 Build Data Preprocessing System

#### Step 3.1: Create Data Loaders
- [ ] **SequenceLoader class**
  - Load FASTA files
  - Clean sequences (remove special chars)
  - Store in efficient format (pandas DataFrame)

- [ ] **LabelLoader class**
  - Load TSV labels
  - Create protein-to-terms mapping
  - Create term-to-proteins mapping
  - Handle multi-label format

- [ ] **OntologyLoader class**
  - Load GO graph
  - Extract ancestors/descendants for each term
  - Implement propagation (child → parent)

**Code Structure:**
```python
class SequenceLoader:
    def load_fasta(self, filepath):
        # Parse FASTA and return DataFrame
        pass
    
    def get_sequence(self, protein_id):
        # Return sequence for protein_id
        pass

class LabelLoader:
    def load_labels(self, filepath):
        # Load TSV and create mappings
        pass
    
    def get_labels_for_protein(self, protein_id):
        # Return all GO terms for protein
        pass
    
    def propagate_labels(self, go_graph):
        # Add ancestor terms to each protein
        pass
```

#### Step 3.2: Feature Engineering
- [ ] **Basic Sequence Features**
  - Sequence length
  - Amino acid composition (20 features)
  - Di-peptide frequency (400 features)
  - Sequence entropy

- [ ] **K-mer Features**
  - Extract k-mers (k=3, 4, 5)
  - Count k-mer frequencies
  - Normalize by sequence length

- [ ] **Physicochemical Properties**
  - Hydrophobicity
  - Charge
  - Molecular weight
  - Isoelectric point

**Code Template:**
```python
def calculate_amino_acid_composition(sequence):
    """Calculate frequency of each amino acid"""
    aa_counts = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    for aa in sequence:
        if aa in aa_counts:
            aa_counts[aa] += 1
    
    # Normalize by length
    total = len(sequence)
    aa_freq = {aa: count/total for aa, count in aa_counts.items()}
    return aa_freq
```

#### Step 3.3: Dataset Splitting
- [ ] Create train/validation split (80/20)
  - Random split
  - Stratified by term frequency (if possible)

- [ ] Save splits to disk
  ```python
  train_ids, val_ids = train_test_split(protein_ids, test_size=0.2)
  ```

- [ ] Create PyTorch DataLoaders
  - Custom Dataset class
  - Batch loading
  - Handle variable-length sequences

**Code Template:**
```python
from torch.utils.data import Dataset, DataLoader

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer=None):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            seq = self.tokenizer(seq)
        
        return seq, label
```

#### Step 3.4: Label Encoding
- [ ] Create binary label matrix
  - Rows = proteins
  - Columns = GO terms
  - Values = 0 or 1

- [ ] Handle label propagation
  - If protein has GO:123456, add all ancestors

- [ ] Save encoder mapping
  - term_to_index dictionary
  - index_to_term dictionary

**Code Template:**
```python
from sklearn.preprocessing import MultiLabelBinarizer

# Fit encoder
mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(protein_labels)

print(f"Label matrix shape: {label_matrix.shape}")
# Output: (num_proteins, num_go_terms)
```

---

## 4. Baseline Model

### 🚀 Build Simple Baseline First

#### Step 4.1: Baseline 1 - Frequency Baseline
- [ ] Count term frequencies in training data
- [ ] For each test protein, predict top-N most common terms
- [ ] Assign confidence scores based on frequency

**Logic:**
```
If GO:0005515 appears in 30% of training proteins:
  → Predict it for ALL test proteins with confidence 0.30
```

**Why This Works:**
- Some GO terms are very common (like "protein binding")
- Gives you a baseline score to beat

#### Step 4.2: Baseline 2 - BLAST-based Prediction
- [ ] Install BLAST locally
- [ ] Create BLAST database from training sequences
- [ ] For each test protein:
  - Find top-K similar proteins (by sequence similarity)
  - Transfer their GO terms
  - Weight by BLAST score

**Code Flow:**
```python
1. makeblastdb -in train_sequences.fasta -dbtype prot
2. For each test_protein:
     blastp -query test_protein -db train_db -outfmt 6
     Get top 10 hits
     Aggregate their GO terms
     Weight by E-value
```

#### Step 4.3: Baseline 3 - K-mer + Logistic Regression
- [ ] Extract k-mer features (3-mers, 4-mers)
- [ ] Train separate logistic regression for each GO term
- [ ] Predict probabilities for test set

**Code Template:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# Extract features
X_train = extract_kmers(train_sequences)  # Shape: (n_samples, n_features)
y_train = label_matrix  # Shape: (n_samples, n_go_terms)

# Train multi-label classifier
clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train)

# Predict
y_pred_proba = clf.predict_proba(X_test)
```

#### Step 4.4: Evaluate Baselines
- [ ] Calculate metrics for each baseline
  - Precision, Recall, F1 (per subontology)
  - Overall weighted F1

- [ ] Create comparison table
  ```
  Model                  | BPO F1 | MFO F1 | CCO F1 | Overall
  --------------------- |--------|--------|--------|--------
  Frequency Baseline     | 0.15   | 0.20   | 0.18   | 0.18
  BLAST Baseline         | 0.32   | 0.38   | 0.35   | 0.35
  K-mer + LogReg         | 0.28   | 0.34   | 0.30   | 0.31
  ```

**Goal:** Establish baseline score to improve upon

---

## 5. Advanced Models

### 🧠 Deep Learning Approaches

#### Step 5.1: Model Architecture Planning

**Option A: CNN-based Model**
- [ ] Design 1D CNN architecture
  - Conv1D layers to capture local patterns
  - Max pooling
  - Fully connected layers
  - Sigmoid output (multi-label)

**Option B: Transformer-based Model**
- [ ] Use pre-trained protein language model
  - ProtBERT, ESM-2, or ProtTrans
  - Fine-tune on your data
  - Add classification head

**Option C: Graph Neural Network**
- [ ] Incorporate GO graph structure
  - Encode GO hierarchy
  - Message passing between terms
  - Predict based on graph embeddings

#### Step 5.2: Start with CNN Model

**Architecture:**
```python
import torch.nn as nn

class ProteinCNN(nn.Module):
    def __init__(self, vocab_size=21, num_classes=10000):
        super().__init__()
        
        # Embedding layer (amino acid → vector)
        self.embedding = nn.Embedding(vocab_size, 128)
        
        # Conv layers
        self.conv1 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected
        self.fc1 = nn.Linear(1024, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, 128)
        x = x.transpose(1, 2)  # (batch, 128, seq_len)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = self.pool(x).squeeze(-1)  # (batch, 1024)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))  # (batch, num_classes)
        
        return x
```

#### Step 5.3: Training Setup
- [ ] Define loss function
  - Binary Cross-Entropy Loss (BCELoss)
  - Or BCEWithLogitsLoss for stability

- [ ] Choose optimizer
  - Adam (lr=0.001)
  - Or AdamW with weight decay

- [ ] Set up learning rate scheduler
  - ReduceLROnPlateau
  - Or CosineAnnealingLR

**Training Loop:**
```python
import torch
import torch.nn as nn
from torch.optim import Adam

# Initialize
model = ProteinCNN(num_classes=num_go_terms)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_seqs, batch_labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_seqs)
        loss = criterion(predictions, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Validation
    val_metrics = evaluate_model(model, val_loader)
    print(f"Val F1: {val_metrics['f1']:.4f}")
```

#### Step 5.4: Protein Language Model (Advanced)

- [ ] Load pre-trained model
  ```python
  from transformers import AutoTokenizer, AutoModel
  
  tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
  model = AutoModel.from_pretrained("Rostlab/prot_bert")
  ```

- [ ] Fine-tune on CAFA data
  - Freeze early layers (optional)
  - Add classification head
  - Train with lower learning rate (1e-5)

- [ ] Extract embeddings
  - Use as features for downstream model
  - Or fine-tune end-to-end

**Code Template:**
```python
class ProtBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained("Rostlab/prot_bert")
        self.classifier = nn.Linear(1024, num_classes)  # ProtBERT hidden size = 1024
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(pooled)
        return self.sigmoid(logits)
```

#### Step 5.5: Ensemble Methods
- [ ] Train multiple models
  - CNN
  - ProtBERT
  - BLAST-based

- [ ] Combine predictions
  - Simple average
  - Weighted average (based on validation performance)
  - Stacking (meta-model)

**Ensemble Logic:**
```python
pred_cnn = model_cnn(test_seq)       # Shape: (n_samples, n_terms)
pred_bert = model_bert(test_seq)
pred_blast = blast_predict(test_seq)

# Simple average
pred_ensemble = (pred_cnn + pred_bert + pred_blast) / 3

# Weighted average (if CNN performs best)
pred_ensemble = 0.5 * pred_cnn + 0.3 * pred_bert + 0.2 * pred_blast
```

---

## 6. Evaluation System

### 📏 Build Evaluation Pipeline

#### Step 6.1: Implement Metrics
- [ ] Precision @ k
  - For each protein, check if predicted terms are correct
  
- [ ] Recall @ k
  - For each protein, check if all true terms are predicted

- [ ] F1 Score
  - Harmonic mean of precision and recall

- [ ] Weighted F1 (IA-weighted)
  - Use information accretion weights
  - Rare terms contribute more

**Code Template:**
```python
def calculate_weighted_f1(y_true, y_pred, ia_weights, threshold=0.5):
    """
    y_true: (n_samples, n_terms) - binary labels
    y_pred: (n_samples, n_terms) - probabilities
    ia_weights: (n_terms,) - IA weight for each term
    """
    # Binarize predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate TP, FP, FN for each term
    tp = ((y_true == 1) & (y_pred_binary == 1)).sum(axis=0)
    fp = ((y_true == 0) & (y_pred_binary == 1)).sum(axis=0)
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum(axis=0)
    
    # Weighted precision and recall
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # Weight by IA
    weighted_precision = (precision * ia_weights).sum() / ia_weights.sum()
    weighted_recall = (recall * ia_weights).sum() / ia_weights.sum()
    
    # F1
    f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall + 1e-10)
    
    return f1
```

#### Step 6.2: Threshold Optimization
- [ ] Try different confidence thresholds
  - 0.1, 0.2, 0.3, ..., 0.9
  - Find threshold that maximizes F1

- [ ] Per-term threshold optimization
  - Different threshold for each GO term
  - Based on term frequency

**Code:**
```python
best_threshold = 0.5
best_f1 = 0

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    f1 = calculate_weighted_f1(y_val_true, y_val_pred, ia_weights, threshold)
    print(f"Threshold {threshold}: F1 = {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, Best F1: {best_f1:.4f}")
```

#### Step 6.3: Per-Ontology Evaluation
- [ ] Split labels by BPO/MFO/CCO
- [ ] Calculate metrics separately
- [ ] Report overall average

**Code:**
```python
def evaluate_by_ontology(y_true, y_pred, term_to_ontology, ia_weights):
    results = {}
    
    for ont in ['BPO', 'MFO', 'CCO']:
        # Get term indices for this ontology
        ont_indices = [i for i, term in enumerate(all_terms) 
                       if term_to_ontology[term] == ont]
        
        # Subset labels and predictions
        y_true_ont = y_true[:, ont_indices]
        y_pred_ont = y_pred[:, ont_indices]
        ia_ont = ia_weights[ont_indices]
        
        # Calculate F1
        f1 = calculate_weighted_f1(y_true_ont, y_pred_ont, ia_ont)
        results[ont] = f1
    
    results['Overall'] = np.mean(list(results.values()))
    return results
```

#### Step 6.4: Create Evaluation Report
- [ ] Generate HTML/PDF report with:
  - Overall metrics
  - Per-ontology breakdown
  - Confusion matrices (top terms)
  - Prediction confidence distribution
  - Example predictions (good and bad)

---

## 7. Submission Pipeline

### 📤 Prepare Final Submission

#### Step 7.1: Generate Predictions
- [ ] Load best trained model
- [ ] Load test sequences from `testsuperset.fasta`
- [ ] Generate predictions for all proteins

**Code:**
```python
# Load test data
test_sequences = load_fasta('Test/testsuperset.fasta')

# Predict
model.eval()
all_predictions = []

with torch.no_grad():
    for batch in test_loader:
        preds = model(batch)
        all_predictions.append(preds.cpu().numpy())

all_predictions = np.vstack(all_predictions)
# Shape: (n_test_proteins, n_go_terms)
```

#### Step 7.2: Format Predictions
- [ ] Convert probability matrix to submission format
- [ ] Apply confidence threshold
- [ ] Keep only top predictions (max 1500 per protein)

**Submission Format:**
```
ProteinID    GO_Term    Confidence
P12345       GO:0003677    0.95
P12345       GO:0005634    0.87
P12345       GO:0006281    0.73
```

**Code:**
```python
def create_submission(protein_ids, predictions, go_terms, threshold=0.1, max_per_protein=1500):
    rows = []
    
    for i, protein_id in enumerate(protein_ids):
        # Get predictions for this protein
        pred_probs = predictions[i]
        
        # Filter by threshold
        mask = pred_probs >= threshold
        confident_indices = np.where(mask)[0]
        confident_probs = pred_probs[confident_indices]
        
        # Sort by confidence (descending)
        sorted_idx = np.argsort(confident_probs)[::-1]
        top_indices = confident_indices[sorted_idx[:max_per_protein]]
        top_probs = confident_probs[sorted_idx[:max_per_protein]]
        
        # Add to rows
        for idx, prob in zip(top_indices, top_probs):
            go_term = go_terms[idx]
            rows.append({
                'protein_id': protein_id,
                'go_term': go_term,
                'confidence': f"{prob:.3f}"  # 3 significant figures
            })
    
    df = pd.DataFrame(rows)
    return df
```

#### Step 7.3: Propagate Predictions
- [ ] Ensure hierarchical consistency
  - If predicting GO:123456, also predict all ancestors
  - Child confidence ≥ Parent confidence

**Code:**
```python
def propagate_predictions(predictions_df, go_graph):
    """
    For each predicted term, add all ancestors with max confidence
    """
    propagated = []
    
    for protein_id, group in predictions_df.groupby('protein_id'):
        term_confidences = {}
        
        for _, row in group.iterrows():
            term = row['go_term']
            conf = float(row['confidence'])
            
            # Add this term
            term_confidences[term] = max(term_confidences.get(term, 0), conf)
            
            # Add all ancestors
            ancestors = nx.ancestors(go_graph, term)
            for ancestor in ancestors:
                term_confidences[ancestor] = max(term_confidences.get(ancestor, 0), conf)
        
        # Add to result
        for term, conf in term_confidences.items():
            propagated.append({
                'protein_id': protein_id,
                'go_term': term,
                'confidence': f"{conf:.3f}"
            })
    
    return pd.DataFrame(propagated)
```

#### Step 7.4: Add Optional Text Predictions
- [ ] Generate function descriptions (optional)
  - Use LLM (GPT-4, Claude) to generate text
  - Or template-based generation
  
**Format:**
```
P12345    Text    0.85    This protein is involved in DNA repair and binds to damaged DNA sites
P12345    Text    0.75    Functions as part of the nucleotide excision repair pathway
```

#### Step 7.5: Validate Submission
- [ ] Check format requirements
  - Tab-separated
  - No header
  - Confidence in (0, 1]
  - 3 significant figures
  - Max 1500 terms per protein

- [ ] Run validation script
  ```python
  def validate_submission(submission_df):
      # Check format
      assert list(submission_df.columns) == ['protein_id', 'go_term', 'confidence']
      
      # Check confidence range
      assert submission_df['confidence'].astype(float).between(0, 1).all()
      
      # Check max per protein
      terms_per_protein = submission_df.groupby('protein_id').size()
      assert terms_per_protein.max() <= 1500
      
      print("✅ Submission is valid!")
  ```

#### Step 7.6: Save and Submit
- [ ] Save to TSV file (no header)
  ```python
  submission_df.to_csv('submissions/submission_v1.tsv', 
                       sep='\t', header=False, index=False)
  ```

- [ ] Compress if needed
  ```powershell
  Compress-Archive -Path submissions/submission_v1.tsv -DestinationPath submission_v1.zip
  ```

- [ ] Upload to competition platform

---

## 8. Optimization & Iteration

### 🔄 Improve Your Model

#### Step 8.1: Error Analysis
- [ ] Analyze validation predictions
  - Which proteins are predicted poorly?
  - Which GO terms are hard to predict?
  - Common error patterns?

- [ ] Create error analysis notebook
  ```python
  # Find worst predictions
  errors = []
  for i, (true, pred) in enumerate(zip(y_val_true, y_val_pred)):
      f1 = f1_score(true, (pred > 0.5).astype(int), average='micro')
      errors.append((protein_ids[i], f1))
  
  worst_proteins = sorted(errors, key=lambda x: x[1])[:20]
  ```

#### Step 8.2: Hyperparameter Tuning
- [ ] Learning rate
  - Try: 1e-5, 1e-4, 1e-3, 1e-2

- [ ] Batch size
  - Try: 16, 32, 64, 128

- [ ] Model architecture
  - Number of layers
  - Hidden dimensions
  - Dropout rate

- [ ] Training epochs
  - Early stopping based on validation F1

**Use tools:**
- Grid search
- Random search
- Optuna (Bayesian optimization)

#### Step 8.3: Data Augmentation
- [ ] Sequence-level augmentation
  - Random mutations (flip amino acids)
  - Random cropping (for long sequences)
  - Random masking (MLM-style)

- [ ] Label smoothing
  - Instead of 0/1, use 0.1/0.9

#### Step 8.4: Advanced Techniques
- [ ] Multi-task learning
  - Predict all 3 ontologies together
  - Share lower layers, separate heads

- [ ] Attention mechanisms
  - Self-attention over sequence
  - Cross-attention with GO terms

- [ ] Incorporate external data
  - Protein-protein interaction networks
  - Domain annotations (Pfam)
  - Structure predictions (AlphaFold)

#### Step 8.5: Monitor Progress
- [ ] Create experiment tracking
  - Use MLflow or Weights & Biases
  - Log metrics, hyperparameters, model artifacts

- [ ] Maintain experiment log
  ```markdown
  | Exp ID | Model | LR | Batch | Val F1 | Test F1 | Notes |
  |--------|-------|-----|-------|--------|---------|-------|
  | exp001 | CNN   | 1e-3| 32    | 0.42   | -       | Baseline |
  | exp002 | CNN   | 1e-4| 64    | 0.45   | -       | Lower LR helps |
  | exp003 | BERT  | 1e-5| 16    | 0.52   | -       | Best so far! |
  ```

---

## 🎯 Milestones & Checkpoints

### Week 1: Setup & EDA
- [ ] Environment setup complete
- [ ] Data exploration done
- [ ] EDA notebook with visualizations
- [ ] Understanding of problem clear

### Week 2: Data Processing
- [ ] Data loaders working
- [ ] Feature engineering implemented
- [ ] Train/val split created
- [ ] Label encoding done

### Week 3: Baseline Models
- [ ] Frequency baseline (F1 > 0.15)
- [ ] BLAST baseline (F1 > 0.30)
- [ ] K-mer + LogReg (F1 > 0.25)

### Week 4: Deep Learning
- [ ] CNN model trained (F1 > 0.40)
- [ ] ProtBERT fine-tuned (F1 > 0.50)
- [ ] Evaluation pipeline working

### Week 5: Optimization
- [ ] Hyperparameter tuning
- [ ] Ensemble methods (F1 > 0.55)
- [ ] Error analysis complete

### Week 6: Submission
- [ ] Final predictions generated
- [ ] Submission formatted correctly
- [ ] Validation checks passed
- [ ] Submitted to platform

---

## 📚 Resources & References

### Essential Papers
- [ ] Read CAFA assessment papers
  - Jiang et al. (2016) - Evaluation methods
  - Radivojac et al. (2013) - Original CAFA paper

### Code Repositories
- [ ] Browse past CAFA solutions on GitHub
- [ ] Check Kaggle notebooks (similar competitions)

### Libraries & Tools
- **BioPython**: FASTA parsing
- **OBOnet**: GO ontology parsing
- **PyTorch**: Deep learning
- **Transformers**: Protein language models
- **Scikit-learn**: ML utilities

### Pre-trained Models
- ProtBERT: `Rostlab/prot_bert`
- ESM-2: `facebook/esm2_t33_650M_UR50D`
- ProtTrans: `Rostlab/prot_t5_xl_uniref50`

---

## 🐛 Troubleshooting Guide

### Common Issues

**Issue: Out of Memory (OOM)**
- ✅ Reduce batch size
- ✅ Use gradient accumulation
- ✅ Use mixed precision training (fp16)
- ✅ Truncate long sequences

**Issue: Model not converging**
- ✅ Lower learning rate
- ✅ Check label distribution (imbalanced?)
- ✅ Try different optimizer (SGD vs Adam)
- ✅ Add learning rate scheduler

**Issue: Overfitting**
- ✅ Add dropout
- ✅ Use weight decay
- ✅ Reduce model complexity
- ✅ Get more data (augmentation)

**Issue: Low recall**
- ✅ Lower confidence threshold
- ✅ Use label propagation
- ✅ Increase training epochs

**Issue: Low precision**
- ✅ Increase confidence threshold
- ✅ Use stricter evaluation
- ✅ Better feature engineering

---

## 🎉 Final Checklist

### Before Submission
- [ ] Model achieves F1 > 0.50 on validation
- [ ] Predictions propagated correctly
- [ ] Submission format validated
- [ ] File size < upload limit
- [ ] Confident about top predictions

### After Submission
- [ ] Document model architecture
- [ ] Save trained weights
- [ ] Write summary of approach
- [ ] Note lessons learned
- [ ] Plan for next iteration

---

## 📝 Notes for ADHD-Friendly Workflow

### Tips to Stay Focused
✅ **Work in sprints**: 25 min work, 5 min break (Pomodoro)
✅ **One task at a time**: Don't jump between sections
✅ **Visual progress**: Use checkboxes, celebrate each ✅
✅ **Break big tasks**: Each checkpoint = mini-goal
✅ **Code in notebooks**: Experiment interactively
✅ **Save often**: Git commit after each completed section

### When Stuck
1. Take a break (walk, stretch)
2. Ask for help (forums, ChatGPT)
3. Skip and come back later
4. Review EDA notebook for insights

### Stay Motivated
🎯 **Remember the goal**: Build an AI that predicts protein function
🏆 **Track progress**: Maintain experiment log
📈 **Celebrate wins**: Each improved F1 score is progress!

---

## 🚀 LET'S BUILD THIS!

Start with **Step 1.1** and work through systematically. 

Good luck! 🧬🤖✨
