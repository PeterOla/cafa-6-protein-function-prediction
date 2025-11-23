# CAFA-6 End-to-End Pipeline Sketch

## 🗂️ Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RAW INPUT DATA (Train/)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ train_        │          │ train_terms.tsv  │          │ go-basic.obo     │
│ sequences.    │          │                  │          │                  │
│ fasta         │          │ EntryID | term   │          │ [Term]           │
│               │          │ ─────────────    │          │ id: GO:0008150   │
│ >T100...001   │          │ T10..1  GO:0..1  │          │ name: bio proc   │
│ MKKLAVAA...   │          │ T10..1  GO:0..2  │          │ namespace: BP    │
│ >T100...002   │          │ T10..2  GO:0..3  │          │ is_a: GO:0..X    │
│ ATGGCCTA...   │          │ ...              │          │ ...              │
└───────────────┘          └──────────────────┘          └──────────────────┘
        │                             │                             │
        │                             │                             │
        └─────────────────────────────┴─────────────────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────┐
                        │   PREPROCESSING LAYER       │
                        │                             │
                        │ • Parse FASTA sequences     │
                        │ • Map GO terms → aspects    │
                        │   (F→MF, P→BP, C→CC)        │
                        │ • Build GO graph structure  │
                        │ • Train/Val split (80/20)   │
                        └─────────────────────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
                ▼                                           ▼
    ┌───────────────────────┐                 ┌───────────────────────┐
    │  TRAINING DATA        │                 │  VALIDATION DATA      │
    │                       │                 │                       │
    │  • 113K proteins      │                 │  • 29K proteins       │
    │  • 490K annotations   │                 │  • 122K annotations   │
    │  • 26K unique GO terms│                 │  • Unseen proteins    │
    └───────────────────────┘                 └───────────────────────┘
```

---

## 🔄 Four Parallel Modelling Pipelines

### **Pipeline 1: Frequency Baseline** (`01_baseline_frequency.ipynb`)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        FREQUENCY BASELINE                                 │
└──────────────────────────────────────────────────────────────────────────┘

    TRAINING DATA (train_terms.tsv)
            │
            ▼
    ┌───────────────────────────────┐
    │  Count Term Frequencies       │
    │                               │
    │  GO:0008150 → 2,319 times     │
    │  GO:0003824 → 33,713 times    │
    │  GO:0005575 → 13,283 times    │
    │  ...                          │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │  Select Top 10,000 Terms      │
    │                               │
    │  • MF: 17.5% (1,750 terms)    │
    │  • BP: 57.5% (5,750 terms)    │
    │  • CC: 25.0% (2,500 terms)    │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │  PREDICTION STRATEGY          │
    │                               │
    │  For ANY protein:             │
    │  Predict same 10K terms       │
    │  with their frequencies       │
    │  as probability scores        │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │  OUTPUT                       │
    │                               │
    │  142K proteins × 10K terms    │
    │  = 1.4 million predictions    │
    │  (chunked processing)         │
    └───────────────────────────────┘
            │
            ▼
    Per-aspect F1 evaluation
    (expected: MF high, BP low, CC medium)
```

**Key Insight:** Ignores sequence content — purely statistical baseline

---

### **Pipeline 2: K-Nearest Neighbors** (`02_baseline_knn.ipynb`)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         KNN BASELINE                                      │
└──────────────────────────────────────────────────────────────────────────┘

    TRAINING DATA                    VALIDATION DATA
    ┌──────────────┐                ┌──────────────┐
    │ Sequences    │                │ Query protein│
    │ + GO labels  │                │ MKKLAVAA...  │
    └──────────────┘                └──────────────┘
            │                               │
            ▼                               ▼
    ┌───────────────────────────────────────────────┐
    │     SEQUENCE SIMILARITY COMPUTATION           │
    │                                               │
    │     • BLAST alignment / k-mer overlap         │
    │     • Find K=5 most similar proteins          │
    │     • Compute similarity scores (0-1)         │
    └───────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────────────┐
    │     AGGREGATE NEIGHBOR LABELS                 │
    │                                               │
    │     Neighbor 1 (sim=0.95): GO:001, GO:002    │
    │     Neighbor 2 (sim=0.89): GO:001, GO:003    │
    │     Neighbor 3 (sim=0.82): GO:002, GO:004    │
    │     Neighbor 4 (sim=0.78): GO:001            │
    │     Neighbor 5 (sim=0.71): GO:003, GO:005    │
    │                                               │
    │     Weighted vote:                            │
    │     GO:001 → (0.95+0.89+0.78)/3 = 0.873      │
    │     GO:002 → (0.95+0.82)/2 = 0.885           │
    │     GO:003 → (0.89+0.71)/2 = 0.800           │
    │     ...                                       │
    └───────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────────────┐
    │     OUTPUT: Sequence-aware predictions        │
    │                                               │
    │     • BP F1 expected to IMPROVE significantly │
    │     • Rare BP terms can be predicted via      │
    │       similar sequences                       │
    └───────────────────────────────────────────────┘
            │
            ▼
    Per-aspect F1 evaluation
```

**Key Insight:** Leverages sequence similarity — should fix BP problem

---

### **Pipeline 3: ESM-2 Fine-Tuned** (`03_model_esm_finetuned.ipynb`)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    ESM-2 FINE-TUNED MODEL                                 │
└──────────────────────────────────────────────────────────────────────────┘

    PROTEIN SEQUENCE                 PRE-TRAINED MODEL
    ┌──────────────┐                ┌─────────────────────┐
    │ MKKLAVAA...  │                │  ESM-2 8M params    │
    │              │                │  (Facebook/Meta AI) │
    │              │                │                     │
    │              │                │  Trained on 250M    │
    │              │                │  protein sequences  │
    └──────────────┘                └─────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
                ┌───────────────────────────┐
                │   TOKEN EMBEDDING         │
                │                           │
                │   M → [0.23, -0.15, ...]  │
                │   K → [0.41, 0.09, ...]   │
                │   K → [0.41, 0.09, ...]   │
                │   ...                     │
                └───────────────────────────┘
                            │
                            ▼
                ┌───────────────────────────┐
                │   TRANSFORMER LAYERS      │
                │                           │
                │   • Self-attention (6x)   │
                │   • Learn sequence context│
                │   • Output: 320-dim vector│
                └───────────────────────────┘
                            │
                            ▼
                ┌───────────────────────────┐
                │   CLASSIFICATION HEAD     │
                │   (trainable)             │
                │                           │
                │   Linear: 320 → 26,125    │
                │   (one output per GO term)│
                └───────────────────────────┘
                            │
                            ▼
                ┌───────────────────────────┐
                │   SIGMOID ACTIVATION      │
                │                           │
                │   GO:0001 → 0.92 ✓        │
                │   GO:0002 → 0.05          │
                │   GO:0003 → 0.78 ✓        │
                │   ...                     │
                └───────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  TRAINING PROCESS                     │
        │                                       │
        │  Loss: Binary Cross-Entropy           │
        │  Optimizer: AdamW                     │
        │  Learning Rate: 1e-4                  │
        │  Batch Size: 8                        │
        │  Epochs: 10                           │
        │                                       │
        │  Each epoch:                          │
        │  • Forward pass (predict)             │
        │  • Compute loss vs true labels        │
        │  • Backprop gradients                 │
        │  • Update classification head weights │
        │  • Validate on held-out set           │
        └───────────────────────────────────────┘
                            │
                            ▼
        Per-aspect F1 evaluation during training
```

**Key Insight:** Deep learning captures sequence patterns frequency/KNN miss

---

### **Pipeline 4: Label Propagation** (`04_label_propagation.ipynb`)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      LABEL PROPAGATION                                    │
└──────────────────────────────────────────────────────────────────────────┘

    BASE MODEL PREDICTIONS          GO ONTOLOGY GRAPH
    (from Pipeline 3)               (from go-basic.obo)
    ┌──────────────────┐            ┌─────────────────────────┐
    │ GO:0006355 → 0.8 │            │                         │
    │ (specific term)  │            │   GO:0008150 (root)     │
    └──────────────────┘            │      │                  │
            │                       │      ├─ GO:0065007      │
            │                       │      │    │             │
            └───────────────────────┼──────┴─── GO:0006355   │
                                    │           (leaf)        │
                                    │                         │
                                    │  is_a relationships     │
                                    │  form directed graph    │
                                    └─────────────────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │  PROPAGATION ALGORITHM        │
                            │                               │
                            │  IF predict GO:0006355 (0.8)  │
                            │  THEN also predict:           │
                            │                               │
                            │  • GO:0065007 (parent) → 0.8  │
                            │  • GO:0008150 (root)   → 0.8  │
                            │                               │
                            │  Rule: ancestors inherit      │
                            │  max score of descendants     │
                            └───────────────────────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │  ENHANCED PREDICTIONS         │
                            │                               │
                            │  Original: 150 terms          │
                            │  After prop: 210 terms        │
                            │                               │
                            │  • Ensures biological validity│
                            │  • Fixes "orphan" predictions │
                            │  • Expected: +0.02-0.04 F1    │
                            └───────────────────────────────┘
                                            │
                                            ▼
                            Per-aspect F1 evaluation
```

**Key Insight:** Enforces GO hierarchy constraints — free performance boost

---

## 📊 Evaluation Framework (All Pipelines)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   PER-ASPECT CAFA METRIC                                  │
└──────────────────────────────────────────────────────────────────────────┘

    MODEL PREDICTIONS                VALIDATION LABELS
    ┌─────────────────┐              ┌─────────────────┐
    │ GO:0001 → 0.92  │              │ GO:0001 ✓       │
    │ GO:0002 → 0.05  │              │ GO:0003 ✓       │
    │ GO:0003 → 0.78  │              │ GO:0005 ✓       │
    │ GO:0004 → 0.15  │              └─────────────────┘
    │ GO:0005 → 0.88  │
    └─────────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────┐
    │  STEP 1: Map terms to aspects               │
    │                                             │
    │  GO:0001 (F) → MF                          │
    │  GO:0002 (P) → BP                          │
    │  GO:0003 (F) → MF                          │
    │  GO:0004 (C) → CC                          │
    │  GO:0005 (P) → BP                          │
    └─────────────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────┐
    │  STEP 2: Split by aspect                    │
    │                                             │
    │  MF predictions: [0.92, 0.78]              │
    │  BP predictions: [0.05, 0.88]              │
    │  CC predictions: [0.15]                    │
    │                                             │
    │  MF labels: [GO:0001, GO:0003]             │
    │  BP labels: [GO:0005]                      │
    │  CC labels: []                             │
    └─────────────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────┐
    │  STEP 3: Apply threshold (e.g., 0.5)        │
    │                                             │
    │  MF: [0.92✓, 0.78✓] → predict both         │
    │  BP: [0.05✗, 0.88✓] → predict GO:0005 only │
    │  CC: [0.15✗] → predict nothing             │
    └─────────────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────┐
    │  STEP 4: Compute IA-weighted F1 per aspect  │
    │                                             │
    │  Precision_MF = TP / (TP + FP)             │
    │  Recall_MF = TP / (TP + FN)                │
    │  F1_MF = 2 × (P × R) / (P + R)             │
    │  (weighted by IA scores)                    │
    │                                             │
    │  Repeat for BP, CC                         │
    └─────────────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────┐
    │  STEP 5: Average across aspects             │
    │                                             │
    │  F1_overall = (F1_MF + F1_BP + F1_CC) / 3  │
    │                                             │
    │  Example:                                   │
    │  F1_MF = 0.42                              │
    │  F1_BP = 0.15                              │
    │  F1_CC = 0.38                              │
    │  Overall = 0.317                           │
    └─────────────────────────────────────────────┘
```

---

## 🎯 Final Submission Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TEST SET PREDICTION                                │
└──────────────────────────────────────────────────────────────────────────┘

    TEST DATA (Test/)
    ┌──────────────────────┐
    │ testsuperset.fasta   │
    │                      │
    │ >T200...001          │
    │ MGGKLAAAA...         │
    │ >T200...002          │
    │ ATAGGCCTA...         │
    │ ...                  │
    │ (142,000 proteins)   │
    └──────────────────────┘
            │
            ▼
    ┌──────────────────────────────────────┐
    │  BEST MODEL (Pipeline 3 or 4)        │
    │                                      │
    │  • Trained ESM-2 + propagation       │
    │  • Optimal threshold per aspect      │
    │  • Generate predictions              │
    └──────────────────────────────────────┘
            │
            ▼
    ┌──────────────────────────────────────┐
    │  FORMAT SUBMISSION                   │
    │                                      │
    │  EntryID    term      score          │
    │  ────────────────────────────        │
    │  T200..1    GO:0001   0.92           │
    │  T200..1    GO:0003   0.78           │
    │  T200..2    GO:0005   0.88           │
    │  ...                                 │
    └──────────────────────────────────────┘
            │
            ▼
    sample_submission.tsv
    (upload to Kaggle)
```

---

## 🔗 Data Dependencies Map - **THE LAYMAN'S GUIDE**

### **Think of it like a Restaurant Database System:**

```
┌──────────────────────────────────────────────────────────────────────────┐
│  YOUR MISSION: Build a system that looks at a customer (protein)         │
│  and predicts what dishes they'll order (GO functions)                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### **1. train_sequences.fasta** = **CUSTOMER LIST** 👥

```
What it looks like:
──────────────────
>T100900000001
MKKLAVAATVMSLLIACSASSAAKENVTNFKTEQSTPQAAA
>T100900000002
ATGGCCTATATCGGTGCCAAGGACGGCGACTACAAAGACGATGAC

What it ACTUALLY means:
───────────────────────
Customer #1 (ID: T100900000001)
  Description: "Tall, brown hair, wears glasses, has tattoo"
  (In reality: protein's amino acid sequence = its physical structure)

Customer #2 (ID: T100900000002)  
  Description: "Short, blonde, athletic build"
  (In reality: another protein's unique sequence)

🧑‍🍳 Analogy: 
   Each line of random letters (MKKLAVAA...) is like describing a person's appearance.
   Just like you can recognize someone by "tall + glasses + tattoo", 
   the model recognizes protein function from "MKKL..." sequence pattern.

🔬 Reality:
   - 113,000 proteins (customers) in training set
   - Each sequence is 100-5000 letters long
   - Letters = amino acids (building blocks of proteins)
   - The SEQUENCE determines what the protein DOES
```

---

### **2. train_terms.tsv** = **ORDER HISTORY** 🍕🍔🍜

```
What it looks like:
──────────────────
T100900000001   GO:0008150
T100900000001   GO:0003824
T100900000001   GO:0005737
T100900000002   GO:0016020
T100900000002   GO:0005575

What it ACTUALLY means:
───────────────────────
Customer #1 ordered:
  • GO:0008150 (biological_process) = "Meal category: Main course"
  • GO:0003824 (catalytic_activity) = "Specific dish: Pepperoni Pizza"
  • GO:0005737 (cytoplasm) = "Location: Dine-in"

Customer #2 ordered:
  • GO:0016020 (membrane) = "Location: Takeaway"
  • GO:0005575 (cellular_component) = "Meal category: Dessert"

🧑‍🍳 Analogy:
   Your job: Look at Customer #1's appearance (sequence MKKLAVAA...)
   and predict they'll order pizza + main course + dine-in.
   
   This file tells you "Customer #1 DID order these things in the past"
   → Training data to learn patterns

🔬 Reality:
   - 490,000 rows (order records)
   - Each protein has 3-50 GO terms (functions)
   - GO terms = biological jobs like "cuts DNA", "makes energy", "lives in nucleus"
   - THIS IS WHAT YOU'RE TRYING TO PREDICT for new proteins!
```

---

### **3. go-basic.obo** = **MENU HIERARCHY** 📋

```
What it looks like:
──────────────────
[Term]
id: GO:0006355
name: regulation of transcription, DNA-templated
namespace: biological_process
is_a: GO:0065007 ! regulation of biological process
is_a: GO:0051252 ! regulation of RNA metabolic process

What it ACTUALLY means:
───────────────────────
Dish: "Margherita Pizza" (GO:0006355)
  ↳ is_a: "Pizza" (GO:0065007)
     ↳ is_a: "Main Course" (GO:0051252)
        ↳ is_a: "Food" (root category)

🧑‍🍳 Analogy:
   Restaurant menu has hierarchy:
   
   FOOD (root)
   ├── Main Course
   │   ├── Pizza
   │   │   ├── Margherita Pizza ← specific
   │   │   └── Pepperoni Pizza  ← specific
   │   └── Pasta
   └── Dessert
       └── Ice Cream

   Rules:
   • If someone orders "Margherita Pizza" → they ALSO ordered "Pizza", "Main Course", "Food"
   • Can't order "Margherita" without it being a "Pizza"
   • Parent categories are IMPLIED by child orders

🔬 Reality:
   - 47,000 GO terms in ontology
   - 3 main branches: Molecular Function (MF), Biological Process (BP), Cellular Component (CC)
   - "is_a" relationships form a tree structure
   - Used in Pipeline 4 (label propagation) to add missing parent terms
   
Example:
  Model predicts: GO:0006355 (transcription)
  But forgets: GO:0065007 (biological regulation) ← its parent
  
  go-basic.obo says "transcription is_a biological regulation"
  → Propagation adds the parent automatically
```

---

### **4. IA.tsv** = **DISH RARITY SCORES** ⭐💎

```
What it looks like:
──────────────────
GO:0008150    1.000
GO:0003824    2.145
GO:0006355    7.851
GO:0043167    9.825

What it ACTUALLY means:
───────────────────────
GO:0008150 (biological_process root) → IA = 1.0
  = "Food" category → EVERYONE orders this → boring, no credit

GO:0003824 (catalytic activity) → IA = 2.1
  = "Pizza" → 70% of customers order this → common, little credit

GO:0006355 (transcription) → IA = 7.9
  = "Truffle Risotto" → Only 5% order this → rare, HIGH CREDIT!

GO:0043167 (ion binding) → IA = 9.8
  = "Molecular Gastronomy Foam" → Only 0.5% order → VERY RARE, HUGE CREDIT!

🧑‍🍳 Analogy:
   You're a waiter trying to predict orders.
   
   Scenario A: You predict "Customer will order food"
      → Correct! But everyone orders food. Score: 1/10 (useless prediction)
   
   Scenario B: You predict "Customer will order Truffle Risotto"
      → Correct! Very few order this. Score: 8/10 (impressive!)
   
   Scenario C: You predict "Customer will order Molecular Foam"
      → Correct! Almost nobody orders this. Score: 10/10 (master waiter!)

🔬 Reality:
   - IA = -log₂(frequency)
   - Common terms (appear 50% of time) → IA ≈ 1.0
   - Rare terms (appear 0.1% of time) → IA ≈ 10.0
   - Used to WEIGHT the F1 score
   - Predicting rare = more valuable than predicting common
   - Prevents model from cheating by only predicting "biological_process" for everyone
```

---

### **HOW THEY ALL CONNECT - THE COMPLETE STORY:**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    RESTAURANT PREDICTION SYSTEM                         │
└────────────────────────────────────────────────────────────────────────┘

STEP 1: Load Customer Database
───────────────────────────────
train_sequences.fasta:
  Customer T100...001: "Tall, glasses, tattoo" (protein sequence MKKLAVAA...)
  Customer T100...002: "Short, blonde, athletic" (protein sequence ATGGCC...)
  
  → These are your TRAINING CUSTOMERS


STEP 2: Load Order History
───────────────────────────
train_terms.tsv:
  Customer T100...001 previously ordered:
    • GO:0003824 (Pepperoni Pizza)
    • GO:0005737 (Dine-in)
    
  Customer T100...002 previously ordered:
    • GO:0016020 (Takeaway)
    • GO:0005575 (Dessert)
    
  → This is your TRAINING DATA (what they actually ordered)


STEP 3: Study Menu Structure
─────────────────────────────
go-basic.obo:
  Pizza is_a Main Course
  Main Course is_a Food
  Pepperoni Pizza is_a Pizza
  
  → This is the MENU HIERARCHY (how dishes relate)


STEP 4: Load Rarity Scores
───────────────────────────
IA.tsv:
  "Food" → IA = 1.0 (everyone orders, no credit)
  "Pizza" → IA = 2.1 (common, little credit)
  "Truffle Risotto" → IA = 7.9 (rare, big credit)
  "Molecular Foam" → IA = 9.8 (very rare, huge credit)
  
  → This is the SCORING SYSTEM (how much credit for correct predictions)


STEP 5: TRAIN THE MODEL
────────────────────────
"Look at customer appearance (sequence) + past orders (train_terms)
 Learn patterns like:
 • Tall customers with tattoos → usually order Pizza
 • Athletic customers → usually order Salad
 • Customers wearing suits → usually order Wine"


STEP 6: PREDICT NEW CUSTOMER
─────────────────────────────
New customer walks in: "Tall, glasses, tattoo" (test protein sequence)

Model thinks:
  "Hmm, this looks like Customer T100...001 from training"
  "T100...001 ordered Pizza + Dine-in"
  "I predict this new customer will order Pizza + Dine-in"
  
Model outputs:
  GO:0003824 (Pizza) → confidence 0.92
  GO:0005737 (Dine-in) → confidence 0.78


STEP 7: PROPAGATE HIERARCHY
────────────────────────────
go-basic.obo says:
  "Pizza is_a Main Course is_a Food"
  
Propagation adds:
  GO:0003824 (Pizza) → 0.92  ← model predicted
  GO:xxxxxxx (Main Course) → 0.92  ← added by propagation
  GO:0008150 (Food) → 0.92  ← added by propagation


STEP 8: SCORE WITH IA WEIGHTS
──────────────────────────────
True labels: Customer ordered Pizza + Truffle Risotto

Model predicted: Pizza + Dine-in

Standard F1:
  TP = 1 (Pizza correct)
  FP = 1 (Dine-in wrong)
  FN = 1 (missed Truffle Risotto)
  F1 = 0.50

IA-weighted F1:
  TP_weight = IA(Pizza) = 2.1
  FP_weight = IA(Dine-in) = 3.0
  FN_weight = IA(Truffle Risotto) = 7.9
  
  Precision = 2.1 / (2.1 + 3.0) = 0.41
  Recall = 2.1 / (2.1 + 7.9) = 0.21
  F1 = 0.28  ← LOWER because missed rare dish (Truffle Risotto)
  
  → Penalty for missing rare items!
```

---

### **WHY THE DATA LOOKS RANDOM:**

```
❓ "Why does train_sequences.fasta look like gibberish?"
───────────────────────────────────────────────────────
MKKLAVAATVMSLLIACSASSAAKENVTNFKTEQSTPQAAA...

Answer: It's NOT random! It's a LANGUAGE.
  
  M = Methionine (amino acid)
  K = Lysine (amino acid)
  K = Lysine (amino acid)
  L = Leucine (amino acid)
  A = Alanine (amino acid)
  V = Valine (amino acid)
  ...
  
  Just like "HELLO" means something in English,
  "MKKLAVAA" means something in Protein Language.
  
  The model learns:
    "MKKL" at start → signal peptide → protein goes to membrane
    "KDEL" at end → ER retention signal → protein stays in ER
    "CxxC" pattern → zinc finger domain → DNA binding protein
  
  Same way you recognize words → model recognizes sequence motifs!


❓ "Why GO:0008150 instead of normal names?"
─────────────────────────────────────────────
Answer: GO IDs are UNIQUE and STANDARDIZED.
  
  "biological process" could mean different things
  GO:0008150 ALWAYS means same thing worldwide
  
  Like:
    ISBN numbers for books (GO IDs)
    vs
    "Harry Potter" (common name - which book? which edition?)


❓ "How do proteins (sequences) link to GO terms?"
──────────────────────────────────────────────────
Answer: train_terms.tsv is the BRIDGE!

  train_sequences.fasta says: T100...001 = MKKLAVAA...
  train_terms.tsv says: T100...001 has GO:0003824
  
  So: MKKLAVAA... → GO:0003824
  
  Model learns: "This sequence pattern → catalytic activity"
  
  Like: Customer appearance → past orders
        Tall + glasses → ordered Pizza last time
        MKKLAVAA... → has catalytic activity


❓ "Why do we need IA.tsv?"
───────────────────────────
Answer: Without it, model would CHEAT!
  
  Model could predict "biological_process" (GO:0008150) for EVERYONE
  → 100% correct (all proteins do SOME biological process)
  → But totally useless (doesn't tell us WHAT process)
  
  IA.tsv says: GO:0008150 = 1.0 (no credit)
                GO:0006355 = 7.9 (big credit)
  
  Forces model to be SPECIFIC, not just correct but vague.
  
  Like: Waiter predicting "customer will order food" vs "customer will order Truffle Risotto"
        Both might be correct, but second is USEFUL!
```

---

### **QUICK REFERENCE - FILE CONNECTIONS:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FILE RELATIONSHIPS                                │
└─────────────────────────────────────────────────────────────────────────┘

Train/
├── train_sequences.fasta ──┬──→ WHO (protein identities)
│   "Customer appearance"   │    
│                           │
├── train_terms.tsv ────────┼──→ WHAT (protein functions)
│   "Order history"         │    Links WHO to WHAT
│                           │    
├── go-basic.obo ───────────┼──→ HOW (functions relate)
│   "Menu hierarchy"        │    Links WHAT to WHAT
│                           │    
└── train_taxonomy.tsv      └──→ WHERE (species info - optional)
    "Customer demographics"      

Competition Files/
├── IA.tsv ─────────────────────→ VALUE (scoring system)
│   "Dish rarity scores"         How VALUABLE each WHAT is
│                           
└── sample_submission.tsv ──────→ FORMAT (output template)
    "Order receipt format"
```

---

## 📈 Expected Performance Trajectory

```
Model Pipeline          MF F1    BP F1    CC F1    Overall F1   Notes
─────────────────────────────────────────────────────────────────────────
1. Frequency            0.28     0.00     0.22     0.167        Baseline
   (ignores sequence)                                            BP fails

2. KNN                  0.35     0.18     0.30     0.277        +0.11
   (sequence-aware)                                              BP fixed!

3. ESM-2 Fine-tuned     0.42     0.25     0.38     0.350        +0.07
   (deep learning)                                               SOTA

4. + Label Propagation  0.44     0.27     0.40     0.370        +0.02
   (enforce hierarchy)                                           Free boost
```

---

## 🧠 Key Conceptual Links

### Why BP Fails in Frequency Baseline:
```
BP terms: 16,858 unique → spread thin → individually RARE
MF terms: 6,616 unique  → concentrated → individually FREQUENT

Frequency baseline predicts SAME terms for ALL proteins
→ Only picks most frequent terms
→ Most frequent terms are MF/CC (GO:0003824, GO:0005575)
→ BP terms like GO:0006355 (2,319 occurrences) get filtered out
→ BP F1 = 0.00
```

### Why KNN Fixes BP:
```
KNN looks at SIMILAR sequences
→ Similar proteins often have similar BP functions
→ Can predict rare BP terms via neighbors
→ Example: DNA-binding protein → neighbors likely have DNA-related BP terms
→ BP F1 jumps from 0.00 → 0.18
```

### Why ESM-2 Does Better:
```
Transformer learns patterns like:
• "KDEL motif at C-terminus" → ER retention (GO:0006621)
• "Zinc finger domains" → transcription regulation (GO:0006355)
• "Transmembrane helices" → membrane localization (GO:0016020)

Can predict BP terms WITHOUT finding similar training examples
→ Generalizes to unseen sequence patterns
→ BP F1 jumps to 0.25
```

### Why Propagation Helps:
```
Model predicts: GO:0006355 (transcription, DNA-templated)
But forgets parent: GO:0065007 (biological regulation)

Propagation enforces:
IF child predicted → THEN ancestors should be predicted too
→ Adds ~40-60 ancestor terms per protein
→ Fixes "incomplete" predictions
→ +0.02 F1 boost (free lunch!)
```

---

## 🎯 Information Accretion (IA) Weights - Deep Dive

### What is IA?

**Information Accretion** measures how **specific** a GO term is in the ontology hierarchy.

```
IA Score Logic:
─────────────────
High IA (9-10)  → Very SPECIFIC term (leaf nodes, rare annotations)
Medium IA (4-8) → Moderately specific term (mid-level)
Low IA (1-2)    → Very GENERAL term (root nodes, common annotations)
```

### Why IA Matters in CAFA Evaluation

```
WITHOUT IA weighting:
─────────────────────
Predicting GO:0008150 (biological_process - root)     → Easy, uninformative
Predicting GO:0006355 (DNA-templated transcription)   → Hard, informative

Both count equally in F1 → Model just predicts easy root terms → Useless!

WITH IA weighting:
──────────────────
Predicting GO:0008150 (IA=1.00)  → Low reward
Predicting GO:0006355 (IA=7.85)  → High reward (7.85× more valuable!)

F1 calculation weights by IA → Model incentivized to predict SPECIFIC terms
```

### IA Calculation Formula

```
IA(term) = -log₂(P(term))

Where:
P(term) = frequency of term in training annotations / total annotations

Example:
────────
GO:0003674 (molecular_function root):
  • Appears in 128K annotations out of 490K total
  • P = 128K/490K = 0.261
  • IA = -log₂(0.261) = 1.94  ← LOW (very common)

GO:0043167 (ion binding):
  • Appears in 450 annotations out of 490K total
  • P = 450/490K = 0.00092
  • IA = -log₂(0.00092) = 10.08  ← HIGH (very rare)
```

### How IA Integrates into Evaluation

```
┌─────────────────────────────────────────────────────────────────────┐
│               PRECISION CALCULATION WITH IA                          │
└─────────────────────────────────────────────────────────────────────┘

Predictions:        True Labels:         IA Weights:
GO:0001 (0.9) ✓     GO:0001 ✓           GO:0001 → 8.5
GO:0002 (0.8) ✗     GO:0003 ✓           GO:0002 → 6.2
GO:0003 (0.7) ✓                         GO:0003 → 9.1

Standard Precision = 2/3 = 0.667

IA-weighted Precision:
  TP_weight = IA(GO:0001) + IA(GO:0003) = 8.5 + 9.1 = 17.6
  FP_weight = IA(GO:0002) = 6.2
  
  Precision_IA = 17.6 / (17.6 + 6.2) = 0.739

→ Correctly predicting rare terms (GO:0003, IA=9.1) increases precision more
→ False positives on rare terms hurt more than on common terms
```

### IA Distribution Across Aspects

```
Load IA.tsv and analyze:

                Min IA    Median IA    Max IA    Interpretation
              ────────────────────────────────────────────────────
Molecular     1.00      4.23         9.85      MF has many common
Function                                       terms (enzymes)

Biological    1.00      5.67         10.12     BP has most specific
Process                                        terms (rare pathways)

Cellular      1.00      3.98         8.94      CC terms moderately
Component                                      specific (organelles)

→ BP has highest IA scores → Predicting BP correctly is most valuable
→ Explains why BP F1 low in frequency baseline hurts overall score so much
```

### Practical Example in Code

```python
# Load IA weights
ia_weights = pd.read_csv('IA.tsv', sep='\t', header=None, names=['term', 'IA'])
ia_dict = dict(zip(ia_weights['term'], ia_weights['IA']))

# During evaluation
def compute_weighted_f1(y_true, y_pred, terms, ia_dict):
    """
    y_true: [0, 1, 0, 1, 0]  (ground truth labels)
    y_pred: [1, 1, 0, 0, 1]  (model predictions)
    terms: ['GO:0001', 'GO:0002', 'GO:0003', 'GO:0004', 'GO:0005']
    """
    
    tp_weight = sum(ia_dict[term] for term, true, pred in zip(terms, y_true, y_pred)
                    if true == 1 and pred == 1)  # Correct positives
    
    fp_weight = sum(ia_dict[term] for term, true, pred in zip(terms, y_true, y_pred)
                    if true == 0 and pred == 1)  # False positives
    
    fn_weight = sum(ia_dict[term] for term, true, pred in zip(terms, y_true, y_pred)
                    if true == 1 and pred == 0)  # Missed positives
    
    precision = tp_weight / (tp_weight + fp_weight) if (tp_weight + fp_weight) > 0 else 0
    recall = tp_weight / (tp_weight + fn_weight) if (tp_weight + fn_weight) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1
```

### Strategic Implications

```
1. ROOT TERMS ARE WORTHLESS
   ──────────────────────────
   GO:0008150, GO:0003674, GO:0005575 all have IA ≈ 1.0
   → Predicting these gives almost NO credit
   → Always propagate to MORE SPECIFIC children

2. RARE BP TERMS ARE GOLD
   ───────────────────────
   BP terms like GO:0006355 (transcription) have IA > 7.0
   → Getting these right massively boosts F1
   → This is why frequency baseline fails — misses rare BP terms

3. THRESHOLD TUNING MATTERS
   ─────────────────────────
   • Too high → Miss rare high-IA terms → Low recall → Low F1
   • Too low → Predict common low-IA terms → Low precision → Low F1
   • Optimal threshold balances high-IA true positives vs false positives

4. ASPECT-SPECIFIC THRESHOLDS
   ──────────────────────────
   MF: threshold 0.5 (common terms, need high confidence)
   BP: threshold 0.1 (rare terms, accept lower confidence)
   CC: threshold 0.3 (moderate)
   
   → Can tune per-aspect for +0.02-0.05 F1 improvement
```

### Visualization of IA Impact

```
Scenario: Model predicts 10 terms, 5 are correct

Case 1: Predicts 5 COMMON terms correctly (IA avg = 2.0)
        TP_weight = 5 × 2.0 = 10.0
        F1 ≈ 0.25

Case 2: Predicts 5 RARE terms correctly (IA avg = 8.0)
        TP_weight = 5 × 8.0 = 40.0
        F1 ≈ 0.68

→ Same NUMBER of correct predictions, but 2.7× BETTER F1 score!
→ CAFA rewards biological insight (rare terms) over naive prediction (common terms)
```

---

## 💾 Intermediate File Outputs

```
Pipeline 1 (Frequency):
└── predictions_temp.parquet ─→ Chunked predictions (1.4M rows)

Pipeline 2 (KNN):
└── similarity_matrix.npy ─────→ Pairwise sequence similarities

Pipeline 3 (ESM-2):
├── model_checkpoint_best.pt ──→ Trained model weights
└── training_history.json ─────→ Loss/F1 curves per epoch

Pipeline 4 (Propagation):
├── go_graph.pkl ───────────────→ Parsed GO ontology graph
└── propagated_predictions.tsv ─→ Enhanced predictions

Final:
└── submission.tsv ─────────────→ Kaggle upload file
```

---

## 🎓 Learning Progression

```
START: "I have protein sequences + GO labels. What now?"
   ↓
Step 1: Frequency baseline
   → Learn: aspect distribution, term frequencies, evaluation metric
   → Outcome: Understand why naive approach fails for BP
   ↓
Step 2: KNN baseline
   → Learn: sequence similarity matters, k-mer matching, weighted voting
   → Outcome: See BP performance improve dramatically
   ↓
Step 3: ESM-2 fine-tuning
   → Learn: transfer learning, transformers, embeddings
   → Outcome: Beat handcrafted features with deep learning
   ↓
Step 4: Label propagation
   → Learn: GO hierarchy, graph algorithms, biological constraints
   → Outcome: Enforce domain knowledge for free gains
   ↓
END: Competitive CAFA-6 submission with interpretable pipeline
```

