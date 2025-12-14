# CAFA-6 — Execution Plan (Rank-1 Style Stack)

This is the end-to-end, runnable plan for the **modular Level-1 ensemble → Level-2 GCN stacker → strict GO post-processing** approach.

Principles:
- **Per-aspect evaluation is mandatory**: BP/MF/CC are scored separately then averaged.
- **IA weights matter everywhere**: use `IA.tsv` to prioritise rare, high-value terms.
- **Separate concerns**: learn probabilities first; enforce GO constraints after (min/max propagation).

---
## Phase 0 — Kaggle kernel setup (single notebook)

Deliverables:
- One Kaggle notebook that contains:
  - Input discovery (under `/kaggle/input`)
  - Paths + artefact locations
  - Sanity checks + diagnostics plots

Run:
- `notebooks/00_setup_kaggle_colab.ipynb`

Notes (pragmatic):
- We are **Kaggle-first**: no external config files are required to run.
- If you have multiple datasets mounted, set `DATASET_SLUG` inside the notebook.

Checkpointing (resumable across providers):
- Use a Kaggle Dataset as the canonical artefact store.
- The notebook pulls checkpoints on startup (`STORE.pull()`) and publishes milestone versions after each stage (`STORE.push(stage, paths)`).
- Control the target dataset via `CAFA_CHECKPOINT_DATASET_ID` (or `CAFA_KAGGLE_DATASET_ID`) and authenticate via `KAGGLE_USERNAME` + `KAGGLE_KEY` (from `kaggle.json`).
- If `STORE.pull()` fails with **HTTP 403 Forbidden**, the dataset is not accessible (usually private / not shared). On Kaggle, prefer attaching the dataset as a Notebook Input.
- By default, checkpoint pulls are **fail-fast**. Set `CAFA_CHECKPOINT_REQUIRED=0` if you want best-effort warning-only pulls.

Runbook:
- See `docs/RUNBOOK_CHECKPOINTS.md` for the straight-through “run here, resume there” workflow.

---
## Phase 1 — Data artefacts + features

### 1.1 Parse FASTA → efficient tables
Inputs:
- `Train/train_sequences.fasta`
- `Test/testsuperset.fasta`

Outputs:
- `train_seq.feather`
- `test_seq.feather`

### 1.2 Build targets + priors + IA weights
Inputs:
- `Train/train_terms.tsv`
- `IA.tsv`

Outputs:
- `real_targets.parquet` (labels)
- `priors.pkl` (term prior means)
- `nulls.pkl` (term NaN rates)

### 1.3 External GOA labels (optional but high leverage)
Goal:
- Download GOA GAFs, split by evidence, then **propagate** via `Train/go-basic.obo`.

Outputs (example naming):
- `prop_train_no_kaggle.tsv`, `prop_test_no_kaggle.tsv` (electronic labels; features)

Storage-safe recommendation (do this locally, then upload to Kaggle):
- Run the streaming filter script to keep only CAFA proteins and write **compressed** output:
  - `scripts/01_build_goa_features.py` → `goa_filtered_iea.tsv.gz` (recommended: `--only-iea`)
- Upload the resulting `.tsv.gz` as a Kaggle Dataset and mount it under `/kaggle/input/...`

Why this matters:
- GOA `goa_uniprot_all.gaf.*.gz` is huge; unzipping or writing unfiltered TSVs on Kaggle often hits disk limits.

Kaggle operational note:
- Write heavy intermediates to `/kaggle/temp` (bigger, ephemeral), not the notebook working directory.
- If you download models, set caches to `/kaggle/temp` (e.g., `HF_HOME`, `TRANSFORMERS_CACHE`, `TORCH_HOME`) to avoid filling the default disk.

Risk / sceptic note:
- External labels are powerful but can leak weird biases; treat them as *features* for stackers, not unquestioned truth.

### 1.4 Embeddings (start minimal, scale later)
Minimum recommended modalities:
- T5 embeddings
- ESM2 embeddings (650M)
- ESM2-3B embeddings
- Ankh embeddings
- TF-IDF text embeddings (EntryID→text corpus)
- Taxonomy features from `Train/train_taxonomy.tsv` + `Test/testsuperset-taxon-list.tsv`

Optional:
- None for the core pipeline (add only if clearly justified).

Colab notebooks (offline artefact generation):
- `notebooks/Colab_01_build_entryid_text_uniprot_pubmed.ipynb` (build `entryid_text.tsv`)
- `notebooks/Colab_02_generate_optional_embeddings.ipynb` (legacy/offline generation of embeddings artefacts)
- `notebooks/Colab_03_text_plus_solution.ipynb` (single notebook: FASTA→feather → corpus → TF-IDF → solution handoff)
- `notebooks/Colab_04_all_in_one.ipynb` (single notebook: inlines solution + corpus + TF-IDF code; no `!python` calls)

Reliability note:
- PubMed fetching is hardened in `Colab_04_all_in_one.ipynb` (sanitise invalid XML + retry/backoff + batch splitting) so long runs don’t crash on transient/invalid responses.

Option B operational note (Kaggle final stop):
- `notebooks/Colab_04_all_in_one.ipynb` runs in **strict mode** for TF-IDF + external GOA priors and includes an artefact manifest diagnostics cell.
- Publishing/checkpointing is handled exclusively via `STORE.push(...)` (no duplicated “publish cell” flow).
- `KAGGLE_USERNAME` + `KAGGLE_KEY` are required when the runtime needs to create/version or download the checkpoint dataset via the Kaggle API.

Outputs:
- An `embeds/` directory with one file per modality (format TBD by implementation).

---
## Phase 2 — Level-1 models (OOF predictions)

Goal:
- Train diverse base models and save **OOF predictions** (training features for GCN) and **test predictions**.

Recommended base set (minimum viable):
- Logistic regression (fast, strong baseline)
- One small GBDT variant

Scale set (when pipeline is solid):
- Multiple GBDT variants (raw vs conditional targets; different feature mixes)
- DNN ensemble (multi-input)

Deliverables per model:
- `oof_pred.pkl` (or `.npy/.parquet`)
- `test_pred.pkl`

---
## Phase 3 — Level-2 stacker (GCN per ontology)

Goal:
- Train 3 independent GCNs (BP/MF/CC) using:
  - Node features = concatenated Level-1 OOF predictions
  - Graph = GO hierarchy (from `Train/go-basic.obo`)

Deliverables:
- `gcn_bp.pt`, `gcn_mf.pt`, `gcn_cc.pt`
- TTA predictions: `pred_tta_*.tsv`
- Aggregated: `pred.tsv`

---
## Phase 4 — Strict post-processing + submission

Goal:
- Enforce GO hierarchy constraints post-hoc.

Steps:
1. **Max propagation** (upward consistency): parent score >= child score
2. **Min propagation** (downward consistency): child score <= parent score
3. Combine and format as `submission.tsv` (also enforce per-protein term cap)

Deliverables:
- `pred_max.tsv`
- `pred_min.tsv`
- `submission.tsv`

---
## First runnable slice (what we implement first)

Deliverables:
1. `config.yaml` + artefact directories
2. FASTA parsing + targets + IA weights
3. One embedding modality (or reuse existing baseline embeddings) + taxonomy
4. One Level-1 model producing OOF + test preds
5. Tiny GCN (one ontology first) + max-prop post-processing
6. Sanity-check submission format
