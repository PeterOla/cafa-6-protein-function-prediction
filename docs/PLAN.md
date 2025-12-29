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

Status:
- ✅ `notebooks/05_cafa_e2e.ipynb`: removed obsolete `CAFA_INPUT_ROOT`/`INPUT_ROOT` wiring; notebook passes syntax smoke-check.
- ✅ `notebooks/05_cafa_e2e.ipynb`: removed Kaggle dataset emergency restore cell entirely (no full-tree republish path).

Run:
- `notebooks/00_setup_kaggle_colab.ipynb`

Notes (pragmatic):
- We are **Kaggle-first**: no external config files are required to run.
- If you have multiple datasets mounted, set `DATASET_SLUG` inside the notebook.

Checkpointing (resumable across providers):
- Use a Hugging Face Hub dataset repo as the canonical artefact store.
- The notebook pulls checkpoints on startup (`STORE.pull(required_files=...)`) and **uploads immediately after each artefact/stage is generated** (`STORE.maybe_push(stage, required_paths, ...)`).
- Control the target repo via `CAFA_HF_REPO_ID` (e.g. `PeterOla/cafa6-checkpoints`) and authenticate via `HF_TOKEN` / `HUGGINGFACE_TOKEN`.
- Colab-only rule: fetch secrets exclusively via `from google.colab import userdata; userdata.get('...')`.
- Local/other notebooks: rely on env vars (optionally loaded from a repo `.env`, gitignored).
- Bulk snapshot tool: `scripts/hf_upload_cafa6_data.py` uploads the whole `cafa6_data/` tree when you want a one-off sync.



Practical note (non-negotiable publishing rule):
- The notebook must **never publish downloaded/pulled files** (competition inputs, checkpoint pulls, mounted inputs).
- It must **only publish freshly-built artefacts from the current runtime**.
- Implementation detail: publishing happens from an isolated staging directory containing only the newly built outputs (so the Kaggle CLI never zips the whole working tree).
- This behaviour is enforced in-code even if you enable checkpointing via env vars (publishing is still blocked for downloaded/pulled or non-fresh artefacts).

Operational note (important):
- `CAFA_CHECKPOINT_PULL` / `CAFA_CHECKPOINT_PUSH` are env-controlled; safe defaults are used on local runs (pull off, push off).
- `CAFA_CHECKPOINT_PULL_STAGE` selects which milestone to hydrate (cumulative). Useful stages in `notebooks/05_cafa_e2e.ipynb`:
  - `stage_01_parsed` (parsed tables)
  - `stage_03_tfidf_text` (TF-IDF vectoriser + `.npy`)
  - `stage_04_external_goa_priors` (propagated IEA priors)
  - `stage_06_embeddings_core` (T5/ESM2/ESM2-3B/Ankh)
  - `stage_07_level1_preds` (Level-1 OOF + test preds)
  - `stage_08_stacker_gcn` (writes `features/test_pred_gcn.npy` needed for submission)
- Kaggle API “single file” downloads are reliable for top-level files, but can 404 for nested paths on some datasets.
  For deterministic file-by-file hydration across environments, either publish a *flat* checkpoint dataset (e.g. `features__x.npy` naming) or attach the dataset as a Kaggle Notebook Input (mounted filesystem copy).

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
- TF-IDF text embeddings (EntryID→text corpus)
- Taxonomy features from `Train/train_taxonomy.tsv` + `Test/testsuperset-taxon-list.tsv`

Option A (first submission, faster / safer):
- Skip Ankh entirely (see `notebooks/Colab_04b_first_submission_no_ankh.ipynb`) until Ankh artefacts are regenerated and pass non-finite checks.
- The no-Ankh notebook is trimmed for reuse-artefacts runs: checkpoint pushes are disabled by default; optional cleanup republish is gated by `CAFA_CLEAN_CHECKPOINT_REMOVE_ANKH=1`.

Main pipeline (05 / Colab_04):
- Ankh embeddings are mandatory (fail-fast if missing).

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

Validated (local run, `notebooks/05_cafa_e2e.ipynb`):
- EntryID→text coverage is 100% non-empty for both train and test.
- Test texts are shorter on average (median chars lower), which explains lower TF-IDF nnz/row on test without implying missing text.

---
## Phase 2 — Level-1 models (OOF predictions)

Goal:
- Train diverse base models on the embeddings to predict GO term probabilities.
- **Target Scope**: 13,500 GO terms (Champion Strategy).
  - **Breakdown**: 10,000 BP + 2,000 MF + 1,500 CC.
  - **Validation**: Analysis of CAFA 6 data confirms this split covers >90% of annotations per aspect (BP: 95%, MF: 90%, CC: 98%).
  - **Implementation**: Requires RAPIDS (GPU) to handle the ~1.7B parameters.

Hard requirements (current pipeline):
- GBDT via `py_boost` is mandatory: the notebook fails fast if the package is missing.
- If `py_boost` GPU kernels fail to initialise on a CUDA VM (e.g. `feature_grouper_kernel is None`), use the Python 3.11 setup in `docs/ENV_GPU_PYBOOST_PY311.md`.
- For runtime, GBDT trains as a single multi-output model per fold (1 call to `predict(X_test)` per fold), rather than 1,585 independent per-target predicts.
- Checkpoint publishing uses `STORE.push(stage, required_paths, note)`; split per-model cells must pass `required_paths=`.
- Colab_04b Phase 2a LogReg avoids `X[idx_tr]`/`fit_transform` full copies by using disk-backed folds + streamed scaling.
- `notebooks/05_cafa_e2e.ipynb` trains LogReg **per aspect** (BP/MF/CC) and writes per-aspect predictions under `features/level1_preds/` (also assembles combined `oof_pred_logreg.npy` / `test_pred_logreg.npy` for the 13,500-term contract).
- `notebooks/05_cafa_e2e.ipynb` LogReg GPU path keeps fold-scaled `X_test` on GPU when feasible to avoid repeating host→device transfer per target-chunk (major runtime win on A100-class GPUs).
- To run MF and CC in parallel (without copying `05_cafa_e2e.ipynb`), use wrapper notebooks: `notebooks/05_cafa_e2e_run_MF.ipynb` and `notebooks/05_cafa_e2e_run_CC.ipynb`.
- `notebooks/05_cafa_e2e.ipynb` Phase 2 setup mirrors the 04b target-selection logic and builds disk-backed `features/X_train_mmap.npy` + `features/X_test_mmap.npy` so downstream per-model cells can stay RAM-safe.
- Colab_04b Phase 2a LogReg defaults to RAPIDS/cuML when available (`USE_RAPIDS_LOGREG=True`).
- Colab_04b target selection normalises `train_terms.aspect` (namespace strings → BP/MF/CC) and fails fast if the split is missing (prevents silent global fallback).

Recommended base set (minimum viable):

Scale set (when pipeline is solid):

Deliverables per model:

 Diagnostics (keep it visual, keep it cheap):
 - The all-in-one notebook plots **OOF probability histograms** and **IA-F1 vs threshold** curves per Level-1 model.
 - Embeddings are sanity-checked via **train vs test L2-norm histograms** (sampled).
 - Embedding generation **fails fast** if any non-finite values (NaN/Inf) are detected (prevents silently saving broken Ankh arrays).
 - Control sampling cost via `CAFA_DIAG_N` (default: 20000 rows).

 End-of-run analysis (what helps vs hurts):
 - The all-in-one notebook includes a final **OOF ablation** cell that computes leave-one-out deltas and greedy forward selection for a simple mean-ensemble.
 - This is **read-only** and is intended to answer: “which model is actually contributing to the score?”

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
