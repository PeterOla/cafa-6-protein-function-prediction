# 🧬 CAFA-6 Protein Function Prediction — Overview

**Date:** 11 Dec 2025  
**Best validated baseline:** KNN + per-aspect thresholds — F1 = 0.2579 (MF=0.40, BP=0.20, CC=0.40)  
**Decision:** KNN alone is not competitive enough → pivot to a CAFA5 Rank-1 style stack (modular Level-1 ensemble → Level-2 GCN stacker → strict GO post-processing).  
**Detailed execution:** `docs/PLAN.md`

---
## 1. Problem
Predict multi-label Gene Ontology (GO) terms for protein sequences across MF/BP/CC.  
Metric: IA-weighted F1, averaged over MF/BP/CC (per-aspect evaluation is mandatory).

---
## 2. Where we are (truth, not vibes)
- Validation + metric are now correct locally (MF/BP/CC split).
- KNN got us to F1=0.2579, but it plateaus: it transfers homology, not biology.
- Label propagation on KNN is counterproductive (noise gets amplified up the hierarchy).

---
## 3. New approach (what we are actually building)
**Core idea:** treat strong but diverse base predictors as feature generators, then let a GCN learn how to reconcile them using the GO graph.

### Phase 0 — Environments + config
Deliverables:
- Two environments: `rapids-env` (preproc + linear/GBDT) and `pytorch-env` (embeddings + GCN).
- Kaggle-first setup (single notebook) with paths and artefact directories.

### Phase 1 — Feature engineering at scale
Deliverables:
- Parsed sequences + targets in efficient formats.
- External GO annotations (especially electronic) parsed and hierarchy-propagated.
- Multimodal embeddings (core: T5 + ESM2-650M + ESM2-3B + Ankh + TF-IDF text + taxonomy).

### Phase 2 — Level-1 models (OOF features)
Deliverables:
- Out-of-fold (OOF) predictions for each base model (GBDTs + logistic regression + DNN ensemble).
- Test predictions for each base model.
- **Target Scope**: 1,500 GO terms (Top-K). Scalable to 13k+ via RAPIDS.

### Phase 3 — Level-2 stacker (GCN per ontology)
Deliverables:
- Three GCNs (BP/MF/CC) trained on Level-1 OOF predictions.
- Test-time augmentation (TTA) predictions and averaged `pred.tsv`.

### Phase 4 — Strict post-processing + submission
Deliverables:
- Max-propagation + min-propagation over GO graph.
- Final `submission.tsv` that is hierarchy-consistent and format-valid.

---
## 4. Immediate next steps (implementation order)
1. Add `docs/PLAN.md` and align repo paths + artefact locations (Windows-friendly).
2. Decide the minimal “first runnable slice”:
   - Start with (T5 + taxonomy) → (logreg + small GBDT) → (tiny GCN) → (max-prop only).
3. Implement the data artefact pipeline (parse FASTA, build label matrices, IA weights).
4. Only then scale: more modalities, more base models, full GCN + TTA + min/max postproc.

---
## 4a. ✅ Progress checklist (single source of truth)
- [x] Data ingestion (FASTA, GO terms, ontology, taxonomy, IA weights)
- [x] Baseline: Frequency (per-aspect CAFA metric implemented)
- [x] Baseline: KNN (per-aspect CAFA metric implemented) — best F1=0.2579
- [x] Label propagation on KNN tested — failed (amplifies errors)
- [x] CAFA5-style stacker prototype notebook exists (`notebooks/06_cafa5_style_stack.ipynb`)
- [x] Overview cleaned and pivoted to Rank-1 stack plan
- [x] Create `docs/PLAN.md` with end-to-end execution steps
- [x] Kaggle setup notebook added (`notebooks/CAFA6_Rank1_Solution.ipynb`)
- [x] Competition rules snapshot captured (`rules.md`)
- [x] Phase 0: verify Kaggle GPU and run notebook end-to-end
- [x] Phase 1: Data Structuring (FASTA -> Feather, OBO parsing, Priors)
- [x] Phase 1: generate multimodal embeddings (T5 + ESM2 implemented)
- [x] Phase 1: Taxonomy features implemented
- [x] Phase 1: external GO features (UniProt GAF - optional)
- [x] Local GOA precompute notebook + filtered artefact (`goa_filtered_iea.tsv.gz`)
- [x] Kaggle GOA artefact discovery hardened (auto-scan `/kaggle/input`; supports `.tsv` and `.tsv.gz`)
- [x] Phase 1: external GOA hierarchy propagation (IEA) + injected into GCN stacker inputs
- [x] Phase 2: train Level-1 models + save OOF predictions (IA-weighted DNN loss + IA-F1 threshold diagnostics)
- [x] Offline embedding generator supports ESM2-3B + Ankh + 10279D text (TF-IDF)
- [x] TF-IDF embeddings (Stage 03) validated non-zero and pushed to checkpoint dataset (`stage_03_tfidf_text`)
- [x] TF-IDF text coverage audited (train vs test): 100% non-empty; test median text shorter (explains nnz/row shift)
- [x] Optional: build `EntryID -> text` corpus from UniProt + PubMed (`scripts/03_build_entryid_text_from_uniprot_pubmed.py`)
- [x] Colab notebook: build `entryid_text.tsv` (UniProt + PubMed) (`notebooks/Colab_01_build_entryid_text_uniprot_pubmed.ipynb`)
- [x] Colab notebook: generate optional embeddings artefacts (`notebooks/Colab_02_generate_optional_embeddings.ipynb`)
- [x] Colab notebooks normalised to Jupyter schema (cell `metadata.id`, `metadata.language`, `nbformat` keys)
- [x] Colab notebook: single-file text + TF-IDF + solution handoff (`notebooks/Colab_03_text_plus_solution.ipynb`)
- [x] Colab notebook: all-in-one (solution + text corpus + TF-IDF inline, no script calls) (`notebooks/Colab_04_all_in_one.ipynb`)
- [x] Option B: strict mode + Kaggle API dataset publishing integrated into `Colab_04` (requires `KAGGLE_USERNAME` + `KAGGLE_KEY`)
- [x] Colab_04: artefact manifest diagnostics cell (sizes + Option B required artefacts sanity check)
- [x] Colab_04: Kaggle Dataset-backed milestone checkpointing (`STORE.pull()` + `STORE.push(stage, ...)`)
- [x] Colab_04: unified secrets getter (env → Kaggle Secrets → Colab `userdata`) for Kaggle creds + dataset ID
- [x] Runbook: resumable runs across Kaggle/Colab/local (`docs/RUNBOOK_CHECKPOINTS.md`)
- [x] Colab_04: clearer failure mode for checkpoint pull HTTP 403 (private/not-shared dataset guidance)
- [x] Colab_04: checkpoint pull exports Kaggle creds + surfaces Kaggle CLI stderr on failure (incl. 404 guidance)
- [x] Colab_04: compact notebook formatting (remove accidental excessive blank lines)
- [x] Colab_04: removed markdown cells + clarified embeddings generator label
- [x] Colab_04: added CAFA_FORCE_REBUILD + improved Colab dataset discovery (CAFA_DATASET_ROOT)
- [x] Colab_04: Colab-only Kaggle API auto-download for competition files (CAFA_COLAB_AUTO_DOWNLOAD)
- [x] Colab_04b: strict file-by-file hydration for Data Explorer artefacts (v28)
- [x] Colab_04: Colab competition download unzips via Python (no `kaggle --unzip`)
- [x] Colab_04: fix Cell 3 diagnostics plot syntax (smoke-check clean)
- [x] Colab_04: fail-fast FASTA readability checks after setup
- [x] Colab_04: checkpoint publishing uploads folders (--dir-mode zip)
- [x] Colab_04: checkpoint publishing skips unchanged stages (CAFA_CHECKPOINT_FORCE_PUSH)
- [x] Colab_04b: Fix missing features download (file-by-file strategy for unzipped dataset)
- [x] Colab_04: fix GOA propagation ID matching (normalise UniProt headers)
- [x] Colab_04: resolve CAFA_FORCE_REBUILD from Kaggle Secrets
- [x] Colab_04: avoid re-uploading checkpoints when embeddings already exist (opt-in via `CAFA_CHECKPOINT_PUSH_EXISTING=1`)
- [x] Colab_04: ESM2 cell generates ESM2-3B + Ankh (skip-if-exists + checkpoint push)
- [x] Colab_04: require ESM2-3B + Ankh + taxonomy (fail-fast if missing)
- [x] Colab_04: split Level-1 training into per-model cells + per-model checkpoint pushes
- [x] Colab_04: make GBDT mandatory (fail if `py_boost` missing) + fix split `STORE.push(required_paths=...)`
- [x] GPU VM: documented Python 3.11 + CuPy + py_boost setup for kernel-init failures (`docs/ENV_GPU_PYBOOST_PY311.md`)
- [x] Colab_04: remove unused embedding-generator cell; TF-IDF run cell is self-contained
- [x] 05: no-repo notebook bootstrap + mandatory `cafa6_data/` root + mandatory file-by-file downloads + coherent `STORE.pull/push`
- [x] 05: checkpoint publishing is strict: stage-only uploads; never publish downloaded/pulled files; only publish freshly-built outputs
- [x] 05: checkpoint pull/push are env-controlled with safe local defaults (avoid accidental Kaggle pulls/pushes)
- [x] 05: staged checkpoint requirements (CAFA_CHECKPOINT_PULL_STAGE) + GCN stacker emits `test_pred_gcn.npy`
- [x] 05: final audit — `cafa6_data/` is canonical; checkpoint store de-duped + manifest fixed; notebook passes syntax + name smoke-checks
- [x] 05: Phase 2 setup aligned with 04b (aspect Top-K split + guardrail) + Ankh mandatory + disk-backed `X_train_mmap.npy`/`X_test_mmap.npy`
- [x] 05: local run verified up to (excluding) embeddings generation (pre-Ankh)
- [x] 05: added verbose heartbeat + ETA logs for Level-1 DNN and KNN
- [x] 05: GBDT uses contiguous RAM arrays for mini-fit + fold slices (prevents numba TypingError)
- [x] 05: GBDT uses fold validation via `eval_sets` (enables `es=` early stopping) + fixes resume print path
- [x] 05: GBDT refactor to multi-output per fold (predict `X_test` once per fold, not 1,585×)
- [x] 05: LogReg materialises `X_test` + fold `X_val` into RAM (reduces memmap I/O bottlenecks)
- [x] `notebooks/05_cafa_e2e.ipynb`: LogReg upgraded to RAM-safe + RAPIDS-when-feasible implementation (tqdm ETA + disk-backed folds)
- [x] `notebooks/05_cafa_e2e.ipynb`: Phase 2 LogReg split by aspect (BP/MF/CC) + per-aspect OOF/test artefacts + combined 13,500-term output
- [x] `notebooks/05_cafa_e2e.ipynb`: Phase 2 LogReg RAM cliff fix (no `Y_aspect` fancy-index copy; mandatory per-aspect runs; batched val/test)
- [x] `notebooks/05_cafa_e2e.ipynb`: Phase 2 LogReg deep GPU telemetry (CuPy `memGetInfo` + pool stats around `X_tr_gpu` and first cuML fit)
- [x] `notebooks/05_cafa_e2e.ipynb`: LogReg RAPIDS/cuML OVR inference hardened (no `predict_proba` assumption; `decision_function`→sigmoid fallback) + CPU fallback avoids joblib subprocesses
- [x] `notebooks/05_cafa_e2e.ipynb`: LogReg runtime: pre-scale `X_test` once per fold + keep `X_test` on GPU per fold when possible (avoid repeated PCIe transfers) + larger GPU/CPU test batches (no quality change)
- [x] Wrapper notebooks for parallel per-aspect LogReg runs (`notebooks/05_cafa_e2e_run_MF.ipynb`, `notebooks/05_cafa_e2e_run_CC.ipynb`)
- [x] `scripts/audit_gbdt_target_scale.py`: Audit expanding GBDT targets 4,500→6,000 (frequency drop-off, <50 noise floor rate, IA contribution)
- [x] `scripts/audit_optimal_topk_by_aspect.py`: Per-aspect (BP/MF/CC) optimal Top-K audit under IA vs noise-floor trade-off
- [x] `scripts/verify_signal_horizon_1585.py`: Foolproof verification of the 1,585 stable targets from Phase-1 artefacts (namespace integrity, ≥50 positives, IA coverage)
- [x] Colab_04: PubMed fetch hardened (sanitize invalid XML + retry/backoff + recursive batch split)
- [x] Colab_04: ProtT5 checkpoint pushes after train and after test (granular, crash-safe)
- [x] Colab_04: ESM2/ESM2-3B/Ankh checkpoint pushes after train and after test (granular, crash-safe; Ankh keyed by model via `CAFA_ANKH_MODEL`)
- [x] Phase 2: add KNN as Level-1 model (OOF + test preds) and wire into GCN stacker
- [x] Colab_04b: Phase 2a LogReg uses disk-backed folds + streamed scaling (avoids Colab RAM OOM)
- [x] Local: Phase 2a streaming LR mini-run (disk-backed X/Y + tiny fold completes)
- [x] Colab_04b: normalise `train_terms.aspect` (namespace strings → BP/MF/CC) + fail-fast guardrail (prevents silent target fallback)
- [x] Colab_04b: enable RAPIDS/cuML LogReg path by default (`USE_RAPIDS_LOGREG=True`)
- [x] Colab_04: visual diagnostics for embeddings + Level-1 models (histograms + IA-F1 curves)
- [x] IA-weighted evaluation of Level-1 OOF predictions (Rank-1 thresholds) (`notebooks/06_evaluate_oof_predictions (1).ipynb`)
- [x] Ankh embeddings: AMP fallback + non-finite guard (prevents silently saving all-NaN `.npy` artefacts)
- [x] Option A: first-submission notebook that skips Ankh (`notebooks/Colab_04b_first_submission_no_ankh.ipynb`)
- [x] Option A: first-submission notebook trimmed (reuse artefacts) + optional checkpoint republish without Ankh/ankr `.npy` (Colab_04b)
- [x] Colab: secrets fetched only via `from google.colab import userdata; userdata.get('...')`
- [x] Colab_04_all_in_one: enforce userdata-only secrets + rerun Kaggle CLI on empty error output
- [x] Phase 3: train GCN stacker (BP/MF/CC) + emit `test_pred_gcn.npy`
- [x] Phase 4: strict min/max propagation + final submission generation
- [x] HF: migrate checkpoint publishing to Hugging Face Hub (HF-only; no Kaggle publish) + bulk uploader (`scripts/hf_upload_cafa6_data.py`)
- [x] HF: verify uploaded artefacts and LFS status (`scripts/hf_verify_cafa6_data_upload.py`)
- [x] `notebooks/05_cafa_e2e.ipynb`: remove Kaggle dataset emergency restore (no in-notebook full-tree republish)
- [x] `notebooks/05_cafa_e2e.ipynb`: DNN refactored to per-modality multi-branch heads + 5 seeds × 5 folds (25 models)
- [x] `notebooks/05_cafa_e2e.ipynb`: GCN stacker uses concatenated Level-1 OOF features (not mean) + Phase 4 max/min propagation iterates >=10
