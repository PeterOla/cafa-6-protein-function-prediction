# 🧬 CAFA-6 Protein Function Prediction — Overview

**Date:** 11 Dec 2025  
**Best validated baseline:** KNN + per-aspect thresholds — F1 = 0.2579 (MF=0.40, BP=0.20, CC=0.40)  
**Decision:** KNN is not competitive enough → pivot to a CAFA5 Rank-1 style stack (modular Level-1 ensemble → Level-2 GCN stacker → strict GO post-processing).  
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
- Multimodal embeddings (at minimum: T5 + ESM2 + taxonomy; optional: Ankh/text).

### Phase 2 — Level-1 models (OOF features)
Deliverables:
- Out-of-fold (OOF) predictions for each base model (GBDTs + logistic regression + DNN ensemble).
- Test predictions for each base model.

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
- [x] Optional: offline embedding generator supports ESM2-3B + Ankh + 10279D text (TF-IDF)
- [x] Optional: build `EntryID -> text` corpus from UniProt + PubMed (`scripts/03_build_entryid_text_from_uniprot_pubmed.py`)
- [x] Colab notebook: build `entryid_text.tsv` (UniProt + PubMed) (`notebooks/Colab_01_build_entryid_text_uniprot_pubmed.ipynb`)
- [x] Colab notebook: generate optional embeddings artefacts (`notebooks/Colab_02_generate_optional_embeddings.ipynb`)
- [x] Colab notebooks normalised to Jupyter schema (cell `metadata.id`, `metadata.language`, `nbformat` keys)
- [ ] Phase 3: train GCN stacker (BP/MF/CC) + TTA aggregation
- [ ] Phase 4: strict min/max propagation + final submission generation
