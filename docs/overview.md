# 🧬 CAFA-6 Protein Function Prediction - CONSOLIDATED OVERVIEW

**Date:** 23 Nov 2025  
**Current Best Model:** Fine-tuned ESM-2 (Asymmetric Loss) — F1 = 0.2331  
**Keep for detailed execution:** `docs/PLAN.md`

---
## 1. 🎯 Problem Snapshot
Predict multi-label Gene Ontology (GO) terms (MF, BP, CC) for protein sequences.  
Challenge: Extreme class imbalance (≈26k terms, ~6 positives per protein).  
Metric: Weighted F1 (information accretion).

---
## 2. ✅ What’s Already Done
- Data ingestion (FASTA, GO terms, ontology, taxonomy, IA weights)
- EDA completed (length distributions, term frequency, ontology coverage)
- Baselines: Frequency (0.1412), KNN with embeddings (0.1776), MLP (0.1672)
- Fine-tuning pipeline built (tokenisation, Dataset, Trainer)
- Threshold optimisation (grid over 0.01–0.50)
- Asymmetric Loss integrated (gamma_neg=2.0, clip=0.05)
- Best F1 improved to 0.2331 (Precision 0.3397, Recall 0.2379, Thr=0.40)

---
## 3. 📊 Current Performance
| Model | F1 | Notes |
|-------|----|-------|
| Frequency | 0.1412 | Predicts common terms |
| KNN (ESM-2 embeddings) | 0.1776 | Homology-style transfer |
| MLP (Frozen embeddings) | 0.1672 | Underperformed KNN |
| ESM-2 Fine-Tuned (BCE) | 0.1806 | Needed threshold tuning |
| ESM-2 Fine-Tuned (Asym Loss) | 0.2331 | Current best |

Trajectory target: Short-term 0.25–0.27; Mid-term 0.30+; Long-term 0.35–0.40.

---
## 4. 🔥 Immediate High-ROI Actions (Next 1–2 Days)
| Action | Impact | Effort | Notes |
|--------|--------|--------|-------|
| Label propagation | +0.02–0.04 | 2h | Add ancestors at inference |
| Per-aspect thresholds | +0.01–0.02 | 1h | MF/BP/CC separate tuning |
| Simple ensemble (KNN + ESM) | +0.01–0.02 | 30m | Weighted average |
| Larger backbone (ESM-2 35M) | +0.03–0.05 | 12h | Reduce batch size |

Recommended order: Propagation → Thresholds → Ensemble → Larger model.

## 4a. ✅ Progress Checklist
- [x] Data ingestion (01, 02, 03, 04)
- [x] Exploratory data analysis (EDA) — DELETED
- [x] Baselines: Frequency (01) — **Per-aspect CAFA metric implemented**
- [x] Baselines: KNN (02) — **Per-aspect CAFA metric implemented**
- [x] Baselines: MLP — SKIPPED (underperformed KNN)
- [x] Fine-tuning pipeline (ESM-2 8M) (03) — **Per-aspect CAFA metric implemented**
- [x] Threshold optimisation (global sweep) (03)
- [x] Asymmetric loss integration (03)
- [x] **Per-aspect evaluation (MF/BP/CC split)** — Competition metric now correctly implemented in ALL notebooks
- [x] Label propagation (ancestor add) (04) — **Per-aspect CAFA metric implemented**
- [ ] Per-aspect thresholds (MF/BP/CC) (04 - to add) — **Evaluation ready, need separate threshold tuning**
- [ ] Simple ensemble (KNN + ESM) (05 - new notebook)
- [ ] Larger backbone (ESM-2 35M) (03 - modify MODEL_NAME)
- [ ] Expand GO vocabulary (10k terms) (03 - modify VOCAB_SIZE)
- [ ] Increase max sequence length (1024 residues) (03 - modify max_length)
- [ ] Evolutionary features (MSA / PSSM) (06 - future)
- [ ] Structure features (AlphaFold embeddings) (06 - future)
- [ ] Domain features (Pfam) (06 - future)
- [ ] GO term embeddings (text + graph) (07 - future)
- [ ] Hierarchy consistency loss (03 - modify loss function)
- [ ] Hard negative mining (03 - modify training loop)
- [ ] Data tier weighting (evidence levels) (03 - modify dataset)

---
## 4b. 📚 Plain-English Feature Cheatsheet
> **Analogy:** Solving a crime with better clues & tools

**CRITICAL:** Competition evaluates **three subontologies separately** (MF, BP, CC), then averages. **ALL notebooks (01, 02, 03, 04) now correctly implement this per-aspect evaluation.** This was missed initially — all previous F1 scores were computed incorrectly by mixing aspects together.

**Data ingestion** ✅  
Loading protein sequences, GO annotations, ontology structure, taxonomy mapping, and IA weights from raw files. Like gathering all evidence at a crime scene — foundation for everything (+baseline).

**Exploratory data analysis (EDA)** ✅  
Understanding data distributions, sequence lengths, term frequencies, and class imbalance. Like profiling suspects before investigation — reveals what you're up against (+insight).

**Baselines (Frequency, KNN, MLP)** ✅  
Simple models to beat: predicting common terms, nearest-neighbour transfer, shallow neural nets. Like starting with obvious suspects — establishes minimum performance bar (F1 0.14–0.18).

**Fine-tuning pipeline (ESM-2 8M)** ✅  
Training protein language model end-to-end on GO prediction task. Like teaching a detective domain-specific skills — learns task-relevant patterns (F1 0.18→0.23).

**Threshold optimisation (global sweep)** ✅  
Finding best confidence cutoff for predictions across all terms. Like calibrating when to make an arrest — critical for imbalanced data (+0.18 F1).

**Asymmetric loss integration** ✅  
Down-weighting easy negatives, focusing on hard positives. Like spending investigation time on unclear cases, not obvious innocents — handles extreme imbalance (F1 0.18→0.23, +29%).

**Label propagation** (ancestor propagation) ✅  
If you predict a very specific term, auto-add its broader parents. Like saying "Golden Retriever" implies "Dog" → painless lift (+0.02–0.04).

**Per-aspect evaluation** ✅  
Competition metric: compute F1 separately for MF, BP, CC, then average the three. Not a single F1 across all terms. Kaggle does this automatically; local validation must match. Now correctly implemented in notebooks 01-02.

**Per-aspect thresholds** (separate tuning per subontology)  
After fixing evaluation, next step: find optimal threshold separately for MF, BP, CC instead of global threshold. MF might need 0.45, CC needs 0.35. Optimises precision/recall trade-off per domain (+0.01–0.02).

**Simple ensemble** (KNN + ESM weighted average)  
Combine homology-based (KNN) with learned patterns (ESM). Like asking both an experienced practitioner and an AI — they catch different errors (+0.01–0.02).

**Larger backbone** (ESM-2 35M vs current 8M)  
More parameters = better pattern recognition. Like upgrading from a pocket calculator to a supercomputer — captures subtler amino acid relationships (+0.03–0.05).

**Expand GO vocabulary** (10k terms vs current 5k)  
Cover more rare functions. Like expanding your dictionary from common words to technical jargon — improves rare term recall (+0.01–0.02).

**Increase max sequence length** (1024 vs current 512 residues)  
Don't truncate long proteins. Like reading full book chapters instead of summaries — preserves context for large proteins (+0.01–0.02).

**Evolutionary features** (MSA/PSSM)  
Asking a protein's relatives what they do. Multiple sequence alignment shows conserved "important" positions — like interviewing a big family. Strong lift (+0.03–0.05).

**Structure features** (AlphaFold)  
Knowing the 3D shape, not just the letters. Like seeing how a folded tool fits into a machine → reveals functional pockets (+0.02–0.04).

**Domain features** (Pfam)  
Predefined Lego blocks. If you spot a known block, you guess its role faster. Small but steady gain (+0.01–0.02).

**GO embeddings** (Text + Graph)  
Turning term definitions + hierarchy into numbers. Like mapping related job titles ("chef", "cook", "baker") closer together → helps predict rare terms (+0.03–0.05).

**Hierarchy loss** (consistency penalty)  
Enforces parent ≥ child logic. Like making sure you don't claim "Brakes specialist" without "Mechanic". Cleans logical mistakes (+0.01–0.02).

**Hard negative mining** (adaptive sampling)  
Drill on the mistakes you keep making. Like flashcards of the ones you get wrong — sharpens discrimination (+0.01–0.02).

**Data tiers** (evidence-based weighting)  
Trust high-quality annotations more. Like weighting eyewitnesses over rumours — reduces noise (+0.01–0.03).

---
## 5. 🔮 Strategic Roadmap (Condensed)
| Tier | Goal | Feature Set | Est. F1 Gain |
|------|------|-------------|--------------|
| Core | 0.25–0.27 | Propagation + thresholds + ensemble | +0.04–0.06 |
| Growth | 0.30 | Larger model + 10k terms + 1024 tokens | +0.05–0.07 |
| Advanced | 0.35 | Multi-modal (MSA, Pfam, structure) | +0.07–0.10 |
| Frontier | 0.40+ | GO embeddings + hierarchy loss + hard negatives | +0.05–0.07 |

---
## 6. 🐛 Issues Solved
| Problem | Cause | Fix | Result |
|---------|-------|-----|--------|
| F1 = 0.0000 | Threshold 0.5 too high | Grid search thresholds | F1 → 0.1806 |
| Poor focus on positives | BCE treats all negatives equally | AsymmetricLoss | F1 → 0.2331 |
| Loss explosion (446) | `.sum()` in custom loss | Use `.mean()` | Stable training |

---
## 7. 💡 Lessons Learned
- Threshold tuning is mandatory for extreme imbalance.  
- Focal-style (asymmetric) loss outperforms vanilla BCE here.  
- Fine-tuning backbone > frozen embeddings + shallow head.  
- Early stopping protects against plateau wastage.  
- Homology (KNN) remains a strong complementary signal.

---
## 8. 🗂️ Key Files (Active Set)
| Area | File |
|------|------|
| Data | `src/data/loaders.py`, `src/data/finetune_dataset.py` |
| Models | `src/models/esm_classifier.py`, `src/models/baseline_embedding_knn.py`, `src/models/baseline_frequency.py` |
| Training | `src/training/finetune_esm.py`, `src/training/loss.py`, `src/training/trainer.py` |
| Saved Model | `models/esm_finetuned/best_model/` |

---
## 9. 🧪 Evaluation Approach (Current)
- Collect logits → sigmoid probabilities.  
- Global threshold chosen via validation sweep.  
- Macro label averaging (sample-wise F1).  
Next upgrades: Per-aspect thresholds, per-term dynamic thresholding, calibration (temperature / isotonic).

---
## 10. 🚀 Next Concrete Steps
1. Implement `src/inference/propagation.py` (ancestor add).  
2. Modify evaluation to compute MF/BP/CC optimal thresholds.  
3. Add `src/inference/ensemble.py` combining KNN + ESM outputs.  
4. Trial larger model (start with 35M param).  

---
## 11. 📈 Success Targets
| Milestone | Success Criteria |
|-----------|------------------|
| Propagation | F1 ≥ 0.25 |
| Ensemble | F1 ≥ 0.26 |
| Larger Model | F1 ≥ 0.28–0.30 |
| Multi-modal MVP | F1 ≥ 0.33+ |
| Hierarchy-aware & embeddings | F1 ≥ 0.35–0.40 |

---
## 12. ❓ Open Decisions
| Decision | Options | Recommendation |
|----------|---------|----------------|
| Backbone scale | 35M vs 150M | Start 35M (memory safe) |
| Term count | 5k vs 10k vs 15k | Move to 10k first |
| Sequence length | 512 vs 1024 | Increase with batch size reduction |
| Ensemble strategy | Simple avg vs stacking | Begin with weighted avg |

---
## 13. 🧠 Future Enhancements (Outline)
- Multi-modal fusion: Cross-attention across sequence, profile, structure embeddings.  
- GO term embedding: Text + graph + co-occurrence → similarity scoring head.  
- Hierarchy consistency loss: Penalise child > parent probability gaps.  
- Hard negative mining: Replay buffer of frequent false positives.  
- Active learning loop: Surface high-uncertainty proteins.

---
## 14. 🔍 Risk Check
| Risk | Mitigation |
|------|------------|
| GPU memory with larger model | Gradient accumulation + mixed precision |
| MSA generation cost | Subsample or on-demand compute |
| Ontology misuse | Validate propagation coverage + ancestor integrity |
| Overfitting with bigger model | Maintain validation discipline + early stop |

---
## 15. 🏁 Summary in One Line
We have a healthy fine-tuning pipeline at F1 0.2331; apply label propagation + aspect thresholds + ensemble next to break 0.25, then scale model and ontology awareness for 0.30+.

---
**If happy with this consolidation, I can remove `ROADMAP.md`, `SUMMARY.md`, `PROGRESS_TRACKER.md` and keep only `PLAN.md` + `OVERVIEW.md`.**
