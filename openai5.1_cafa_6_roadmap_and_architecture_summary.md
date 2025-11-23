# CAFA6 Roadmap and Architecture Summary

**Date:** 23 November 2025

This document describes a practical, high impact roadmap and a concrete system architecture to push your CAFA model from F1 around 0.23 toward the theoretical ceiling, using multi modal signals, ontology aware modelling, and production grade training and inference.

# 1 Goals

1. Maximise weighted F1 on the CAFA style benchmark
2. Build a pipeline that is ontology aware, multi modal, and actively improves with human in the loop curation
3. Make progress measurable and repeatable with a curated high quality holdout set

# 2 High level system overview

Components

1. Data curation and augmentation service, responsible for evidence based label tiers, homology driven label transfer, and curated holdout management
2. Feature extraction service, producing: LM embeddings, MSA profiles, PSSMs, Pfam domain hits, transmembrane and signal peptide flags, AlphaFold or ESMFold structure embeddings and structure fingerprints, taxon embeddings
3. Fusion model backbone, a multi modal encoder that combines sequence LM embeddings, structure fingerprints, and evolutionary profiles via cross attention
4. GO embedding service, producing vector representations for GO terms from: textual definitions, synonyms, and graph positional encodings
5. Task head, a similarity based compatibility scorer between protein embeddings and GO term embeddings, with hierarchical and asymmetric losses
6. Training and monitoring orchestrator, handling batching, hard negative mining, contrastive objectives and model checkpoints
7. Inference engine, performing ensemble calibration, hierarchy consistent decoding, and per term thresholding
8. Active learning and human in the loop UI, exposing high uncertainty cases for curator review

# 3 Data organisation and quality tiers

* Tier 1, high confidence: experimentally validated annotations with direct evidence codes
* Tier 2, medium confidence: high quality computational or orthology transferred annotations, tracked with provenance
* Tier 3, low confidence: inferred or automatic annotations

Store each protein record with: sequence, taxon id, list of evidence coded GO annotations, PSSM or MSA paths, structure path and per residue confidence scores

# 4 Feature extraction pipeline

* Sequence LM embeddings
  * Use ProtT5 XL or ESM2 150M as primary backbones
  * Produce per residue and pooled sequence embeddings
* Evolutionary features
  * Run HHblits or JackHMMER to produce MSAs and PSSMs
  * Save compressed profiles for reuse
* Structural features
  * Predict structure with AlphaFold or ESMFold where available
  * Extract graph features from residue contact map and per residue surface descriptors
* Domain and motif features
  * Pfam HMM scan, signal peptide prediction, transmembrane helices
* Taxon and metadata embeddings
  * Encode taxonomic lineage and any available expression or localisation priors

# 5 Model architecture, succinct description

## 5.1 Backbone encoders

* Sequence encoder, pre trained LM
  * Input: raw sequence or token windows for long proteins
  * Output: per residue embeddings and pooled embedding
* Structure encoder
  * Input: coordinates or contact map
  * Output: structure fingerprint vector and per residue structural embeddings
* Profile encoder
  * Input: PSSM or MSA derived features
  * Output: profile embedding

## 5.2 Fusion encoder

* Cross attention block that attends across modality embeddings, producing a unified protein embedding vector P
* Learnable pooling layer to summarise variable length proteins into a fixed size vector

## 5.3 GO embedding module

* Create GO term vectors G_i by combining: encoded textual definition, positional graph embeddings from Node2Vec or GNN, and information accretion weight

## 5.4 Compatibility head

* Compute compatibility score s_i = f(P, G_i) using dot product or small MLP on concatenated vectors
* Convert s_i to probability via temperature scaled sigmoid or softmax variant for grouped terms

# 6 Losses and training objectives

* Main loss, asymmetric focal like loss emphasising positives and down weighting easy negatives
* Hierarchical consistency penalty, applying a small cost when child probability exceeds parent probability in an inconsistent way
* Contrastive loss at batch level, pulling proteins with similar GO sets together and pushing apart dissimilar ones
* Calibration objective on validation set, used post training to set per term thresholds

# 7 Inference and decoding

1. Produce per term probability estimates from ensemble of models
2. Calibrate probabilities using temperature scaling or isotonic regression per aspect
3. Apply per term thresholds, learned from validation, regularised by IA weight and term frequency
4. Propagate positives up the GO DAG to ensure ancestor coverage and hierarchy consistency
5. Apply final hierarchy smoothing: if many children are predicted but parent misses threshold, boost parent to minimal pass threshold

# 8 Hard negative mining and active learning

* Maintain a buffer of hard negatives discovered during validation and training
* Sample them with elevated probability for subsequent training epochs
* Score model uncertainty by ensemble disagreement and margin sampling
* Expose top k uncertain proteins to the curation UI for human review and provenance assignment

# 9 Ensemble and stacking

* Base models should be diverse in architecture and features
  * Example set: ProtT5 / ESM2 model, structure aware model, profile heavy model, graph neural network on residue graph
* Meta learner receives base model confidences, IA weights, term priors and returns final calibrated scores

# 10 Evaluation and monitoring

* Maintain two validation sets
  * Noisy validation for quick iterations
  * Curated holdout for true measurement
* Stratified metrics by term frequency, aspect, taxon and sequence length
* Track per term precision recall and IA weighted F1

# 11 Files and code organisation suggestions

* src/data: loaders, curators, augmentation scripts
* src/features: wrappers for HHblits, AlphaFold, Pfam scanning
* src/models: backbones, fusion encoder, GO embedding module, task head
* src/training: losses, schedulers, miners, experiment runner
* src/inference: calibration, propagation, threshold store
* notebooks: experiments and error analysis

# 12 Minimal viable implementation plan and timeline

Phase A, 2 weeks

1. Build curated tiered dataset and high quality holdout set
2. Implement feature extraction for profiles and Pfam
3. Implement GO embedding module, label propagation logic and basic compatibility head
4. Train a single fusion model with ProtT5 or ESM2 150M

Phase B, 3 weeks

1. Add structure based features and fuse them in the encoder
2. Implement hard negative mining and contrastive objective
3. Add per term calibration and hierarchy aware thresholds
4. Create simple stacking ensembling and meta learner

Phase C, ongoing

1. Active learning loop with curator UI
2. Iterative retraining with newly curated labels
3. Scale models and experiments to find practical ceiling

# 13 Code snippets and pseudocode

*Compatibility scoring pseudocode*

```
# P is protein embedding, G is matrix of GO embeddings
scores = P @ G.T  # dot product similarity
probs = sigmoid(scores / temperature)
```

*Hierarchy propagation pseudocode*

```
for each protein:
  predicted_terms = {i | prob_i >= thresh_i}
  for each term in predicted_terms:
    add all ancestors of term to predicted_terms
```

# 14 Risks and mitigations

* Risk: computing AlphaFold for large datasets is expensive
  * Mitigation: restrict structure predictions to proteins above uncertainty threshold or to a representative subset, or use ESMFold
* Risk: overfitting on curated holdout
  * Mitigation: maintain separate unseen expert curated test set and rotate only limited experiments against it

# 15 Next steps

1. Confirm choice of primary backbone, compute budget and curated holdout staffing
2. I will produce the detailed architecture diagram and example implementation files for the fusion model, GO embedding module, hierarchical loss, and inference pipeline when you confirm



---

End of summary

