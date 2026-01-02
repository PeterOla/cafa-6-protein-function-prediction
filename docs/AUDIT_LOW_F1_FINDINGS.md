# Audit: Near-zero IA-F1 — Evidence-led Findings (local, 2026-01-02)

This note documents **observed facts** (with reproducible commands/outputs) explaining why IA-weighted F1 is near-zero in the current local artefacts.

Scope:
- No assumptions about “mislabelling” or “mismatched joins” unless demonstrated.
- Focus on proving (a) what is *not* broken (raw data integrity) and (b) what *is* broken (prediction distributions + thresholds), using direct measurements.

Repro command:
- `python scripts/audit_dataset_consistency.py`

---

## 1) Raw training data integrity checks (TSV schemas + joins)

**Fact 1.1 — `train_terms.tsv` schema and formats are clean.**

Audit output (direct):
- `annotations: 537,027`
- `proteins: 82,404`
- `terms: 26,125`
- `bad_go_id: 0`
- `bad_aspect: 0`
- `bad_entryid: 0`

Interpretation (minimal):
- The label file is parseable and consistent with expected formats.

**Fact 1.2 — `train_taxonomy.tsv` aligns 1:1 with `train_terms.tsv` EntryIDs.**

Audit output (direct):
- `rows: 82,404`
- `proteins: 82,404`
- `bad_taxon_format: 0`
- `EntryID overlap train_terms vs train_taxonomy: 82,404 / 82,404`
- `missing_in_taxonomy: 0`
- `extra_in_taxonomy: 0`

Interpretation (minimal):
- No missing/extra protein IDs between core label table and taxonomy table.

---

## 2) GO universe consistency (Train terms vs IA.tsv vs OBO)

**Fact 2.1 — Every training GO term exists in IA.tsv and the GO OBO graph.**

Audit output (direct):
- `train GO terms: 26,125`
- `IA terms: 40,122`
- `OBO terms: 48,101`
- `train terms missing in IA: 0`
- `train terms missing in OBO: 0`
- `namespace_mismatches (where namespace known): 0 / 26,125`

Interpretation (minimal):
- Near-zero IA-F1 is **not** explained by “train terms absent from IA” or “terms not in the GO graph”.

---

## 3) Submission template landmine (format heterogeneity)

**Fact 3.1 — `sample_submission.tsv` contains mixed record types and 4-column lines.**

Audit output (direct):
- `lines_total: 20004`
- `lines_go_like: 20000`
- `lines_text: 4`
- `n_lines_with_3_cols: 20000`
- `n_lines_with_4_cols: 4`
- `bad_go_format: 0`

Interpretation (minimal):
- Any code assuming a uniform 3-column GO-only TSV can break.
- This is a **proven format hazard**, but it is not automatically implicated in the OOF IA-F1 evaluation unless that evaluation parses this file (the audit does not claim that).

---

## 4) FASTA identifiers are inconsistent (train vs test)

**Fact 4.1 — Train FASTA headers are UniProt-style (`sp|ACC|NAME`), test FASTA is accession-only.**

Audit output (direct):
- `train_sequences.fasta headers: 82,404 (sample ids: ['sp|A0A0C5B5G6|MOTSC_HUMAN', ...])`
- `testsuperset.fasta headers: 224,309 (sample ids: ['A0A0C5B5G6', ...])`

Interpretation (minimal):
- Any join/filter that expects accession IDs must normalise train FASTA headers.

---

## 5) Row-order mismatch exists, but does **not** explain near-zero IA-F1

**Fact 5.1 — Parsed sequence order and `gbdt_state` order differ.**

Audit output (direct):
- `row_order: mismatch`
- `first_mismatch_index: 0`
- `seq_id_at_mismatch: A0A0C5B5G6`
- `state_id_at_mismatch: A0A023FBW4`

**Fact 5.2 — IA-F1 is essentially unchanged under different row orders.**

Audit output (direct):
- `LogReg IA-F1 vs gbdt_state order: 0.0003 (P=0.0002, R=0.0982)`
- `LogReg IA-F1 vs train_seq order:  0.0003 (P=0.0002, R=0.0988)`

Interpretation (minimal):
- The presence of a row-order mismatch is **proven**, but it is **not the primary driver** of the near-zero IA-F1 in the current artefacts.

---

## 6) The dominant failure mode is prediction distribution pathology + thresholding

### 6.1 LogReg OOF distributions have a massive spike at exactly 0.5

**Fact 6.1 — LogReg BP shows a very large mass at exactly 0.5.**

Audit output (direct, sample-based on 20,000,000 values):
- `LogReg BP p95: 0.5`
- `LogReg BP spike-audit: frac_eq_0.5: 0.29225265`
- `LogReg BP spike-audit: frac_in_[.49,.51]: 0.29233245`

**Fact 6.2 — LogReg MF/CC also show spikes at 0.5 (smaller but still material).**

Audit output (direct):
- MF: `p95: 0.5`, `p99: 0.5`, `frac_eq_0.5: 0.06487725`
- CC: `p95: 0.5`, `p99: 0.5`, `frac_eq_0.5: 0.09865766666666667`

Interpretation (minimal):
- Large exact 0.5 mass strongly indicates a systematic behaviour (e.g., degenerate/constant predictors, failure-to-fit for many targets, or a probability fallback that returns 0.5).
- This is not “noise”; it’s a repeatable measurable property of the stored arrays.

### 6.2 Notebook thresholds create extreme overprediction (precision collapse)

**Fact 6.3 — At the notebook thresholds, predicted positives are enormous.**

Audit output (direct):
- `LogReg BP @0.25: total_pred_positives=250,170,331 terms_with_any_pred=7,775`
- `LogReg MF @0.35: total_pred_positives=11,452,733 terms_with_any_pred=1,517`
- `LogReg CC @0.35: total_pred_positives=12,290,844 terms_with_any_pred=896`

**Fact 6.4 — IA-weighted predicted mass is orders of magnitude larger than IA-weighted true mass.**

Audit output (direct, for LogReg vs gbdt_state order):
- `weighted_true: 663,910.3433`
- `weighted_pred: 402,145,507.8407`
- `weighted_tp: 65,202.3290`
- This gives: `P≈0.0002`, `R≈0.0982`, `IA-F1≈0.0003`

Interpretation (minimal):
- The near-zero IA-F1 is explained numerically by **precision collapsing** due to extreme overprediction.

### 6.3 Simple threshold increases do not fix precision (because 0.5 spike dominates)

**Fact 6.5 — Raising thresholds to 0.50 does not materially reduce weighted_pred.**

Audit output (direct):
- At `(0.25,0.35,0.35)`: `weighted_pred=402,145,507.8`, IA-F1 `0.0003`
- At `(0.50,0.50,0.50)`: `weighted_pred=400,670,912.6`, IA-F1 `0.0003`

Interpretation (minimal):
- If a large fraction of probabilities equal exactly 0.5, then moving the threshold from 0.25/0.35 to 0.50 barely changes the decision boundary for those values.
- This is direct evidence that the distribution itself (not just threshold choice) is pathological.

---

## 7) GBDT is extremely sparse / near-zero (recall collapse)

**Fact 7.1 — GBDT OOF probabilities are near-zero for most entries.**

Audit output (direct):
- `GBDT p95: 2.758e-05`
- `GBDT p99: 1.139e-04`
- `GBDT mean: 1.990e-05`
- Spike audit: `frac_eq_0: 0.882680962962963` (sample-based)

**Fact 7.2 — At threshold 0.25, GBDT predicts almost nothing.**

Audit output (direct):
- `GBDT (thr=0.25): total_pred_positives=13,669 terms_with_any_pred=468`

Interpretation (minimal):
- GBDT contributes to low IA-F1 via very low recall at standard thresholds.

---

## 8) What this proves / what it does not

### Proven (by direct measurement)
- Raw training labels and taxonomy tables are consistent, parseable, and fully joined.
- GO term universe is consistent across training labels, IA.tsv, and GO OBO.
- There is a real row-order mismatch between `parsed/train_seq.feather` and `gbdt_state_elite1585.json`.
- That row-order mismatch does **not** materially change IA-F1 for the current LogReg artefacts.
- LogReg OOF arrays contain a massive spike at exactly 0.5 (especially BP).
- With notebook thresholds, LogReg predicts an enormous number of positives, producing `weighted_pred` ≫ `weighted_true`, collapsing precision and F1.
- GBDT outputs are near-zero for most entries; at 0.25 threshold it predicts almost nothing.

### Not proven (requires further targeted checks)
- The root cause of the 0.5 spike (e.g., failure-to-fit for many targets, predict_proba fallback, class imbalance handling, or bug in probability extraction) is not identified here.
- The audit does **not** claim a specific fix, only documents the failure mode in stored artefacts.

---

## 9) Quick “proof without training” options (minimal runtime)

These are *verification* steps, not retraining:

1) Re-run the audit (cheap; uses memmap sampling):
   - `python scripts/audit_dataset_consistency.py`

2) If you want a single extra smoking gun without full training:
   - Inspect the proportion of targets whose column is almost constant 0.5 (can be added as a follow-up audit if desired). This is still read-only on existing `.npy` arrays.

