from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp


RE_GO = re.compile(r"^GO:\d{7}$")


def extract_entry_id(token: str) -> str:
    """Normalise protein IDs across CAFA files.

    - Train FASTA commonly uses UniProt headers: sp|A0A0C5B5G6|MOTSC_HUMAN
    - CAFA TSVs (train_terms/train_taxonomy) use the accession only: A0A0C5B5G6
    """
    s = str(token).strip()
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 3 and parts[0] in {"sp", "tr"} and parts[1]:
            return parts[1]
    return s


@dataclass(frozen=True)
class AuditPaths:
    repo_root: Path
    dataset_root: Path

    train_terms_tsv: Path
    train_taxonomy_tsv: Path
    train_sequences_fasta: Path
    go_obo: Path

    test_sequences_fasta: Path
    test_taxon_list_tsv: Path

    ia_tsv: Path
    sample_submission_tsv: Path

    parsed_dir: Path
    features_dir: Path
    level1_preds_dir: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _paths() -> AuditPaths:
    repo = _repo_root()
    dataset = repo / "cafa6_data"
    return AuditPaths(
        repo_root=repo,
        dataset_root=dataset,
        train_terms_tsv=dataset / "Train" / "train_terms.tsv",
        train_taxonomy_tsv=dataset / "Train" / "train_taxonomy.tsv",
        train_sequences_fasta=dataset / "Train" / "train_sequences.fasta",
        go_obo=dataset / "Train" / "go-basic.obo",
        test_sequences_fasta=dataset / "Test" / "testsuperset.fasta",
        test_taxon_list_tsv=dataset / "Test" / "testsuperset-taxon-list.tsv",
        ia_tsv=dataset / "IA.tsv",
        sample_submission_tsv=dataset / "sample_submission.tsv",
        parsed_dir=dataset / "parsed",
        features_dir=dataset / "features",
        level1_preds_dir=dataset / "features" / "level1_preds",
    )


def _head(path: Path, n: int = 5) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            out.append(line.rstrip("\n"))
    return out


def _count_fasta_headers(path: Path, *, max_headers_to_capture: int = 10) -> tuple[int, list[str]]:
    n = 0
    ids: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                n += 1
                if len(ids) < max_headers_to_capture:
                    header = line[1:].strip()
                    # CAFA/UniProt typically puts EntryID as the first token
                    ids.append(header.split()[0])
    return n, ids


def _read_train_terms(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    expected = ["EntryID", "term", "aspect"]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)} (expected {expected})")

    df["EntryID"] = df["EntryID"].astype(str)
    df["term"] = df["term"].astype(str)
    df["aspect"] = df["aspect"].astype(str)
    return df


def _read_train_taxonomy(path: Path) -> pd.DataFrame:
    # No header in your file (2 columns), but we verify.
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if df.shape[1] != 2:
        raise ValueError(f"Unexpected column count in {path}: {df.shape[1]} (expected 2)")
    df.columns = ["EntryID", "taxon"]
    df["EntryID"] = df["EntryID"].astype(str)
    df["taxon"] = df["taxon"].astype(str)
    return df


def _read_ia(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["term", "ia"], dtype={"term": str})
    df["ia"] = pd.to_numeric(df["ia"], errors="coerce")
    df = df.dropna(subset=["term", "ia"]).copy()
    df["term"] = df["term"].astype(str)
    return df


def _parse_go_obo_namespaces(path: Path) -> tuple[set[str], dict[str, str], set[str], dict[str, str]]:
    """Return (terms, namespace_by_term, obsolete_terms, alt_id_to_primary)."""
    terms: set[str] = set()
    namespace: dict[str, str] = {}
    obsolete: set[str] = set()
    alt_id_to_primary: dict[str, str] = {}

    cur_id: str | None = None
    cur_ns: str | None = None
    cur_obsolete = False
    cur_alt_ids: list[str] = []

    def flush() -> None:
        nonlocal cur_id, cur_ns, cur_obsolete, cur_alt_ids
        if cur_id is None:
            return
        terms.add(cur_id)
        if cur_ns is not None:
            namespace[cur_id] = cur_ns
        if cur_obsolete:
            obsolete.add(cur_id)
        for alt in cur_alt_ids:
            if alt not in alt_id_to_primary:
                alt_id_to_primary[alt] = cur_id
        cur_id = None
        cur_ns = None
        cur_obsolete = False
        cur_alt_ids = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line == "[Term]":
                flush()
                continue
            if not line or line.startswith("!"):
                continue
            if line.startswith("id: "):
                cur_id = line.split("id: ", 1)[1].strip()
                continue
            if line.startswith("namespace: "):
                cur_ns = line.split("namespace: ", 1)[1].strip()
                continue
            if line.startswith("is_obsolete: "):
                val = line.split("is_obsolete: ", 1)[1].strip().lower()
                cur_obsolete = val == "true"
                continue
            if line.startswith("alt_id: "):
                cur_alt_ids.append(line.split("alt_id: ", 1)[1].strip())
                continue

    flush()
    return terms, namespace, obsolete, alt_id_to_primary


def _regex_audit_terms(train_terms: pd.DataFrame) -> dict[str, int]:
    bad_go = (~train_terms["term"].map(lambda s: bool(RE_GO.match(str(s))))).sum()
    bad_aspect = (~train_terms["aspect"].isin(["P", "F", "C"])).sum()
    # EntryID: don’t overfit regex; just reject whitespace/empty.
    bad_entry = (
        train_terms["EntryID"].isna()
        | (train_terms["EntryID"].astype(str).str.len() == 0)
        | (train_terms["EntryID"].astype(str).str.contains(r"\s", regex=True))
    ).sum()
    return {
        "bad_go_id": int(bad_go),
        "bad_aspect": int(bad_aspect),
        "bad_entryid": int(bad_entry),
    }


def _sample_submission_column_audit(path: Path) -> dict[str, int]:
    col_counts = Counter()
    n_text_rows = 0
    n_go_rows = 0
    n_other_rows = 0
    n_bad_go = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            col_counts[len(parts)] += 1
            if len(parts) >= 2 and parts[1] == "Text":
                n_text_rows += 1
            elif len(parts) >= 2 and RE_GO.match(parts[1] or ""):
                n_go_rows += 1
                if len(parts) < 3:
                    n_other_rows += 1
            else:
                n_other_rows += 1
                if len(parts) >= 2 and parts[1].startswith("GO:") and not RE_GO.match(parts[1]):
                    n_bad_go += 1

    out = {
        "lines_total": int(sum(col_counts.values())),
        "lines_go_like": int(n_go_rows),
        "lines_text": int(n_text_rows),
        "lines_other": int(n_other_rows),
        "bad_go_format": int(n_bad_go),
    }
    for k, v in col_counts.most_common():
        out[f"n_lines_with_{k}_cols"] = int(v)
    return out


def _load_memmap(path: Path) -> np.memmap:
    return np.load(path, mmap_mode="r")


def _memmap_stats(mm: np.memmap, name: str) -> dict[str, float | int]:
    # Sample a subset to keep it fast.
    n_rows = mm.shape[0]
    n_cols = mm.shape[1]

    rng = np.random.default_rng(0)
    sample_rows = min(2000, n_rows)
    idx = rng.choice(n_rows, size=sample_rows, replace=False)
    sample = np.asarray(mm[idx, :], dtype=np.float32)

    # Flattened stats
    flat = sample.ravel()
    qs = np.quantile(flat, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "name": name,
        "shape_rows": int(n_rows),
        "shape_cols": int(n_cols),
        "min": float(qs[0]),
        "p01": float(qs[1]),
        "p05": float(qs[2]),
        "p50": float(qs[3]),
        "p95": float(qs[4]),
        "p99": float(qs[5]),
        "max": float(qs[6]),
        "mean": float(np.mean(flat)),
    }


def _prob_spike_audit(mm: np.memmap, name: str, eps: float = 1e-6) -> dict[str, float | int]:
    """Cheaply detect suspicious probability mass at exact 0, 0.5, 1.0.

    We sample rows (like _memmap_stats) to keep runtime bounded.
    """

    n_rows = mm.shape[0]
    rng = np.random.default_rng(0)
    sample_rows = min(2000, n_rows)
    idx = rng.choice(n_rows, size=sample_rows, replace=False)
    sample = np.asarray(mm[idx, :], dtype=np.float32).ravel()

    def frac_near(x: float) -> float:
        return float(np.mean(np.abs(sample - x) <= eps))

    def frac_in(lo: float, hi: float) -> float:
        return float(np.mean((sample >= lo) & (sample <= hi)))

    return {
        "name": name,
        "sample_n": int(sample.size),
        "frac_eq_0": frac_near(0.0),
        "frac_eq_05": frac_near(0.5),
        "frac_eq_1": frac_near(1.0),
        "frac_in_[.49,.51]": frac_in(0.49, 0.51),
        "frac_in_[.99,1]": frac_in(0.99, 1.0),
    }


def _count_above_threshold(mm: np.memmap, thr: float, batch_rows: int = 1024) -> tuple[int, int, int]:
    """Return (n_pred_positives_total, n_terms_with_any_pred, n_rows)."""
    n_rows, n_cols = mm.shape
    pred_counts = np.zeros(n_cols, dtype=np.int64)

    for start in range(0, n_rows, batch_rows):
        end = min(n_rows, start + batch_rows)
        xb = np.asarray(mm[start:end, :], dtype=np.float32)
        pred_counts += (xb >= thr).sum(axis=0).astype(np.int64)

    return int(pred_counts.sum()), int(np.sum(pred_counts > 0)), int(n_rows)


def _load_term_contract(level1_preds: Path) -> tuple[list[str], dict[str, slice], dict[str, list[str]]]:
    top_bp = json.load(open(level1_preds / "top_terms_BP.json", "r"))
    top_mf = json.load(open(level1_preds / "top_terms_MF.json", "r"))
    top_cc = json.load(open(level1_preds / "top_terms_CC.json", "r"))
    top_terms_all = top_bp + top_mf + top_cc
    slices = {
        "BP": slice(0, len(top_bp)),
        "MF": slice(len(top_bp), len(top_bp) + len(top_mf)),
        "CC": slice(len(top_bp) + len(top_mf), len(top_terms_all)),
    }
    return top_terms_all, slices, {"BP": top_bp, "MF": top_mf, "CC": top_cc}


def _load_ia_weights_contract(ia_tsv: Path, top_terms_all: list[str]) -> np.ndarray:
    ia_df = pd.read_csv(ia_tsv, sep="\t", header=None, names=["term", "ia"])
    ia_df["ia"] = pd.to_numeric(ia_df["ia"], errors="coerce")
    ia_df = ia_df.dropna(subset=["term", "ia"]).copy()
    ia_series = ia_df.set_index("term")["ia"]
    reindexed = ia_series.reindex(top_terms_all)
    return reindexed.fillna(0.0).astype(np.float32).to_numpy()


def _build_y_true_sparse(train_df: pd.DataFrame, protein_ids: list[str], top_terms_all: list[str]) -> sp.csr_matrix:
    n_samples = len(protein_ids)
    n_terms = len(top_terms_all)
    row_index = pd.Series(np.arange(n_samples, dtype=np.int32), index=protein_ids)
    col_index = pd.Series(np.arange(n_terms, dtype=np.int32), index=top_terms_all)

    df = train_df[["EntryID", "term"]].copy()
    df["row"] = df["EntryID"].map(row_index)
    df["col"] = df["term"].map(col_index)
    df = df.dropna(subset=["row", "col"])

    rows = df["row"].astype(np.int32).to_numpy()
    cols = df["col"].astype(np.int32).to_numpy()
    data = np.ones(len(df), dtype=np.int8)

    y_true = sp.coo_matrix((data, (rows, cols)), shape=(n_samples, n_terms), dtype=np.int8).tocsr()
    y_true.sum_duplicates()
    y_true.data[:] = 1
    return y_true


def _ia_f1_streaming(
    y_prob_getter,
    y_true_csr: sp.csr_matrix,
    ia_weights: np.ndarray,
    thr_vec: np.ndarray,
    batch_size: int = 1024,
) -> tuple[float, float, float, dict[str, float | int]]:
    n_samples, n_terms = y_true_csr.shape
    if ia_weights.shape[0] != n_terms or thr_vec.shape[0] != n_terms:
        raise ValueError("ia_weights/thr_vec must match number of terms")

    true_counts = np.asarray(y_true_csr.sum(axis=0)).ravel().astype(np.int64)
    weighted_true = float(np.dot(ia_weights, true_counts))

    pred_counts = np.zeros(n_terms, dtype=np.int64)
    tp_counts = np.zeros(n_terms, dtype=np.int64)

    for start in range(0, n_samples, batch_size):
        end = min(n_samples, start + batch_size)
        y_prob = y_prob_getter(start, end)
        if y_prob.shape != (end - start, n_terms):
            raise ValueError(f"y_prob_getter returned {y_prob.shape}, expected {(end-start, n_terms)}")

        y_pred = (y_prob >= thr_vec)
        pred_counts += y_pred.sum(axis=0).astype(np.int64)

        y_true_batch = y_true_csr[start:end]
        tp_batch = y_true_batch.multiply(y_pred)
        tp_counts += np.asarray(tp_batch.sum(axis=0)).ravel().astype(np.int64)

    weighted_tp = float(np.dot(ia_weights, tp_counts))
    weighted_pred = float(np.dot(ia_weights, pred_counts))

    precision = (weighted_tp / weighted_pred) if weighted_pred > 0 else 0.0
    recall = (weighted_tp / weighted_true) if weighted_true > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    debug = {
        "weighted_tp": weighted_tp,
        "weighted_pred": weighted_pred,
        "weighted_true": weighted_true,
        "pred_nonzero_terms": int(np.sum(pred_counts > 0)),
        "true_nonzero_terms": int(np.sum(true_counts > 0)),
    }
    return f1, precision, recall, debug


def _row_order_check(parsed_train_seq: Path, gbdt_state_path: Path) -> dict[str, int | str]:
    # parsed train seq ids
    seq = pd.read_feather(parsed_train_seq)
    if "id" not in seq.columns:
        raise ValueError(f"Expected column 'id' in {parsed_train_seq}, got {list(seq.columns)}")
    ids_seq = [extract_entry_id(x) for x in seq["id"].astype(str).tolist()]

    # gbdt state ids
    with gbdt_state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    ids_state = [str(x) for x in state.get("train_protein_ids", [])]

    out: dict[str, int | str] = {
        "train_seq_n": int(len(ids_seq)),
        "gbdt_state_train_ids_n": int(len(ids_state)),
    }

    n = min(len(ids_seq), len(ids_state))
    if n == 0:
        out["row_order"] = "empty"
        return out

    # find first mismatch
    first_mismatch = None
    for i in range(n):
        if ids_seq[i] != ids_state[i]:
            first_mismatch = i
            break

    if first_mismatch is None and len(ids_seq) == len(ids_state):
        out["row_order"] = "exact_match"
        out["first_mismatch_index"] = -1
    else:
        out["row_order"] = "mismatch"
        out["first_mismatch_index"] = int(first_mismatch if first_mismatch is not None else n)
        out["seq_id_at_mismatch"] = str(ids_seq[first_mismatch]) if first_mismatch is not None else ""
        out["state_id_at_mismatch"] = str(ids_state[first_mismatch]) if first_mismatch is not None else ""

    return out


def main() -> None:
    p = _paths()

    print("== Paths ==")
    print(f"repo_root:    {p.repo_root}")
    print(f"dataset_root: {p.dataset_root}")

    # 1) Raw schema audits
    print("\n== Raw file header peeks ==")
    print(f"train_terms.tsv head: {p.train_terms_tsv}")
    print("  " + "\n  ".join(_head(p.train_terms_tsv, 3)))
    print(f"train_taxonomy.tsv head: {p.train_taxonomy_tsv}")
    print("  " + "\n  ".join(_head(p.train_taxonomy_tsv, 3)))
    print(f"IA.tsv head: {p.ia_tsv}")
    print("  " + "\n  ".join(_head(p.ia_tsv, 3)))
    print(f"sample_submission.tsv head: {p.sample_submission_tsv}")
    print("  " + "\n  ".join(_head(p.sample_submission_tsv, 6)))

    print("\n== Sample submission format audit ==")
    ss_audit = _sample_submission_column_audit(p.sample_submission_tsv)
    for k in sorted(ss_audit.keys()):
        print(f"{k}: {ss_audit[k]}")

    print("\n== Train terms TSV audit ==")
    train_terms = _read_train_terms(p.train_terms_tsv)
    print(f"annotations: {len(train_terms):,}")
    print(f"proteins:    {train_terms['EntryID'].nunique():,}")
    print(f"terms:       {train_terms['term'].nunique():,}")
    print("aspects:")
    print(train_terms["aspect"].value_counts(dropna=False).to_string())
    rx = _regex_audit_terms(train_terms)
    for k, v in rx.items():
        print(f"{k}: {v}")

    print("\n== Train taxonomy TSV audit ==")
    tax = _read_train_taxonomy(p.train_taxonomy_tsv)
    print(f"rows:     {len(tax):,}")
    print(f"proteins: {tax['EntryID'].nunique():,}")
    print(f"taxa:     {tax['taxon'].nunique():,}")
    bad_taxon = (~tax["taxon"].astype(str).str.fullmatch(r"\d+"))
    print(f"bad_taxon_format: {int(bad_taxon.sum())}")

    print("\n== Like-for-like joins (raw) ==")
    ids_terms = set(train_terms["EntryID"].unique())
    ids_tax = set(tax["EntryID"].unique())
    print(f"EntryID overlap train_terms vs train_taxonomy: {len(ids_terms & ids_tax):,} / {len(ids_terms):,} terms_ids")
    print(f"missing_in_taxonomy: {len(ids_terms - ids_tax):,}")
    print(f"extra_in_taxonomy:   {len(ids_tax - ids_terms):,}")

    print("\n== FASTA header counts (streaming) ==")
    n_train_fa, train_fa_ids = _count_fasta_headers(p.train_sequences_fasta)
    n_test_fa, test_fa_ids = _count_fasta_headers(p.test_sequences_fasta)
    print(f"train_sequences.fasta headers: {n_train_fa:,}  (sample ids: {train_fa_ids[:5]})")
    print(f"testsuperset.fasta headers:    {n_test_fa:,}  (sample ids: {test_fa_ids[:5]})")
    print(f"train_terms proteins:          {len(ids_terms):,}")
    print(f"train_taxonomy proteins:       {len(ids_tax):,}")

    # 2) Term universe checks: IA + OBO
    print("\n== Term universe checks ==")
    ia = _read_ia(p.ia_tsv)
    ia_terms = set(ia["term"].unique())
    train_go_terms = set(train_terms["term"].unique())

    obo_terms, ns_by_term, obsolete_terms, alt_map = _parse_go_obo_namespaces(p.go_obo)

    missing_in_ia = train_go_terms - ia_terms
    missing_in_obo = train_go_terms - obo_terms
    alt_resolvable = {t for t in missing_in_obo if t in alt_map}

    print(f"train GO terms:      {len(train_go_terms):,}")
    print(f"IA terms:            {len(ia_terms):,}")
    print(f"OBO terms:           {len(obo_terms):,}")
    print(f"train terms missing in IA:  {len(missing_in_ia):,}")
    print(f"train terms missing in OBO: {len(missing_in_obo):,}")
    print(f"...of which alt_id-resolvable: {len(alt_resolvable):,}")

    # Aspect/namespace agreement (rough): map P/F/C -> BP/MF/CC using OBO namespace
    aspect_alias = {"P": "biological_process", "F": "molecular_function", "C": "cellular_component"}
    mismatched_namespace = 0
    n_ns_known = 0
    for a, grp in train_terms.groupby("aspect"):
        want = aspect_alias.get(a)
        if want is None:
            continue
        terms_a = grp["term"].unique()
        for t in terms_a:
            ns = ns_by_term.get(t)
            if ns is None:
                continue
            n_ns_known += 1
            if ns != want:
                mismatched_namespace += 1
    print(f"namespace_mismatches (where namespace known): {mismatched_namespace:,} / {n_ns_known:,}")

    # 3) Parsed/feature artefacts row order and prediction distributions
    print("\n== Parsed artefacts checks ==")
    parsed_train_seq = p.parsed_dir / "train_seq.feather"
    gbdt_state = p.level1_preds_dir / "gbdt_state_elite1585.json"

    ids_state: list[str] = []
    ids_seq_entry: list[str] = []
    if parsed_train_seq.exists() and gbdt_state.exists():
        row_check = _row_order_check(parsed_train_seq, gbdt_state)
        for k in sorted(row_check.keys()):
            print(f"{k}: {row_check[k]}")

        with gbdt_state.open("r", encoding="utf-8") as f:
            ids_state = [str(x) for x in json.load(f).get("train_protein_ids", [])]

        seq = pd.read_feather(parsed_train_seq)
        ids_seq_entry = [extract_entry_id(x) for x in seq["id"].astype(str).tolist()]
    else:
        print("Missing parsed/train_seq.feather or gbdt_state_elite1585.json; skipping row-order check")

    print("\n== OOF prediction array checks ==")
    oof_gbdt = p.level1_preds_dir / "oof_pred_gbdt.npy"
    oof_bp = p.level1_preds_dir / "oof_pred_logreg_BP.npy"
    oof_mf = p.level1_preds_dir / "oof_pred_logreg_MF.npy"
    oof_cc = p.level1_preds_dir / "oof_pred_logreg_CC.npy"

    if oof_gbdt.exists():
        mm = _load_memmap(oof_gbdt)
        print("GBDT stats:")
        for k, v in _memmap_stats(mm, "oof_pred_gbdt").items():
            print(f"  {k}: {v}")

    # LogReg per-aspect stats
    for name, path in [("BP", oof_bp), ("MF", oof_mf), ("CC", oof_cc)]:
        if path.exists():
            mm = _load_memmap(path)
            print(f"LogReg {name} stats:")
            for k, v in _memmap_stats(mm, f"oof_pred_logreg_{name}").items():
                print(f"  {k}: {v}")
            print(f"LogReg {name} spike-audit (sample-based):")
            for k, v in _prob_spike_audit(mm, f"oof_pred_logreg_{name}").items():
                print(f"  {k}: {v}")

    # Threshold sanity: replicate notebook thresholds (BP=0.25, MF=0.35, CC=0.35)
    print("\n== Threshold sanity (counts above notebook thresholds) ==")
    if oof_bp.exists():
        mm = _load_memmap(oof_bp)
        tot, n_terms_any, n_rows = _count_above_threshold(mm, 0.25)
        print(f"LogReg BP: total_pred_positives={tot:,} across {n_rows:,} rows; terms_with_any_pred={n_terms_any:,}")
    if oof_mf.exists():
        mm = _load_memmap(oof_mf)
        tot, n_terms_any, n_rows = _count_above_threshold(mm, 0.35)
        print(f"LogReg MF: total_pred_positives={tot:,} across {n_rows:,} rows; terms_with_any_pred={n_terms_any:,}")
    if oof_cc.exists():
        mm = _load_memmap(oof_cc)
        tot, n_terms_any, n_rows = _count_above_threshold(mm, 0.35)
        print(f"LogReg CC: total_pred_positives={tot:,} across {n_rows:,} rows; terms_with_any_pred={n_terms_any:,}")

    if oof_gbdt.exists():
        # Need contract-specific thr vector; approximate with 0.25 (lower bound) to detect sparsity.
        mm = _load_memmap(oof_gbdt)
        tot, n_terms_any, n_rows = _count_above_threshold(mm, 0.25)
        print(f"GBDT (thr=0.25): total_pred_positives={tot:,} across {n_rows:,} rows; terms_with_any_pred={n_terms_any:,}")
        print("GBDT spike-audit (sample-based):")
        for k, v in _prob_spike_audit(mm, "oof_pred_gbdt").items():
            print(f"  {k}: {v}")

    print("\n== IA-F1 sanity: does protein row order explain low scores? ==")
    if not (ids_state and ids_seq_entry and oof_bp.exists() and oof_mf.exists() and oof_cc.exists()):
        print("Skipping IA-F1 sanity (missing ids and/or OOF preds)")
        print("\nDone.")
        return

    top_terms_all, _, top_terms_by_aspect = _load_term_contract(p.level1_preds_dir)
    ia_weights = _load_ia_weights_contract(p.ia_tsv, top_terms_all)
    thr_vec = np.concatenate(
        [
            np.full(len(top_terms_by_aspect["BP"]), 0.25, dtype=np.float32),
            np.full(len(top_terms_by_aspect["MF"]), 0.35, dtype=np.float32),
            np.full(len(top_terms_by_aspect["CC"]), 0.35, dtype=np.float32),
        ]
    )

    y_true_state = _build_y_true_sparse(train_terms, ids_state, top_terms_all)
    y_true_seq = _build_y_true_sparse(train_terms, ids_seq_entry, top_terms_all)

    lr_bp_mm = _load_memmap(oof_bp)
    lr_mf_mm = _load_memmap(oof_mf)
    lr_cc_mm = _load_memmap(oof_cc)

    def get_logreg(start: int, end: int) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(lr_bp_mm[start:end], dtype=np.float32),
                np.asarray(lr_mf_mm[start:end], dtype=np.float32),
                np.asarray(lr_cc_mm[start:end], dtype=np.float32),
            ],
            axis=1,
        )

    f1_state, p_state, r_state, dbg_state = _ia_f1_streaming(get_logreg, y_true_state, ia_weights, thr_vec)
    f1_seq, p_seq, r_seq, dbg_seq = _ia_f1_streaming(get_logreg, y_true_seq, ia_weights, thr_vec)
    print(f"LogReg IA-F1 vs gbdt_state order: {f1_state:.4f} (P={p_state:.4f}, R={r_state:.4f})")
    print(f"  debug: {dbg_state}")
    print(f"LogReg IA-F1 vs train_seq order:  {f1_seq:.4f} (P={p_seq:.4f}, R={r_seq:.4f})")
    print(f"  debug: {dbg_seq}")

    print("\n== IA-F1 sensitivity (LogReg) to higher thresholds ==")
    # Keep the same per-aspect triple structure, but probe higher cutoffs.
    probes = [
        (0.25, 0.35, 0.35),
        (0.50, 0.50, 0.50),
        (0.75, 0.75, 0.75),
        (0.90, 0.90, 0.90),
        (0.95, 0.95, 0.95),
        (0.99, 0.99, 0.99),
    ]
    for thr_bp, thr_mf, thr_cc in probes:
        thr_vec_probe = np.concatenate(
            [
                np.full(len(top_terms_by_aspect["BP"]), thr_bp, dtype=np.float32),
                np.full(len(top_terms_by_aspect["MF"]), thr_mf, dtype=np.float32),
                np.full(len(top_terms_by_aspect["CC"]), thr_cc, dtype=np.float32),
            ]
        )
        f1p, pp, rp, dbgp = _ia_f1_streaming(get_logreg, y_true_state, ia_weights, thr_vec_probe)
        print(f"thr(BP,MF,CC)=({thr_bp:.2f},{thr_mf:.2f},{thr_cc:.2f}) -> IA-F1={f1p:.4f} (P={pp:.4f}, R={rp:.4f}) weighted_pred={dbgp['weighted_pred']:.1f}")

    if oof_gbdt.exists():
        gbdt_mm = _load_memmap(oof_gbdt)

        def get_gbdt(start: int, end: int) -> np.ndarray:
            return np.asarray(gbdt_mm[start:end], dtype=np.float32)

        g_f1_state, g_p_state, g_r_state, g_dbg_state = _ia_f1_streaming(get_gbdt, y_true_state, ia_weights, thr_vec)
        g_f1_seq, g_p_seq, g_r_seq, g_dbg_seq = _ia_f1_streaming(get_gbdt, y_true_seq, ia_weights, thr_vec)
        print(f"GBDT IA-F1 vs gbdt_state order: {g_f1_state:.4f} (P={g_p_state:.4f}, R={g_r_state:.4f})")
        print(f"  debug: {g_dbg_state}")
        print(f"GBDT IA-F1 vs train_seq order:  {g_f1_seq:.4f} (P={g_p_seq:.4f}, R={g_r_seq:.4f})")
        print(f"  debug: {g_dbg_seq}")

    print("\nDone.")


if __name__ == "__main__":
    main()
