import argparse
import csv
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def _norm_uniprot_accession(raw_id: str) -> str:
    s = str(raw_id).strip()
    parts = s.split("|")
    # Common UniProt FASTA header: sp|P12345|NAME_HUMAN
    if len(parts) >= 2 and parts[0] in {"sp", "tr"}:
        return parts[1]
    if len(parts) >= 3:
        return parts[1]
    return s


def _detect_id_col(df: pd.DataFrame) -> str:
    for cand in ["id", "EntryID", "entry_id", "accession"]:
        if cand in df.columns:
            return cand
    return df.columns[0]


def _read_ids(feather_path: Path) -> list[str]:
    df = pd.read_feather(feather_path)
    col = _detect_id_col(df)
    return df[col].astype(str).tolist()


def _strip_go_leakage(text: str) -> str:
    # Remove explicit GO identifiers and obvious "GO:" tokens.
    text = re.sub(r"\bGO:\d{7}\b", " ", text)
    text = text.replace("GO:", " ")
    return re.sub(r"\s+", " ", text).strip()


def _make_session(total_retries: int = 5) -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=total_retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


@dataclass
class UniProtRow:
    accession: str
    protein_name: str
    organism_name: str
    keywords: str
    protein_families: str
    cc_function: str
    cc_subcellular_location: str
    pubmed_ids: list[str]


def fetch_uniprot_tsv(
    session: requests.Session,
    accessions: list[str],
    sleep_s: float,
    fields: str,
) -> dict[str, UniProtRow]:
    """Fetch UniProt rows for a small batch of accessions using the search endpoint (TSV)."""

    # Query is a disjunction of accessions.
    # Keep the batch modest to avoid URL length limits.
    query = " OR ".join([f"accession:{a}" for a in accessions])

    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"({query})",
        "format": "tsv",
        "fields": fields,
        "size": 500,
    }

    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    if sleep_s:
        time.sleep(sleep_s)

    lines = resp.text.splitlines()
    if not lines:
        return {}

    reader = csv.DictReader(lines, delimiter="\t")
    out: dict[str, UniProtRow] = {}

    for row in reader:
        acc = (row.get("Entry") or row.get("accession") or "").strip()
        if not acc:
            continue

        pmids_raw = (row.get("PubMed ID") or row.get("lit_pubmed_id") or "").strip()
        pmids: list[str] = []
        if pmids_raw:
            # UniProt typically uses ';' separators for multi-valued TSV cells.
            pmids = [p.strip() for p in re.split(r"[;\s]+", pmids_raw) if p.strip().isdigit()]

        out[acc] = UniProtRow(
            accession=acc,
            protein_name=(row.get("Protein names") or row.get("protein_name") or "").strip(),
            organism_name=(row.get("Organism") or row.get("organism_name") or "").strip(),
            keywords=(row.get("Keywords") or row.get("keyword") or "").strip(),
            protein_families=(row.get("Protein families") or row.get("protein_families") or "").strip(),
            cc_function=(row.get("Function [CC]") or row.get("cc_function") or "").strip(),
            cc_subcellular_location=(
                row.get("Subcellular location [CC]") or row.get("cc_subcellular_location") or ""
            ).strip(),
            pubmed_ids=pmids,
        )

    return out


def _pubmed_abstract_from_xml(xml_text: str) -> dict[str, tuple[str, str]]:
    """Return {pmid: (title, abstract)} from PubMed efetch XML."""
    out: dict[str, tuple[str, str]] = {}
    root = ET.fromstring(xml_text)

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        if pmid_el is None or pmid_el.text is None:
            continue
        pmid = pmid_el.text.strip()

        title = ""
        title_el = article.find(".//Article/ArticleTitle")
        if title_el is not None:
            title = "".join(title_el.itertext()).strip()

        abs_texts: list[str] = []
        for abs_el in article.findall(".//Article/Abstract/AbstractText"):
            abs_texts.append("".join(abs_el.itertext()).strip())

        abstract = " ".join([t for t in abs_texts if t])
        out[pmid] = (title, abstract)

    return out


def fetch_pubmed_abstracts(
    session: requests.Session,
    pmids: list[str],
    sleep_s: float,
    email: str | None,
    api_key: str | None,
) -> dict[str, tuple[str, str]]:
    """Fetch PubMed abstracts for a batch of PMIDs."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    if sleep_s:
        time.sleep(sleep_s)

    return _pubmed_abstract_from_xml(resp.text)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build EntryID->text corpus from UniProt (fields) + PubMed abstracts for --mode text."  # noqa: E501
    )
    ap.add_argument(
        "--artefacts-dir",
        type=Path,
        default=Path("artefacts_local") / "artefacts",
        help="Path containing parsed/ (default: artefacts_local/artefacts)",
    )
    ap.add_argument(
        "--out-path",
        type=Path,
        default=Path("artefacts_local") / "artefacts" / "external" / "entryid_text.tsv",
        help="Output TSV path (default: artefacts_local/artefacts/external/entryid_text.tsv)",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artefacts_local") / "artefacts" / "external" / "uniprot_pubmed_cache",
        help="Cache dir for UniProt/PubMed responses (default: artefacts_local/artefacts/external/uniprot_pubmed_cache)",
    )
    ap.add_argument(
        "--max-ids",
        type=int,
        default=0,
        help="Limit number of unique accessions (debug). 0 = all.",
    )
    ap.add_argument(
        "--uniprot-batch-size",
        type=int,
        default=200,
        help="Accessions per UniProt request (default: 200)",
    )
    ap.add_argument(
        "--pubmed-batch-size",
        type=int,
        default=200,
        help="PMIDs per PubMed efetch request (default: 200)",
    )
    ap.add_argument(
        "--max-pubmed-per-protein",
        type=int,
        default=5,
        help="Max PMIDs to use per protein (default: 5)",
    )
    ap.add_argument(
        "--max-abstract-chars",
        type=int,
        default=2000,
        help="Truncate each abstract to this many chars (default: 2000)",
    )
    ap.add_argument(
        "--sleep-uniprot",
        type=float,
        default=0.1,
        help="Sleep seconds between UniProt requests (default: 0.1)",
    )
    ap.add_argument(
        "--sleep-pubmed",
        type=float,
        default=0.34,
        help="Sleep seconds between PubMed requests (default: 0.34 ~ 3 req/s)",
    )
    ap.add_argument(
        "--email",
        type=str,
        default=None,
        help="NCBI contact email (recommended)",
    )
    ap.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="NCBI API key (optional; increases rate limits)",
    )
    ap.add_argument(
        "--strip-go",
        action="store_true",
        help="Strip explicit GO identifiers (GO:xxxxxxx) from text to reduce leakage.",
    )

    args = ap.parse_args()

    artefacts_dir: Path = args.artefacts_dir
    parsed_dir = artefacts_dir / "parsed"
    train_feather = parsed_dir / "train_seq.feather"
    test_feather = parsed_dir / "test_seq.feather"
    if not train_feather.exists() or not test_feather.exists():
        raise FileNotFoundError(f"Expected {train_feather} and {test_feather}. Run Phase 1 parsing first.")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Read IDs in row order (we only need uniques for fetching).
    print("Loading IDs...")
    train_ids_raw = _read_ids(train_feather)
    test_ids_raw = _read_ids(test_feather)

    all_ids_raw = train_ids_raw + test_ids_raw
    accessions_all = [_norm_uniprot_accession(x) for x in all_ids_raw]

    # Unique accessions for fetch.
    uniq = list(dict.fromkeys([a for a in accessions_all if a]))
    if args.max_ids and args.max_ids > 0:
        uniq = uniq[: args.max_ids]

    print(f"Unique accessions: {len(uniq)}")

    # UniProt fields (TSV).
    # Source: UniProtKB return fields
    # - accession, protein_name, organism_name, keyword, protein_families, cc_function, cc_subcellular_location, lit_pubmed_id
    uniprot_fields = ",".join(
        [
            "accession",
            "protein_name",
            "organism_name",
            "keyword",
            "protein_families",
            "cc_function",
            "cc_subcellular_location",
            "lit_pubmed_id",
        ]
    )

    session = _make_session()

    # Fetch UniProt annotations (with caching).
    uniprot_cache = args.cache_dir / "uniprot_rows.tsv"
    uniprot_rows: dict[str, UniProtRow] = {}

    if uniprot_cache.exists():
        print(f"Loading UniProt cache: {uniprot_cache}")
        df_cache = pd.read_csv(uniprot_cache, sep="\t", dtype=str)
        for _, r in df_cache.iterrows():
            pmids = []
            if isinstance(r.get("lit_pubmed_id"), str) and r["lit_pubmed_id"].strip():
                pmids = [p for p in re.split(r"[;\s]+", r["lit_pubmed_id"].strip()) if p.isdigit()]
            uniprot_rows[str(r["accession"])]= UniProtRow(
                accession=str(r["accession"]),
                protein_name=str(r.get("protein_name", "")) if r.get("protein_name") is not None else "",
                organism_name=str(r.get("organism_name", "")) if r.get("organism_name") is not None else "",
                keywords=str(r.get("keyword", "")) if r.get("keyword") is not None else "",
                protein_families=str(r.get("protein_families", "")) if r.get("protein_families") is not None else "",
                cc_function=str(r.get("cc_function", "")) if r.get("cc_function") is not None else "",
                cc_subcellular_location=str(r.get("cc_subcellular_location", "")) if r.get("cc_subcellular_location") is not None else "",
                pubmed_ids=pmids,
            )

    missing = [a for a in uniq if a not in uniprot_rows]
    if missing:
        print(f"Fetching UniProt rows (missing={len(missing)})...")

        # Stream to cache as we go (append).
        write_header = not uniprot_cache.exists()
        with uniprot_cache.open("a", newline="", encoding="utf-8") as f:
            writer = None

            for i in tqdm(range(0, len(missing), args.uniprot_batch_size), desc="UniProt"):
                batch = missing[i : i + args.uniprot_batch_size]
                rows = fetch_uniprot_tsv(
                    session=session,
                    accessions=batch,
                    sleep_s=args.sleep_uniprot,
                    fields=uniprot_fields,
                )

                # Initialise writer lazily (need consistent fieldnames).
                if writer is None:
                    writer = csv.DictWriter(
                        f,
                        delimiter="\t",
                        fieldnames=[
                            "accession",
                            "protein_name",
                            "organism_name",
                            "keyword",
                            "protein_families",
                            "cc_function",
                            "cc_subcellular_location",
                            "lit_pubmed_id",
                        ],
                    )
                    if write_header:
                        writer.writeheader()
                        write_header = False

                for acc, r in rows.items():
                    uniprot_rows[acc] = r
                    writer.writerow(
                        {
                            "accession": r.accession,
                            "protein_name": r.protein_name,
                            "organism_name": r.organism_name,
                            "keyword": r.keywords,
                            "protein_families": r.protein_families,
                            "cc_function": r.cc_function,
                            "cc_subcellular_location": r.cc_subcellular_location,
                            "lit_pubmed_id": ";".join(r.pubmed_ids),
                        }
                    )

    # Determine PubMed IDs to fetch (cap per protein).
    pmid_sets: dict[str, list[str]] = {}
    pmids_all: list[str] = []
    for acc, r in uniprot_rows.items():
        pmids = r.pubmed_ids[: max(0, int(args.max_pubmed_per_protein))] if args.max_pubmed_per_protein else r.pubmed_ids
        # Keep deterministic order
        pmids = [p for p in pmids if p.isdigit()]
        pmid_sets[acc] = pmids
        pmids_all.extend(pmids)

    pmids_uniq = list(dict.fromkeys(pmids_all))
    print(f"Unique PubMed IDs to fetch (capped): {len(pmids_uniq)}")

    # Load / populate PubMed cache
    pubmed_cache = args.cache_dir / "pubmed_abstracts.tsv"
    pubmed_map: dict[str, tuple[str, str]] = {}
    if pubmed_cache.exists():
        print(f"Loading PubMed cache: {pubmed_cache}")
        df_p = pd.read_csv(pubmed_cache, sep="\t", dtype=str)
        for _, r in df_p.iterrows():
            pmid = str(r["pmid"])
            pubmed_map[pmid] = (str(r.get("title", "")) if r.get("title") is not None else "", str(r.get("abstract", "")) if r.get("abstract") is not None else "")

    missing_pmids = [p for p in pmids_uniq if p not in pubmed_map]
    if missing_pmids:
        print(f"Fetching PubMed abstracts (missing={len(missing_pmids)})...")

        write_header = not pubmed_cache.exists()
        with pubmed_cache.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=["pmid", "title", "abstract"])
            if write_header:
                writer.writeheader()

            for i in tqdm(range(0, len(missing_pmids), args.pubmed_batch_size), desc="PubMed"):
                batch = missing_pmids[i : i + args.pubmed_batch_size]
                got = fetch_pubmed_abstracts(
                    session=session,
                    pmids=batch,
                    sleep_s=args.sleep_pubmed,
                    email=args.email,
                    api_key=args.api_key,
                )

                for pmid, (title, abstract) in got.items():
                    if args.max_abstract_chars and abstract:
                        abstract = abstract[: int(args.max_abstract_chars)]
                    pubmed_map[pmid] = (title, abstract)
                    writer.writerow({"pmid": pmid, "title": title, "abstract": abstract})

    # Build final EntryID -> text
    print("Writing entryid_text.tsv...")
    with args.out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["EntryID", "text"])

        n_with_uniprot = 0
        n_with_pubmed = 0

        for acc in tqdm(uniq, desc="Assemble"):
            r = uniprot_rows.get(acc)
            if r is None:
                w.writerow([acc, ""])
                continue

            parts: list[str] = []

            # UniProt text (short, curated)
            uniprot_bits = [r.protein_name, r.organism_name, r.keywords, r.protein_families, r.cc_function, r.cc_subcellular_location]
            uniprot_text = " ".join([b for b in uniprot_bits if b])
            if uniprot_text:
                n_with_uniprot += 1
                parts.append(uniprot_text)

            # PubMed abstracts (richer)
            pmids = pmid_sets.get(acc, [])
            abs_parts: list[str] = []
            for pmid in pmids:
                title, abstract = pubmed_map.get(pmid, ("", ""))
                if title or abstract:
                    abs_parts.append(f"{title}. {abstract}".strip(" ."))

            if abs_parts:
                n_with_pubmed += 1
                parts.append(" ".join(abs_parts))

            text = " ".join(parts).strip()
            if args.strip_go and text:
                text = _strip_go_leakage(text)

            w.writerow([acc, text])

    print(f"Saved: {args.out_path}")
    print(f"Coverage: UniProt text {n_with_uniprot}/{len(uniq)} = {n_with_uniprot/ max(1,len(uniq)):.3f}")
    print(f"Coverage: PubMed text  {n_with_pubmed}/{len(uniq)} = {n_with_pubmed/ max(1,len(uniq)):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
