"""Build storage-safe GOA/GAF-derived features.

Key idea: never uncompress GOA GAF to disk.
- Stream the `.gaf.gz` file
- Filter early to the CAFA protein IDs (from Train/Test FASTAs)
- Write a compact compressed artefact (`.tsv.gz`)

This avoids Kaggle "No space left on device" failures.

Example (local):
  python scripts/01_build_goa_features.py \
    --gaf-gz artefacts_local/artefacts/external/goa_uniprot_all.gaf.216.gz \
    --train-fasta Train/train_sequences.fasta \
    --test-fasta Test/testsuperset.fasta \
    --out artefacts_local/artefacts/external/goa_filtered_iea.tsv.gz \
    --only-iea

Example (Kaggle): write outputs to /kaggle/temp then (optionally) save as a Dataset.

"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, Set, Tuple


def _extract_entry_id(token: str) -> str:
    """Normalise FASTA header tokens into a CAFA EntryID.

    Train FASTA commonly uses UniProt headers like `sp|A0A0C5B5G6|MOTSC_HUMAN`,
    whereas CAFA TSVs and GOA GAF use the accession only (`A0A0C5B5G6`).
    """
    s = str(token).strip()
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 3 and parts[0] in {"sp", "tr"} and parts[1]:
            return parts[1]
    return s


def iter_fasta_ids(fasta_path: Path) -> Iterator[str]:
    """Yield FASTA record IDs (first token after '>')."""
    with fasta_path.open("rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                if not header:
                    continue
                yield _extract_entry_id(header.split()[0])


def load_cafa_ids(train_fasta: Path, test_fasta: Path) -> Set[str]:
    ids: Set[str] = set()
    ids.update(iter_fasta_ids(train_fasta))
    ids.update(iter_fasta_ids(test_fasta))
    return ids


def iter_gaf_rows(gaf_gz: Path) -> Iterator[Tuple[str, str, str]]:
    """Yield (db_object_id, go_id, evidence) from a GAF 2.x file."""
    with gzip.open(gaf_gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            # GAF 2.1: DB, DB_Object_ID, DB_Object_Symbol, Qualifier, GO_ID, DB:Reference, Evidence_Code, ...
            if len(parts) < 7:
                continue
            yield parts[1], parts[4], parts[6]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter GOA GAF to CAFA proteins and write compressed TSV.")
    parser.add_argument("--gaf-gz", type=Path, required=True, help="Path to GOA `.gaf.gz` file")
    parser.add_argument("--train-fasta", type=Path, required=True, help="Train FASTA path")
    parser.add_argument("--test-fasta", type=Path, required=True, help="Test FASTA path")
    parser.add_argument("--out", type=Path, required=True, help="Output `.tsv.gz` path")
    parser.add_argument(
        "--only-iea",
        action="store_true",
        help="Keep only IEA (electronic) annotations (recommended for no-leakage features).",
    )
    parser.add_argument(
        "--exclude-exp",
        action="store_true",
        help="Exclude classic experimental codes (EXP/IDA/IPI/IMP/IGI/IEP).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Optional safety cap (debug): stop after writing N lines (0 = no cap).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    gaf_gz = args.gaf_gz
    out = args.out

    if not gaf_gz.exists():
        raise FileNotFoundError(f"GAF not found: {gaf_gz}")

    out.parent.mkdir(parents=True, exist_ok=True)

    # Evidence policy
    exp_codes = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}

    # Load CAFA protein IDs (this is the critical early filter)
    print("Loading CAFA protein IDs from FASTA...")
    cafa_ids = load_cafa_ids(args.train_fasta, args.test_fasta)
    print(f"Loaded {len(cafa_ids):,} unique protein IDs")

    written = 0
    seen = 0
    kept = 0

    print(f"Streaming GAF: {gaf_gz}")
    print(f"Writing: {out}")

    # Write compressed TSV
    with gzip.open(out, "wt", encoding="utf-8") as f_out:
        f_out.write("EntryID\tterm\tevidence\n")

        for obj_id, go_id, evidence in iter_gaf_rows(gaf_gz):
            seen += 1

            if obj_id not in cafa_ids:
                continue

            if args.only_iea and evidence != "IEA":
                continue

            if args.exclude_exp and evidence in exp_codes:
                continue

            kept += 1
            f_out.write(f"{obj_id}\t{go_id}\t{evidence}\n")
            written += 1

            if args.max_lines and written >= args.max_lines:
                print("Hit --max-lines cap; stopping early.")
                break

            if written % 1_000_000 == 0:
                print(f"  wrote {written:,} lines (seen {seen:,}, kept {kept:,})")

    print("Done.")
    print(f"Seen rows: {seen:,}")
    print(f"Kept rows (post-filter): {kept:,}")
    print(f"Written lines: {written:,}")
    print(f"Output size: {out.stat().st_size / (1024**3):.2f} GiB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
