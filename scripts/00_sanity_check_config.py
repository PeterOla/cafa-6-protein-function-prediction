from __future__ import annotations

import argparse
from pathlib import Path

from src.cafa6.config import ensure_work_dirs, load_config, resolve_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="CAFA-6 sanity check: config + inputs")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    ensure_work_dirs(paths, cfg)

    required_files = {
        "IA.tsv": paths.ia_tsv,
        "sample_submission.tsv": paths.sample_submission_tsv,
        "Train/train_sequences.fasta": paths.train_sequences_fasta,
        "Train/train_terms.tsv": paths.train_terms_tsv,
        "Train/train_taxonomy.tsv": paths.train_taxonomy_tsv,
        "Train/go-basic.obo": paths.go_obo,
        "Test/testsuperset.fasta": paths.test_sequences_fasta,
        "Test/testsuperset-taxon-list.tsv": paths.test_taxonomy_tsv,
    }

    missing = {name: p for name, p in required_files.items() if not p.exists()}

    print(f"Base dir: {paths.base_dir}")
    print(f"Work dir: {paths.work_dir}")

    if missing:
        print("\nMissing required inputs:")
        for name, p in missing.items():
            print(f"- {name}: {p}")
        print("\nFix by either:")
        print("- Setting paths.base_dir in config.yaml to the folder containing Train/ and Test/")
        print("- Or (Kaggle) set paths.kaggle_input_dirs to your dataset folder(s)")
        return 2

    print("\nAll required inputs found.")
    print("Work directories created under work_dir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
