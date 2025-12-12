from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import yaml


@dataclass(frozen=True)
class ResolvedPaths:
    base_dir: Path
    work_dir: Path

    ia_tsv: Path
    sample_submission_tsv: Path

    train_sequences_fasta: Path
    train_terms_tsv: Path
    train_taxonomy_tsv: Path
    go_obo: Path

    test_sequences_fasta: Path
    test_taxonomy_tsv: Path


def _is_kaggle() -> bool:
    return (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or Path("/kaggle/input").exists()
    )


def _is_colab() -> bool:
    return os.environ.get("COLAB_GPU") is not None or os.environ.get("COLAB_RELEASE_TAG") is not None


def _first_existing_dir(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def load_config(config_path: str | os.PathLike[str] = "config.yaml") -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict[str, Any]) -> ResolvedPaths:
    paths_cfg = cfg.get("paths", {})
    inputs_cfg = cfg.get("inputs", {})

    base_dir_raw = (paths_cfg.get("base_dir") or "").strip()
    if base_dir_raw:
        base_dir = Path(base_dir_raw).expanduser().resolve()
    else:
        # Prefer an explicit input root when running on Kaggle/Colab.
        candidates: list[Path] = []

        if _is_kaggle():
            for p in paths_cfg.get("kaggle_input_dirs", []) or []:
                candidates.append(Path(p))
            candidates.append(Path("/kaggle/input"))

        if _is_colab():
            for p in paths_cfg.get("colab_input_dirs", []) or []:
                candidates.append(Path(p))

        for p in paths_cfg.get("local_input_dirs", []) or []:
            candidates.append(Path(p))

        base_dir = _first_existing_dir([c.expanduser() for c in candidates]) or Path.cwd()

    work_dir = (base_dir / (paths_cfg.get("work_dir") or "artefacts")).resolve()

    def p(rel: str) -> Path:
        return (base_dir / rel).resolve()

    ia_tsv = p(inputs_cfg["ia_tsv"])
    sample_submission_tsv = p(inputs_cfg["sample_submission_tsv"])

    train_cfg = inputs_cfg["train"]
    test_cfg = inputs_cfg["test"]

    return ResolvedPaths(
        base_dir=base_dir,
        work_dir=work_dir,
        ia_tsv=ia_tsv,
        sample_submission_tsv=sample_submission_tsv,
        train_sequences_fasta=p(train_cfg["sequences_fasta"]),
        train_terms_tsv=p(train_cfg["terms_tsv"]),
        train_taxonomy_tsv=p(train_cfg["taxonomy_tsv"]),
        go_obo=p(train_cfg["go_obo"]),
        test_sequences_fasta=p(test_cfg["sequences_fasta"]),
        test_taxonomy_tsv=p(test_cfg["taxonomy_tsv"]),
    )


def ensure_work_dirs(paths: ResolvedPaths, cfg: dict[str, Any]) -> None:
    paths.work_dir.mkdir(parents=True, exist_ok=True)

    artefacts_cfg = cfg.get("artefacts", {})
    for group in artefacts_cfg.values():
        if not isinstance(group, dict):
            continue
        for rel in group.values():
            out_path = paths.work_dir / str(rel)
            out_path.parent.mkdir(parents=True, exist_ok=True)
