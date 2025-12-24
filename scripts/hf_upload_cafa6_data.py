"""Upload the local cafa6_data artefact tree to Hugging Face Hub.

Why this exists:
- Kaggle dataset versioning tends to reupload multi-GB files frequently.
- HF Hub (dataset repo) supports incremental uploads and sane versioning.

Security:
- Reads token from env only (HUGGINGFACE_TOKEN / HF_TOKEN).
- Optionally loads a local .env file without printing secrets.

Typical usage:
  python scripts/hf_upload_cafa6_data.py --repo-id <user_or_org>/<repo> --work-root cafa6_data --repo-type dataset

Dry-run first:
  python scripts/hf_upload_cafa6_data.py --repo-id ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _posix_join(a: str, b: str) -> str:
    a = (a or "").strip().strip("/")
    b = (b or "").strip().lstrip("/")
    if not a:
        return b
    if not b:
        return a
    return f"{a}/{b}"


def _load_dotenv_if_present(dotenv_path: Path) -> None:
    """Minimal .env loader (no dependencies)."""
    try:
        dotenv_path = Path(dotenv_path)
        if not dotenv_path.exists():
            return
        for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        return


def _get_token() -> str:
    return (os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or "").strip()


def _iter_files(work_root: Path, ignore_globs: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in work_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(work_root)
        rel_posix = rel.as_posix()
        if any(Path(rel_posix).match(g) for g in ignore_globs):
            continue
        files.append(p)
    return files


def _repo_path_sizes(api, repo_id: str, repo_type: str) -> dict[str, int]:
    """Best-effort map of path_in_repo -> size in bytes."""
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except Exception:
        return {}

    siblings = getattr(info, "siblings", None)
    if siblings is None:
        return {}

    out: dict[str, int] = {}
    for s in siblings:
        # huggingface_hub versions differ: sibling can be an object or a dict.
        if isinstance(s, dict):
            name = (s.get("rfilename") or s.get("path") or "").strip()
            size = s.get("size")
        else:
            name = (getattr(s, "rfilename", None) or getattr(s, "path", None) or "").strip()
            size = getattr(s, "size", None)
        if not name:
            continue
        try:
            out[name] = int(size) if size is not None else -1
        except Exception:
            out[name] = -1
    return out


def _chunk_batches(
    items: list[tuple[Path, str, int]],
    *,
    max_files_per_commit: int,
    max_bytes_per_commit: int,
    single_file_over_bytes: int,
) -> list[list[tuple[Path, str, int]]]:
    """Greedy batching. Large files (>= threshold) become single-file commits."""
    batches: list[list[tuple[Path, str, int]]] = []
    cur: list[tuple[Path, str, int]] = []
    cur_bytes = 0

    for p, rel, sz in items:
        if sz >= single_file_over_bytes:
            if cur:
                batches.append(cur)
                cur = []
                cur_bytes = 0
            batches.append([(p, rel, sz)])
            continue

        if (max_files_per_commit and len(cur) >= max_files_per_commit) or (
            max_bytes_per_commit and cur and (cur_bytes + sz) > max_bytes_per_commit
        ):
            batches.append(cur)
            cur = []
            cur_bytes = 0
        cur.append((p, rel, sz))
        cur_bytes += sz

    if cur:
        batches.append(cur)
    return batches


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Upload cafa6_data folder to Hugging Face Hub (dataset repo).")
    ap.add_argument("--repo-id", type=str, default=os.environ.get("CAFA_HF_REPO_ID", "").strip(), help="HF repo id: <user_or_org>/<name>")
    ap.add_argument("--repo-type", type=str, default=os.environ.get("CAFA_HF_REPO_TYPE", "dataset").strip() or "dataset", choices=["dataset", "model", "space"], help="HF repo type (default: dataset)")
    ap.add_argument("--work-root", type=Path, default=Path(os.environ.get("CAFA_WORK_ROOT", "cafa6_data")), help="Local artefacts folder (default: cafa6_data)")
    ap.add_argument("--path-in-repo", type=str, default=os.environ.get("CAFA_HF_PATH_IN_REPO", "").strip(), help="Optional subfolder in the HF repo")
    ap.add_argument("--commit-message", type=str, default="Upload cafa6_data artefacts", help="Commit message")
    ap.add_argument("--create-repo", action="store_true", help="Create the repo if missing")
    ap.add_argument("--dry-run", action="store_true", help="List files that would upload but do not upload")

    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already present on HF with matching size (recommended).",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip existing repo files (force reupload attempts).",
    )
    ap.add_argument(
        "--max-files-per-commit",
        type=int,
        default=int(os.environ.get("CAFA_HF_MAX_FILES_PER_COMMIT", "50")),
        help="Max files per HF commit (default: 50).",
    )
    ap.add_argument(
        "--max-mb-per-commit",
        type=int,
        default=int(os.environ.get("CAFA_HF_MAX_MB_PER_COMMIT", "1024")),
        help="Max total MB per HF commit for small files (default: 1024).",
    )
    ap.add_argument(
        "--single-file-over-mb",
        type=int,
        default=int(os.environ.get("CAFA_HF_SINGLE_FILE_OVER_MB", "512")),
        help="If a file is >= this size, upload it in its own commit (default: 512).",
    )
    ap.add_argument(
        "--num-threads",
        type=int,
        default=int(os.environ.get("CAFA_HF_NUM_THREADS", "1")),
        help="Threads for HF preupload LFS (default: 1 for stability).",
    )

    ap.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Glob to ignore, relative to work-root (can repeat).",
    )

    args = ap.parse_args(argv)

    work_root = Path(args.work_root).resolve()
    if not work_root.exists():
        print(f"ERROR: work-root not found: {work_root}")
        return 2

    # Load .env (repo root + CWD) without printing values.
    cwd = Path.cwd()
    _load_dotenv_if_present(cwd / ".env")
    _load_dotenv_if_present(cwd.parent / ".env")

    ignore_globs = [
        "_tmp_*/*",
        "_tmp_*",
        "_publish_tmp/*",
        "_publish_tmp/**",
        "_publish_restore/*",
        "_publish_restore/**",
        "hf_cache/*",
        "hf_cache/**",
        "torch_cache/*",
        "torch_cache/**",
        "**/.ipynb_checkpoints/**",
    ] + list(args.ignore)

    files = _iter_files(work_root, ignore_globs=ignore_globs)
    # Prefer largest first so progress is visible and failures waste less time.
    files = sorted(files, key=lambda p: (p.stat().st_size, str(p).lower()), reverse=True)

    print(f"Work root: {work_root}")
    if args.repo_id:
        print(f"Repo: {args.repo_type}:{args.repo_id}")
        if args.path_in_repo:
            print(f"Path in repo: {args.path_in_repo}")
    else:
        print("Repo: (not set)")
    print(f"Files selected: {len(files)}")

    # Resolve resume default (resume unless explicitly disabled)
    resume = True
    if args.no_resume:
        resume = False
    if args.resume:
        resume = True

    if args.dry_run:
        preview = 50
        print(f"Resume: {resume}")
        print(
            "Batching: max_files_per_commit=", args.max_files_per_commit,
            "max_mb_per_commit=", args.max_mb_per_commit,
            "single_file_over_mb=", args.single_file_over_mb,
        )
        for p in files[:preview]:
            rel = p.relative_to(work_root).as_posix()
            print(f"  - {rel} ({p.stat().st_size / (1024**2):.2f} MB)")
        if len(files) > preview:
            print(f"  ... (+{len(files) - preview} more)")
        print("DRY RUN: not uploading.")
        return 0

    token = _get_token()
    if not token:
        print("ERROR: Missing HUGGINGFACE_TOKEN (or HF_TOKEN) in environment.")
        print("Set it via env vars, or add it to .env as HUGGINGFACE_TOKEN and rerun.")
        return 2

    # Soft validation: HF tokens usually start with 'hf_'. Don't print the token.
    if not token.startswith("hf_"):
        print("WARNING: HUGGINGFACE_TOKEN/HF_TOKEN does not look like a Hugging Face token (expected prefix 'hf_').")
        print("Uploads/auth may fail; regenerate a Write token at https://huggingface.co/settings/tokens")

    if not args.repo_id:
        print("ERROR: Missing --repo-id (or set CAFA_HF_REPO_ID).")
        return 2

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except Exception as e:
        print("ERROR: huggingface_hub is not installed in this environment.")
        print("Install with: pip install -U huggingface_hub")
        print(f"Underlying error: {e!r}")
        return 2

    api = HfApi(token=token)

    if args.create_repo:
        try:
            api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=True, exist_ok=True)
            print("Repo ensured (create_repo).")
        except Exception as e:
            print(f"ERROR: Failed to create/ensure repo: {e}")
            return 2

    # Batched uploads (resumable): avoid one huge commit and reduce wasted time on retry.
    try:
        from huggingface_hub import CommitOperationAdd
    except Exception:
        CommitOperationAdd = None  # type: ignore

    path_in_repo = (args.path_in_repo or "").strip().strip("/")
    repo_sizes: dict[str, int] = {}
    if resume:
        repo_sizes = _repo_path_sizes(api, repo_id=args.repo_id, repo_type=args.repo_type)

    items: list[tuple[Path, str, int]] = []
    skipped = 0
    for p in files:
        rel = p.relative_to(work_root).as_posix()
        dst = _posix_join(path_in_repo, rel)
        sz = int(p.stat().st_size)
        if resume and dst in repo_sizes and repo_sizes[dst] == sz:
            skipped += 1
            continue
        items.append((p, dst, sz))

    print(f"Resume: {resume} (skip already-present exact-size files)")
    if resume:
        print(f"Repo index: {len(repo_sizes)} paths")
    print(f"To upload: {len(items)} (skipped: {skipped})")

    max_bytes = max(1, int(args.max_mb_per_commit)) * 1024 * 1024
    single_over = max(1, int(args.single_file_over_mb)) * 1024 * 1024
    max_files = max(1, int(args.max_files_per_commit))

    batches = _chunk_batches(
        items,
        max_files_per_commit=max_files,
        max_bytes_per_commit=max_bytes,
        single_file_over_bytes=single_over,
    )

    print(f"Batches: {len(batches)}")

    try:
        for bi, batch in enumerate(batches, 1):
            mb = sum(sz for _, _, sz in batch) / (1024**2)
            msg = f"{args.commit_message} (batch {bi}/{len(batches)}; {len(batch)} files; ~{mb:.1f} MB)"
            print(f"\n==> Commit {bi}/{len(batches)}: {len(batch)} files (~{mb:.1f} MB)")

            if CommitOperationAdd is None:
                # Fallback: upload one file per call.
                for p, dst, _sz in batch:
                    api.upload_file(
                        repo_id=args.repo_id,
                        repo_type=args.repo_type,
                        path_or_fileobj=str(p),
                        path_in_repo=dst,
                        commit_message=msg,
                    )
            else:
                ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(p)) for p, dst, _sz in batch]
                # Try to pass num_threads if supported by this hub version.
                try:
                    api.create_commit(
                        repo_id=args.repo_id,
                        repo_type=args.repo_type,
                        operations=ops,
                        commit_message=msg,
                        num_threads=int(args.num_threads),
                    )
                except TypeError:
                    api.create_commit(
                        repo_id=args.repo_id,
                        repo_type=args.repo_type,
                        operations=ops,
                        commit_message=msg,
                    )
    except KeyboardInterrupt:
        print("\nInterrupted (KeyboardInterrupt).")
        print("Re-run the same command to resume; already-uploaded files will be skipped (size check).")
        return 130
    except HfHubHTTPError as e:
        print(f"ERROR: HF Hub HTTP error: {e}")
        print("Re-run the same command to resume; already-uploaded files will be skipped (size check).")
        return 1
    except Exception as e:
        print(f"ERROR: Upload failed: {e}")
        print("Re-run the same command to resume; already-uploaded files will be skipped (size check).")
        return 1

    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
