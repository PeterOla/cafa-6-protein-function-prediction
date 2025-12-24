"""Verify that local cafa6_data/ artefacts exist on Hugging Face Hub with matching sizes.

This is a safety check after uploads that complete suspiciously quickly.

Usage:
  python scripts/hf_verify_cafa6_data_upload.py --repo-id <user>/<repo> --work-root cafa6_data

Notes:
- Requires HF token in env (HF_TOKEN or HUGGINGFACE_TOKEN) for private repos.
- Does not print secrets.
"""

from __future__ import annotations

import argparse
import os
import sys
from fnmatch import fnmatch
from pathlib import Path


def _load_dotenv_if_present(dotenv_path: Path) -> None:
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
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()


def _posix_join(a: str, b: str) -> str:
    a = (a or "").strip().strip("/")
    b = (b or "").strip().lstrip("/")
    if not a:
        return b
    if not b:
        return a
    return f"{a}/{b}"


def _repo_path_sizes(api, repo_id: str, repo_type: str) -> dict[str, int]:
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    siblings = getattr(info, "siblings", None) or []

    out: dict[str, int] = {}
    for s in siblings:
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


def _remote_size_bytes(repo_file) -> int | None:
    """Best-effort size for a remote file from HF metadata."""
    size = getattr(repo_file, "size", None)
    if size is not None:
        try:
            return int(size)
        except Exception:
            pass

    lfs = getattr(repo_file, "lfs", None)
    if isinstance(lfs, dict) and lfs.get("size") is not None:
        try:
            return int(lfs.get("size"))
        except Exception:
            return None
    if lfs is not None:
        try:
            lfs_size = getattr(lfs, "size", None)
            return int(lfs_size) if lfs_size is not None else None
        except Exception:
            return None

    return None


def _safe_lfs_info(repo_file) -> tuple[bool, str]:
    """Return (is_lfs, summary)."""
    lfs = getattr(repo_file, "lfs", None)
    if not lfs:
        return False, ""

    # huggingface_hub may return a dict or a small object.
    if isinstance(lfs, dict):
        sha256 = lfs.get("sha256") or ""
        size = lfs.get("size")
    else:
        sha256 = getattr(lfs, "sha256", "") or ""
        size = getattr(lfs, "size", None)

    size_s = "?" if size is None else str(size)
    sha_s = sha256[:12] if sha256 else ""
    extra = f"size={size_s}" + (f", sha256={sha_s}…" if sha_s else "")
    return True, extra


def _iter_local_files(work_root: Path) -> list[Path]:
    return [p for p in work_root.rglob("*") if p.is_file()]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", type=str, default=os.environ.get("CAFA_HF_REPO_ID", "").strip())
    ap.add_argument(
        "--repo-type",
        type=str,
        default=(os.environ.get("CAFA_HF_REPO_TYPE", "dataset").strip() or "dataset"),
        choices=["dataset", "model", "space"],
    )
    ap.add_argument("--work-root", type=Path, default=Path(os.environ.get("CAFA_WORK_ROOT", "cafa6_data")))
    ap.add_argument("--path-in-repo", type=str, default=os.environ.get("CAFA_HF_PATH_IN_REPO", "").strip())
    ap.add_argument("--top", type=int, default=20, help="Show top-N largest local files")
    ap.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Glob pattern (relative to work-root) to ignore in local->remote comparison. Can be repeated.",
    )
    ap.add_argument(
        "--lfs-report",
        action="store_true",
        help="Print LFS metadata for top-N largest remote files (uses HF API paths-info).",
    )
    ap.add_argument("--lfs-top", type=int, default=10, help="Top-N largest remote files to report LFS status for")
    args = ap.parse_args(argv)

    # Local convenience
    cwd = Path.cwd()
    _load_dotenv_if_present(cwd / ".env")
    _load_dotenv_if_present(cwd.parent / ".env")

    token = _get_token()
    if not token:
        print("ERROR: missing HF_TOKEN/HUGGINGFACE_TOKEN (needed to inspect private repos)")
        return 2

    if not args.repo_id:
        print("ERROR: missing --repo-id (or set CAFA_HF_REPO_ID)")
        return 2

    work_root = Path(args.work_root).resolve()
    if not work_root.exists():
        print(f"ERROR: work-root not found: {work_root}")
        return 2

    try:
        from huggingface_hub import HfApi
    except Exception as e:
        print("ERROR: huggingface_hub not installed")
        print(repr(e))
        return 2

    api = HfApi(token=token)

    # Fetch repo index
    sizes = _repo_path_sizes(api, repo_id=args.repo_id, repo_type=args.repo_type)
    print(f"Repo: {args.repo_type}:{args.repo_id}")
    print(f"Repo paths indexed: {len(sizes)}")

    # Show largest local files (sanity)
    local_files = _iter_local_files(work_root)
    local_files_sorted = sorted(local_files, key=lambda p: p.stat().st_size, reverse=True)
    print(f"Local files: {len(local_files_sorted)}")
    print("Top local files:")
    for p in local_files_sorted[: max(0, args.top)]:
        rel = p.relative_to(work_root).as_posix()
        print(f"  - {rel}: {p.stat().st_size / (1024**3):.3f} GB")

    # Compare local -> remote using authoritative per-path metadata.
    prefix = (args.path_in_repo or "").strip().strip("/")
    ignore_globs = [g.strip() for g in (args.ignore or []) if g.strip()]
    missing: list[tuple[str, int]] = []
    mismatch: list[tuple[str, int, int]] = []
    unknown_remote_size: list[str] = []

    local_dst_paths: list[str] = []
    local_by_dst: dict[str, int] = {}
    for p in local_files_sorted:
        rel = p.relative_to(work_root).as_posix()
        if any(fnmatch(rel, g) for g in ignore_globs):
            continue
        dst = _posix_join(prefix, rel)
        local_sz = int(p.stat().st_size)
        local_dst_paths.append(dst)
        local_by_dst[dst] = local_sz

    remote_infos = api.get_paths_info(repo_id=args.repo_id, repo_type=args.repo_type, paths=local_dst_paths)
    remote_by_path = {getattr(i, "path", getattr(i, "rfilename", "")): i for i in remote_infos}

    for dst in local_dst_paths:
        local_sz = local_by_dst[dst]
        ri = remote_by_path.get(dst)
        if ri is None:
            missing.append((dst, local_sz))
            continue
        remote_sz = _remote_size_bytes(ri)
        if remote_sz is None:
            unknown_remote_size.append(dst)
            continue
        if remote_sz != local_sz:
            mismatch.append((dst, local_sz, remote_sz))

    if args.lfs_report:
        print("\nRemote LFS report (largest files):")
        # Sort by remote size (authoritative), fall back to local size.
        items: list[tuple[str, int]] = []
        for dst in local_dst_paths:
            ri = remote_by_path.get(dst)
            if ri is None:
                continue
            rs = _remote_size_bytes(ri)
            items.append((dst, rs if rs is not None else local_by_dst.get(dst, 0)))
        items.sort(key=lambda x: x[1], reverse=True)
        for dst, rs in items[: max(0, args.lfs_top)]:
            ri = remote_by_path.get(dst)
            if ri is None:
                continue
            is_lfs, extra = _safe_lfs_info(ri)
            sz_gb = f"{rs / (1024**3):.3f} GB" if rs is not None else "?"
            flag = "LFS" if is_lfs else "non-LFS"
            suffix = f" ({extra})" if extra else ""
            print(f"  - {dst}: {sz_gb} [{flag}]{suffix}")

    print("\nComparison:")
    print(f"  Missing on HF: {len(missing)}")
    print(f"  Size mismatches: {len(mismatch)}")
    if unknown_remote_size:
        print(f"  Unknown remote sizes: {len(unknown_remote_size)}")

    if missing:
        print("\nMissing (first 20):")
        for dst, local_sz in missing[:20]:
            print(f"  - {dst} (local {local_sz / (1024**3):.3f} GB)")

    if mismatch:
        print("\nMismatched sizes (first 20):")
        for dst, local_sz, remote_sz in mismatch[:20]:
            print(
                f"  - {dst} (local {local_sz / (1024**3):.3f} GB vs remote {remote_sz / (1024**3):.3f} GB)"
            )

    if unknown_remote_size:
        print("\nUnknown remote sizes (first 20):")
        for dst in unknown_remote_size[:20]:
            print(f"  - {dst}")

    if not missing and not mismatch and not unknown_remote_size:
        print("\nOK: all local files are present on HF with matching sizes.")
        return 0

    print("\nNOT OK: HF repo does not match local files yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
