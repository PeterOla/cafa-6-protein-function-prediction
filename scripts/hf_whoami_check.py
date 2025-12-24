"""Quick Hugging Face auth sanity check.

- Loads token from env (HUGGINGFACE_TOKEN / HF_TOKEN).
- If not present, tries to load .env from repo root.
- Calls https://huggingface.co/api/whoami-v2 with an explicit timeout.

Never prints the token.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _load_dotenv_if_present(dotenv_path: Path) -> None:
    try:
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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv_if_present(repo_root / ".env")

    token = (os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        print("HF whoami FAILED: missing token (HUGGINGFACE_TOKEN/HF_TOKEN)")
        return 2

    req = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "cafa6-hf-whoami-check/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
            name = payload.get("name") or payload.get("fullname") or payload.get("username") or "(unknown)"
            print(f"HF whoami OK: {name} (HTTP {response.status})")
            return 0
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"HF whoami FAILED: HTTP {e.code} {body[:300]}")
        return 1
    except Exception as e:
        print(f"HF whoami FAILED: {type(e).__name__} {str(e)[:300]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
