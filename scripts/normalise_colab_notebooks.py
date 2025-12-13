import hashlib
import json
from pathlib import Path
from typing import Any


def _stable_cell_id(cell: dict[str, Any]) -> str:
    cell_type = str(cell.get("cell_type", ""))
    src = cell.get("source", [])
    if isinstance(src, list):
        payload = "\n".join(str(x) for x in src)
    else:
        payload = str(src)
    h = hashlib.sha1(f"{cell_type}\n{payload}".encode("utf-8", errors="replace")).hexdigest()
    return h[:8]


def _normalise_cell(cell: dict[str, Any]) -> dict[str, Any]:
    # Move VS Code top-level cell id -> metadata.id
    top_id = cell.pop("id", None)
    meta = cell.get("metadata") or {}
    if top_id and "id" not in meta:
        # VS Code tends to write "#VSC-<hex>".
        meta["id"] = str(top_id).replace("#VSC-", "")

    # Ensure required metadata.id exists for existing cells
    if "id" not in meta or not str(meta.get("id") or "").strip():
        meta["id"] = _stable_cell_id(cell)

    # Ensure required metadata.language exists
    if "language" not in meta or not str(meta.get("language") or "").strip():
        meta["language"] = "python" if cell.get("cell_type") == "code" else "markdown"

    cell["metadata"] = meta

    # Ensure Jupyter-friendly code cell fields
    if cell.get("cell_type") == "code":
        cell.setdefault("execution_count", None)
        cell.setdefault("outputs", [])

    return cell


def _patch_colab01_sanity(nb: dict[str, Any]) -> None:
    # Make sure the "Quick sanity check" cell doesn't crash if the TSV is missing.
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source")
        if not isinstance(src, list):
            continue
        joined = "\n".join(src)
        if "Quick sanity check" in joined and "entryid_text.tsv" in joined:
            cell["source"] = [
                "# Quick sanity check (won’t crash if the TSV isn’t produced yet)\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "\n",
                "p = Path('artefacts_local/artefacts/external/entryid_text.tsv')\n",
                "if not p.is_file():\n",
                "    print('Missing:', p)\n",
                "    ext_dir = p.parent\n",
                "    print('External dir exists:', ext_dir.exists(), ext_dir)\n",
                "    if ext_dir.exists():\n",
                "        print('External dir contents:', sorted([x.name for x in ext_dir.iterdir()])[:50])\n",
                "    print('Run the previous cell to build the corpus. If it failed, scroll that cell output.')\n",
                "else:\n",
                "    df = pd.read_csv(p, sep='\\t')\n",
                "    print('Shape:', df.shape)\n",
                "    print(df.head(3))\n",
                "    print('Non-empty text rows:', int((df['text'].fillna('').str.len() > 0).sum()))\n",
            ]
            return


def normalise_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))

    # Normalise cells
    nb["cells"] = [_normalise_cell(c) for c in nb.get("cells", [])]

    # Ensure top-level nbformat + metadata
    nb.setdefault("metadata", {})
    nb.setdefault("nbformat", 4)
    nb.setdefault("nbformat_minor", 4)

    # Minimal but valid metadata
    md = nb["metadata"]
    md.setdefault(
        "kernelspec",
        {"display_name": "Python 3", "language": "python", "name": "python3"},
    )
    md.setdefault("language_info", {"name": "python"})

    # Ensure no saved error outputs linger
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        # If outputs exist, drop them (portability + avoids stored tracebacks)
        cell["outputs"] = []
        cell["execution_count"] = None

    # Notebook-specific tweaks
    if path.name == "Colab_01_build_entryid_text_uniprot_pubmed.ipynb":
        _patch_colab01_sanity(nb)

    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    nb_dir = repo_root / "notebooks"
    targets = [
        nb_dir / "Colab_01_build_entryid_text_uniprot_pubmed.ipynb",
        nb_dir / "Colab_02_generate_optional_embeddings.ipynb",
    ]

    for p in targets:
        if not p.exists():
            raise FileNotFoundError(p)
        normalise_notebook(p)
        print("Normalised:", p)


if __name__ == "__main__":
    main()
