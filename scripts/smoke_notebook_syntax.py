import ast
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: smoke_notebook_syntax.py <path-to-ipynb>")
        return 2

    notebook_path = Path(sys.argv[1])
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    bad: list[tuple[int, str, int | None]] = []
    code_cells = 0

    for i, cell in enumerate(cells, 1):
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        raw = "".join(cell.get("source", []))
        # Notebook cells may contain IPython magics/shell escapes that aren't valid Python syntax.
        # Strip the common ones for a lightweight syntax smoke-check.
        src_lines = [
            ln
            for ln in raw.splitlines(True)
            if not ln.lstrip().startswith(("%", "!", "??", "?"))
        ]
        src = "".join(src_lines)
        try:
            ast.parse(src)
        except SyntaxError as e:
            bad.append((i, e.msg, e.lineno))

    print(f"code_cells={code_cells}")
    if bad:
        print("syntax_errors=")
        for i, msg, lineno in bad:
            print(f"  cell={i} line={lineno} msg={msg}")
        return 1

    print("syntax_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
