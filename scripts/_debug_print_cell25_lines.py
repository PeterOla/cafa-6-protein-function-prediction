import json
from pathlib import Path


def main() -> None:
    p = Path('notebooks/05_cafa_e2e.ipynb')
    nb = json.loads(p.read_text(encoding='utf-8'))
    cell = nb['cells'][24]  # cell 25 (1-based)
    src = ''.join(cell.get('source', []) or [])
    lines = src.splitlines()
    err_line = 659
    start = max(1, err_line - 40)
    end = min(len(lines), err_line + 40)
    print('cell25_total_lines', len(lines))
    for i in range(start, end + 1):
        print(f"{i:04d}: {lines[i-1]}")


if __name__ == '__main__':
    main()
