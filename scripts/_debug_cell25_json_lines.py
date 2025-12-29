import json
from pathlib import Path


def main() -> None:
    nb = json.loads(Path('notebooks/05_cafa_e2e.ipynb').read_text(encoding='utf-8'))
    cell = nb['cells'][24]
    src = list(cell.get('source', []) or [])
    for i in range(652, 662):
        s = src[i]
        print(i, json.dumps(s, ensure_ascii=False))


if __name__ == '__main__':
    main()
