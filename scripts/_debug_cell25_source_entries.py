import json
from pathlib import Path


def main() -> None:
    nb = json.loads(Path('notebooks/05_cafa_e2e.ipynb').read_text(encoding='utf-8'))
    cell = nb['cells'][24]  # cell 25
    src_list = list(cell.get('source', []) or [])

    hits = [i for i, s in enumerate(src_list) if '[PROGRESS]' in s or 'fit=_fmt_seconds' in s or 'avg≈' in s]
    print('hits', hits)
    for h in hits:
        lo = max(0, h - 5)
        hi = min(len(src_list), h + 6)
        print('\n--- context around source index', h, '---')
        for j in range(lo, hi):
            print(f"{j:04d}: {src_list[j]!r}")


if __name__ == '__main__':
    main()
