from pathlib import Path


def main() -> None:
    p = Path('notebooks/05_cafa_e2e.ipynb')
    lines = p.read_text(encoding='utf-8').splitlines(True)
    idxs = [i for i, ln in enumerate(lines) if '[PROGRESS]' in ln or 'fit={_fmt_seconds' in ln or 'avg≈' in ln]
    print('match_file_line_idxs', idxs[:20], 'count', len(idxs))
    for i in idxs[:5]:
        lo = max(0, i-3)
        hi = min(len(lines), i+6)
        print('\n--- file context around', i, '(0-based) ---')
        for j in range(lo, hi):
            print(f"{j:05d}: {lines[j]!r}")


if __name__ == '__main__':
    main()
