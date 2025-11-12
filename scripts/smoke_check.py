import sys
from pathlib import Path
from Bio import SeqIO
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
train_fasta = ROOT / 'Train' / 'train_sequences.fasta'
train_terms = ROOT / 'Train' / 'train_terms.tsv'

print('Python:', sys.version)
print('Filenames:')
print('  ', train_fasta)
print('  ', train_terms)

# Read a few sequences quickly
n_seq = 0
if train_fasta.exists():
    for i, rec in enumerate(SeqIO.parse(str(train_fasta), 'fasta')):
        n_seq += 1
        if i < 3:
            print(f"  seq[{i}] id={rec.id} len={len(rec.seq)} preview={str(rec.seq)[:20]}")
        if i >= 99:
            break
else:
    print('Missing:', train_fasta)

# Read a few labels
if train_terms.exists():
    df = pd.read_csv(train_terms, sep='\t', header=0, usecols=['EntryID','term']).head(5)
    print('Label sample:')
    print(df)
else:
    print('Missing:', train_terms)

print('Done.')
