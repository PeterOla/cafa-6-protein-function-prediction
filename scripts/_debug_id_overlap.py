import pandas as pd

# Load train sequence IDs
train_ids = pd.read_feather('cafa6_data/parsed/train_seq.feather')['id'].astype(str)
print(f'Total train_ids: {len(train_ids)}')
print(f'First 10: {train_ids.head(10).tolist()}')

# Apply ID cleaning (as notebook does)
clean = train_ids.str.extract(r'\|(.*?)\|')[0].fillna(train_ids)
print(f'\nCleaned IDs (first 10): {clean.head(10).tolist()}')

# Load official train_terms
terms = pd.read_csv('Train/train_terms.tsv', sep='\t')
print(f'\nTotal EntryIDs in train_terms: {terms["EntryID"].nunique()}')
print(f'First 10: {terms["EntryID"].head(10).tolist()}')

# Check overlap
overlaps = set(clean) & set(terms['EntryID'])
print(f'\nOverlap: {len(overlaps)} IDs')
print(f'Missing from terms (in train_seq but not train_terms): {len(set(clean)) - len(overlaps)}')
print(f'Extra in terms (in train_terms but not train_seq): {len(set(terms["EntryID"])) - len(overlaps)}')

# Show some missing IDs
missing_from_terms = set(clean) - set(terms['EntryID'])
if missing_from_terms:
    print(f'\nFirst 10 missing from terms:')
    for i, mid in enumerate(sorted(missing_from_terms)[:10], 1):
        print(f'  {i}. {mid}')
