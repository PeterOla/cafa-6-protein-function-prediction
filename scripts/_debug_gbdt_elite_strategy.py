import pandas as pd
import json

# Load data
train_terms = pd.read_csv('Train/train_terms.tsv', sep='\t')
top_terms = json.load(open('cafa6_data/features/top_terms_13500.json'))
stable_terms = json.load(open('cafa6_data/features/stable_terms_1585.json'))

# Count positives per term
term_counts = train_terms.groupby('term').size().to_dict()

# Analyze the 13,500 target terms
counts_for_targets = [(t, term_counts.get(t, 0)) for t in top_terms]
df = pd.DataFrame(counts_for_targets, columns=['term', 'count'])

print('=== Full Target Set (13,500 terms) ===')
print(f'Total terms: {len(df)}')
print(f'\nTerms with ≥50 positives (GBDT trains on these): {(df["count"] >= 50).sum()}')
print(f'Terms with <50 positives (GBDT zero-predicts): {(df["count"] < 50).sum()}')

print(f'\nPercentile distribution:')
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f'  {p}th: {df["count"].quantile(p/100):.0f} positives')

print(f'\nMin: {df["count"].min()}, Max: {df["count"].max()}, Mean: {df["count"].mean():.1f}')

# Check if stable_terms is exactly the ≥50 subset
rare_terms = df[df['count'] < 50]['term'].tolist()
stable_set = set(stable_terms)
top_set = set(top_terms)

print(f'\n=== Stable Terms (1,585 elite) ===')
print(f'Stable terms in top_terms: {len(stable_set & top_set)}')
print(f'Stable terms missing from top_terms: {len(stable_set - top_set)}')

# Show rarest terms
print(f'\n=== Rarest 20 Terms in Target Set ===')
rarest = df.nsmallest(20, 'count')[['term', 'count']]
print(rarest.to_string(index=False))

# Check GBDT's prediction strategy
print(f'\n=== GBDT Strategy Summary ===')
print(f'Training: 1,585 terms (≥50 positives)')
print(f'Zero-predictions: {len(rare_terms)} terms (<50 positives)')
print(f'Total contract: 13,500 terms')
print(f'\nThis means GBDT emits zeros for {100 * len(rare_terms) / len(top_terms):.1f}% of the target space!')
