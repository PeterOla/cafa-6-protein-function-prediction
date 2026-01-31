"""
Visualisation: GBDT Elite Subset Training Strategy
Shows the distribution of term frequencies and highlights which terms GBDT trains on vs predicts zero for.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

# Load data
train_terms = pd.read_csv('Train/train_terms.tsv', sep='\t')
top_terms = json.load(open('cafa6_data/features/top_terms_13500.json'))
stable_terms = set(json.load(open('cafa6_data/features/stable_terms_1585.json')))

# Count positives per term
term_counts = train_terms.groupby('term').size().to_dict()
counts_for_targets = [(t, term_counts.get(t, 0), t in stable_terms) for t in top_terms]
df = pd.DataFrame(counts_for_targets, columns=['term', 'count', 'is_stable'])

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Histogram of term frequencies (log scale)
ax1 = axes[0]
bins = np.logspace(np.log10(df['count'].min()), np.log10(df['count'].max()), 50)
ax1.hist(df[df['is_stable']]['count'], bins=bins, alpha=0.7, label='GBDT Trains (≥50 positives)', color='green', edgecolor='black')
ax1.hist(df[~df['is_stable']]['count'], bins=bins, alpha=0.7, label='GBDT Zero-Predicts (<50)', color='red', edgecolor='black')
ax1.axvline(50, color='black', linestyle='--', linewidth=2, label='Noise Floor (50 positives)')
ax1.set_xscale('log')
ax1.set_xlabel('Number of Positive Samples (log scale)', fontsize=13)
ax1.set_ylabel('Number of Terms', fontsize=13)
ax1.set_title('GBDT Training Strategy: Term Frequency Distribution\n(13,500 Target Terms)', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)

# Add annotation
ax1.text(0.02, 0.95, 
         f'Elite (≥50): {df["is_stable"].sum()} terms (11.7%)\nZero-Predicted: {(~df["is_stable"]).sum()} terms (88.3%)',
         transform=ax1.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Cumulative Distribution Function
ax2 = axes[1]
sorted_counts = df.sort_values('count', ascending=False)['count'].values
cumsum = np.cumsum(sorted_counts) / sorted_counts.sum()
x = np.arange(len(cumsum))
ax2.plot(x, cumsum * 100, color='blue', linewidth=2)
ax2.axvline(1585, color='green', linestyle='--', linewidth=2, label='Elite Terms (1,585)')
ax2.axhline(100 * cumsum[1584], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Term Rank (by frequency)', fontsize=13)
ax2.set_ylabel('Cumulative % of All Positive Samples', fontsize=13)
ax2.set_title('Coverage Analysis: Elite 1,585 Terms Cover What % of Total Annotations?', fontsize=15, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

# Add annotation
elite_coverage = cumsum[1584] * 100
ax2.text(0.55, 0.25, 
         f'Elite 1,585 terms cover:\n{elite_coverage:.1f}% of all positive samples\n\nRemaining 11,915 terms:\n{100 - elite_coverage:.1f}% of positives',
         transform=ax2.transAxes, fontsize=12,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 3: Percentile breakdown
ax3 = axes[2]
percentiles = [10, 25, 50, 75, 90, 95, 99]
values = [df['count'].quantile(p/100) for p in percentiles]
colors = ['red' if v < 50 else 'green' for v in values]
bars = ax3.barh(percentiles, values, color=colors, edgecolor='black', alpha=0.7)
ax3.axvline(50, color='black', linestyle='--', linewidth=2, label='Noise Floor (50)')
ax3.set_xlabel('Number of Positive Samples', fontsize=13)
ax3.set_ylabel('Percentile', fontsize=13)
ax3.set_title('Percentile Breakdown: How Rare Are Most Terms?', fontsize=15, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (p, v, c) in enumerate(zip(percentiles, values, colors)):
    label_x = v + (5 if v < 200 else 20)
    ax3.text(label_x, i, f'{int(v)}', va='center', fontsize=11, fontweight='bold')

# Add summary stats box
stats_text = f"""Summary Statistics:
Min:    {df['count'].min():.0f}
Median: {df['count'].median():.0f}
Mean:   {df['count'].mean():.1f}
Max:    {df['count'].max():.0f}
"""
ax3.text(0.7, 0.15, stats_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig('artefacts_local/audits/gbdt_elite_subset_analysis.png', dpi=150, bbox_inches='tight')
print('Saved: artefacts_local/audits/gbdt_elite_subset_analysis.png')
plt.show()

# Additional analysis: aspect breakdown
print('\n=== Aspect Breakdown (Elite vs Rare) ===')
aspect_col = 'aspect' if 'aspect' in train_terms.columns else None
if aspect_col:
    # Map to standard codes
    aspect_map = {'P': 'BP', 'F': 'MF', 'C': 'CC', 'BP': 'BP', 'MF': 'MF', 'CC': 'CC'}
    train_terms['aspect_norm'] = train_terms[aspect_col].map(aspect_map)
    
    # Get aspect for each target term
    term_to_aspect = train_terms.groupby('term')['aspect_norm'].first().to_dict()
    df['aspect'] = df['term'].map(lambda t: term_to_aspect.get(t, 'UNK'))
    
    # Count by aspect
    for asp in ['BP', 'MF', 'CC']:
        asp_df = df[df['aspect'] == asp]
        elite_count = asp_df['is_stable'].sum()
        rare_count = (~asp_df['is_stable']).sum()
        total = len(asp_df)
        print(f'{asp}: {total:>5} terms | Elite: {elite_count:>4} ({100*elite_count/total:>5.1f}%) | Rare: {rare_count:>5} ({100*rare_count/total:>5.1f}%)')
