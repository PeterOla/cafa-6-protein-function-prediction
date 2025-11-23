"""Baseline models"""

import numpy as np
from collections import Counter


class FrequencyBaseline:
    """Predict most frequent GO terms"""
    
    def __init__(self):
        self.term_frequencies = {}
        
    def fit(self, labels_df):
        """Learn term frequencies from training data"""
        term_counts = labels_df['term'].value_counts()
        total = len(labels_df['EntryID'].unique())
        
        self.term_frequencies = {
            term: count / total 
            for term, count in term_counts.items()
        }
        
        print(f"Learned frequencies for {len(self.term_frequencies)} terms")
    
    def predict(self, protein_ids, top_k=100):
        """Predict top-k most frequent terms for all proteins"""
        predictions = {}
        
        # Get top-k terms
        top_terms = sorted(self.term_frequencies.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:top_k]
        
        for protein_id in protein_ids:
            predictions[protein_id] = top_terms
        
        return predictions
