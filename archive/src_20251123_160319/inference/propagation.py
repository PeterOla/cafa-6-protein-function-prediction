"""
Label propagation using GO ontology hierarchy.

Key idea: If a model predicts a specific term (e.g., "DNA helicase activity"),
automatically add all ancestor terms (e.g., "helicase activity", "catalytic activity").
This ensures predictions are ontologically consistent and captures implied functions.
"""

import numpy as np
import torch
from typing import Dict, List, Set, Union
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data.loaders import OntologyLoader


def propagate_predictions(
    predictions: Union[np.ndarray, torch.Tensor],
    term_list: List[str],
    ontology_loader: OntologyLoader,
    strategy: str = 'max'
) -> Union[np.ndarray, torch.Tensor]:
    """
    Propagate predictions to ancestor terms using GO hierarchy.
    
    Args:
        predictions: (N, K) array of probabilities for K GO terms
        term_list: List of K GO term IDs corresponding to prediction columns
        ontology_loader: Loaded GO ontology graph
        strategy: How to set ancestor probabilities
            - 'max': Take max of (current, max(children))
            - 'copy': Copy max child probability if ancestor not predicted
            - 'threshold': Set to 1.0 if any child above threshold
    
    Returns:
        Propagated predictions with same shape as input
    
    Example:
        >>> preds = np.array([[0.1, 0.9, 0.05]])  # 3 terms
        >>> terms = ['GO:0003674', 'GO:0016787', 'GO:0004518']  
        >>> # If GO:0004518 (nuclease) is child of GO:0016787 (hydrolase)
        >>> propagated = propagate_predictions(preds, terms, ontology)
        >>> # Result: [[0.1, 0.9, 0.05]] -> [[0.1, 0.9, 0.05]] with ancestors boosted
    """
    # Convert to numpy for easier manipulation
    is_torch = isinstance(predictions, torch.Tensor)
    if is_torch:
        device = predictions.device
        predictions = predictions.cpu().numpy()
    
    propagated = predictions.copy()
    
    # Build term index mapping
    term_to_idx = {term: idx for idx, term in enumerate(term_list)}
    
    # For each term in vocabulary, find its ancestors
    for child_idx, child_term in enumerate(term_list):
        # Get ancestors from ontology
        ancestors = ontology_loader.get_ancestors(child_term)
        
        # Find which ancestors are in our vocabulary
        ancestor_indices = [
            term_to_idx[anc] for anc in ancestors 
            if anc in term_to_idx
        ]
        
        if not ancestor_indices:
            continue
            
        # Propagate child predictions to ancestors
        # predictions shape: (N_samples, N_terms)
        child_probs = propagated[:, child_idx:child_idx+1]  # (N, 1)
        
        for anc_idx in ancestor_indices:
            if strategy == 'max':
                # Ancestor prob = max(current, child_prob)
                propagated[:, anc_idx] = np.maximum(
                    propagated[:, anc_idx],
                    child_probs.squeeze()
                )
            elif strategy == 'copy':
                # Only update if ancestor currently low
                mask = propagated[:, anc_idx] < child_probs.squeeze()
                propagated[mask, anc_idx] = child_probs.squeeze()[mask]
            elif strategy == 'threshold':
                # Set ancestor to 1.0 if any child predicted
                mask = child_probs.squeeze() > 0.5
                propagated[mask, anc_idx] = 1.0
    
    # Convert back to torch if needed
    if is_torch:
        propagated = torch.from_numpy(propagated).to(device)
    
    return propagated


def get_propagated_terms(
    predicted_terms: Set[str],
    ontology_loader: OntologyLoader
) -> Set[str]:
    """
    Given a set of predicted GO terms, return the expanded set including all ancestors.
    
    Args:
        predicted_terms: Set of predicted GO term IDs
        ontology_loader: Loaded GO ontology
    
    Returns:
        Expanded set of terms including all ancestors
    
    Example:
        >>> predicted = {'GO:0004518'}  # nuclease activity
        >>> expanded = get_propagated_terms(predicted, ontology)
        >>> # Returns: {'GO:0004518', 'GO:0016787', 'GO:0003824', ...}
    """
    propagated = set(predicted_terms)
    
    for term in predicted_terms:
        ancestors = ontology_loader.get_ancestors(term)
        propagated.update(ancestors)
    
    return propagated


def propagate_submission(
    submission_df,
    ontology_loader: OntologyLoader,
    confidence_strategy: str = 'inherit'
):
    """
    Propagate predictions in a CAFA submission DataFrame.
    
    Args:
        submission_df: DataFrame with columns [EntryID, term, confidence]
        ontology_loader: Loaded GO ontology
        confidence_strategy: How to assign confidence to ancestors
            - 'inherit': Copy child confidence
            - 'max': Max confidence among children
            - 'fixed': Use fixed value (0.5)
    
    Returns:
        Expanded DataFrame with ancestor terms added
    """
    import pandas as pd
    
    rows = []
    
    # Group by protein
    for protein_id, group in submission_df.groupby('EntryID'):
        predicted_terms = set(group['term'])
        
        # Get all terms + ancestors
        all_terms = get_propagated_terms(predicted_terms, ontology_loader)
        
        # For each term, determine confidence
        for term in all_terms:
            if term in predicted_terms:
                # Original prediction - keep confidence
                conf = group[group['term'] == term]['confidence'].iloc[0]
            else:
                # Ancestor term - assign based on strategy
                # Find children of this ancestor that were predicted
                children = [
                    t for t in predicted_terms 
                    if term in ontology_loader.get_ancestors(t)
                ]
                
                if confidence_strategy == 'max' and children:
                    # Max confidence among children
                    child_confs = [
                        group[group['term'] == c]['confidence'].iloc[0]
                        for c in children
                    ]
                    conf = max(child_confs)
                elif confidence_strategy == 'inherit' and children:
                    # Use first child's confidence
                    conf = group[group['term'] == children[0]]['confidence'].iloc[0]
                else:
                    # Fixed confidence for ancestors
                    conf = 0.5
            
            rows.append({
                'EntryID': protein_id,
                'term': term,
                'confidence': conf
            })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    """Test propagation functionality."""
    import sys
    from pathlib import Path
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    obo_path = base_dir / "Train" / "go-basic.obo"
    
    if not obo_path.exists():
        print(f"GO ontology not found at {obo_path}")
        sys.exit(1)
    
    print("Loading GO ontology...")
    ontology = OntologyLoader(obo_path)
    
    # Test 1: Simple propagation with dummy predictions
    print("\n=== Test 1: Array propagation ===")
    # Create dummy predictions for 5 terms
    test_terms = [
        'GO:0003674',  # molecular_function (root)
        'GO:0016787',  # hydrolase activity
        'GO:0004518',  # nuclease activity (child of hydrolase)
        'GO:0008150',  # biological_process (root)
        'GO:0006281',  # DNA repair
    ]
    
    # Simulate predictions: high confidence for nuclease
    preds = np.array([[0.1, 0.2, 0.9, 0.05, 0.3]])
    
    print(f"Original predictions: {preds}")
    print(f"Terms: {test_terms}")
    
    propagated = propagate_predictions(preds, test_terms, ontology, strategy='max')
    print(f"Propagated: {propagated}")
    
    # Test 2: Set expansion
    print("\n=== Test 2: Set expansion ===")
    predicted = {'GO:0004518'}  # nuclease activity
    expanded = get_propagated_terms(predicted, ontology)
    print(f"Predicted: {predicted}")
    print(f"Expanded ({len(expanded)} terms): {list(expanded)[:5]}...")
    
    # Test 3: Check ancestor relationship
    print("\n=== Test 3: Verify hierarchy ===")
    nuclease = 'GO:0004518'
    hydrolase = 'GO:0016787'
    ancestors_of_nuclease = ontology.get_ancestors(nuclease)
    
    print(f"Is {hydrolase} an ancestor of {nuclease}? {hydrolase in ancestors_of_nuclease}")
    print(f"Nuclease has {len(ancestors_of_nuclease)} ancestors")
    
    print("\n✅ Propagation module tests complete!")
