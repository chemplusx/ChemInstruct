import pandas as pd
import numpy as np
from typing import List, Dict, Set
from rdkit import Chem
from pubchempy import get_compounds
import re

class ChemicalNEREvaluator:
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the evaluator with custom weights for the combined metric.
        
        Args:
            weights (Dict[str, float]): Weights for HR, CCS, and CVS (should sum to 1)
        """
        self.weights = weights or {'α': 0.4, 'β': 0.3, 'γ': 0.3}
        self.iupac_pattern = re.compile(r'^[A-Za-z0-9\-\(\)]+$')
        
    def load_chemical_database(self) -> Set[str]:
        """
        Mock function to load a set of known chemical names.
        In practice, this should connect to a real chemical database.
        """
        # Example known chemicals - replace with actual database
        return {
            'methanol', 'ethanol', 'sodium chloride', 'hydrogen peroxide',
            'acetone', 'benzene', 'glucose', 'sulfuric acid'
        }
        
    def verify_chemical_existence(self, chemical: str) -> bool:
        """
        Verify if a chemical exists using PubChem.
        
        Args:
            chemical (str): Chemical name to verify
            
        Returns:
            bool: True if chemical exists in PubChem
        """
        try:
            results = get_compounds(chemical, 'name')
            return len(results) > 0
        except:
            return False
            
    def validate_iupac_name(self, chemical: str) -> bool:
        """
        Basic IUPAC name validation.
        
        Args:
            chemical (str): Chemical name to validate
            
        Returns:
            bool: True if name follows basic IUPAC patterns
        """
        return bool(self.iupac_pattern.match(chemical))
        
    def calculate_hallucination_rate(self, 
                                   predicted: List[str], 
                                   ground_truth: List[str]) -> float:
        """
        Calculate hallucination rate based on ground truth comparison.
        
        Args:
            predicted (List[str]): Predicted chemical entities
            ground_truth (List[str]): Ground truth chemical entities
            
        Returns:
            float: Hallucination rate
        """
        pred_set = set(predicted)
        true_set = set(ground_truth)
        false_entities = len(pred_set - true_set)
        return false_entities / len(pred_set) if pred_set else 0
        
    def check_context_consistency(self, 
                                text: str, 
                                entity: str, 
                                context_window: int = 50) -> bool:
        """
        Check if the entity appears in a valid chemical context.
        
        Args:
            text (str): Full text
            entity (str): Chemical entity
            context_window (int): Number of characters to check before/after
            
        Returns:
            bool: True if context is valid
        """
        # Chemical context keywords
        chemical_contexts = {
            'solution', 'compound', 'reaction', 'concentration',
            'dissolved', 'mixed', 'added', 'contains', 'substance'
        }
        
        # Find entity position
        pos = text.lower().find(entity.lower())
        if pos == -1:
            return False
            
        # Extract context window
        start = max(0, pos - context_window)
        end = min(len(text), pos + len(entity) + context_window)
        context = text[start:end].lower()
        
        # Check for chemical context keywords
        return any(keyword in context for keyword in chemical_contexts)
        
    def calculate_ccs(self, 
                     text: str, 
                     predicted_entities: List[str]) -> float:
        """
        Calculate Context Consistency Score.
        
        Args:
            text (str): Full text
            predicted_entities (List[str]): Predicted chemical entities
            
        Returns:
            float: Context Consistency Score
        """
        if not predicted_entities:
            return 0
            
        consistent_count = sum(
            self.check_context_consistency(text, entity)
            for entity in predicted_entities
        )
        return consistent_count / len(predicted_entities)
        
    def calculate_cvs(self, predicted_entities: List[str]) -> float:
        """
        Calculate Chemical Validity Score.
        
        Args:
            predicted_entities (List[str]): Predicted chemical entities
            
        Returns:
            float: Chemical Validity Score
        """
        if not predicted_entities:
            return 0
            
        validity_scores = []
        for entity in predicted_entities:
            # Equal weights for IUPAC validation and database existence
            iupac_valid = self.validate_iupac_name(entity)
            exists_in_db = self.verify_chemical_existence(entity)
            validity_scores.append(0.5 * iupac_valid + 0.5 * exists_in_db)
            
        return sum(validity_scores) / len(predicted_entities)
        
    def calculate_combined_metric(self, 
                                hr: float, 
                                ccs: float, 
                                cvs: float) -> float:
        """
        Calculate combined metric using weighted sum.
        
        Args:
            hr (float): Hallucination Rate
            ccs (float): Context Consistency Score
            cvs (float): Chemical Validity Score
            
        Returns:
            float: Combined metric
        """
        return (
            self.weights['α'] * hr +
            self.weights['β'] * (1 - ccs) +
            self.weights['γ'] * (1 - cvs)
        )
        
    def evaluate(self, 
                 text: str, 
                 predicted_entities: List[str], 
                 ground_truth_entities: List[str]) -> Dict[str, float]:
        """
        Perform complete evaluation of chemical NER results.
        
        Args:
            text (str): Full text
            predicted_entities (List[str]): Predicted chemical entities
            ground_truth_entities (List[str]): Ground truth chemical entities
            
        Returns:
            Dict[str, float]: Dictionary containing all metrics
        """
        hr = self.calculate_hallucination_rate(
            predicted_entities, 
            ground_truth_entities
        )
        ccs = self.calculate_ccs(text, predicted_entities)
        cvs = self.calculate_cvs(predicted_entities)
        combined = self.calculate_combined_metric(hr, ccs, cvs)
        
        return {
            'HR': hr,
            'CCS': ccs,
            'CVS': cvs,
            'Combined_Metric': combined,
            'Rounded': round(combined, 2)
        }

# Example usage
def main():
    # Sample data
    text = """
    The solution contains methanol and sodium chloride.
    """
    
    predicted_entities = [
        'methanol',
        'sodium chloride',
        'ethylxenon'  # hallucinated
    ]
    
    ground_truth_entities = [
        'methanol',
        'sodium chloride'
    ]
    
    # Initialize evaluator
    evaluator = ChemicalNEREvaluator()
    
    # Calculate metrics
    results = evaluator.evaluate(
        text,
        predicted_entities,
        ground_truth_entities
    )
    
    # Print results
    print("\nChemical NER Evaluation Results:")
    print("-" * 30)
    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    main()