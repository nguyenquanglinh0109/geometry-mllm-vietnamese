"""
Math problem format normalization for Qwen/LLaVA fine-tuning.

Supports two problem types:
1. Calculation: Multiple choice problem with image
2. Proving: Step-by-step proof problem with image

Normalized format for both:
{
    "id": str,
    "image": numpy.ndarray or PIL.Image,
    "problem_type": "calculation" | "proving",
    "question": str,  # The question/problem statement
    "language": "zh" | "en",  # Chinese or English
    "image_path": str,  # Optional: path to image file
    "cot": str
    
    # For calculation problems
    "choices": List[str],  # Answer choices (optional for calculation)
    "answer": str,  # Final answer
    "solution": str,  # Detailed solution steps
    
    # For proving problems
    "given": str,  # Given conditions
    "to_prove": str,  # What to prove
    "statements": List[str],  # Step-by-step statements
    "reasons": List[str],  # Reasoning for each step (parallel to statements)
    "elements": List[str],  # Key geometric elements
    
    # Common fields
    "metadata": {
        "problem_type_category": str,  # e.g., "parallel_lines", "triangles"
        "reasoning_skill": str,  # e.g., "D6-proofs-involving-parallel-lines"
        "difficulty": int,  # Optional
        "source_dataset": str,  # e.g., "UniGeo", "formalgeo7k"
    }
}
"""

from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from PIL import Image
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd

@dataclass
class NormalizedMathProblem:
    """Normalized format for math problems with images."""
    id: str
    problem_type: str  # "calculation" or "proving"
    question: str
    image_path: Optional[str] = None
    cot: Optional[str] = None
    
    # For calculation
    choices: Optional[List[str]] = None
    answer: Optional[str] = None
    solution: Optional[str] = None
    
    # For proving
    statements: Optional[List[str]] = None
    reasons: Optional[List[str]] = None
    elements: Optional[List[str]] = None
    
    # Metadata
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return asdict(self)
    
    def get_prompt_format(self) -> str:
        """Get formatted prompt for LLM input."""
        if self.problem_type == "calculation":
            prompt = f"Question: {self.question}\n\nSolution: {self.solution}\n\nChoices: {', '.join(self.choices) if self.choices else 'N/A'}\n\nAnswer: {self.answer}"
        else:  # proving
            prompt = f"Question: {self.question}\n\nStatements: {self.statements}\n\nReason: {self.reasons}"
        return prompt


class MathCalculationFormatter:
    """Convert calculation format to normalized format."""
    
    @staticmethod
    def normalize(data: Dict, source_dataset: str) -> NormalizedMathProblem:
        """
        Convert calculation problem to normalized format.
        
        Args:
            data: Original calculation problem dict
            
        Returns:
            NormalizedMathProblem instance
        """
        problem_id = str(data.get('id', ''))
        
        # Get question in preferred language
        question = data.get('English_problem', '')
        if not question:
            question = data.get('subject', '')
        
        # Get image
        image_path = os.path.join(source_dataset.split('.')[0], str(problem_id), '.png')
        
        # Extract choices and answer
        choices = data.get('choices', [])
        label = data.get('label')
        answer = choices[label] if label is not None and label < len(choices) else ""
        
        # Build solution from answer text
        solution = data.get('answer', '')
        
        # Metadata
        metadata = {
            'problem_type_category': data.get('problem_form', 'calculation'),
            'formal_point': list(data.get('formal_point', set())) if isinstance(data.get('formal_point'), set) else data.get('formal_point', []),
            'numbers': data.get('numbers', []),
            'manual_program': data.get('manual_program', []),
            'source_dataset': source_dataset,
        }
        
        return NormalizedMathProblem(
            id=problem_id,
            problem_type='calculation',
            question=question,
            image_path=image_path,
            choices=choices,
            answer=answer,
            solution=solution,
            metadata=metadata
        )


class MathProvingFormatter:
    """Convert proving format to normalized format."""
    
    @staticmethod
    def normalize(data: Dict, source_dataset: str) -> NormalizedMathProblem:
        """
        Convert proving problem to normalized format.
        
        Args:
            data: Original proving problem dict
            
        Returns:
            NormalizedMathProblem instance
        """
        problem_id = str(data.get('id', ''))
        
        # Get question
        question = data.get('question', '') or data.get('input_text', '')
        
        # Get image
        image_path = os.path.join(source_dataset.split('.')[0], str(problem_id), '.png')
        
        # Extract statements and reasons
        statements = data.get('statement', [])
        reasons = data.get('reason', [])
        elements = data.get('elements', [])
        
        # Extract given and to_prove
        # given = statements[0] if statements else ""
        # to_prove = statements[-1] if statements else ""
        
        # Metadata
        metadata = {
            'problem_type_category': data.get('problem_form', 'proving'),
            'reasoning_skill': data.get('reasoning_skill', ''),
            'proving_sequence': data.get('proving_sequence', []),
            'source_dataset': 'proving_dataset',
        }
        
        return NormalizedMathProblem(
            id=problem_id,
            problem_type='proving',
            question=question,
            image_path=image_path,
            statements=statements,
            reasons=reasons,
            elements=elements,
            metadata=metadata
        )


class MathProblemNormalizer:
    """Main class to normalize math problems."""
    
    @staticmethod
    def normalize(data: Dict,  source_dataset: str, problem_type: Optional[str] = None) -> NormalizedMathProblem:
        """
        Normalize math problem based on type.
        
        Args:
            data: Problem data dict
            problem_type: 'calculation' or 'proving'. Auto-detect if None.
            
        Returns:
            NormalizedMathProblem instance
        """
        # Auto-detect problem type if not specified
        if problem_type is None:
            if 'problem_form' in data:
                problem_type = data['problem_form']
            elif 'choices' in data:
                problem_type = 'calculation'
            elif 'statement' in data and 'reason' in data:
                problem_type = 'proving'
            else:
                raise ValueError("Cannot determine problem type. Please specify explicitly.")
        
        if problem_type == 'calculation':
            return MathCalculationFormatter.normalize(data, source_dataset)
        elif problem_type == 'proving':
            return MathProvingFormatter.normalize(data, source_dataset)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")    

    @staticmethod
    def normalize_batch(data: List[Dict], source_dataset: str, problem_type: Optional[str] = None) -> List[NormalizedMathProblem]:
        return [MathProblemNormalizer.normalize(problem, source_dataset, problem_type) for problem in data]
    
def test_normalize_calculation():
    data_path = r"dataset\UniGeo_data\UniGeo\calculation_test.pk"


    with open(data_path, "rb") as f:
        data = pickle.load(f)

    data = data[0]
    normalized_problem = MathProblemNormalizer.normalize(data, source_dataset='calculation_test.pk', problem_type='calculation')
    print(normalized_problem)
    print("=" * 100)
    print(normalized_problem.get_prompt_format())
    
def test_normalize_proving():
    data_path = r"dataset\UniGeo_data\UniGeo\proving_test.pk"

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    data = data[0:10]
    normalized_problem = MathProblemNormalizer.normalize_batch(data, source_dataset='proving_test.pk', problem_type='proving')
    print(normalized_problem)
    print("=" * 100)
    normalized_problem = [np.get_prompt_format() for np in normalized_problem]
    for idx, np in enumerate(normalized_problem):
        print(f"Problem {idx}:")
        print(np)
        print("=" * 50)
    # df = pd.DataFrame([problem.__dict__ for problem in normalized_problem])
    # df.to_csv("dataset/UniGeo_data/dataframe/proving_test.csv", index=False)
    
def save_normalized_data(data_path: str, save_path: str):
    try: 
        with open(data_path, "rb") as f:
            data = pickle.load(f)
            
        data_path = os.path.basename(data_path)
        problem_type = data_path.split('.')[0].split('_')[0]
        normalized_problem = MathProblemNormalizer.normalize_batch(data, source_dataset=data_path, problem_type=problem_type)

        df = pd.DataFrame([problem.__dict__ for problem in normalized_problem])

        df.to_csv(save_path, index=False)
        
        print(f"Data saved to {save_path}")
    except Exception as e:
        print(f"Error when saving data: {e}")    
    
    
if __name__ == "__main__":
    ## 1. Sample test
    # test_normalize_calculation()
    # test_normalize_proving()
    
    ## 2. Save normalized data
    data_dir = "dataset/UniGeo_data/UniGeo"    
    save_dir = "dataset/UniGeo_data/dataframe"
    
    for file in os.listdir(data_dir):
        true_file = any(x in file for x in ['test', 'val', 'train'])
        if file.endswith('.pk') and true_file:
            data_path = os.path.join(data_dir, file)
            save_path = os.path.join(save_dir, file[:-3] + ".csv")
            save_normalized_data(data_path, save_path)