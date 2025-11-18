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


@dataclass
class NormalizedMathProblem:
    """Normalized format for math problems with images."""
    id: str
    problem_type: str  # "calculation" or "proving"
    question: str
    language: str  # "zh" or "en"
    image: Optional[Union[np.ndarray, Image.Image]] = None
    image_path: Optional[str] = None
    cot: Optional[str] = None
    
    # For calculation
    choices: Optional[List[str]] = None
    answer: Optional[str] = None
    solution: Optional[str] = None
    
    # For proving
    given: Optional[str] = None
    to_prove: Optional[str] = None
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
            prompt = f"{self.question}\n\nChoices: {', '.join(self.choices) if self.choices else 'N/A'}"
        else:  # proving
            prompt = f"Given: {self.given}\nTo prove: {self.to_prove}\n\n{self.question}"
        return prompt


class MathCalculationFormatter:
    """Convert calculation format to normalized format."""
    
    @staticmethod
    def normalize(data: Dict) -> NormalizedMathProblem:
        """
        Convert calculation problem to normalized format.
        
        Args:
            data: Original calculation problem dict
            
        Returns:
            NormalizedMathProblem instance
        """
        problem_id = str(data.get('id', ''))
        
        # Get question in preferred language
        question = data.get('subject', '')  # Chinese by default
        if not question:
            question = data.get('English_problem', '')
        
        language = 'zh' if data.get('subject') else 'en'
        
        # Get image
        image = data.get('image') or data.get('img')
        
        # Extract choices and answer
        choices = data.get('choices', [])
        label = data.get('label')
        answer = choices[label] if label is not None and label < len(choices) else ""
        
        # Build solution from answer text
        solution = data.get('answer', '')
        
        # Metadata
        metadata = {
            'problem_type_category': data.get('problem_type', 'calculation'),
            'formal_point': list(data.get('formal_point', set())) if isinstance(data.get('formal_point'), set) else data.get('formal_point', []),
            'numbers': data.get('numbers', []),
            'manual_program': data.get('manual_program', []),
            'source_dataset': 'calculation_dataset',
        }
        
        return NormalizedMathProblem(
            id=problem_id,
            problem_type='calculation',
            question=question,
            language=language,
            image=image,
            choices=choices,
            answer=answer,
            solution=solution,
            metadata=metadata
        )


class MathProvingFormatter:
    """Convert proving format to normalized format."""
    
    @staticmethod
    def normalize(data: Dict) -> NormalizedMathProblem:
        """
        Convert proving problem to normalized format.
        
        Args:
            data: Original proving problem dict
            
        Returns:
            NormalizedMathProblem instance
        """
        problem_id = str(data.get('id', ''))
        
        # Get question
        question = data.get('input_text') or data.get('question', '')
        
        # Detect language
        language = 'zh' if any(ord(c) > 0x4E00 for c in question) else 'en'
        
        # Get image
        image = data.get('img') or data.get('image')
        
        # Extract statements and reasons
        statements = data.get('statement', [])
        reasons = data.get('reason', [])
        elements = data.get('elements', [])
        
        # Extract given and to_prove
        given = statements[0] if statements else ""
        to_prove = statements[-1] if statements else ""
        
        # Metadata
        metadata = {
            'problem_type_category': data.get('problem_type', 'proving'),
            'reasoning_skill': data.get('reasoning_skill', ''),
            'proving_sequence': data.get('proving_sequence', []),
            'source_dataset': 'proving_dataset',
        }
        
        return NormalizedMathProblem(
            id=problem_id,
            problem_type='proving',
            question=question,
            language=language,
            image=image,
            given=given,
            to_prove=to_prove,
            statements=statements,
            reasons=reasons,
            elements=elements,
            metadata=metadata
        )


class MathProblemNormalizer:
    """Main class to normalize math problems."""
    
    @staticmethod
    def normalize(data: Dict, problem_type: Optional[str] = None) -> NormalizedMathProblem:
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
            return MathCalculationFormatter.normalize(data)
        elif problem_type == 'proving':
            return MathProvingFormatter.normalize(data)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")    
