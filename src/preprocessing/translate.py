import os
import sys
import json
import pandas as pd
from pathlib import Path
from src.config import Config
from pandas.core.frame import DataFrame
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompt import TRANSLATE_PROMPT

os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY


def get_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")
    return df


def get_chat_model(model_name: str = "gpt-4o-mini"):
    """Initialize and return chat model."""
    llm = None
    if 'gemini' in model_name:
        llm = ChatGoogleGenerativeAI(model=model_name)
    elif 'gpt' in model_name:
        llm = init_chat_model(
            model_name,
            temperature=0.1,
            timeout=30,
            max_tokens=10000
        )
    return llm


def translate_single(text: str, index: int, llm: BaseChatModel, text_type: str = "question", retry_count: int = 3) -> tuple:
    """
    Translate a single text with retry logic.
    
    Args:
        text: Text to translate
        index: Index of item
        llm: Language model
        text_type: "question" or "answer"
        retry_count: Number of retries
    
    Returns:
        Tuple of (index, translated_text, error)
    """
    prompt = TRANSLATE_PROMPT 
    
    for attempt in range(retry_count):
        try:
            response = llm.invoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ])
            translated_text = response.content if hasattr(response, 'content') else str(response)
            return index, translated_text, None
        except Exception as e:
            if attempt == retry_count - 1:
                print(f"✗ Error translating item {index} ({text_type}) after {retry_count} attempts: {e}")
                return index, None, str(e)
            time.sleep(1)  # Wait before retry
    return index, None, "Max retries exceeded"


def translate_batch(df: DataFrame, llm: BaseChatModel, batch_size: int = 10, 
                   save_checkpoint: str = None, skip_existing: bool = True,
                   translate_answers: bool = True) -> tuple:
    """
    Batch translate questions and answers with checkpoint support for large datasets.
    
    Args:
        df: DataFrame with 'question' and 'answer' column
        llm: Language model
        batch_size: Number of items per batch
        save_checkpoint: Path to save checkpoint file
        skip_existing: Skip already translated items if checkpoint exists
        translate_answers: Whether to translate answers too
    
    Returns:
        Tuple of (translated_questions, translated_answers, error_log)
    """
    questions = df['question'].to_list()
    answers = df.get('answer', pd.Series([None] * len(df))).to_list()
    total = len(questions)
    translated_questions = [None] * total
    translated_answers = [None] * total
    error_log = []
    
    # Load checkpoint if exists and matches size
    checkpoint_path = Path(save_checkpoint) if save_checkpoint else None
    start_idx = 0
    
    if checkpoint_path and skip_existing and checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                checkpoint_total = checkpoint_data.get('total', 0)
                
                # Only load checkpoint if size matches
                if checkpoint_total == total:
                    translated_questions = checkpoint_data.get('translations_q', [None] * total)
                    translated_answers = checkpoint_data.get('translations_a', [None] * total)
                    error_log = checkpoint_data.get('errors', [])
                    start_idx = checkpoint_data.get('last_completed_idx', -1) + 1
                    print(f"✓ Loaded checkpoint from {checkpoint_path}")
                    print(f"  Resuming from item {start_idx}/{total}")
                else:
                    print(f"⚠ Checkpoint size mismatch (expected {total}, got {checkpoint_total}). Starting fresh.")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}. Starting from beginning.")
    
    # Process remaining items in batches
    print(f"\nTranslating {total - start_idx} items (batch size: {batch_size})...")
    if translate_answers:
        print("  Including answers in translation")
    
    try:
        for batch_start in range(start_idx, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_indices = range(batch_start, batch_end)
            
            # Parallel translation within batch
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}
                
                # Submit question translations
                for idx in batch_indices:
                    futures[executor.submit(
                        translate_single,
                        questions[idx],
                        idx,
                        llm,
                        "question"
                    )] = ("question", idx)
                
                # Submit answer translations if enabled
                if translate_answers:
                    for idx in batch_indices:
                        ans = answers[idx]
                        # Check if answer exists and is not None/NaN
                        if ans is not None and (isinstance(ans, str) or pd.notna(ans)):
                            futures[executor.submit(
                                translate_single,
                                str(ans),
                                idx,
                                llm,
                                "answer"
                            )] = ("answer", idx)
                
                pbar = tqdm(
                    as_completed(futures), 
                    total=len(futures),
                    desc=f"Batch {batch_start//batch_size + 1}"
                )
                
                for future in pbar:
                    text_type, idx = futures[future]
                    idx_result, translated_text, error = future.result()
                    
                    if translated_text is not None:
                        if text_type == "question":
                            translated_questions[idx] = translated_text
                        else:
                            translated_answers[idx] = translated_text
                    else:
                        error_msg = {
                            'index': idx,
                            'type': text_type,
                            'text': questions[idx] if text_type == "question" else answers[idx],
                            'error': error
                        }
                        error_log.append(error_msg)
            
            # Save checkpoint after each batch
            if checkpoint_path:
                checkpoint_data = {
                    'last_completed_idx': batch_end - 1,
                    'total': total,
                    'timestamp': datetime.now().isoformat(),
                    'translations_q': translated_questions,
                    'translations_a': translated_answers,
                    'errors': error_log
                }
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Translation completed!")
        print(f"  Questions: {sum(1 for t in translated_questions if t is not None)}/{total}")
        if translate_answers:
            print(f"  Answers: {sum(1 for t in translated_answers if t is not None)}/{total}")
        if error_log:
            print(f"  Errors: {len(error_log)}")
        
    except KeyboardInterrupt:
        print("\nTranslation interrupted. Checkpoint saved.")
        if checkpoint_path:
            checkpoint_data = {
                'last_completed_idx': batch_start - 1,
                'total': total,
                'timestamp': datetime.now().isoformat(),
                'translations_q': translated_questions,
                'translations_a': translated_answers,
                'errors': error_log
            }
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        raise
    
    return translated_questions, translated_answers, error_log


def save_results(df: DataFrame, translated_questions: list, translated_answers: list = None, 
                error_log: list = None, save_path: str = None):
    """Save translation results to CSV with error tracking."""
    df = df.copy()
    
    # Ensure lengths match
    if len(translated_questions) != len(df):
        print(f"⚠ Length mismatch: translated_questions={len(translated_questions)}, df={len(df)}")
        translated_questions = translated_questions[:len(df)]
    
    if translated_answers and len(translated_answers) != len(df):
        print(f"⚠ Length mismatch: translated_answers={len(translated_answers)}, df={len(df)}")
        translated_answers = translated_answers[:len(df)]
    
    df['question_vn'] = translated_questions
    df['translation_status_q'] = ['success' if t else 'failed' for t in translated_questions]
    
    if translated_answers:
        df['answer_vn'] = translated_answers
        df['translation_status_a'] = ['success' if t else 'failed' for t in translated_answers]
    
    if save_path:
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✓ Results saved to {save_path}")
    
    # Save error log if any
    if error_log:
        error_path = Path(save_path).parent / f"{Path(save_path).stem}_errors.json" if save_path else "translation_errors.json"
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_log, f, ensure_ascii=False, indent=2)
        print(f"✓ Error log saved to {error_path}")

def main():
    """Main execution function with optimization for large datasets."""
    # Configuration
    data_path = "dataset/UniGeo_data/dataframe/calculation_test.csv"
    output_path = "dataset/UniGeo_data/dataframe/translated_questions.csv"
    checkpoint_path = "dataset/UniGeo_data/dataframe/translation_checkpoint.json"
    batch_size = 10
    num_questions = None  # Set to None to translate all, or specify number
    translate_answers = True  # Enable answer translation
    
    try:
        # Load data
        df = get_data(data_path)
        df = df.iloc[:20]
        
        # Limit to num_questions if specified
        if num_questions:
            df = df.head(num_questions)
            print(f"Processing first {num_questions} questions")
        
        # Initialize model
        print("Initializing chat model...")
        llm = get_chat_model("gpt-4o-mini")
        
        print(f"\nTotal items to translate: {len(df)}")
        
        # Batch translate with checkpoint support
        translated_questions, translated_answers, error_log = translate_batch(
            df, 
            llm, 
            batch_size=batch_size,
            save_checkpoint=checkpoint_path,
            skip_existing=True,
            translate_answers=translate_answers
        )
        
        # Save results
        save_results(df, translated_questions, translated_answers, error_log, output_path)
        
        # Display sample results
        print("\n" + "="*80)
        print("SAMPLE RESULTS")
        print("="*80)
        for i in range(min(3, len(df))):
            if translated_questions[i]:
                print(f"\n{i+1}. Original Q: {df.iloc[i]['question'][:80]}...")
                print(f"   Vietnamese Q: {translated_questions[i][:80]}...")
                if translate_answers and translated_answers[i]:
                    print(f"   Original A: {str(df.iloc[i].get('answer', ''))[:80]}...")
                    print(f"   Vietnamese A: {translated_answers[i][:80]}...")
        
        print("\n✓ Translation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Translation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()