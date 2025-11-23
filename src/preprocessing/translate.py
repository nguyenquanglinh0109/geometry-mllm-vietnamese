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
import ast
from pydantic import BaseModel
from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompt import TRANSLATE_PROMPT_BATCH

os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY

class TranslatedSentence(BaseModel):
    translated_sentences: List[str]

def get_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")
    return df


def get_chat_model(model_name: str = "gemini-2.5-flash"):
    """Initialize and return chat model. gpt-4o-mini'"""
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


def translate_batch(
    df: DataFrame, 
    collumn_name: str, 
    llm: BaseChatModel, 
    batch_size: int = 10, 
) -> tuple:
    """
    Batch translate questions and answers with checkpoint support for large datasets.
    
    Args:
        df: DataFrame with 'question' and 'answer' column
        collumn_name: Name of column to translate (questions or answers)
        llm: Language model
        batch_size: Number of items per batch
    
    Returns:
        List of translated texts
    """
    sentences = df[collumn_name].to_list()
    total_sentences = len(sentences)
    translated_sentences = [None] * total_sentences
    llm = llm.with_structured_output(TranslatedSentence)

    print(f"Total items to translate: {total_sentences}")
    try:
        for i in tqdm(range(0, total_sentences, batch_size)):
            batch = sentences[i:i + batch_size]
            user_content = ""
            for j, q in enumerate(batch, start=1):
                user_content += f"{j}. {q}\n"

            response = llm.invoke([
                {"role": "system", "content": TRANSLATE_PROMPT_BATCH},
                {"role": "user", "content": user_content}
            ])
   
            # content = ast.literal_eval(response.content)
            content = response.translated_sentences
            translated_sentences[i:i + batch_size] = content
        
        new_df = df.copy()
        new_df[collumn_name + "_translated"] = translated_sentences
            
    except KeyboardInterrupt:
        print("\nTranslation interrupted. Checkpoint saved.")
        raise
    
    return new_df


def translate_and_save(
    data_path: str, 
    llm: BaseChatModel,
    output_path: str, 
    batch_size: int = 10, 
    limit: int = None, 
    translate_answers: bool = True,
):
    print(f"Translating {data_path}...")
    df = get_data(data_path)
    if limit:
        df = df.iloc[:limit]
    
    new_df = translate_batch(
        df=df, 
        collumn_name='question', 
        llm=llm, 
        batch_size=batch_size, 
    )
    
    if translate_answers:
        new_df = translate_batch(
            df=new_df, 
            collumn_name='answer', 
            llm=llm, 
            batch_size=batch_size,
        )    
    
    new_df.to_csv(output_path, index=False)
    print(f"Translation saved to {output_path}")
    print("=" * 100, "\n")

 
def main():
    """Main execution function with optimization for large datasets."""
    # Configuration
    limit = None
    batch_size = min(64, limit) if limit else 64
    translate_answers = True 
    llm = get_chat_model("gemini-2.0-flash")
    
    # Dataset paths
    data_dir = "dataset/UniGeo_data/dataframe"
    # 'calculation_test.csv', 
    all_files = ['calculation_train.csv', 'calculation_val.csv', 'proving_test.csv', 'proving_train.csv', 'proving_val.csv',]
    
    for file in all_files:
        file_name = file.split('.')[0]
        data_path = os.path.join(data_dir, file)
        output_path = os.path.join(data_dir, 'translated', file_name + "_translated.csv")
        
        translate_and_save(
            data_path=data_path,
            llm=llm,
            output_path=output_path,
            batch_size=batch_size,
            limit=limit,
            translate_answers=translate_answers
        )


if __name__ == "__main__":
    main()