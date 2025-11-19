import os
import sys
import pandas as pd
from langchain.chat_models import init_chat_model
from src.constant import OPENAI_API_KEY

def get_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")
    return df

def get_chat_model():
    """Initialize and return chat model."""
    model = init_chat_model(
        "gpt-4o-mini",
        temperature=0.1,
        timeout=30,
        max_tokens=100,
        openai_api_key=OPENAI_API_KEY    
    )
    return model

def translate_questions(questions: list, llm, save_path: str = None) -> list:
    """Translate questions to Vietnamese using LLM."""
    TRANSLATE_PROMPT = """Translate the following geometry question into natural Vietnamese. 
Keep all mathematical symbols, angle notations (∠), parallel symbols (∥), and element names exactly as they are.
Only translate the English words and phrases into natural, fluent Vietnamese.
Translate directly without any explanation or preamble."""
    translations = []
    
    for idx, question in enumerate(questions, 1):
        try:
            # Correct message format: user provides the question
            response = llm.invoke([
                {"role": "system", "content": TRANSLATE_PROMPT},
                {"role": "user", "content": question}
            ])
            
            # Extract translated text
            translated_text = response.content if hasattr(response, 'content') else str(response)
            translations.append(translated_text)
            
            print(f"[{idx}/{len(questions)}] Translated: {question[:50]}...")
            
        except Exception as e:
            print(f"Error translating question {idx}: {e}")
            translations.append(None)
    
    # Save translations if path provided
    if save_path:
        result_df = pd.DataFrame({
            'original': questions,
            'translation': translations
        })
        result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\nTranslations saved to {save_path}")
    
    return translations

def main():
    """Main execution function."""
    # Configuration
    data_path = "dataset/UniGeo_data/dataframe/proving_test.csv"
    output_path = "dataset/UniGeo_data/dataframe/translated_questions.csv"
    num_questions = 10
    
    try:
        # Load data
        df = get_data(data_path)
        
        # Initialize model
        print("Initializing chat model...")
        llm = get_chat_model()
        
        # Get questions to translate
        questions = df["question"].tolist()[:num_questions]
        print(f"\nTranslating {len(questions)} questions...\n")
        
        # Translate questions
        translations = translate_questions(questions, llm, save_path=output_path)
        
        # Display results
        print("\n" + "="*80)
        print("TRANSLATION RESULTS")
        print("="*80)
        for i, (orig, trans) in enumerate(zip(questions, translations), 1):
            print(f"\n{i}. Original: {orig}")
            print(f"   Vietnamese: {trans}")
        
        print("\nTranslation completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()