import os
from dotenv import load_dotenv
_ = load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TRANSLATE_PROMPT = ""
