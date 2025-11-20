from dotenv import load_dotenv
import os
_ = load_dotenv(override=True)


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")