# ============================================================
# src/config.py
#
# WHAT THIS FILE DOES:
# It reads all the settings out of your .env file (API keys,
# chunk size, model names, etc.) ONE TIME, and makes them
# available anywhere in the project as `settings.SOMETHING`.
#
# Why do this instead of just typing values directly in each file?
# - You never hardcode secrets in your code.
# - You can change a setting in ONE place (.env) instead of
#   hunting through every file that uses it.
# ============================================================

from pydantic_settings import BaseSettings
# BaseSettings is a special Pydantic class that automatically reads
# environment variables (and .env file values) and validates their types.


class Settings(BaseSettings):
    # ---- Each line below is: NAME: TYPE = DEFAULT_VALUE ----
    # Pydantic will look for an environment variable with this exact
    # name (e.g. OPENAI_API_KEY) and use it. If it's not found, it
    # uses the default value shown here.

    OPENAI_API_KEY: str = ""
    VECTOR_STORE_PATH: str = "./data/lancedb_store"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    SIMILARITY_THRESHOLD: float = 0.3

    class Config:
        # This tells Pydantic: "also read values from a file named .env"
        env_file = ".env"
        env_file_encoding = "utf-8"


# We create ONE instance of Settings here, when this file is first imported.
# Every other file in the project will import THIS variable:
#     from src.config import settings
#     print(settings.OPENAI_API_KEY)
settings = Settings()
