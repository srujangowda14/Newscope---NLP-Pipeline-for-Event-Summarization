from pydantic settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Model configurations
    SUMMARIZER_MODEL: Literal["bart","t5"] = "bart"
    BART_MODEL_NAME: str = "facebook/bart-large-cnn"
    T5_MODEL_NAME: str = "t5-base"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    #Summarization parameters
    MAX_INPUT_LENGTH: int = 1024
    MAX_SUMMARY_LENGTH: int = 150
    MIN_SUMMARY_LENGTH: int = 50

    #Depduplication parameters
    SIMILARITY_THRESHOLD float = 0.85
    EMBEDDING_BATCH_SIZE: int = 32

    #API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    #Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str= "redis://localhost:6379/0"

    #Processing settings
    BATCH_SIZE: int = 16
    MAX_ARTICLES_PER_DAY: int = 10000

    #logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    #Device
    DEVICE: str = "cuda"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()


