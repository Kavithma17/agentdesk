from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.rag.chunking import DEFAULT_MAX_TOKENS, DEFAULT_MIN_TOKENS, DEFAULT_OVERLAP_TOKENS


class Settings(BaseSettings):
    """App settings, loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["gemini"] = "gemini"
    gemini_api_key: SecretStr
    llm_model: str = "gemini-3.5-flash"
    log_level: str = "INFO"

    docs_dir: Path = Path("data/docs")
    chunk_max_tokens: int = Field(DEFAULT_MAX_TOKENS, gt=0)
    chunk_overlap_tokens: int = Field(DEFAULT_OVERLAP_TOKENS, ge=0)
    chunk_min_tokens: int = Field(DEFAULT_MIN_TOKENS, ge=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
