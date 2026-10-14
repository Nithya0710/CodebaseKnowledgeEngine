from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    # Ollama's default context window can be as small as 2048 tokens and it
    # truncates longer prompts silently from the front, which drops the
    # instructions. Generation prompts reach ~3k tokens and RAGAS judge
    # prompts are larger, so the window is set explicitly.
    ollama_num_ctx: int = 8192
    chroma_persist_dir: str = "./data/chroma"
    log_level: str = "INFO"
    openai_api_key: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"

    ragas_judge_backend: Literal["ollama", "groq"] = "ollama"
    ragas_judge_model: str = "qwen2.5:7b"
    groq_api_key: str = ""
    groq_judge_model: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
