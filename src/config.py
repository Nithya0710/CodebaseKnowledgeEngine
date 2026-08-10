from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    chroma_persist_dir: str = "./data/chroma"
    log_level: str = "INFO"
    openai_api_key: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()