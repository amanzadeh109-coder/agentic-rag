from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    llm_api_key: str = ""
    embed_dim: int = 384
    upload_dir: str = "/data/uploads"
    max_upload_mb: int = 50
    chunk_words: int = 300
    chunk_overlap_words: int = 40


settings = Settings()