from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    llm_api_key: str = ""
    embed_dim: int = 384


settings = Settings()