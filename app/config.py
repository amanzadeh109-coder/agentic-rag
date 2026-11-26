from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    llm_api_key: str = ""
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    candidates_k: int = 30      # from each of dense and lexical
    rerank_k: int = 20          # fused shortlist the reranker reads
    final_k: int = 5            # results returned
    rerank_threshold: float = 0.3
    embed_dim: int = 384
    upload_dir: str = "/data/uploads"
    max_upload_mb: int = 50
    chunk_words: int = 300
    chunk_overlap_words: int = 40


settings = Settings()