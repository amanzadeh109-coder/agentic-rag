from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    # --- main LLM: any OpenAI-compatible API (Claude, OpenAI, Groq...) ---
    llm_base_url: str = "https://api.anthropic.com/v1/"
    llm_api_key: str = ""
    llm_model: str = "claude-haiku-5-5"
    llm_price_in: float = 0.10     # USD per 1M input tokens
    llm_price_out: float = 0.50    # USD per 1M output tokens
    llm_fake: bool = False         # tests: a scripted LLM, no network

    # --- fallback LLM when the main one fails (Ollama speaks the same API) ---
    fallback_base_url: str = "http://ollama:11434/v1"
    fallback_model: str = "qwen2.5:3b"

    # --- retrieval ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    candidates_k: int = 30      # from each of dense and lexical
    rerank_k: int = 20          # fused shortlist the reranker reads
    final_k: int = 5            # results returned
    rerank_threshold: float = 0.3

    # --- agent limits: it must always stop ---
    agent_max_steps: int = 6          # LLM calls per question
    agent_max_tokens: int = 60000     # prompt + completion tokens per question
    history_max_tokens: int = 3000    # past conversation kept in the prompt

    # --- users ---
    user_budget_usd: float = 0.50     # total spend allowed per user id

    # --- ingestion ---
    upload_dir: str = "/data/uploads"
    max_upload_mb: int = 50
    chunk_words: int = 300
    chunk_overlap_words: int = 40


settings = Settings()