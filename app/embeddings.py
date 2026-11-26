from functools import lru_cache

from fastembed import TextEmbedding

from app.config import settings


@lru_cache
def get_model() -> TextEmbedding:
    return TextEmbedding(model_name=settings.embed_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in get_model().passage_embed(texts, batch_size=32)]


def embed_query(query: str) -> list[float]:
    return next(iter(get_model().query_embed(query))).tolist()