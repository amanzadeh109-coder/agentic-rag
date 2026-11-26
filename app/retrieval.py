import math
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder
from sqlalchemy import Text, cast, func, select

from app.config import settings
from app.db import Chunk, ChunkEmbedding, Document, SessionLocal
from app.embeddings import embed_query


def dense_search(db, query: str, k: int, files: list[str] | None) -> list[str]:
    """Chunk ids ordered by meaning: cosine distance between vectors."""
    qvec = embed_query(query)
    q = (select(Chunk.id)
         .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
         .join(Document, Document.id == Chunk.document_id)
         .order_by(ChunkEmbedding.embedding.cosine_distance(qvec))
         .limit(k))
    if files:
        q = q.where(Document.filename.in_(files))
    return list(db.scalars(q))



def lexical_search(db, query: str, k: int, files: list[str] | None) -> list[str]:
    """Chunk ids ordered by shared words (Postgres full-text, BM25-like)."""
    # plainto_tsquery joins words with AND (&): a long question then matches nothing.
    # Replacing & with | (OR) lets ts_rank reward chunks that share MORE of the words.
    tsq = func.to_tsquery("english", func.replace(
        cast(func.plainto_tsquery("english", query), Text), "&", "|"))
    q = (select(Chunk.id)
         .join(Document, Document.id == Chunk.document_id)
         .where(Chunk.tsv.op("@@")(tsq))
         .order_by(func.ts_rank_cd(Chunk.tsv, tsq).desc())
         .limit(k))
    if files:
        q = q.where(Document.filename.in_(files))
    return list(db.scalars(q))


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: a chunk ranked high in either list ends up high."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True) 



@lru_cache
def get_reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=settings.rerank_model)


def rerank(query: str, texts: list[str]) -> list[float]:
    """Relevance 0..1 for each text: the cross-encoder reads query and text together."""
    logits = list(get_reranker().rerank(query, texts))
    return [1 / (1 + math.exp(-x)) for x in logits]



def search(query: str, k: int | None = None, files: list[str] | None = None) -> list[dict]:
    k = k or settings.final_k
    with SessionLocal() as db:
        dense = dense_search(db, query, settings.candidates_k, files)
        lexical = lexical_search(db, query, settings.candidates_k, files)
        shortlist = rrf([dense, lexical])[: settings.rerank_k]
        if not shortlist:
            return []
        rows = db.execute(
            select(Chunk, Document.filename)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(shortlist))
        ).all()

    by_id = {c.id: (c, fname) for c, fname in rows}
    ordered = [by_id[i] for i in shortlist if i in by_id]
    scores = rerank(query, [c.text for c, _ in ordered])

    results = [
        {"chunk_id": c.id, "file": fname, "page": c.page, "section": c.section,
         "text": c.text, "score": round(s, 3)}
        for (c, fname), s in zip(ordered, scores)
        if s >= settings.rerank_threshold  # below this, the chunk is not evidence
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]