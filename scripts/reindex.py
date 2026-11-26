"""Rebuild the vector index from the chunks table. No PDF is parsed again."""
from sqlalchemy import delete, func, select

from app.db import Chunk, ChunkEmbedding, SessionLocal
from app.embeddings import embed_passages

BATCH = 256

with SessionLocal() as db:
    db.execute(delete(ChunkEmbedding))
    total = db.scalar(select(func.count()).select_from(Chunk))
    done = 0
    while done < total:
        rows = db.execute(select(Chunk.id, Chunk.text).order_by(Chunk.id)
                          .offset(done).limit(BATCH)).all()
        vectors = embed_passages([r.text for r in rows])
        db.add_all(ChunkEmbedding(chunk_id=r.id, embedding=v) for r, v in zip(rows, vectors))
        done += len(rows)
        print(f"{done}/{total}")
    db.commit()
print("index rebuilt")