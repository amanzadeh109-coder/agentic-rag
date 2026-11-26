import logging
import os

from sqlalchemy import delete, update

from app.chunking import chunk_pages
from app.config import settings
from app.db import Chunk, ChunkEmbedding, Document, SessionLocal
from app.embeddings import embed_passages
from app.parsing import parse_pdf

log = logging.getLogger(__name__)


def pdf_path(filename: str) -> str:
    return os.path.join(settings.upload_dir, filename)


def ingest_document(doc_id: str) -> None:
    with SessionLocal() as db:
        doc = db.get(Document, doc_id)
        if doc is None:          # deleted before the job started
            return
        doc.status, doc.error = "processing", None
        db.commit()
        try:
            pages = parse_pdf(pdf_path(doc.filename))
            if not any(p.markdown for p in pages):
                raise ValueError("no text found: the PDF may be scanned images (needs OCR)")
            parts = chunk_pages(pages, doc.filename,
                                max_words=settings.chunk_words,
                                overlap=settings.chunk_overlap_words)
            db.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            chunks = [Chunk(document_id=doc_id, chunk_index=c.chunk_index, page=c.page,
                            section=c.section, text=c.text) for c in parts]
            db.add_all(chunks)
            db.flush()  # gives every chunk its id before we link embeddings to it

            vectors = embed_passages([c.text for c in chunks])
            db.add_all(ChunkEmbedding(chunk_id=c.id, embedding=v)
                       for c, v in zip(chunks, vectors))

            doc.num_pages = len(pages)
            doc.status = "ready"
            db.commit()  # chunks and embeddings are saved together, or not at all
            log.info("ingested %s: %d pages, %d chunks", doc.filename, len(pages), len(chunks))
        except Exception as e:
            db.rollback()
            log.exception("ingestion failed for %s", doc_id)
            db.execute(update(Document).where(Document.id == doc_id)
                       .values(status="failed", error=str(e)[:500]))
            db.commit()


def fail_interrupted_jobs() -> None:
    """On startup: a job that was running when the API stopped is lost."""
    with SessionLocal() as db:
        db.execute(update(Document)
                   .where(Document.status.in_(["pending", "processing"]))
                   .values(status="failed", error="interrupted by a restart; use /reingest"))
        db.commit()