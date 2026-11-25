import os
import re

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from sqlalchemy import select

from app.config import settings
from app.db import Chunk, Document, SessionLocal
from app.ingest import ingest_document, pdf_path

router = APIRouter(prefix="/documents", tags=["documents"])


def to_dict(d: Document) -> dict:
    return {"id": d.id, "filename": d.filename, "status": d.status, "error": d.error,
            "num_pages": d.num_pages, "created_at": d.created_at.isoformat()}


def safe_name(name: str) -> str:
    name = os.path.basename(name or "")
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def get_or_404(db, doc_id: str) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found.")
    return doc


@router.post("", status_code=202)
async def upload(file: UploadFile, background: BackgroundTasks):
    name = safe_name(file.filename)
    if not name.lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are accepted.")
    data = await file.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "The file does not look like a PDF.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")

    with SessionLocal() as db:
        if db.scalar(select(Document).where(Document.filename == name)):
            raise HTTPException(409, f"'{name}' is already uploaded. Delete it first.")
        os.makedirs(settings.upload_dir, exist_ok=True)
        with open(pdf_path(name), "wb") as f:
            f.write(data)
        doc = Document(filename=name)
        db.add(doc)
        db.commit()
        result = to_dict(doc)

    background.add_task(ingest_document, result["id"])
    return result


@router.get("")
def list_documents():
    with SessionLocal() as db:
        return [to_dict(d) for d in db.scalars(select(Document).order_by(Document.created_at))]


@router.get("/{doc_id}")
def get_document(doc_id: str):
    with SessionLocal() as db:
        return to_dict(get_or_404(db, doc_id))


@router.get("/{doc_id}/chunks")
def get_chunks(doc_id: str, page: int | None = None):
    with SessionLocal() as db:
        get_or_404(db, doc_id)
        q = select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.chunk_index)
        if page is not None:
            q = q.where(Chunk.page == page)
        return [{"id": c.id, "page": c.page, "section": c.section, "text": c.text}
                for c in db.scalars(q)]


@router.post("/{doc_id}/reingest", status_code=202)
def reingest(doc_id: str, background: BackgroundTasks):
    with SessionLocal() as db:
        doc = get_or_404(db, doc_id)
        if doc.status == "processing":
            raise HTTPException(409, "Already processing.")
        doc.status, doc.error = "pending", None
        db.commit()
        result = to_dict(doc)
    background.add_task(ingest_document, doc_id)
    return result


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    with SessionLocal() as db:
        doc = get_or_404(db, doc_id)
        path = pdf_path(doc.filename)
        db.delete(doc)
        db.commit()
    if os.path.exists(path):
        os.remove(path)