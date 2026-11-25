import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db import engine, init_db
from app.ingest import fail_interrupted_jobs
from app.routers import documents

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    fail_interrupted_jobs()
    yield


app = FastAPI(title="Agentic RAG", lifespan=lifespan)
app.include_router(documents.router)


@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "db": "ok"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "down", "db": str(e)[:200]})