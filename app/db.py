import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON, Computed, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    create_engine, text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def new_id() -> str:
    return str(uuid.uuid4())


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String, unique=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    num_pages: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
    tsv = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )

    __table_args__ = (Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),)


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    embedding = mapped_column(Vector(settings.embed_dim))

    __table_args__ = (
        Index(
            "ix_embeddings_hnsw", "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String, default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[list | None] = mapped_column(JSON)
    steps: Mapped[list | None] = mapped_column(JSON)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    model: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)