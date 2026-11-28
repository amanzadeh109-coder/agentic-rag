"""Chat endpoints.

POST /chat                       ask a question; streams events (SSE) or returns JSON
GET  /conversations              the user's conversations
GET  /conversations/{id}         one conversation with its messages
DELETE /conversations/{id}
GET  /usage                      the user's spend and remaining budget
GET  /chunks/{id}                the passage behind a citation

The user is identified by the X-User-Id header (no real authentication).
"""
import json

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sse_starlette.sse import EventSourceResponse

from app.agent import run_agent, run_pipeline
from app.config import settings
from app.db import Chunk, Conversation, Document, Message, SessionLocal

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    mode: str = Field("agent", pattern="^(agent|pipeline)$")
    stream: bool = False


def spent_usd(db, user_id: str) -> float:
    return db.scalar(select(func.coalesce(func.sum(Message.cost_usd), 0.0))
                     .join(Conversation, Conversation.id == Message.conversation_id)
                     .where(Conversation.user_id == user_id)) or 0.0


def history_for(db, conv_id: str) -> list[dict]:
    """Most recent turns that fit in history_max_tokens (about 4 characters per token)."""
    msgs = db.scalars(select(Message).where(Message.conversation_id == conv_id)
                      .order_by(Message.created_at.desc())).all()
    budget, kept = settings.history_max_tokens * 4, []
    for m in msgs:
        if len(m.content) > budget:
            break
        budget -= len(m.content)
        kept.append({"role": m.role, "content": m.content})
    return list(reversed(kept))


def start_turn(req: ChatRequest, user_id: str) -> tuple[str, list[dict]]:
    with SessionLocal() as db:
        if spent_usd(db, user_id) >= settings.user_budget_usd:
            raise HTTPException(429, f"Budget of ${settings.user_budget_usd:.2f} for user "
                                     f"'{user_id}' is used up.")
        if req.conversation_id:
            conv = db.get(Conversation, req.conversation_id)
            if conv is None or conv.user_id != user_id:
                raise HTTPException(404, "Conversation not found.")
        else:
            conv = Conversation(user_id=user_id, title=req.question[:60])
            db.add(conv)
            db.commit()
        history = history_for(db, conv.id)
        db.add(Message(conversation_id=conv.id, role="user", content=req.question))
        db.commit()
        return conv.id, history


def save_answer(conv_id: str, done: dict) -> None:
    with SessionLocal() as db:
        db.add(Message(conversation_id=conv_id, role="assistant", content=done["answer"],
                       sources=done["sources"], steps=done["steps"],
                       prompt_tokens=done["usage"]["prompt_tokens"],
                       completion_tokens=done["usage"]["completion_tokens"],
                       cost_usd=done["usage"]["usd"], model=done["model"]))
        db.commit()


def events(req: ChatRequest, conv_id: str, history: list[dict]):
    runner = run_agent if req.mode == "agent" else run_pipeline
    try:
        for ev in runner(req.question, history):
            if ev["type"] == "done":
                ev["conversation_id"] = conv_id
                ev["citations"] = [f"{s['file']} p.{s['page']}" for s in ev["sources"]]
                save_answer(conv_id, ev)
            yield ev
    except Exception as e:  # LLM and fallback both down, etc.
        yield {"type": "error", "message": f"The answer failed: {e}"[:500]}


@router.post("/chat")
def chat(req: ChatRequest, x_user_id: str = Header("anonymous")):
    conv_id, history = start_turn(req, x_user_id)
    if req.stream:
        return EventSourceResponse(
            {"event": ev["type"], "data": json.dumps(ev)} for ev in events(req, conv_id, history))
    final = None
    for ev in events(req, conv_id, history):
        if ev["type"] in ("done", "error"):
            final = ev
    if final is None or final["type"] == "error":
        raise HTTPException(502, final["message"] if final else "No answer was produced.")
    return final


@router.get("/conversations")
def list_conversations(x_user_id: str = Header("anonymous")):
    with SessionLocal() as db:
        convs = db.scalars(select(Conversation).where(Conversation.user_id == x_user_id)
                           .order_by(Conversation.created_at.desc())).all()
        return [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()} for c in convs]


@router.get("/conversations/{conv_id}")
def get_conversation(conv_id: str, x_user_id: str = Header("anonymous")):
    with SessionLocal() as db:
        conv = db.get(Conversation, conv_id)
        if conv is None or conv.user_id != x_user_id:
            raise HTTPException(404, "Conversation not found.")
        msgs = db.scalars(select(Message).where(Message.conversation_id == conv_id)
                          .order_by(Message.created_at)).all()
        return {"id": conv.id, "title": conv.title, "messages": [
            {"role": m.role, "content": m.content, "sources": m.sources, "steps": m.steps,
             "tokens": m.prompt_tokens + m.completion_tokens, "cost_usd": m.cost_usd,
             "model": m.model} for m in msgs]}


@router.delete("/conversations/{conv_id}", status_code=204)
def delete_conversation(conv_id: str, x_user_id: str = Header("anonymous")):
    with SessionLocal() as db:
        conv = db.get(Conversation, conv_id)
        if conv is None or conv.user_id != x_user_id:
            raise HTTPException(404, "Conversation not found.")
        db.delete(conv)
        db.commit()


@router.get("/usage")
def usage(x_user_id: str = Header("anonymous")):
    with SessionLocal() as db:
        spent = spent_usd(db, x_user_id)
    return {"user_id": x_user_id, "spent_usd": round(spent, 6),
            "budget_usd": settings.user_budget_usd,
            "remaining_usd": round(max(0.0, settings.user_budget_usd - spent), 6)}


@router.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    with SessionLocal() as db:
        row = db.execute(select(Chunk, Document.filename)
                         .join(Document, Document.id == Chunk.document_id)
                         .where(Chunk.id == chunk_id)).first()
        if row is None:
            raise HTTPException(404, "Passage not found.")
        c, fname = row
        return {"id": c.id, "file": fname, "page": c.page, "section": c.section, "text": c.text}