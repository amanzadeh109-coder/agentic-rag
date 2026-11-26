from fastapi import APIRouter, Query

from app.retrieval import search

router = APIRouter(tags=["search"])


@router.get("/search")
def search_endpoint(q: str = Query(min_length=2), k: int = 5, file: list[str] | None = Query(None)):
    """Debug view of retrieval: what the agent's search tool will see."""
    return search(q, k=k, files=file)