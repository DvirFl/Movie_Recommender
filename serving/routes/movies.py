"""GET /movies/search — lightweight title search for UI autocomplete (e.g. cold-start favorites)."""
from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import select

from db.connection import get_session
from db.models import ItemFeature, RawMovie

router = APIRouter(prefix="/movies", tags=["movies"])


@router.get("/search")
def search_movies(q: str = Query(min_length=1), limit: int = Query(default=10, ge=1, le=50)) -> list[dict]:
    with get_session() as session:
        rows = session.execute(
            select(RawMovie.movie_id, RawMovie.title, ItemFeature.release_year)
            .join(ItemFeature, ItemFeature.movie_id == RawMovie.movie_id, isouter=True)
            .where(RawMovie.title.ilike(f"%{q}%"))
            .limit(limit)
        ).all()
        return [{"movie_id": r.movie_id, "title": r.title, "year": r.release_year} for r in rows]