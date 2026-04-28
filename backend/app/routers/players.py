from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from propcast.ingest.schema import GameLog

from ..db import get_db

router = APIRouter(prefix="/players", tags=["players"])


class PlayerResult(BaseModel):
    player_id: int
    player_name: str


@router.get("/search", response_model=list[PlayerResult])
def search_players(
    q: str = Query(..., min_length=2, description="Player name fragment"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(GameLog.player_id, GameLog.player_name)
        .where(GameLog.player_name.ilike(f"%{q}%"))
        .distinct()
        .order_by(GameLog.player_name)
        .limit(limit)
    ).all()
    return [{"player_id": r.player_id, "player_name": r.player_name} for r in rows]
