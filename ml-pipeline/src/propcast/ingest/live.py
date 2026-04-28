from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .historical import _upsert, normalize
from .nba_client import fetch_season_logs
from .schema import GameLog, init_db

logger = logging.getLogger(__name__)

CURRENT_SEASON = "2024-25"


def _last_ingested_date(engine: Engine) -> Optional[date]:
    """Return the most recent game_date stored for the current season, or None."""
    with Session(engine) as session:
        return session.execute(
            select(func.max(GameLog.game_date)).where(
                GameLog.season == CURRENT_SEASON
            )
        ).scalar()


def update(engine: Engine) -> None:
    """Incrementally ingest the current season.

    On first run: fetches the entire 2024-25 season.
    On subsequent runs: fetches only games played since the last stored date,
    so the typical nightly call hits a very small payload.
    """
    init_db(engine)

    last_date = _last_ingested_date(engine)

    if last_date is not None:
        # Advance by one day so we don't re-fetch games already in the DB.
        # nba_api date_from_nullable expects 'MM/DD/YYYY'.
        fetch_from = (last_date + timedelta(days=1)).strftime("%m/%d/%Y")
        logger.info("[%s] incremental update from %s", CURRENT_SEASON, fetch_from)
    else:
        fetch_from = None
        logger.info("[%s] no existing data — fetching full season", CURRENT_SEASON)

    raw = fetch_season_logs(CURRENT_SEASON, date_from=fetch_from)

    if raw.empty:
        logger.info("[%s] no new games since %s", CURRENT_SEASON, last_date)
        return

    df = normalize(raw, CURRENT_SEASON)
    inserted = _upsert(df, engine)
    logger.info("[%s] %d fetched  %d inserted", CURRENT_SEASON, len(df), inserted)
