from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.engine import Engine

from .nba_client import fetch_season_logs
from .schema import GameLog, init_db

logger = logging.getLogger(__name__)

# 2015-16 is the earliest season with reliable DK closing-line data for CLV.
BACKFILL_SEASONS: list[str] = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24",
]

# Parquet cache sits under ml-pipeline/data/raw/game_logs/{season}.parquet
RAW_CACHE_DIR = Path(__file__).parents[3] / "data" / "raw" / "game_logs"

# LeagueGameLog raw column name → canonical snake_case name
_COLUMN_MAP: dict[str, str] = {
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ID": "team_id",
    "TEAM_ABBREVIATION": "team_abbreviation",
    "GAME_ID": "game_id",
    "MATCHUP": "matchup",
    "WL": "wl",
    "MIN": "min",
    "FGM": "fgm",
    "FGA": "fga",
    "FG_PCT": "fg_pct",
    "FG3M": "fg3m",
    "FG3A": "fg3a",
    "FG3_PCT": "fg3_pct",
    "FTM": "ftm",
    "FTA": "fta",
    "FT_PCT": "ft_pct",
    "OREB": "oreb",
    "DREB": "dreb",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "tov",
    "PF": "pf",
    "PTS": "pts",
    "PLUS_MINUS": "plus_minus",
}

_INSERT_CHUNK = 500   # stay well under SQLite's 32 766-variable limit


def normalize(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Normalize a raw LeagueGameLog DataFrame to the canonical GameLog schema.

    This is the single source of truth for column names and types that the
    features/ layer depends on.  Both historical and live ingestion paths run
    through here so their outputs are byte-for-byte identical in structure.
    """
    # Select only the columns we care about (drops SEASON_ID, TEAM_NAME, etc.)
    out = df[["GAME_DATE"] + list(_COLUMN_MAP)].copy()
    out = out.rename(columns=_COLUMN_MAP)

    # ISO dates from LeagueGameLog; pd.to_datetime handles any format variant
    out["game_date"] = pd.to_datetime(out["GAME_DATE"]).dt.date
    out = out.drop(columns=["GAME_DATE"])

    out["season"] = season

    return out


def _upsert(df: pd.DataFrame, engine: Engine) -> int:
    """Bulk-insert rows, silently skipping any (player_id, game_id) duplicates.

    Returns the number of rows actually written (0 when all were already present).
    """
    # Convert NaN → None so SQLAlchemy maps them to NULL rather than NaN float
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    if engine.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as _insert
    else:
        from sqlalchemy.dialects.postgresql import insert as _insert  # type: ignore[no-redef]

    inserted = 0
    with engine.begin() as conn:
        for start in range(0, len(records), _INSERT_CHUNK):
            chunk = records[start : start + _INSERT_CHUNK]
            stmt = _insert(GameLog.__table__).values(chunk)

            if engine.dialect.name == "sqlite":
                stmt = stmt.on_conflict_do_nothing()
            else:
                stmt = stmt.on_conflict_do_nothing(constraint="uq_player_game")

            result = conn.execute(stmt)
            # rowcount is reliable for SQLite INSERT OR IGNORE
            inserted += result.rowcount

    return inserted


def backfill(
    engine: Engine,
    seasons: Optional[list[str]] = None,
) -> None:
    """Download and store historical game logs for completed seasons.

    Each season is cached as a parquet file under RAW_CACHE_DIR so that
    re-runs avoid redundant API calls.  Upserts are idempotent — safe to call
    multiple times.

    Args:
        engine:   SQLAlchemy engine pointing at the target database.
        seasons:  Override the default BACKFILL_SEASONS list.
    """
    init_db(engine)
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    targets = seasons or BACKFILL_SEASONS

    for season in targets:
        cache_path = RAW_CACHE_DIR / f"{season}.parquet"

        if cache_path.exists():
            logger.info("[%s] cache hit — loading from %s", season, cache_path.name)
            raw = pd.read_parquet(cache_path)
        else:
            logger.info("[%s] fetching from nba_api...", season)
            raw = fetch_season_logs(season)
            raw.to_parquet(cache_path, index=False)
            logger.info("[%s] cached %d rows → %s", season, len(raw), cache_path.name)

        df = normalize(raw, season)
        inserted = _upsert(df, engine)
        logger.info("[%s] %d fetched  %d inserted", season, len(df), inserted)
