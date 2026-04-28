from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

logger = logging.getLogger(__name__)

_MIN_INTERVAL: float = 0.6   # seconds — nba_api rate limit guidance
_last_call_at: float = 0.0


def _rate_limit() -> None:
    """Block until at least _MIN_INTERVAL seconds have passed since the last call."""
    global _last_call_at
    elapsed = time.monotonic() - _last_call_at
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_at = time.monotonic()


def fetch_season_logs(
    season: str,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch all player game logs for *season* in a single API call.

    Args:
        season:     NBA season string, e.g. '2023-24'.
        date_from:  Inclusive start date as 'MM/DD/YYYY', or None for full season.
        date_to:    Inclusive end date as 'MM/DD/YYYY', or None.
        retries:    Maximum number of attempts before re-raising.

    Returns:
        Raw LeagueGameLog DataFrame with original nba_api column names.
    """
    for attempt in range(retries):
        try:
            _rate_limit()
            response = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation="P",
                season_type_all_star="Regular Season",
                date_from_nullable=date_from or "",
                date_to_nullable=date_to or "",
            )
            df = response.get_data_frames()[0]
            logger.info("season=%s  rows=%d  date_from=%s", season, len(df), date_from or "start")
            return df
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)   # 2s, 4s
            logger.warning(
                "attempt %d/%d failed for season=%s: %s — retrying in %ds",
                attempt + 1,
                retries,
                season,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError("unreachable")  # pragma: no cover
