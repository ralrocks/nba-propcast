"""
Tests for the ingest module.

Scope: schema, normalize(), upsert dedup, and the live update() path.
No real API calls — nba_client is patched throughout.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from propcast.ingest.historical import _upsert, normalize
from propcast.ingest.live import CURRENT_SEASON, update
from propcast.ingest.schema import GameLog, init_db, make_engine


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def engine():
    """Fresh in-memory SQLite database for each test."""
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    return engine


# ── helpers ───────────────────────────────────────────────────────────────────


def _raw_row(**overrides) -> dict:
    """Minimal LeagueGameLog-shaped row for a single game."""
    base = {
        "GAME_DATE": "2025-04-11",
        "PLAYER_ID": 2544,
        "PLAYER_NAME": "LeBron James",
        "TEAM_ID": 1610612747,
        "TEAM_ABBREVIATION": "LAL",
        "GAME_ID": "0022401185",
        "MATCHUP": "LAL vs. HOU",
        "WL": "W",
        "MIN": 22,
        "FGM": 6,  "FGA": 11,  "FG_PCT": 0.545,
        "FG3M": 1,  "FG3A": 4,  "FG3_PCT": 0.25,
        "FTM": 1,  "FTA": 1,  "FT_PCT": 1.0,
        "OREB": 0, "DREB": 4, "REB": 4,
        "AST": 8, "STL": 1, "BLK": 0, "TOV": 1, "PF": 1,
        "PTS": 14, "PLUS_MINUS": 5,
    }
    base.update(overrides)
    return base


def _raw_df(*rows: dict) -> pd.DataFrame:
    """Build a LeagueGameLog-shaped DataFrame from one or more row dicts."""
    if not rows:
        rows = (_raw_row(),)
    return pd.DataFrame(list(rows))


# ── schema roundtrip ──────────────────────────────────────────────────────────


class TestSchemaRoundtrip:
    def test_empty_on_creation(self, engine):
        with Session(engine) as s:
            count = s.execute(select(func.count()).select_from(GameLog)).scalar()
        assert count == 0

    def test_insert_and_retrieve(self, engine):
        df = normalize(_raw_df(), "2024-25")
        _upsert(df, engine)

        with Session(engine) as s:
            row = s.execute(select(GameLog)).scalar_one()

        assert row.player_id == 2544
        assert row.pts == 14
        assert row.reb == 4
        assert row.ast == 8
        assert row.game_date == date(2025, 4, 11)
        assert row.season == "2024-25"

    def test_nullable_ft_pct_stored_as_null(self, engine):
        """When FTA == 0, FT_PCT is NaN in the API response → NULL in DB."""
        df = normalize(_raw_df(_raw_row(FTA=0, FTM=0, FT_PCT=float("nan"))), "2024-25")
        _upsert(df, engine)

        with Session(engine) as s:
            row = s.execute(select(GameLog)).scalar_one()

        assert row.ft_pct is None


# ── deduplication ─────────────────────────────────────────────────────────────


class TestDedup:
    def test_same_row_twice_yields_one_db_row(self, engine):
        df = normalize(_raw_df(), "2024-25")
        _upsert(df, engine)
        _upsert(df, engine)   # second call must be a no-op

        with Session(engine) as s:
            count = s.execute(select(func.count()).select_from(GameLog)).scalar()
        assert count == 1

    def test_different_players_same_game_both_stored(self, engine):
        df = normalize(
            _raw_df(
                _raw_row(PLAYER_ID=2544, PLAYER_NAME="LeBron James"),
                _raw_row(
                    PLAYER_ID=203999,
                    PLAYER_NAME="Nikola Jokic",
                    TEAM_ID=1610612743,
                    TEAM_ABBREVIATION="DEN",
                    MATCHUP="DEN vs. LAL",
                ),
            ),
            "2024-25",
        )
        _upsert(df, engine)

        with Session(engine) as s:
            count = s.execute(select(func.count()).select_from(GameLog)).scalar()
        assert count == 2

    def test_same_player_different_games_both_stored(self, engine):
        df = normalize(
            _raw_df(
                _raw_row(GAME_ID="0022401185", GAME_DATE="2025-04-11"),
                _raw_row(GAME_ID="0022401000", GAME_DATE="2025-04-09"),
            ),
            "2024-25",
        )
        _upsert(df, engine)

        with Session(engine) as s:
            count = s.execute(select(func.count()).select_from(GameLog)).scalar()
        assert count == 2


# ── normalize ─────────────────────────────────────────────────────────────────


class TestNormalize:
    _EXPECTED_COLUMNS = {
        "player_id", "player_name", "team_id", "team_abbreviation",
        "game_id", "game_date", "season", "matchup", "wl",
        "min", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
        "ftm", "fta", "ft_pct", "oreb", "dreb", "reb",
        "ast", "stl", "blk", "tov", "pf", "pts", "plus_minus",
    }

    def test_canonical_column_set(self):
        df = normalize(_raw_df(), "2024-25")
        assert set(df.columns) == self._EXPECTED_COLUMNS

    def test_no_raw_uppercase_columns_leak(self):
        df = normalize(_raw_df(), "2024-25")
        assert not any(col.isupper() for col in df.columns)

    def test_game_date_is_python_date(self):
        df = normalize(_raw_df(), "2024-25")
        assert df["game_date"].iloc[0] == date(2025, 4, 11)

    def test_season_injected(self):
        df = normalize(_raw_df(), "2022-23")
        assert df["season"].iloc[0] == "2022-23"

    def test_does_not_mutate_input(self):
        raw = _raw_df()
        cols_before = raw.columns.tolist()
        normalize(raw, "2024-25")
        assert raw.columns.tolist() == cols_before


# ── live update ───────────────────────────────────────────────────────────────


class TestLiveUpdate:
    def test_full_season_fetch_on_empty_db(self, engine):
        raw = _raw_df(
            _raw_row(GAME_ID="0022401001", GAME_DATE="2024-10-22"),
            _raw_row(GAME_ID="0022401002", GAME_DATE="2024-10-23"),
        )
        with patch("propcast.ingest.live.fetch_season_logs", return_value=raw):
            update(engine)

        with Session(engine) as s:
            count = s.execute(select(func.count()).select_from(GameLog)).scalar()
        assert count == 2

    def test_incremental_only_fetches_new_dates(self, engine):
        """After seeding with one game, update() should pass a date_from to the API."""
        seed = normalize(_raw_df(_raw_row(GAME_DATE="2024-10-22")), CURRENT_SEASON)
        _upsert(seed, engine)

        new_game = _raw_df(_raw_row(GAME_ID="0022401999", GAME_DATE="2024-10-24"))

        captured: dict = {}

        def mock_fetch(season, *, date_from=None, date_to=None, **kw):
            captured["date_from"] = date_from
            return new_game

        with patch("propcast.ingest.live.fetch_season_logs", side_effect=mock_fetch):
            update(engine)

        assert captured["date_from"] == "10/23/2024"

    def test_no_new_games_is_a_no_op(self, engine):
        seed = normalize(_raw_df(), CURRENT_SEASON)
        _upsert(seed, engine)

        with patch(
            "propcast.ingest.live.fetch_season_logs",
            return_value=pd.DataFrame(),
        ):
            update(engine)

        with Session(engine) as s:
            count = s.execute(select(func.count()).select_from(GameLog)).scalar()
        assert count == 1
