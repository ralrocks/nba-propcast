"""
Tests for the feature engineering module.

Everything runs on synthetic DataFrames — no DB, no filesystem, no network.
Invariants tested:
  1. No leakage: rolling features never include the current game
  2. Context features parse correctly (home/away, rest, game_number)
  3. build_features enforces minimum-games threshold
  4. build_features output has no raw uppercase columns
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from propcast.features._rolling import (
    STAT_COLS,
    WINDOWS,
    add_context_features,
    add_rolling_stats,
)
from propcast.features.build import build_features


# ── helpers ───────────────────────────────────────────────────────────────────


def _player_df(n: int = 20, pts_values: list[int] | None = None) -> pd.DataFrame:
    """Synthetic single-player DataFrame sorted by game_date."""
    base_date = date(2023, 10, 24)
    rows = []
    for i in range(n):
        rows.append({
            "player_id": 1,
            "player_name": "Test Player",
            "game_id": f"00234{i:04d}",
            "game_date": base_date + timedelta(days=i * 2),
            "season": "2023-24",
            "matchup": "LAL vs. HOU" if i % 2 == 0 else "LAL @ HOU",
            "wl": "W",
            "pts": pts_values[i] if pts_values else i + 10,
            "reb": 5, "ast": 4, "fg3m": 2, "min": 30,
            "fg3a": 4, "fgm": 5, "fga": 10, "ftm": 2, "fta": 2,
            "oreb": 1, "dreb": 4, "stl": 1, "blk": 0, "tov": 2, "plus_minus": 3,
        })
    return pd.DataFrame(rows)


def _multi_player_df(n_players: int = 3, games_each: int = 15) -> pd.DataFrame:
    """Synthetic multi-player DataFrame for build_features tests."""
    base_date = date(2023, 10, 24)
    rows = []
    for pid in range(1, n_players + 1):
        for i in range(games_each):
            rows.append({
                "player_id": pid,
                "player_name": f"Player {pid}",
                "game_id": f"{pid:03d}{i:04d}",
                "game_date": base_date + timedelta(days=i * 2),
                "season": "2023-24",
                "matchup": "LAL vs. HOU",
                "wl": "W",
                "pts": 20, "reb": 5, "ast": 4, "fg3m": 2, "min": 30,
                "fg3a": 4, "fgm": 5, "fga": 10, "ftm": 2, "fta": 2,
                "oreb": 1, "dreb": 4, "stl": 1, "blk": 0, "tov": 2, "plus_minus": 3,
            })
    return pd.DataFrame(rows)


# ── rolling stats ─────────────────────────────────────────────────────────────


class TestRollingStats:
    def test_adds_expected_columns(self):
        df = add_rolling_stats(_player_df())
        for stat in STAT_COLS:
            for w in WINDOWS:
                assert f"{stat}_l{w}_mean" in df.columns
            assert f"{stat}_season_mean" in df.columns

    def test_no_leakage_l5_mean(self):
        """Game i's L5 mean must not include game i's own pts value.

        We use a strictly increasing pts sequence so any leakage would cause
        the mean at row i to be >= pts[i], which is impossible without leakage.
        """
        pts_vals = list(range(10, 30))  # 10, 11, 12, ...
        df = add_rolling_stats(_player_df(n=20, pts_values=pts_vals))

        for i in range(1, len(df)):
            current_pts = df["pts"].iloc[i]
            l5_mean = df["pts_l5_mean"].iloc[i]
            assert l5_mean < current_pts, (
                f"Row {i}: l5_mean={l5_mean} >= current_pts={current_pts} — leakage!"
            )

    def test_first_row_has_no_prior_history(self):
        """Row 0 has no prior games; rolling values should be NaN or based on 1 game."""
        df = add_rolling_stats(_player_df())
        # With min_periods=1, row 0's shifted value is NaN → mean of [NaN] = NaN
        assert pd.isna(df["pts_l5_mean"].iloc[0])

    def test_season_mean_is_expanding(self):
        """Season mean at row i should equal the mean of pts[0..i-1]."""
        pts_vals = [float(v) for v in range(10, 30)]
        df = add_rolling_stats(_player_df(n=20, pts_values=pts_vals))

        for i in range(2, 10):
            expected = np.mean(pts_vals[:i])
            actual = df["pts_season_mean"].iloc[i]
            assert abs(actual - expected) < 1e-9, f"Row {i}: expected {expected}, got {actual}"

    def test_does_not_mutate_input(self):
        original = _player_df()
        cols_before = original.columns.tolist()
        add_rolling_stats(original)
        assert original.columns.tolist() == cols_before


# ── context features ──────────────────────────────────────────────────────────


class TestContextFeatures:
    def test_is_home_vs_pattern(self):
        df = add_context_features(_player_df())
        # Even rows have 'vs.' → home (1); odd rows have '@' → away (0)
        assert df["is_home"].iloc[0] == 1
        assert df["is_home"].iloc[1] == 0

    def test_days_rest_is_positive(self):
        df = add_context_features(_player_df())
        assert (df["days_rest"] >= 0).all()

    def test_back_to_back_flag(self):
        """A player with 0 days rest between game i-1 and i is on a B2B."""
        base = date(2023, 10, 24)
        rows = [
            {"player_id": 1, "player_name": "X", "game_id": "001", "game_date": base,
             "season": "2023-24", "matchup": "LAL vs. HOU", "wl": "W",
             "pts": 20, "reb": 5, "ast": 4, "fg3m": 2, "min": 30,
             "fg3a": 4, "fgm": 5, "fga": 10, "ftm": 2, "fta": 2,
             "oreb": 1, "dreb": 4, "stl": 1, "blk": 0, "tov": 2, "plus_minus": 3},
            {"player_id": 1, "player_name": "X", "game_id": "002",
             "game_date": base + timedelta(days=1),  # next day → B2B
             "season": "2023-24", "matchup": "LAL vs. HOU", "wl": "W",
             "pts": 20, "reb": 5, "ast": 4, "fg3m": 2, "min": 30,
             "fg3a": 4, "fgm": 5, "fga": 10, "ftm": 2, "fta": 2,
             "oreb": 1, "dreb": 4, "stl": 1, "blk": 0, "tov": 2, "plus_minus": 3},
        ]
        df = add_context_features(pd.DataFrame(rows))
        # 1-day gap between games is a back-to-back; 0 is impossible in the NBA
        assert df["is_back_to_back"].iloc[1] == 1
        assert df["is_back_to_back"].iloc[0] == 0  # first game, defaulted rest=3

    def test_game_number_is_sequential(self):
        df = add_context_features(_player_df(n=15))
        assert df["game_number"].tolist() == list(range(1, 16))


# ── build_features integration ────────────────────────────────────────────────


class TestBuildFeatures:
    def test_output_has_no_uppercase_columns(self):
        df = build_features(_multi_player_df())
        assert not any(c != c.lower() for c in df.columns)

    def test_players_below_min_threshold_raises(self):
        """build_features raises when no players clear the minimum-games threshold.

        Returning an empty DataFrame silently would let a training run proceed
        with zero data.  A loud RuntimeError is the correct behavior here.
        """
        df = _multi_player_df(n_players=2, games_each=5)   # 5 < 10 threshold
        with pytest.raises(RuntimeError, match="minimum game threshold"):
            build_features(df)

    def test_players_above_threshold_included(self):
        df = _multi_player_df(n_players=3, games_each=15)
        result = build_features(df)
        assert result["player_id"].nunique() == 3

    def test_rolling_features_present_in_output(self):
        df = _multi_player_df()
        result = build_features(df)
        assert "pts_l5_mean" in result.columns
        assert "min_season_mean" in result.columns
        assert "is_home" in result.columns

    def test_output_sorted_by_date(self):
        df = _multi_player_df()
        result = build_features(df)
        dates = pd.to_datetime(result["game_date"])
        assert (dates.diff().dropna() >= pd.Timedelta(0)).all()
