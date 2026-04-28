"""Pure rolling-window helpers.

All functions operate on a single-player DataFrame that is already sorted by
game_date ascending.  They return a new DataFrame with features added.

Critical invariant: every window uses shift(1) before computing so the
current game's stats are never included.  Violating this leaks the target
into the features.
"""
from __future__ import annotations

import pandas as pd

# Windows used for every counting stat
WINDOWS = (5, 10)
STAT_COLS = ("pts", "reb", "ast", "fg3m", "min")


def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add L5 / L10 / season rolling means and L5 std for each stat.

    Assumes df is sorted by game_date ascending for a single player.
    """
    out = df.copy()

    for stat in STAT_COLS:
        shifted = out[stat].shift(1)   # exclude current game

        for w in WINDOWS:
            out[f"{stat}_l{w}_mean"] = (
                shifted.rolling(w, min_periods=1).mean()
            )

        # Season mean uses all prior games (expanding window)
        out[f"{stat}_season_mean"] = shifted.expanding(min_periods=1).mean()

    # Consistency signal for the primary props
    for stat in ("pts", "reb", "ast", "fg3m"):
        shifted = out[stat].shift(1)
        out[f"{stat}_l5_std"] = shifted.rolling(5, min_periods=2).std()

    return out


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add home/away, rest, and season-position features.

    Assumes df is sorted by game_date ascending for a single player.
    """
    out = df.copy()

    # 'LAL vs. HOU' → home;  'LAL @ HOU' → away
    out["is_home"] = out["matchup"].str.contains(r"\bvs\.", regex=True).astype("int8")

    # Days between consecutive games for this player
    dates = pd.to_datetime(out["game_date"])
    out["days_rest"] = (dates - dates.shift(1)).dt.days.fillna(3).clip(upper=10)
    # 1 day rest = consecutive-day games = back-to-back (0 is impossible in NBA)
    out["is_back_to_back"] = (out["days_rest"] == 1).astype("int8")

    # Ordinal game number within the season (1-indexed, no leakage — it's a
    # schedule property, not a stat; knowing it's game 60 doesn't leak the score)
    out["game_number"] = range(1, len(out) + 1)

    return out
