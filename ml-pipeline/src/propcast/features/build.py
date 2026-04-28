"""Build the feature matrix from stored game logs.

Reads game_logs from the SQLite DB, applies rolling and context features
per player (sorted chronologically), and writes a single parquet file that
the training pipeline consumes.

Usage:
    uv run python -m propcast.features.build
    uv run python -m propcast.features.build --seasons 2022-23 2023-24
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from propcast.ingest.schema import GameLog, make_engine
from ._rolling import add_context_features, add_rolling_stats

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parents[3] / "data" / "processed"
FEATURE_PATH = PROCESSED_DIR / "features.parquet"

# Minimum games a player must have in the dataset to be included.
# Players with fewer games don't produce meaningful rolling features.
_MIN_GAMES = 10


def load_logs(seasons: list[str] | None = None) -> pd.DataFrame:
    """Load raw game logs from the DB into a DataFrame."""
    engine = make_engine()
    with Session(engine) as session:
        q = select(
            GameLog.player_id,
            GameLog.player_name,
            GameLog.game_id,
            GameLog.game_date,
            GameLog.season,
            GameLog.matchup,
            GameLog.wl,
            GameLog.min,
            GameLog.pts,
            GameLog.reb,
            GameLog.ast,
            GameLog.fg3m,
            GameLog.fg3a,
            GameLog.fgm,
            GameLog.fga,
            GameLog.ftm,
            GameLog.fta,
            GameLog.oreb,
            GameLog.dreb,
            GameLog.stl,
            GameLog.blk,
            GameLog.tov,
            GameLog.plus_minus,
        )
        if seasons:
            q = q.where(GameLog.season.in_(seasons))

        rows = session.execute(q).all()

    df = pd.DataFrame(rows)
    logger.info("Loaded %d rows from DB (%s seasons)", len(df), len(df["season"].unique()))
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transformations and return the feature matrix.

    Processing per player ensures rolling windows never bleed across players.
    The sort inside each group is the chronological correctness guarantee.
    """
    # Fill null stats with 0 — DNP/partial games; rare but must not break windows
    stat_cols = ["pts", "reb", "ast", "fg3m", "fg3a", "fgm", "fga", "ftm", "fta",
                 "oreb", "dreb", "stl", "blk", "tov", "plus_minus", "min"]
    df[stat_cols] = df[stat_cols].fillna(0)

    parts: list[pd.DataFrame] = []

    for player_id, group in df.groupby("player_id"):
        if len(group) < _MIN_GAMES:
            continue

        player_df = group.sort_values("game_date").reset_index(drop=True)
        player_df = add_rolling_stats(player_df)
        player_df = add_context_features(player_df)
        parts.append(player_df)

    if not parts:
        raise RuntimeError("No players met the minimum game threshold — run backfill first.")

    out = pd.concat(parts, ignore_index=True)

    # Drop rows where rolling features are NaN (first game per player has no history)
    required_feature_cols = [c for c in out.columns if c.endswith(("_mean", "_std"))]
    before = len(out)
    out = out.dropna(subset=required_feature_cols[:1])  # proxy: first rolling col is enough
    logger.info("Dropped %d rows with no rolling history (%d remain)", before - len(out), len(out))

    return out.sort_values(["game_date", "player_id"]).reset_index(drop=True)


def main(seasons: list[str] | None = None) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_logs(seasons=seasons)
    features = build_features(raw)

    features.to_parquet(FEATURE_PATH, index=False)

    logger.info(
        "Feature matrix saved → %s  shape=%s  players=%d  seasons=%s",
        FEATURE_PATH.name,
        features.shape,
        features["player_id"].nunique(),
        sorted(features["season"].unique()),
    )

    # Quick sanity checks
    _assert_no_leakage(features)
    _print_summary(features)


def _assert_no_leakage(df: pd.DataFrame) -> None:
    """Crash loudly if any rolling feature correlates perfectly with the target.

    A perfect correlation (r=1.0) on the training set is a near-certain sign
    of target leakage — the current game's stat leaked into the feature.
    """
    targets = ["pts", "reb", "ast", "fg3m"]
    for target in targets:
        for col in [f"{target}_l5_mean", f"{target}_l10_mean", f"{target}_season_mean"]:
            if col not in df.columns:
                continue
            corr = df[[target, col]].dropna().corr().iloc[0, 1]
            if abs(corr) > 0.999:
                raise RuntimeError(
                    f"Leakage detected: {col} correlates {corr:.4f} with {target}. "
                    "Check that shift(1) is applied before rolling windows."
                )
    logger.info("Leakage check passed.")


def _print_summary(df: pd.DataFrame) -> None:
    print(f"\nFeature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Players : {df['player_id'].nunique()}")
    print(f"Seasons : {sorted(df['season'].unique())}")
    print(f"Date range : {df['game_date'].min()}  →  {df['game_date'].max()}")
    print("\nTarget distributions:")
    for stat in ("pts", "reb", "ast", "fg3m"):
        s = df[stat]
        print(f"  {stat:<6}  mean={s.mean():.1f}  std={s.std():.1f}  max={s.max()}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="Build PropCast feature matrix")
    parser.add_argument(
        "--seasons", nargs="+", metavar="YYYY-YY",
        help="Limit to specific seasons (default: all in DB)"
    )
    args = parser.parse_args()
    main(seasons=args.seasons)
