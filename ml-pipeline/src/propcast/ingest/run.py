"""CLI entry point for game-log ingestion.

Usage:
    uv run python -m propcast.ingest.run backfill [--seasons 2022-23 2023-24]
    uv run python -m propcast.ingest.run update
    uv run python -m propcast.ingest.run status
"""
from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .historical import BACKFILL_SEASONS, backfill
from .live import CURRENT_SEASON, update
from .schema import DB_PATH, GameLog, init_db, make_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ── commands ──────────────────────────────────────────────────────────────────


def _cmd_backfill(args: argparse.Namespace) -> None:
    engine = make_engine()
    seasons: list[str] | None = args.seasons or None
    logger.info("Starting backfill — seasons: %s", seasons or BACKFILL_SEASONS)
    backfill(engine, seasons=seasons)
    logger.info("Backfill complete.")


def _cmd_update(args: argparse.Namespace) -> None:
    engine = make_engine()
    update(engine)
    logger.info("Update complete.")


def _cmd_status(args: argparse.Namespace) -> None:
    engine = make_engine()
    init_db(engine)

    with Session(engine) as session:
        total: int = session.execute(
            select(func.count()).select_from(GameLog)
        ).scalar_one()

        rows = session.execute(
            select(
                GameLog.season,
                func.count().label("rows"),
                func.count(GameLog.player_id.distinct()).label("players"),
                func.min(GameLog.game_date).label("first_game"),
                func.max(GameLog.game_date).label("last_game"),
            )
            .group_by(GameLog.season)
            .order_by(GameLog.season)
        ).all()

    print(f"\nDB path : {DB_PATH}")
    print(f"Total   : {total:,} rows\n")

    if not rows:
        print("  (no data — run `backfill` first)")
        return

    header = f"{'Season':<10}  {'Rows':>8}  {'Players':>8}  {'First':12}  {'Last':12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.season:<10}  {r.rows:>8,}  {r.players:>8}  "
            f"{str(r.first_game):<12}  {str(r.last_game):<12}"
        )
    print()


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m propcast.ingest.run",
        description="NBA game-log ingestion pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_bf = sub.add_parser("backfill", help="Backfill historical seasons via nba_api")
    p_bf.add_argument(
        "--seasons",
        nargs="+",
        metavar="YYYY-YY",
        help=f"Seasons to backfill (default: {BACKFILL_SEASONS[0]}..{BACKFILL_SEASONS[-1]})",
    )
    p_bf.set_defaults(func=_cmd_backfill)

    p_up = sub.add_parser(
        "update", help=f"Incremental ingest for the current season ({CURRENT_SEASON})"
    )
    p_up.set_defaults(func=_cmd_update)

    p_st = sub.add_parser("status", help="Print a summary of what's in the DB")
    p_st.set_defaults(func=_cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
