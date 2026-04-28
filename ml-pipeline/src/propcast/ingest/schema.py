from __future__ import annotations

from pathlib import Path

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase

DB_PATH = Path(__file__).parents[3] / "data" / "propcast.db"


def make_engine(db_url: str | None = None):
    """Return a SQLAlchemy engine.  Defaults to the project SQLite DB.

    Pass a full URL to override (e.g. a postgres:// URL for production, or
    'sqlite:///:memory:' for tests).
    """
    url = db_url or f"sqlite:///{DB_PATH}"
    engine = create_engine(url, echo=False, future=True)

    if url.startswith("sqlite"):
        # WAL mode allows concurrent reads during a write; NORMAL sync is safe
        # for this workload (worst case: lose the last transaction on crash).
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(conn, _record):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

    return engine


class Base(DeclarativeBase):
    pass


class GameLog(Base):
    """One row = one player's box-score line for one game."""

    __tablename__ = "game_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # identity
    player_id = Column(Integer, nullable=False)
    player_name = Column(String, nullable=False)
    team_id = Column(Integer, nullable=False)
    team_abbreviation = Column(String(3), nullable=False)
    game_id = Column(String(10), nullable=False)
    game_date = Column(Date, nullable=False)
    season = Column(String(7), nullable=False)   # e.g. '2023-24'
    matchup = Column(String, nullable=False)
    wl = Column(String(1))                        # 'W' | 'L' | NULL (rare)

    # box score
    min = Column(Integer)
    fgm = Column(Integer)
    fga = Column(Integer)
    fg_pct = Column(Float)
    fg3m = Column(Integer)
    fg3a = Column(Integer)
    fg3_pct = Column(Float)
    ftm = Column(Integer)
    fta = Column(Integer)
    ft_pct = Column(Float)                        # NULL when fta == 0
    oreb = Column(Integer)
    dreb = Column(Integer)
    reb = Column(Integer)
    ast = Column(Integer)
    stl = Column(Integer)
    blk = Column(Integer)
    tov = Column(Integer)
    pf = Column(Integer)
    pts = Column(Integer)
    plus_minus = Column(Integer)

    __table_args__ = (
        UniqueConstraint("player_id", "game_id", name="uq_player_game"),
        Index("ix_game_logs_player_date", "player_id", "game_date"),
        Index("ix_game_logs_season_date", "season", "game_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<GameLog player={self.player_id} game={self.game_id} "
            f"pts={self.pts} reb={self.reb} ast={self.ast}>"
        )


class PropLine(Base):
    """One row = one DraftKings player prop snapshot at a point in time.

    Multiple rows per (player, stat, game_date) are expected — we scrape
    throughout the day.  The row with the latest fetched_at before game start
    is the closing line used in backtesting.
    """

    __tablename__ = "prop_lines"

    id = Column(Integer, primary_key=True, autoincrement=True)

    player_name = Column(String, nullable=False)
    player_id   = Column(Integer)               # NULL until matched to game_logs
    stat        = Column(String(4), nullable=False)  # 'pts' | 'reb' | 'ast' | 'fg3m'
    line        = Column(Float, nullable=False)
    over_odds   = Column(Integer, nullable=False)    # American (e.g. -110)
    under_odds  = Column(Integer, nullable=False)
    game_date   = Column(Date, nullable=False)
    dk_event_id = Column(Integer)
    fetched_at  = Column(DateTime, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "player_name", "stat", "game_date", "fetched_at",
            name="uq_prop_snapshot",
        ),
        Index("ix_prop_lines_date_stat",    "game_date", "stat"),
        Index("ix_prop_lines_player_date",  "player_name", "game_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<PropLine {self.player_name} {self.stat} "
            f"line={self.line} over={self.over_odds} under={self.under_odds}>"
        )


def init_db(engine) -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(engine)
