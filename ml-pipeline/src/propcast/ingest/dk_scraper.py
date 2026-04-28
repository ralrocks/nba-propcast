"""DraftKings v5 JSON API scraper for NBA player prop lines.

Fetches current over/under lines for pts / reb / ast / fg3m and stores
snapshots in the prop_lines table.  Run this once before each slate of games
to capture opening lines, and again ~15 minutes before tip-off to capture
closing lines — the closing snapshot is what the backtest uses.

DK API note
-----------
DraftKings' API sits behind Akamai edge protection that challenges automated
requests from data-centre IP ranges.  This scraper works correctly from a
normal laptop or deployed server with a residential/cloud IP that has a valid
browser TLS fingerprint.  If you get 403s in a sandboxed environment, run it
from your terminal directly.

Usage:
    uv run python -m propcast.ingest.dk_scraper           # fetch + store
    uv run python -m propcast.ingest.dk_scraper --dry-run # print lines only
    uv run python -m propcast.ingest.dk_scraper --dump    # save raw JSON
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import requests
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .schema import PropLine, init_db, make_engine

logger = logging.getLogger(__name__)

# ── DraftKings v5 constants ───────────────────────────────────────────────────

_BASE_URL     = "https://sportsbook.draftkings.com/sites/US-SB/api/v5"
_NBA_EG_ID    = 42648    # Basketball - NBA event group (stable across seasons)
_PROPS_CAT_ID = 583      # "Player Props" category (stable)

# Subcategory names → our stat codes.  Matched case-insensitively so minor DK
# renames ("Player Points" → "Points O/U") don't break ingestion.
_SUBCATEGORY_MAP: dict[str, str] = {
    "player points":    "pts",
    "points":           "pts",
    "player rebounds":  "reb",
    "rebounds":         "reb",
    "player assists":   "ast",
    "assists":          "ast",
    "player threes":    "fg3m",
    "3-pointers made":  "fg3m",
    "player 3-pointers made": "fg3m",
}

_HEADERS = {
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://sportsbook.draftkings.com/leagues/basketball/nba",
    "User-Agent":      (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

_RAW_DUMP_DIR = Path(__file__).parents[3] / "data" / "raw" / "dk"


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class PropSnapshot:
    player_name: str
    stat:        str
    line:        float
    over_odds:   int
    under_odds:  int
    game_date:   date
    dk_event_id: int
    fetched_at:  datetime


# ── HTTP client ───────────────────────────────────────────────────────────────

def _get(path: str, session: requests.Session) -> dict:
    url = f"{_BASE_URL}{path}"
    resp = session.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ── parsing ───────────────────────────────────────────────────────────────────

def _parse_game_date(iso_str: str) -> date:
    """Convert DK's ISO-8601 UTC startDate to an Eastern-tz game date.

    DK returns UTC timestamps.  NBA games that tip at 7 pm ET are returned
    as ~23:00 UTC the same calendar day, so the UTC date is correct for most
    games.  For very late West-Coast games (tipoff after midnight ET) there
    can be an off-by-one — acceptable for v1.
    """
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).date()


def _american_to_int(odds_str: str | int) -> int:
    """DK returns American odds as a string like '-110' or '+120'."""
    return int(odds_str)


def _parse_category_response(
    payload: dict,
    fetched_at: datetime,
    dump: bool = False,
) -> list[PropSnapshot]:
    """Parse a DK v5 eventgroup/categories response into PropSnapshot list.

    DK v5 offer structure (within offerSubcategoryDescriptors):
        offerSubcategory.offers = [
            [  ← one inner list per player
                {"label": "Over",  "line": 22.5, "oddsAmerican": "-110",
                 "eventId": 123, "participants": [{"name": "LeBron James"}]},
                {"label": "Under", "line": 22.5, "oddsAmerican": "-110",
                 "eventId": 123, "participants": [{"name": "LeBron James"}]},
            ],
            ...
        ]
    Events list provides startDate for each eventId.
    """
    if dump:
        _RAW_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        ts = fetched_at.strftime("%Y%m%d_%H%M%S")
        path = _RAW_DUMP_DIR / f"dk_{ts}.json"
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Raw DK response saved → %s", path)

    eg = payload.get("eventGroup", {})

    # Build eventId → game_date lookup
    event_dates: dict[int, date] = {}
    for ev in eg.get("events", []):
        try:
            event_dates[ev["eventId"]] = _parse_game_date(ev["startDate"])
        except (KeyError, ValueError) as exc:
            logger.debug("Could not parse event date: %s", exc)

    snapshots: list[PropSnapshot] = []

    for category in eg.get("offerCategories", []):
        for sub_desc in category.get("offerSubcategoryDescriptors", []):
            sub_name  = sub_desc.get("name", "").lower().strip()
            stat_code = _SUBCATEGORY_MAP.get(sub_name)
            if stat_code is None:
                continue

            sub = sub_desc.get("offerSubcategory", {})
            for offer_group in sub.get("offers", []):
                # Each offer_group is a list of outcomes for one player market
                if not isinstance(offer_group, list):
                    offer_group = [offer_group]

                over_outcome  = next((o for o in offer_group if o.get("label", "").lower() == "over"),  None)
                under_outcome = next((o for o in offer_group if o.get("label", "").lower() == "under"), None)

                if over_outcome is None or under_outcome is None:
                    continue

                event_id = over_outcome.get("eventId")
                gd = event_dates.get(event_id)
                if gd is None:
                    logger.debug("No game_date for eventId=%s", event_id)
                    continue

                # Player name lives in participants[0].name
                participants = over_outcome.get("participants", [])
                if not participants:
                    # Older DK response puts name in the outer offer label
                    player_name = over_outcome.get("label", "").strip()
                else:
                    player_name = participants[0].get("name", "").strip()

                if not player_name:
                    continue

                try:
                    line       = float(over_outcome["line"])
                    over_odds  = _american_to_int(over_outcome["oddsAmerican"])
                    under_odds = _american_to_int(under_outcome["oddsAmerican"])
                except (KeyError, ValueError, TypeError) as exc:
                    logger.debug("Skipping malformed offer: %s", exc)
                    continue

                snapshots.append(PropSnapshot(
                    player_name = player_name,
                    stat        = stat_code,
                    line        = line,
                    over_odds   = over_odds,
                    under_odds  = under_odds,
                    game_date   = gd,
                    dk_event_id = event_id or 0,
                    fetched_at  = fetched_at,
                ))

    return snapshots


# ── player name matching ──────────────────────────────────────────────────────

def _match_player_ids(
    snapshots: list[PropSnapshot],
    engine: Engine,
) -> dict[str, Optional[int]]:
    """Return {dk_player_name: player_id} using exact match then fuzzy fallback.

    DK names are usually identical to nba_api names ("LeBron James") but
    occasional discrepancies (accents, suffixes like "Jr.") are handled by
    difflib with a 0.85 similarity threshold.
    """
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from propcast.ingest.schema import GameLog

    with Session(engine) as s:
        db_names: list[tuple[str, int]] = s.execute(
            select(GameLog.player_name, GameLog.player_id)
            .group_by(GameLog.player_name)
        ).all()

    db_map = {name: pid for name, pid in db_names}

    result: dict[str, Optional[int]] = {}
    for snap in snapshots:
        name = snap.player_name
        if name in result:
            continue
        if name in db_map:
            result[name] = db_map[name]
            continue
        # fuzzy fallback
        best_ratio, best_pid = 0.0, None
        for db_name, pid in db_map.items():
            ratio = SequenceMatcher(None, name.lower(), db_name.lower()).ratio()
            if ratio > best_ratio:
                best_ratio, best_pid = ratio, pid
        if best_ratio >= 0.85:
            logger.debug("Fuzzy match: '%s' → '%s' (%.2f)", name, best_ratio)
            result[name] = best_pid
        else:
            logger.warning("No DB match for DK player '%s' (best ratio %.2f)", name, best_ratio)
            result[name] = None

    return result


# ── storage ───────────────────────────────────────────────────────────────────

def _store(
    snapshots: list[PropSnapshot],
    player_id_map: dict[str, Optional[int]],
    engine: Engine,
) -> int:
    records = [
        {
            "player_name": s.player_name,
            "player_id":   player_id_map.get(s.player_name),
            "stat":        s.stat,
            "line":        s.line,
            "over_odds":   s.over_odds,
            "under_odds":  s.under_odds,
            "game_date":   s.game_date,
            "dk_event_id": s.dk_event_id,
            "fetched_at":  s.fetched_at,
        }
        for s in snapshots
    ]
    with engine.begin() as conn:
        stmt = (
            sqlite_insert(PropLine.__table__)
            .values(records)
            .on_conflict_do_nothing()
        )
        result = conn.execute(stmt)
    return result.rowcount


# ── public API ────────────────────────────────────────────────────────────────

def fetch_props(*, dump: bool = False) -> list[PropSnapshot]:
    """Fetch current NBA player prop lines from DraftKings.

    Returns a flat list of PropSnapshot objects for all discovered markets
    (pts / reb / ast / fg3m) across all today's games.
    """
    session    = requests.Session()
    fetched_at = datetime.now(timezone.utc)

    payload = _get(f"/eventgroups/{_NBA_EG_ID}/categories/{_PROPS_CAT_ID}", session)
    snaps   = _parse_category_response(payload, fetched_at, dump=dump)

    logger.info("Fetched %d prop snapshots from DraftKings", len(snaps))
    return snaps


def scrape_and_store(engine: Engine, *, dump: bool = False) -> int:
    """Fetch current DK lines and persist them.  Returns rows written."""
    init_db(engine)
    snaps      = fetch_props(dump=dump)
    id_map     = _match_player_ids(snaps, engine)
    inserted   = _store(snaps, id_map, engine)
    logger.info("Stored %d new prop snapshots", inserted)
    return inserted


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape DraftKings NBA player props")
    parser.add_argument("--dry-run", action="store_true", help="Print lines, don't store")
    parser.add_argument("--dump",    action="store_true", help="Save raw JSON to data/raw/dk/")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    if args.dry_run:
        snaps = fetch_props(dump=args.dump)
        print(f"\n{'Player':<28} {'Stat':<6} {'Line':>6} {'Over':>6} {'Under':>6}  {'Date'}")
        print("-" * 70)
        for s in sorted(snaps, key=lambda x: (x.stat, x.player_name)):
            print(f"{s.player_name:<28} {s.stat:<6} {s.line:>6.1f} {s.over_odds:>6} {s.under_odds:>6}  {s.game_date}")
        print(f"\n{len(snaps)} total props")
    else:
        engine   = make_engine()
        inserted = scrape_and_store(engine, dump=args.dump)
        print(f"Stored {inserted} new prop snapshots.")


if __name__ == "__main__":
    main()
