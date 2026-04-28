import json

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from propcast.backtest.run import run_backtest, _generate_sim_lines
from propcast.ingest.schema import GameLog, PropLine, make_engine

from ..config import Settings, get_settings

router = APIRouter(prefix="/backtest", tags=["backtest"])


def _game_logs(db_url: str, season: str | None) -> pd.DataFrame:
    engine = make_engine(db_url)
    with Session(engine) as s:
        q = select(
            GameLog.player_id,
            GameLog.player_name,
            GameLog.game_date,
            GameLog.season,
            GameLog.pts,
            GameLog.reb,
            GameLog.ast,
            GameLog.fg3m,
        )
        if season:
            q = q.where(GameLog.season == season)
        return pd.DataFrame(s.execute(q).all())


def _prop_lines(db_url: str, season: str | None) -> pd.DataFrame:
    engine = make_engine(db_url)
    with Session(engine) as s:
        rows = s.execute(
            select(
                PropLine.player_name,
                PropLine.player_id,
                PropLine.stat,
                PropLine.line,
                PropLine.over_odds,
                PropLine.under_odds,
                PropLine.game_date,
                PropLine.fetched_at,
            )
        ).all()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    closing = (
        df.sort_values("fetched_at")
        .groupby(["player_name", "stat", "game_date"], as_index=False)
        .last()
    )
    if season:
        closing["game_date"] = pd.to_datetime(closing["game_date"])
        yr = int(season[:4])
        closing = closing[
            closing["game_date"].between(f"{yr}-10-01", f"{yr + 1}-06-30")
        ]
    return closing


@router.get("/results")
def get_results(settings: Settings = Depends(get_settings)):
    if not settings.backtest_results_path.exists():
        raise HTTPException(
            404,
            detail="No cached results yet. POST /backtest/run?mode=sim to generate.",
        )
    return json.loads(settings.backtest_results_path.read_text())


@router.post("/run")
def trigger_run(
    mode: str = Query("sim", pattern="^(sim|live)$"),
    season: str | None = Query(None, description="Season filter, e.g. 2023-24"),
    settings: Settings = Depends(get_settings),
):
    logs = _game_logs(settings.database_url, season)
    if logs.empty:
        raise HTTPException(
            422,
            detail="No game logs in DB. Run: uv run python -m propcast.ingest.run backfill",
        )

    if not settings.feature_path.exists():
        raise HTTPException(
            503,
            detail="Feature matrix not found. Run: uv run python -m propcast.features.build",
        )
    features = pd.read_parquet(settings.feature_path)

    if mode == "sim":
        lines = _generate_sim_lines(logs)
    else:
        lines = _prop_lines(settings.database_url, season)
        if lines.empty:
            raise HTTPException(
                422,
                detail="No prop lines in DB. Run dk_scraper daily, then retry with mode=live.",
            )

    results = run_backtest(
        lines,
        logs,
        features,
        models_dir=settings.models_dir,
        mode_label=mode,
    )
    if not results:
        raise HTTPException(
            500,
            detail="Backtest produced no results. Ensure models are trained and features are built.",
        )

    settings.backtest_results_path.parent.mkdir(parents=True, exist_ok=True)
    settings.backtest_results_path.write_text(json.dumps(results, indent=2))
    return results
