from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from propcast.models.predict import predict as ml_predict
from propcast.models.train import FEATURE_COLS

from ..config import Settings, get_settings

router = APIRouter(prefix="/predict", tags=["predict"])

# Loaded once on first request; invalidated if the configured path changes.
_cache: dict = {"df": None, "path": None}


def _get_features(path: Path) -> pd.DataFrame:
    if _cache["df"] is None or _cache["path"] != path:
        if not path.exists():
            raise FileNotFoundError(path)
        _cache["df"] = pd.read_parquet(path)
        _cache["path"] = path
    return _cache["df"]


class PredictRequest(BaseModel):
    player_id: int
    stat: str = Field(..., pattern="^(pts|reb|ast|fg3m)$")
    line: float | None = None


class PredictResponse(BaseModel):
    player_id: int
    stat: str
    point_estimate: float
    ci_low: float
    ci_high: float
    p_over: float | None
    n_games: int
    latest_game_date: str


@router.post("/", response_model=PredictResponse)
def predict(req: PredictRequest, settings: Settings = Depends(get_settings)):
    try:
        features = _get_features(settings.feature_path)
    except FileNotFoundError:
        raise HTTPException(
            503,
            detail="Feature matrix not built. Run: uv run python -m propcast.features.build",
        )

    player_rows = features[features["player_id"] == req.player_id].sort_values("game_date")
    if player_rows.empty:
        raise HTTPException(404, detail=f"No feature data for player_id={req.player_id}")

    latest = player_rows.iloc[-1]
    feat_dict = {col: float(latest[col]) for col in FEATURE_COLS if col in latest.index}

    try:
        pred = ml_predict(
            req.stat,
            feat_dict,
            line=req.line,
            models_dir=settings.models_dir,
        )
    except FileNotFoundError as exc:
        raise HTTPException(503, detail=str(exc))

    return PredictResponse(
        player_id=req.player_id,
        stat=req.stat,
        point_estimate=pred.point_estimate,
        ci_low=pred.ci_low,
        ci_high=pred.ci_high,
        p_over=pred.p_over,
        n_games=len(player_rows),
        latest_game_date=str(latest["game_date"]),
    )
