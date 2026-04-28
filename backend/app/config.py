from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).parents[1]   # backend/  (config.py is at backend/app/config.py)
_REPO_ROOT   = _BACKEND_DIR.parent         # nba-propcast/
_ML_DATA     = _REPO_ROOT / "ml-pipeline" / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url:           str  = f"sqlite:///{_ML_DATA / 'propcast.db'}"
    models_dir:             Path = _ML_DATA / "models"
    feature_path:           Path = _ML_DATA / "processed" / "features.parquet"
    backtest_results_path:  Path = _ML_DATA / "backtest_results.json"
    cors_origins:           list[str] = ["*"]

    @field_validator("models_dir", "feature_path", "backtest_results_path", mode="before")
    @classmethod
    def _coerce_path(cls, v):
        return Path(v)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
