"""Smoke tests for the PropCast FastAPI backend.

Tests are split into two groups:
- "no_data" tests: run without any ML artifacts; verify error handling.
- Regular tests: require the trained models + feature matrix to be present.
  They are skipped automatically when those files don't exist.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    from main import app
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="session")
def settings():
    from app.config import get_settings
    return get_settings()


def _artifacts_exist(settings) -> bool:
    return (
        settings.models_dir.exists()
        and (settings.models_dir / "pts.joblib").exists()
        and settings.feature_path.exists()
    )


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ------------------------------------------------------------------
# /players/search — requires a populated DB
# ------------------------------------------------------------------

def test_player_search_requires_min_chars(client):
    r = client.get("/players/search", params={"q": "a"})
    assert r.status_code == 422


def test_player_search_returns_list(client, settings):
    if not Path(str(settings.database_url).replace("sqlite:///", "")).exists():
        pytest.skip("No DB available")
    r = client.get("/players/search", params={"q": "LeBron"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    if r.json():
        assert "player_id" in r.json()[0]
        assert "player_name" in r.json()[0]


def test_player_search_limit(client, settings):
    if not Path(str(settings.database_url).replace("sqlite:///", "")).exists():
        pytest.skip("No DB available")
    r = client.get("/players/search", params={"q": "james", "limit": 3})
    assert r.status_code == 200
    assert len(r.json()) <= 3


# ------------------------------------------------------------------
# POST /predict — requires models + features
# ------------------------------------------------------------------

@pytest.mark.parametrize("stat", ["pts", "reb", "ast", "fg3m"])
def test_predict_known_player(client, settings, stat):
    if not _artifacts_exist(settings):
        pytest.skip("ML artifacts not present")

    import pandas as pd
    features = pd.read_parquet(settings.feature_path)
    player_id = int(features["player_id"].iloc[0])

    r = client.post("/predict/", json={"player_id": player_id, "stat": stat})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stat"] == stat
    assert body["ci_low"] <= body["point_estimate"] <= body["ci_high"]
    assert body["n_games"] > 0
    assert body["p_over"] is None  # no line provided


def test_predict_with_line(client, settings):
    if not _artifacts_exist(settings):
        pytest.skip("ML artifacts not present")

    import pandas as pd
    features = pd.read_parquet(settings.feature_path)
    player_id = int(features["player_id"].iloc[0])

    r = client.post("/predict/", json={"player_id": player_id, "stat": "pts", "line": 20.5})
    assert r.status_code == 200
    body = r.json()
    assert body["p_over"] is not None
    assert 0.0 <= body["p_over"] <= 1.0


def test_predict_unknown_player(client, settings):
    if not _artifacts_exist(settings):
        pytest.skip("ML artifacts not present")
    r = client.post("/predict/", json={"player_id": 9999999, "stat": "pts"})
    assert r.status_code == 404


def test_predict_invalid_stat(client):
    r = client.post("/predict/", json={"player_id": 123, "stat": "xyz"})
    assert r.status_code == 422


# ------------------------------------------------------------------
# /backtest — results endpoint
# ------------------------------------------------------------------

def _with_settings(override: dict):
    """Return a FastAPI dependency that yields a Settings copy with overrides applied."""
    from app.config import get_settings
    custom = get_settings().model_copy(update=override)
    return lambda: custom


def test_backtest_results_404_when_missing(tmp_path):
    from main import app
    from app.config import get_settings
    from fastapi.testclient import TestClient

    app.dependency_overrides[get_settings] = _with_settings(
        {"backtest_results_path": tmp_path / "missing.json"}
    )
    try:
        with TestClient(app) as c:
            r = c.get("/backtest/results")
        assert r.status_code == 404
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_backtest_results_returns_cached(tmp_path):
    from main import app
    from app.config import get_settings
    from fastapi.testclient import TestClient

    fake = {"pts": {"brier": 0.21, "clv": 0.02, "mae": 4.5, "roi": 0.01, "n_bets": 100, "n_samples": 500, "mode": "sim"}}
    results_file = tmp_path / "backtest_results.json"
    results_file.write_text(json.dumps(fake))

    app.dependency_overrides[get_settings] = _with_settings(
        {"backtest_results_path": results_file}
    )
    try:
        with TestClient(app) as c:
            r = c.get("/backtest/results")
        assert r.status_code == 200
        assert r.json()["pts"]["brier"] == pytest.approx(0.21)
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_backtest_run_sim(client, settings):
    if not _artifacts_exist(settings):
        pytest.skip("ML artifacts not present")
    if not Path(str(settings.database_url).replace("sqlite:///", "")).exists():
        pytest.skip("No DB available")
    r = client.post("/backtest/run", params={"mode": "sim"})
    assert r.status_code == 200
    body = r.json()
    assert any(stat in body for stat in ["pts", "reb", "ast", "fg3m"])
    for stat_metrics in body.values():
        assert "brier" in stat_metrics
        assert "clv" in stat_metrics
        assert "mae" in stat_metrics
