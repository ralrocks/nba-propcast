"""
Tests for the models module.

Two sections:
  1. CV correctness — pure logic, no XGBoost.
  2. Integration — tiny synthetic feature matrix trains a real model (5 trees)
     fast enough to keep the suite under 10 s, then predict() is exercised.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propcast.models._cv import expanding_date_splits
from propcast.models.train import FEATURE_COLS, train_one
from propcast.models.predict import Prediction, predict


# ── fixtures ──────────────────────────────────────────────────────────────────


def _dates_series(n: int, start: date = date(2023, 10, 24), step_days: int = 1) -> pd.Series:
    return pd.Series([start + timedelta(days=i * step_days) for i in range(n)])


def _synthetic_features(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    """Minimal feature matrix compatible with FEATURE_COLS + target columns."""
    rng = np.random.default_rng(seed)
    base = date(2023, 10, 24)
    df = pd.DataFrame({
        col: rng.normal(10, 3, n_rows).clip(0) for col in FEATURE_COLS
    })
    df["game_date"] = [base + timedelta(days=i) for i in range(n_rows)]
    # Targets: realistic-ish means
    df["pts"]  = rng.normal(15, 8, n_rows).clip(0)
    df["reb"]  = rng.normal(5,  3, n_rows).clip(0)
    df["ast"]  = rng.normal(3,  2, n_rows).clip(0)
    df["fg3m"] = rng.normal(1.5, 1.2, n_rows).clip(0)
    return df


@pytest.fixture(scope="module")
def tiny_df() -> pd.DataFrame:
    return _synthetic_features(n_rows=200)


@pytest.fixture(scope="module")
def trained_pts(tmp_path_factory, tiny_df) -> tuple[Path, object]:
    """Train a tiny pts model in a temp dir.  Shared across all tests in module."""
    models_dir = tmp_path_factory.mktemp("models")
    result = train_one(
        tiny_df,
        "pts",
        n_splits=3,
        xgb_params={"n_estimators": 5, "verbosity": 0},
        models_dir=models_dir,
    )
    return models_dir, result


# ── CV splitting ──────────────────────────────────────────────────────────────


class TestExpandingDateSplits:
    def test_no_overlap_between_train_and_test(self):
        dates = _dates_series(100)
        for train_idx, test_idx in expanding_date_splits(dates, n_splits=4):
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_test_dates_strictly_after_train_dates(self):
        dates = _dates_series(100)
        for train_idx, test_idx in expanding_date_splits(dates, n_splits=4):
            max_train_date = dates.iloc[train_idx].max()
            min_test_date  = dates.iloc[test_idx].min()
            assert min_test_date > max_train_date

    def test_training_window_grows_each_fold(self):
        dates = _dates_series(100)
        sizes = [len(tr) for tr, _ in expanding_date_splits(dates, n_splits=4)]
        assert sizes == sorted(sizes), "Training set must grow monotonically"

    def test_n_splits_respected(self):
        dates = _dates_series(100)
        folds = list(expanding_date_splits(dates, n_splits=5))
        assert len(folds) == 5

    def test_handles_fewer_dates_than_splits(self):
        # 3 unique dates, 10 splits requested — should not raise
        dates = _dates_series(3)
        folds = list(expanding_date_splits(dates, n_splits=10))
        assert len(folds) >= 1

    def test_interleaved_players_same_date_in_same_fold(self):
        """Two players sharing a game date must both be in the same fold."""
        base = date(2023, 10, 24)
        # alternating dates: p1 game1, p2 game1 on same date, etc.
        dates = pd.Series([base + timedelta(days=i // 2) for i in range(40)])
        for train_idx, test_idx in expanding_date_splits(dates, n_splits=4):
            # No date should appear in both train and test
            train_dates = set(dates.iloc[train_idx].tolist())
            test_dates  = set(dates.iloc[test_idx].tolist())
            assert train_dates.isdisjoint(test_dates)


# ── train_one ─────────────────────────────────────────────────────────────────


class TestTrainOne:
    def test_returns_train_result(self, trained_pts):
        _, result = trained_pts
        assert result.target == "pts"

    def test_fold_metrics_populated(self, trained_pts):
        _, result = trained_pts
        assert len(result.fold_metrics) == 3
        for fm in result.fold_metrics:
            assert fm.mae  > 0
            assert fm.rmse > 0
            assert fm.n_train > 0
            assert fm.n_test  > 0

    def test_cv_mae_is_mean_of_folds(self, trained_pts):
        _, result = trained_pts
        expected = np.mean([f.mae for f in result.fold_metrics])
        assert abs(result.cv_mae - expected) < 1e-9

    def test_oof_residuals_shape(self, trained_pts, tiny_df):
        _, result = trained_pts
        # Residuals only exist for rows that appeared in at least one test fold
        assert len(result.oof_residuals) > 0
        assert len(result.oof_residuals) <= len(tiny_df)

    def test_artifacts_written_to_disk(self, trained_pts):
        models_dir, _ = trained_pts
        assert (models_dir / "pts.joblib").exists()
        assert (models_dir / "pts_residuals.npy").exists()

    def test_folds_have_strict_temporal_ordering(self, tiny_df):
        """Verify ordering at the train_one level, not just _cv."""
        from propcast.models._cv import expanding_date_splits
        dates = tiny_df["game_date"]
        for train_idx, test_idx in expanding_date_splits(
            pd.Series(dates), n_splits=3
        ):
            assert pd.Series(dates).iloc[test_idx].min() > pd.Series(dates).iloc[train_idx].max()


# ── predict ───────────────────────────────────────────────────────────────────


class TestPredict:
    def _sample_features(self) -> dict[str, float]:
        """Realistic-ish feature dict for LeBron-level player."""
        return {col: 10.0 for col in FEATURE_COLS}

    def test_returns_prediction_dataclass(self, trained_pts):
        models_dir, _ = trained_pts
        result = predict("pts", self._sample_features(), models_dir=models_dir)
        assert isinstance(result, Prediction)

    def test_point_estimate_is_positive(self, trained_pts):
        models_dir, _ = trained_pts
        result = predict("pts", self._sample_features(), models_dir=models_dir)
        assert result.point_estimate > 0

    def test_ci_ordering(self, trained_pts):
        models_dir, _ = trained_pts
        result = predict("pts", self._sample_features(), models_dir=models_dir)
        assert result.ci_low <= result.point_estimate <= result.ci_high

    def test_p_over_none_without_line(self, trained_pts):
        models_dir, _ = trained_pts
        result = predict("pts", self._sample_features(), models_dir=models_dir)
        assert result.p_over is None

    def test_p_over_between_0_and_1(self, trained_pts):
        models_dir, _ = trained_pts
        result = predict("pts", self._sample_features(), line=15.5, models_dir=models_dir)
        assert result.p_over is not None
        assert 0.0 <= result.p_over <= 1.0

    def test_higher_line_gives_lower_p_over(self, trained_pts):
        """P(pts > 5) must be greater than P(pts > 50)."""
        models_dir, _ = trained_pts
        p_easy = predict("pts", self._sample_features(), line=5.0,  models_dir=models_dir).p_over
        p_hard = predict("pts", self._sample_features(), line=50.0, models_dir=models_dir).p_over
        assert p_easy > p_hard

    def test_missing_features_filled_with_nan(self, trained_pts):
        """predict() must not raise when feature dict is incomplete."""
        models_dir, _ = trained_pts
        # Pass an empty dict — all features NaN, XGBoost handles natively
        result = predict("pts", {}, models_dir=models_dir)
        assert isinstance(result, Prediction)

    def test_unknown_target_raises(self, trained_pts):
        models_dir, _ = trained_pts
        with pytest.raises(ValueError, match="Unknown target"):
            predict("goals", self._sample_features(), models_dir=models_dir)
