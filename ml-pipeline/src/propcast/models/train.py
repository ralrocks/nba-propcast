"""XGBoost training pipeline with expanding-window CV and isotonic calibration.

Pipeline per stat (pts / reb / ast / fg3m):
  1. Expanding-window CV → per-fold MAE / RMSE + OOF predictions
  2. IsotonicRegression fit on OOF (prediction, actual) pairs — corrects any
     monotone bias in XGBoost's raw output (e.g. under-predicting top scorers)
  3. OOF residuals (after calibration) saved → used for confidence intervals
     and empirical P(over line) at prediction time
  4. Final XGBoost fit on the full dataset + calibrator saved as one artifact

Usage:
    uv run python -m propcast.models.train
    uv run python -m propcast.models.train --target pts
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor

from propcast.features import FEATURE_PATH
from ._cv import expanding_date_splits

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parents[3] / "data" / "models"

TARGETS: list[str] = ["pts", "reb", "ast", "fg3m"]

# All features produced by features/build.py that are valid model inputs.
# These must be available-before-game-time (no current-game stats).
FEATURE_COLS: list[str] = [
    # rolling point-estimate features
    "pts_l5_mean",  "pts_l10_mean",  "pts_season_mean",
    "reb_l5_mean",  "reb_l10_mean",  "reb_season_mean",
    "ast_l5_mean",  "ast_l10_mean",  "ast_season_mean",
    "fg3m_l5_mean", "fg3m_l10_mean", "fg3m_season_mean",
    "min_l5_mean",  "min_l10_mean",  "min_season_mean",
    # rolling variance — consistency signal
    "pts_l5_std", "reb_l5_std", "ast_l5_std", "fg3m_l5_std",
    # game context
    "is_home", "days_rest", "is_back_to_back", "game_number",
]

_XGB_PARAMS: dict = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    objective="reg:squarederror",
    tree_method="hist",     # fastest CPU training method
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


@dataclass
class FoldMetrics:
    fold: int
    n_train: int
    n_test: int
    mae: float
    rmse: float


@dataclass
class TrainResult:
    target: str
    fold_metrics: list[FoldMetrics]
    cv_mae: float           # mean MAE across folds
    cv_rmse: float          # mean RMSE across folds
    oof_residuals: np.ndarray = field(repr=False)   # actual − calibrated_pred


# ── core ──────────────────────────────────────────────────────────────────────


def train_one(
    df: pd.DataFrame,
    target: str,
    *,
    n_splits: int = 5,
    xgb_params: dict | None = None,
    models_dir: Path = MODELS_DIR,
) -> TrainResult:
    """CV → calibrate → final fit → save artifacts for one target stat.

    XGBoost handles NaN feature values natively (learns the optimal branch
    direction for missing values), so no imputation is needed.
    """
    params = {**_XGB_PARAMS, **(xgb_params or {})}

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)
    dates = df["game_date"]

    # ── expanding-window CV ────────────────────────────────────────────────
    fold_metrics: list[FoldMetrics] = []
    oof_preds = np.full(len(df), np.nan, dtype=np.float64)

    for fold, (train_idx, test_idx) in enumerate(
        expanding_date_splits(dates, n_splits=n_splits)
    ):
        model = XGBRegressor(**params)
        model.fit(X[train_idx], y[train_idx])

        preds = model.predict(X[test_idx])
        oof_preds[test_idx] = preds

        mae  = float(np.mean(np.abs(preds - y[test_idx])))
        rmse = float(np.sqrt(np.mean((preds - y[test_idx]) ** 2)))
        fold_metrics.append(FoldMetrics(fold, len(train_idx), len(test_idx), mae, rmse))

        logger.info(
            "[%s] fold %d  train=%d  test=%d  MAE=%.2f  RMSE=%.2f",
            target, fold, len(train_idx), len(test_idx), mae, rmse,
        )

    # ── isotonic calibration on OOF predictions ────────────────────────────
    # IsotonicRegression corrects monotone bias: if XGBoost under-predicts
    # high scorers it will learn to shift those predictions upward.
    has_oof = ~np.isnan(oof_preds)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_preds[has_oof], y[has_oof])

    calibrated_oof = calibrator.predict(oof_preds[has_oof])
    oof_residuals  = (y[has_oof] - calibrated_oof).astype(np.float64)

    # ── final fit on full dataset ──────────────────────────────────────────
    final_model = XGBRegressor(**params)
    final_model.fit(X, y)

    # ── persist artifacts ──────────────────────────────────────────────────
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": final_model, "calibrator": calibrator},
        models_dir / f"{target}.joblib",
    )
    np.save(models_dir / f"{target}_residuals.npy", oof_residuals)

    cv_mae  = float(np.mean([f.mae  for f in fold_metrics]))
    cv_rmse = float(np.mean([f.rmse for f in fold_metrics]))

    logger.info("[%s] CV done — mean MAE=%.2f  mean RMSE=%.2f", target, cv_mae, cv_rmse)

    return TrainResult(target, fold_metrics, cv_mae, cv_rmse, oof_residuals)


def train_all(
    feature_path: Path = FEATURE_PATH,
    models_dir: Path = MODELS_DIR,
    targets: list[str] | None = None,
) -> dict[str, TrainResult]:
    """Train one model per target stat and write a combined metrics.json."""
    df = pd.read_parquet(feature_path)
    logger.info("Loaded feature matrix: %s rows × %s cols", *df.shape)

    results: dict[str, TrainResult] = {}
    for target in (targets or TARGETS):
        results[target] = train_one(df, target, models_dir=models_dir)

    _save_metrics(results, models_dir)
    return results


def _save_metrics(results: dict[str, TrainResult], models_dir: Path) -> None:
    payload = {
        target: {
            "cv_mae":  r.cv_mae,
            "cv_rmse": r.cv_rmse,
            "folds": [asdict(f) for f in r.fold_metrics],
        }
        for target, r in results.items()
    }
    path = models_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Metrics saved → %s", path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _print_results(results: dict[str, TrainResult]) -> None:
    print(f"\n{'Stat':<8}  {'CV MAE':>8}  {'CV RMSE':>9}")
    print("-" * 30)
    for stat, r in results.items():
        print(f"{stat:<8}  {r.cv_mae:>8.2f}  {r.cv_rmse:>9.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PropCast XGBoost models")
    parser.add_argument(
        "--target", choices=TARGETS, default=None,
        help="Train a single stat (default: all)"
    )
    parser.add_argument(
        "--feature-path", type=Path, default=FEATURE_PATH,
        help="Path to feature matrix parquet"
    )
    args = parser.parse_args()

    results = train_all(
        feature_path=args.feature_path,
        targets=[args.target] if args.target else None,
    )
    _print_results(results)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    main()
