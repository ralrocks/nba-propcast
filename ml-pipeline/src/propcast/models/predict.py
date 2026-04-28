"""Load trained artifacts and produce predictions for one player-game.

Prediction pipeline:
  1. XGBoost → raw point estimate
  2. IsotonicRegression calibrator → corrected point estimate
  3. OOF residual distribution → 80 % CI  +  P(stat > line)

P(stat > line) uses the empirical CDF of OOF residuals — no distributional
assumptions.  This is valid and honest as long as the residuals are
approximately stationary across the season (a reasonable assumption).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from .train import FEATURE_COLS, MODELS_DIR, TARGETS


@dataclass(frozen=True)
class Prediction:
    target: str
    point_estimate: float       # calibrated prediction
    ci_low: float               # 80 % CI lower bound
    ci_high: float              # 80 % CI upper bound
    p_over: float | None        # P(stat > line); None when no line provided


@lru_cache(maxsize=len(TARGETS))
def _load_artifacts(target: str, models_dir: Path = MODELS_DIR) -> tuple:
    """Load and cache model + calibrator + residuals for *target*."""
    artifact_path = models_dir / f"{target}.joblib"
    residual_path = models_dir / f"{target}_residuals.npy"

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"No trained model for '{target}'. Run: uv run python -m propcast.models.train"
        )

    bundle     = joblib.load(artifact_path)
    residuals  = np.load(residual_path)

    return bundle["model"], bundle["calibrator"], residuals


def predict(
    target: str,
    features: dict[str, float],
    *,
    line: float | None = None,
    ci_pct: float = 0.80,
    models_dir: Path = MODELS_DIR,
) -> Prediction:
    """Return a calibrated prediction and confidence interval for one player-game.

    Args:
        target:     One of 'pts', 'reb', 'ast', 'fg3m'.
        features:   Dict of feature name → value matching FEATURE_COLS.
                    Missing keys are filled with NaN (XGBoost handles natively).
        line:       DraftKings prop line (e.g. 22.5).  When provided, P(over)
                    is computed; otherwise p_over is None.
        ci_pct:     Width of the confidence interval (default 80 %).
        models_dir: Override the default models directory (useful for tests).

    Returns:
        Prediction dataclass with point_estimate, ci_low, ci_high, p_over.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}'. Choose from {TARGETS}.")

    model, calibrator, residuals = _load_artifacts(target, models_dir)

    x = np.array(
        [features.get(col, np.nan) for col in FEATURE_COLS],
        dtype=np.float32,
    ).reshape(1, -1)

    raw_pred       = float(model.predict(x)[0])
    point_estimate = float(calibrator.predict([raw_pred])[0])

    # CI from empirical residuals
    tail = (1 - ci_pct) / 2
    ci_low  = point_estimate + float(np.percentile(residuals, tail * 100))
    ci_high = point_estimate + float(np.percentile(residuals, (1 - tail) * 100))

    # P(stat > line) via empirical residual CDF
    # actual = prediction + residual  →  actual > line ↔ residual > line − prediction
    p_over: float | None = None
    if line is not None:
        threshold = line - point_estimate
        p_over = float(np.mean(residuals > threshold))

    return Prediction(
        target=target,
        point_estimate=round(point_estimate, 2),
        ci_low=round(ci_low, 2),
        ci_high=round(ci_high, 2),
        p_over=round(p_over, 4) if p_over is not None else None,
    )


def predict_multi(
    features: dict[str, float],
    *,
    lines: dict[str, float] | None = None,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Prediction]:
    """Convenience wrapper — predict all stats for one player-game in one call."""
    lines = lines or {}
    return {
        target: predict(target, features, line=lines.get(target), models_dir=models_dir)
        for target in TARGETS
    }
