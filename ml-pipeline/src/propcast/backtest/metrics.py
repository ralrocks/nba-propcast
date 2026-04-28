"""Pure metric functions for prop-betting model evaluation.

All functions are stateless and take numpy arrays — no DB, no I/O.
This makes them trivially testable and reusable in CV, backtest, and live
monitoring contexts.

Metric definitions
------------------
Brier score     : mean((p_over - outcome)^2).  Lower = better.
                  0.25 = uninformative (always predict 0.5)
                  < 0.20 is strong for player props

CLV             : mean(our_prob - market_no_vig_prob)
                  Positive CLV means we're finding edges the market hasn't
                  priced in.  This is the gold-standard proof of model value.

MAE             : mean(|point_estimate - actual|)
                  Same as CV MAE but computed on games we had a DK line for.

ROI             : simulated return on investment if we bet flat $1 whenever
                  our edge > threshold.  Presented as % (0.05 = +5 %).
"""
from __future__ import annotations

import numpy as np


# ── odds utilities ────────────────────────────────────────────────────────────

def american_to_decimal(american: np.ndarray) -> np.ndarray:
    """Convert American odds array to decimal odds array."""
    american = np.asarray(american, dtype=float)
    pos = american >= 0
    out = np.empty_like(american)
    out[pos]  = american[pos]  / 100.0 + 1.0
    out[~pos] = 100.0 / np.abs(american[~pos]) + 1.0
    return out


def vig_free_prob(over_odds: np.ndarray, under_odds: np.ndarray) -> np.ndarray:
    """Fair (no-vig) probability of the Over outcome.

    We normalise the raw implied probabilities so they sum to 1, stripping
    the bookmaker margin.  This is the correct baseline for CLV computation.
    """
    p_over_raw  = 1.0 / american_to_decimal(over_odds)
    p_under_raw = 1.0 / american_to_decimal(under_odds)
    total = p_over_raw + p_under_raw
    return p_over_raw / total


# ── core metrics ──────────────────────────────────────────────────────────────

def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes.

    Args:
        probs:    Predicted P(over line), shape (N,), values in [0, 1].
        outcomes: Actual result — 1 if player exceeded the line, 0 otherwise.

    Returns:
        Scalar Brier score in [0, 1].  Lower is better.  0.25 = random.
    """
    probs    = np.asarray(probs,    dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def clv(
    our_probs:    np.ndarray,
    market_probs: np.ndarray,
) -> float:
    """Mean edge our model holds over the vig-free market closing probability.

    Args:
        our_probs:    Our model's P(over) estimates, shape (N,).
        market_probs: No-vig market P(over) at close, shape (N,).

    Returns:
        Scalar CLV.  > 0 means the model systematically finds value.
        A CLV of 0.02 means we're finding ~2 % edge on average.
    """
    return float(np.mean(np.asarray(our_probs) - np.asarray(market_probs)))


def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean absolute error between continuous predictions and actual values."""
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(actuals))))


def roi(
    our_probs:  np.ndarray,
    outcomes:   np.ndarray,
    over_odds:  np.ndarray,
    *,
    threshold:  float = 0.52,
) -> float:
    """Simulated flat-bet ROI when our_prob > threshold.

    Returns the fractional ROI (e.g. 0.04 = +4 %).  Bets are sized at $1
    each; payout uses actual DK over odds.  Returns 0.0 if no bets placed.
    """
    mask = np.asarray(our_probs) > threshold
    if not mask.any():
        return 0.0

    dec_odds = american_to_decimal(np.asarray(over_odds)[mask])
    wins      = np.asarray(outcomes)[mask].astype(bool)

    profit = np.where(wins, dec_odds - 1.0, -1.0)   # profit per $1 staked
    return float(profit.mean())


def calibration_summary(
    probs:    np.ndarray,
    outcomes: np.ndarray,
    n_bins:   int = 10,
) -> dict[str, np.ndarray]:
    """Bin predictions and compute mean predicted vs mean actual per bin.

    Returns dict with keys 'bin_center', 'mean_predicted', 'mean_actual',
    'count' — ready to plot as a calibration curve.
    """
    probs    = np.asarray(probs,    dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    bins       = np.linspace(0, 1, n_bins + 1)
    bin_idx    = np.digitize(probs, bins) - 1
    bin_idx    = np.clip(bin_idx, 0, n_bins - 1)

    centers, mean_pred, mean_act, counts = [], [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        centers.append((bins[b] + bins[b + 1]) / 2)
        mean_pred.append(probs[mask].mean())
        mean_act.append(outcomes[mask].mean())
        counts.append(mask.sum())

    return {
        "bin_center":    np.array(centers),
        "mean_predicted": np.array(mean_pred),
        "mean_actual":   np.array(mean_act),
        "count":         np.array(counts),
    }
