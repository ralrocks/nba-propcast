"""
Tests for backtest/metrics.py and the DK scraper parser.

Metrics are pure functions — no DB, no network, no models.
Parser tests use a hard-coded DK v5 response fixture.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pytest

from propcast.backtest.metrics import (
    american_to_decimal,
    brier_score,
    calibration_summary,
    clv,
    mae,
    roi,
    vig_free_prob,
)
from propcast.ingest.dk_scraper import _parse_category_response


# ── odds utilities ────────────────────────────────────────────────────────────


class TestAmericanToDecimal:
    def test_minus_110(self):
        # -110 → 1 + 100/110 ≈ 1.909
        result = american_to_decimal(np.array([-110.0]))
        assert abs(result[0] - (1 + 100 / 110)) < 1e-6

    def test_plus_150(self):
        # +150 → 1 + 150/100 = 2.5
        result = american_to_decimal(np.array([150.0]))
        assert abs(result[0] - 2.5) < 1e-6

    def test_even_odds(self):
        # +100 → 2.0
        result = american_to_decimal(np.array([100.0]))
        assert abs(result[0] - 2.0) < 1e-6

    def test_mixed_array(self):
        result = american_to_decimal(np.array([-110.0, 110.0]))
        assert result[0] < result[1]   # -110 pays less than +110


class TestVigFreeProb:
    def test_balanced_book_is_50pct(self):
        # Both sides at -110 → vig-free prob is exactly 0.50
        p = vig_free_prob(np.array([-110.0]), np.array([-110.0]))
        assert abs(p[0] - 0.5) < 1e-9

    def test_favourite_has_higher_prob(self):
        # Over at -150 means the market thinks Over is more likely
        p = vig_free_prob(np.array([-150.0]), np.array([+130.0]))
        assert p[0] > 0.5

    def test_probs_sum_to_one_after_vig_removal(self):
        over  = np.array([-115.0, -120.0, +105.0])
        under = np.array([-105.0,  +100.0, -125.0])
        p_over  = vig_free_prob(over, under)
        p_under = vig_free_prob(under, over)
        np.testing.assert_allclose(p_over + p_under, 1.0, atol=1e-9)


# ── brier score ───────────────────────────────────────────────────────────────


class TestBrierScore:
    def test_perfect_predictions(self):
        probs    = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(probs, outcomes) == 0.0

    def test_random_model(self):
        # Always predict 0.5 → Brier = 0.25
        probs    = np.full(1000, 0.5)
        outcomes = np.array([1.0, 0.0] * 500)
        assert abs(brier_score(probs, outcomes) - 0.25) < 1e-9

    def test_worst_predictions(self):
        # Predict 1 when outcome 0, and vice versa → Brier = 1.0
        probs    = np.array([1.0, 0.0])
        outcomes = np.array([0.0, 1.0])
        assert brier_score(probs, outcomes) == 1.0

    def test_lower_is_better(self):
        good = brier_score(np.array([0.9, 0.1]), np.array([1.0, 0.0]))
        bad  = brier_score(np.array([0.6, 0.4]), np.array([1.0, 0.0]))
        assert good < bad


# ── CLV ───────────────────────────────────────────────────────────────────────


class TestCLV:
    def test_zero_edge_when_equal(self):
        p = np.full(100, 0.52)
        assert clv(p, p) == pytest.approx(0.0)

    def test_positive_clv_when_we_have_edge(self):
        our_probs = np.full(100, 0.55)
        mkt_probs = np.full(100, 0.50)
        assert clv(our_probs, mkt_probs) == pytest.approx(0.05)

    def test_negative_clv_when_market_better(self):
        our_probs = np.full(100, 0.45)
        mkt_probs = np.full(100, 0.50)
        assert clv(our_probs, mkt_probs) == pytest.approx(-0.05)


# ── MAE ───────────────────────────────────────────────────────────────────────


class TestMAE:
    def test_perfect(self):
        assert mae(np.array([10.0, 20.0]), np.array([10.0, 20.0])) == 0.0

    def test_symmetric(self):
        a = np.array([10.0, 20.0])
        b = np.array([15.0, 25.0])
        assert mae(a, b) == mae(b, a)

    def test_value(self):
        preds   = np.array([20.0, 30.0])
        actuals = np.array([22.0, 27.0])
        assert mae(preds, actuals) == pytest.approx(2.5)


# ── ROI ───────────────────────────────────────────────────────────────────────


class TestROI:
    def test_no_bets_when_nothing_above_threshold(self):
        probs    = np.full(100, 0.48)
        outcomes = np.ones(100)
        odds     = np.full(100, -110.0)
        assert roi(probs, outcomes, odds, threshold=0.52) == 0.0

    def test_positive_roi_when_always_win(self):
        probs    = np.full(10, 0.60)
        outcomes = np.ones(10)          # always win
        odds     = np.full(10, -110.0)  # -110 pays 100/110 ≈ 0.909
        r = roi(probs, outcomes, odds, threshold=0.52)
        assert r == pytest.approx(100 / 110, rel=1e-4)

    def test_negative_roi_when_always_lose(self):
        probs    = np.full(10, 0.60)
        outcomes = np.zeros(10)         # always lose
        odds     = np.full(10, -110.0)
        r = roi(probs, outcomes, odds, threshold=0.52)
        assert r == pytest.approx(-1.0)


# ── calibration summary ───────────────────────────────────────────────────────


class TestCalibrationSummary:
    def test_returns_expected_keys(self):
        probs    = np.linspace(0.05, 0.95, 50)
        outcomes = (probs > 0.5).astype(float)
        result = calibration_summary(probs, outcomes, n_bins=5)
        assert set(result.keys()) == {"bin_center", "mean_predicted", "mean_actual", "count"}

    def test_counts_sum_to_n(self):
        n        = 100
        probs    = np.random.default_rng(0).uniform(0, 1, n)
        outcomes = (probs > 0.5).astype(float)
        result = calibration_summary(probs, outcomes, n_bins=10)
        assert result["count"].sum() == n


# ── DK scraper parser ─────────────────────────────────────────────────────────


def _make_dk_payload(player: str = "LeBron James", line: float = 22.5) -> dict:
    """Minimal DK v5 API response fixture for one player prop."""
    return {
        "eventGroup": {
            "eventGroupId": 42648,
            "events": [
                {
                    "eventId": 99001,
                    "name":    "Lakers vs Warriors",
                    "startDate": "2024-01-15T00:10:00.000Z",
                }
            ],
            "offerCategories": [
                {
                    "offerCategoryId": 583,
                    "name": "Player Props",
                    "offerSubcategoryDescriptors": [
                        {
                            "subcategoryId": 4517,
                            "name": "Player Points",
                            "offerSubcategory": {
                                "offers": [
                                    [
                                        {
                                            "eventId":      99001,
                                            "label":        "Over",
                                            "line":         line,
                                            "oddsAmerican": "-115",
                                            "participants": [{"name": player, "type": "Player"}],
                                        },
                                        {
                                            "eventId":      99001,
                                            "label":        "Under",
                                            "line":         line,
                                            "oddsAmerican": "-105",
                                            "participants": [{"name": player, "type": "Player"}],
                                        },
                                    ]
                                ]
                            },
                        }
                    ],
                }
            ],
        }
    }


class TestDKParser:
    _TS = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_parses_player_name(self):
        snaps = _parse_category_response(_make_dk_payload(), self._TS)
        assert len(snaps) == 1
        assert snaps[0].player_name == "LeBron James"

    def test_parses_line_and_odds(self):
        snaps = _parse_category_response(_make_dk_payload(line=22.5), self._TS)
        assert snaps[0].line       == 22.5
        assert snaps[0].over_odds  == -115
        assert snaps[0].under_odds == -105

    def test_maps_subcategory_name_to_stat_code(self):
        snaps = _parse_category_response(_make_dk_payload(), self._TS)
        assert snaps[0].stat == "pts"

    def test_parses_game_date(self):
        snaps = _parse_category_response(_make_dk_payload(), self._TS)
        assert snaps[0].game_date == date(2024, 1, 15)

    def test_unknown_subcategory_returns_empty(self):
        payload = _make_dk_payload()
        payload["eventGroup"]["offerCategories"][0]["offerSubcategoryDescriptors"][0]["name"] = "Touchdowns"
        snaps = _parse_category_response(payload, self._TS)
        assert snaps == []

    def test_missing_event_id_skips_row(self):
        payload = _make_dk_payload()
        # Remove the event so eventId won't resolve to a date
        payload["eventGroup"]["events"] = []
        snaps = _parse_category_response(payload, self._TS)
        assert snaps == []
