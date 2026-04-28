"""Full backtest pipeline: prop_lines ✕ game_logs → model predictions → metrics.

Two modes
---------
live    Uses real prop_lines rows stored in the DB.
        Requires running dk_scraper daily for 30+ game days first.

sim     Generates synthetic-but-realistic lines from the 2023-24 outcomes so
        the full pipeline can be exercised immediately.  Lines are sampled
        from (season_avg ± 0.5 unit) with standard -110/-110 juice — not
        cherry-picked, just representative.  Results are labeled SIMULATED
        in all output so there is no ambiguity.

Usage:
    uv run python -m propcast.backtest.run --mode sim
    uv run python -m propcast.backtest.run --mode live
    uv run python -m propcast.backtest.run --mode live --season 2024-25
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from propcast.ingest.schema import GameLog, PropLine, make_engine
from propcast.features import FEATURE_PATH
from propcast.models.train import FEATURE_COLS
from propcast.models.predict import predict
from .metrics import brier_score, clv, mae, roi, vig_free_prob

logger = logging.getLogger(__name__)

METRICS_PATH = Path(__file__).parents[3] / "data" / "backtest_results.json"


# ── data loaders ──────────────────────────────────────────────────────────────

def _load_game_logs(season: str | None = None) -> pd.DataFrame:
    engine = make_engine()
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


def _load_prop_lines(season: str | None = None) -> pd.DataFrame:
    """Load the *closing* line per (player_name, stat, game_date).

    Closing = the last snapshot fetched before the game started.
    We approximate this by taking max(fetched_at) per group — correct as long
    as the final scrape runs after lines are posted and before tip-off.
    """
    engine = make_engine()
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

    # Keep only the closing snapshot per (player, stat, game_date)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    return (
        df.sort_values("fetched_at")
          .groupby(["player_name", "stat", "game_date"], as_index=False)
          .last()
    )


def _load_features() -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {FEATURE_PATH}. "
            "Run: uv run python -m propcast.features.build"
        )
    return pd.read_parquet(FEATURE_PATH)


# ── simulation ────────────────────────────────────────────────────────────────

def _generate_sim_lines(
    game_logs: pd.DataFrame,
    stats: list[str] = ("pts", "reb", "ast", "fg3m"),
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic prop lines from actual 2023-24 outcomes.

    Method: for each player-game, the line is the player's season rolling
    mean at that point (i.e. what a smart market would post) ± U[-0.5, 0.5]
    rounded to nearest 0.5.  Odds are set at -110/-110 (standard juice).
    This produces an unbiased simulation — the model's edge (if any) comes
    from features, not from cherry-picked lines.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []

    for stat in stats:
        for player_id, grp in game_logs.groupby("player_id"):
            grp = grp.sort_values("game_date").reset_index(drop=True)
            # Rolling mean of prior games (shift(1) = no leakage)
            season_avg = grp[stat].shift(1).expanding(min_periods=3).mean()

            for i, row in grp.iterrows():
                avg = season_avg.iloc[i]
                if pd.isna(avg):
                    continue
                noise = rng.uniform(-0.5, 0.5)
                # Snap to nearest 0.5 (DK typically posts half-point lines)
                line = round((avg + noise) * 2) / 2
                rows.append({
                    "player_name": row["player_name"],
                    "player_id":   int(row["player_id"]),
                    "stat":        stat,
                    "line":        line,
                    "over_odds":   -110,
                    "under_odds":  -110,
                    "game_date":   row["game_date"],
                })

    df = pd.DataFrame(rows)
    logger.info(
        "[sim] Generated %d synthetic prop lines (%d players, %d stats)",
        len(df), df["player_id"].nunique(), df["stat"].nunique(),
    )
    return df


# ── core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    lines_df:  pd.DataFrame,
    logs_df:   pd.DataFrame,
    features:  pd.DataFrame,
    models_dir: Path | None = None,
    *,
    mode_label: str = "live",
) -> dict:
    """Join lines + outcomes + features, run predictions, compute all metrics.

    Returns a nested dict:
        {stat: {brier, clv, mae, roi, n_bets, n_samples}}
    """
    results: dict[str, dict] = {}

    for stat in ("pts", "reb", "ast", "fg3m"):
        stat_lines = lines_df[lines_df["stat"] == stat].copy()
        if stat_lines.empty:
            logger.warning("[%s] No lines for stat=%s — skipping", mode_label, stat)
            continue

        # Join: lines ← game outcomes
        merged = stat_lines.merge(
            logs_df[["player_id", "game_date", stat]].rename(columns={stat: "actual"}),
            on=["player_id", "game_date"],
            how="inner",
        )

        # Join: merged ← features (to get model inputs)
        merged = merged.merge(
            features[["player_id", "game_date"] + FEATURE_COLS],
            on=["player_id", "game_date"],
            how="inner",
        )

        if len(merged) < 10:
            logger.warning("[%s] Too few matched rows for %s (%d) — skipping", mode_label, stat, len(merged))
            continue

        # Run model predictions for each row
        kwargs = {"models_dir": models_dir} if models_dir else {}
        p_overs, point_preds = [], []
        for _, row in merged.iterrows():
            feat_dict = {col: row[col] for col in FEATURE_COLS}
            pred = predict(stat, feat_dict, line=float(row["line"]), **kwargs)
            p_overs.append(pred.p_over)
            point_preds.append(pred.point_estimate)

        p_overs      = np.array(p_overs,      dtype=float)
        point_preds  = np.array(point_preds,  dtype=float)
        actuals      = merged["actual"].to_numpy(dtype=float)
        over_odds    = merged["over_odds"].to_numpy(dtype=float)
        under_odds   = merged["under_odds"].to_numpy(dtype=float)
        outcomes     = (actuals > merged["line"].to_numpy()).astype(float)
        mkt_probs    = vig_free_prob(over_odds, under_odds)

        bs   = brier_score(p_overs, outcomes)
        edge = clv(p_overs, mkt_probs)
        m    = mae(point_preds, actuals)
        r    = roi(p_overs, outcomes, over_odds, threshold=0.52)
        n_bets = int((p_overs > 0.52).sum())

        results[stat] = {
            "brier":     round(bs,   4),
            "clv":       round(edge, 4),
            "mae":       round(m,    3),
            "roi":       round(r,    4),
            "n_bets":    n_bets,
            "n_samples": len(merged),
            "mode":      mode_label,
        }

        logger.info(
            "[%s] stat=%-4s  n=%d  Brier=%.3f  CLV=%+.3f  MAE=%.2f  ROI=%+.2f%%  n_bets=%d",
            mode_label, stat, len(merged), bs, edge, m, r * 100, n_bets,
        )

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_results(results: dict, mode: str) -> None:
    label = "*** SIMULATED LINES ***" if mode == "sim" else "Live DK Closing Lines"
    print(f"\n{'='*60}")
    print(f"  PropCast Backtest — {label}")
    print(f"{'='*60}")
    print(f"{'Stat':<6}  {'N':>5}  {'Brier':>6}  {'CLV':>6}  {'MAE':>6}  {'ROI':>7}  {'Bets':>5}")
    print("-" * 60)
    for stat, m in results.items():
        print(
            f"{stat:<6}  {m['n_samples']:>5}  {m['brier']:>6.3f}  "
            f"{m['clv']:>+6.3f}  {m['mae']:>6.2f}  "
            f"{m['roi']*100:>+6.1f}%  {m['n_bets']:>5}"
        )
    print()
    if mode == "sim":
        print("  NOTE: Lines are synthetic (season rolling avg ± noise, -110/-110)")
        print("  Run with --mode live once real DK lines are collected (30+ days)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="PropCast backtest")
    parser.add_argument("--mode",   choices=["live", "sim"], default="sim")
    parser.add_argument("--season", default=None, metavar="YYYY-YY")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    logs     = _load_game_logs(season=args.season)
    features = _load_features()

    if args.mode == "sim":
        lines = _generate_sim_lines(logs)
        # For sim, player_id is in the synthetic lines already
    else:
        lines = _load_prop_lines(season=args.season)
        if lines.empty:
            print("No prop lines in DB. Run dk_scraper daily then re-run with --mode live.")
            sys.exit(1)

    results = run_backtest(lines, logs, features, mode_label=args.mode)

    if not results:
        print("No results — check that models are trained and features are built.")
        sys.exit(1)

    _print_results(results, mode=args.mode)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    logger.info("Results saved → %s", METRICS_PATH)


if __name__ == "__main__":
    main()
