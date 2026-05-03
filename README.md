# NBA PropCast

ML-powered NBA player prop projections with market-benchmarked backtesting.

Enter a player + prop type, get a projection, confidence interval, and edge vs the DraftKings closing line — with honest CLV-tracked performance over time.

**Stack:** XGBoost · FastAPI · React · SQLite/PostgreSQL · Render + Vercel

---

## What it does

| Feature | Detail |
|---|---|
| Projections | PTS, REB, AST, 3PM per player per game |
| Model | XGBoost regressor, one per stat, expanding-window time-series CV |
| Calibration | Isotonic regression post-training — no overconfident probabilities |
| Edge calc | Projection vs DraftKings closing line (not opening — closing is honest) |
| Backtest | CLV, Brier score, MAE, ROI vs DK closing lines (sim + live modes) |
| UI | Player search → projection card with CI bar + P(over) + backtest dashboard |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Frontend                         │
│           React + Vite + Tailwind + shadcn/ui           │
│                     (Vercel)                            │
└────────────────────────┬────────────────────────────────┘
                         │ REST
┌────────────────────────▼────────────────────────────────┐
│                     Backend API                         │
│              FastAPI + Pydantic + SQLAlchemy            │
│                     (Render)                            │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
┌──────────▼──────────┐  ┌────────────▼───────────────────┐
│     PostgreSQL      │  │         ML Pipeline             │
│  game logs · lines  │  │  ingest → features → XGBoost   │
│  (Render, SQLite    │  │  → isotonic cal → backtest      │
│   locally)          │  │  (offline, artifacts committed) │
└─────────────────────┘  └────────────────────────────────┘
           ▲                          ▲
           │                          │
┌──────────┴──────────┐  ┌────────────┴───────────────────┐
│      nba_api        │  │      DraftKings v5 API          │
│  game logs +        │  │   closing lines for CLV         │
│  player stats       │  │   backtest                      │
└─────────────────────┘  └────────────────────────────────┘
```

---

## ML Methodology

**No shortcuts:**
- Expanding-window time-series CV only — no `train_test_split`, no data leakage
- Features at time T use only data available before T (`shift(1)` before every rolling window)
- Isotonic calibration fit on OOF predictions before any probabilistic metric is reported
- Backtest vs DK **closing** lines, not opening lines
- Numbers reported as-is — no cherry-picked date ranges

**Feature set (implemented):**
- Rolling means (L5, L10, season) for PTS, REB, AST, 3PM, MIN
- Rolling L5 std for each stat — consistency signal
- Home/away, days rest, back-to-back flag, game number in season

**Backtest modes:**
- `--mode sim` — synthetic lines generated from each player's season rolling average ± noise at -110/-110; exercises the full pipeline immediately
- `--mode live` — real DK closing lines from the DB; requires running `dk_scraper` daily for 30+ game days

---

## Repo Structure

```
nba-propcast/
├── ml-pipeline/          # XGBoost training, backtesting, feature engineering
│   ├── src/propcast/
│   │   ├── ingest/       # nba_api pull (historical + live) + DraftKings scraper
│   │   ├── features/     # rolling stats, context features (home/rest/pace)
│   │   ├── models/       # XGBoost training + isotonic calibration + predict
│   │   └── backtest/     # CLV + Brier + MAE + ROI evaluation
│   └── tests/            # pytest suite (CV logic, train, predict)
├── backend/              # FastAPI REST API
│   └── app/
│       ├── routers/      # /players/search, /predict, /backtest
│       └── config.py     # Pydantic Settings, path resolution
├── frontend/             # React + Vite UI
│   └── src/
│       ├── pages/        # Predict.tsx, Backtest.tsx
│       └── lib/api.ts    # typed fetch client
├── docker-compose.yml    # Local dev: SQLite default, --profile postgres for PG
└── render.yaml           # Render Blueprint for one-click backend deploy
```

---

## Local Setup

**Prerequisites:** Docker, uv (`pip install uv`)

```bash
git clone https://github.com/ralrocks/nba-propcast.git
cd nba-propcast
docker-compose up
```

Backend: `http://localhost:8000` · Frontend: `http://localhost:5173` · API docs: `http://localhost:8000/docs`

**ML pipeline only (no Docker):**

```bash
cd ml-pipeline
uv sync

# Pull historical game logs (2023-24 season)
uv run python -m propcast.ingest.run backfill --seasons 2023-24

# Build feature matrix
uv run python -m propcast.features.build

# Train models (all 4 stats)
uv run python -m propcast.models.train

# Run backtest (sim mode — no DK data required)
uv run python -m propcast.backtest.run --mode sim
```

---

## Build Status

[![CI](https://github.com/ralrocks/nba-propcast/actions/workflows/ci.yml/badge.svg)](https://github.com/ralrocks/nba-propcast/actions/workflows/ci.yml)

4 jobs: ML pipeline tests · Backend tests · Frontend build · Docker smoke test

---

## Backtest Results

Live results require 30+ days of DK closing line data collected via `dk_scraper`.
Sim-mode results (synthetic lines from season rolling avg ± noise, -110/-110) are available after running `uv run python -m propcast.backtest.run --mode sim`.

_Production backtest results will be posted here after the live DK scraper has run through a full slate._

---

## Resume / Portfolio Context

Built to demonstrate:
- **Feature engineering** with time-series discipline — `shift(1)` enforced, leakage check built into the pipeline
- **Calibration** — isotonic regression on OOF predictions, Brier score reported
- **Market benchmarking** — CLV vs DK closing lines, not a toy accuracy metric
- **Full-stack deployment** — containerized (Docker + docker-compose), GitHub Actions CI, Render + Vercel

Not just "built an ML model."
