# NBA PropCast

ML-powered NBA player prop projections with market-benchmarked backtesting.

Enter a player + prop type, get a projection, confidence interval, and edge vs the DraftKings closing line — with honest CLV-tracked performance over time.

**Stack:** XGBoost · FastAPI · React · PostgreSQL · Railway + Vercel

---

## What it does

| Feature | Detail |
|---|---|
| Projections | PTS, REB, AST, 3PM per player per game |
| Model | XGBoost regressor, one per stat, time-series CV only |
| Calibration | Isotonic regression post-training — no overconfident probabilities |
| Edge calc | Projection vs DraftKings closing line (not opening — closing is honest) |
| Backtest | Historical CLV tracking with Brier score + MAE |
| UI | Player + prop search → projection card with CI bar + backtest sparkline |

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
│                     (Railway)                           │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
┌──────────▼──────────┐  ┌────────────▼───────────────────┐
│     PostgreSQL      │  │         ML Pipeline             │
│  game logs · lines  │  │  ingest → features → XGBoost   │
│  projections · bets │  │  → isotonic cal → backtest      │
│     (Railway)       │  │  (offline, artifacts in DB)    │
└─────────────────────┘  └────────────────────────────────┘
           ▲                          ▲
           │                          │
┌──────────┴──────────┐  ┌────────────┴───────────────────┐
│      nba_api        │  │      DraftKings v5 API          │
│  V3 game logs +     │  │   closing lines for CLV         │
│  player stats       │  │   backtest                      │
└─────────────────────┘  └────────────────────────────────┘
```

---

## ML Methodology

**No shortcuts:**
- Time-series cross-validation only — no `train_test_split`
- Features at time T use only data available before T (no leakage)
- Isotonic calibration before any probabilistic metric is reported
- Backtest vs DK **closing** lines, not opening lines
- Numbers are reported as-is — no cherry-picked date ranges

**Feature set (planned):**
- Rolling averages (L5, L10, season) for each stat
- Home/away split, rest days, pace, opponent defensive rating
- Usage rate, minutes trend
- Vegas total + spread as market priors

---

## Repo Structure

```
nba-propcast/
├── ml-pipeline/          # XGBoost training, backtesting, feature engineering
│   ├── src/propcast/
│   │   ├── ingest/       # nba_api + DraftKings data pulls
│   │   ├── features/     # rolling stats, opponent adjustments
│   │   ├── models/       # XGBoost training + isotonic calibration
│   │   └── backtest/     # CLV + Brier score evaluation
│   └── tests/
├── backend/              # FastAPI REST API
│   └── app/
│       ├── routers/      # /projections, /players, /backtest
│       ├── models/       # SQLAlchemy ORM
│       └── schemas/      # Pydantic request/response
├── frontend/             # React + Vite UI
│   └── src/
│       ├── components/   # PlayerSearch, ProjectionCard, BacktestChart
│       └── pages/
└── docker-compose.yml    # Local dev: backend + frontend + postgres
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
uv run python src/propcast/ingest/smoke_test.py   # verify nba_api connection
```

---

## Progress

| Week | Focus | Status |
|------|-------|--------|
| 1 | Data pipeline · nba_api ingest · raw parquet storage | **In progress** |
| 2 | Feature engineering · XGBoost training · time-series CV | Upcoming |
| 3 | FastAPI backend · PostgreSQL schema · projections endpoint | Upcoming |
| 4 | React frontend · backtesting dashboard · Railway + Vercel deploy | Upcoming |

---

## Backtest Results

_Will be populated after Week 2 training run. Target: honest CLV > 0 on held-out 2023-24 season._

---

## Resume / Portfolio Context

Built to demonstrate:
- **Feature engineering** with time-series discipline (no leakage)
- **Calibration** — isotonic regression, Brier score reporting
- **Market benchmarking** — CLV vs DK closing lines, not a toy accuracy metric
- **Full-stack deployment** — containerized, CI/CD, production infra

Not just "built an ML model."
