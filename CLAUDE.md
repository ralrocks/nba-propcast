# NBA PropCast

Solo ML portfolio project. 4-week deadline, full-time focus. Backend/ML engineer resume target.

## What this is
Web app projecting NBA player props (PTS, REB, AST, 3PM) using XGBoost. Users enter player + prop, get projection, CI, and edge vs sportsbook line. Backtest dashboard shows performance vs DK closing lines with CLV.

## Stack (non-negotiable, do not relitigate)
- **Data**: nba_api (V3 endpoints, 0.6s delays), DraftKings v5 JSON endpoint (direct HTTP, not HTML scraping), historical backfill from shufinskiy/nba_data
- **ML**: XGBoost regressors, one per stat. Time-series CV only (NEVER random splits). Isotonic calibration post-training. Metrics: Brier + MAE + CLV
- **Backend**: FastAPI + Pydantic + SQLAlchemy. SQLite dev, Postgres deploy
- **Frontend**: React + Vite + Tailwind + shadcn/ui + Recharts. No Next.js
- **Infra**: Docker + docker-compose. Railway (backend + Postgres), Vercel (frontend). GitHub Actions for CI
- **Python deps**: uv only. Use `uv add`, `uv run`, `uv sync`. Not pip, not poetry.

## Working style
- Give exact commands, not paragraphs
- Explain the "why" for architecture decisions — I'm learning, not just shipping
- Flag overengineering and scope creep aggressively (full-time focus = easier to rabbit-hole)
- Default concise; expand when asked
- Show the plan before writing code for anything non-trivial

## ML rigor (hard rules — do not cut corners)
- Time-series CV only. If you reach for train_test_split, stop
- No target leakage. Features at time T may only use data available before T
- Calibrate (isotonic) before reporting any probabilistic metric
- Backtest against real DK closing lines, not opening lines
- Report honest numbers. 52.1% is 52.1% — never round up, never cherry-pick a date range

## Session rules (for Claude Code)
- Before editing multiple files or adding a dependency, show the plan
- Use `uv add` for new deps, never `pip install`
- Run tests before claiming something works — "it compiles" ≠ "it works"
- If you're about to reach for a library not in the stack above, surface the swap and the tradeoff — don't silently substitute
- Do not modify CLAUDE.md or README.md without asking
- Do not touch .github/workflows/ without asking — CI changes get their own review
- After any code change, show me what to run to verify it works

## Resume bullet target
Must prove: feature engineering + calibration + market-benchmarked backtesting + deployment. Not "built an ML model."

## Repo structure
- backend/ — FastAPI app
- ml-pipeline/ — training, backtesting, feature engineering
- frontend/ — Vite + React
- .github/workflows/ — CI

## Deliverables by Week 4
- Clean monorepo with the structure above
- README: architecture diagram, methodology, honest backtest results, demo link, 15s GIF
- Deployed full-stack (Railway + Vercel)
- docker-compose local dev
- pytest for backend, smoke tests for ML
- 2-min Loom walkthrough
