from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NBA PropCast API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# Routers wired in Week 3:
# from app.routers import projections, players, backtest
# app.include_router(projections.router, prefix="/projections")
# app.include_router(players.router, prefix="/players")
# app.include_router(backtest.router, prefix="/backtest")
