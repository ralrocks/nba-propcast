from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import backtest, players, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure DB tables exist — idempotent with SQLite, required on fresh Postgres.
    from propcast.ingest.schema import init_db, make_engine
    init_db(make_engine(get_settings().database_url))
    yield


settings = get_settings()

app = FastAPI(title="NBA PropCast API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(players.router)
app.include_router(predict.router)
app.include_router(backtest.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}
