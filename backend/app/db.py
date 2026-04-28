from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .config import get_settings


@lru_cache(maxsize=1)
def _engine() -> Engine:
    url = get_settings().database_url
    kwargs = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=kwargs)


def get_db() -> Generator[Session, None, None]:
    with Session(_engine()) as session:
        yield session
