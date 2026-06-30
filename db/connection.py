"""SQLAlchemy engine and session factory.

The connection URL is resolved from the environment:
  RECSYS_DB_URL  — overrides everything (used in tests to point at test DB)
  RECSYS_ENV     — when 'test', appends '_test' to the default DB name

Default URL: postgresql+psycopg2://postgres:postgres@localhost:5432/movielens
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator

from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session


def _build_url() -> str:
    explicit = os.environ.get("RECSYS_DB_URL")
    if explicit:
        return explicit
    base = "postgresql+psycopg2://postgres:postgres@localhost:5432/movielens"
    if os.environ.get("RECSYS_ENV") == "test":
        base = base + "_test"
    return base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = _build_url()
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=os.environ.get("RECSYS_SQL_ECHO", "0") == "1",
    )


def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Returns True if the DB is reachable."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
