"""Database engine, session management, and pgvector registration."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


@event.listens_for(engine, "connect")
def _register_vector_type(dbapi_connection, _connection_record) -> None:
    """Register the pgvector type adapter on every new psycopg2 connection."""
    from pgvector.psycopg2 import register_vector

    register_vector(dbapi_connection)


def init_db() -> None:
    """Create the `vector` extension (if missing) and all ORM tables.

    Safe to call multiple times - both operations are idempotent.
    """
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
