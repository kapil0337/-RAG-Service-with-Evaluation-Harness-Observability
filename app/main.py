"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.db import init_db

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Ensure the pgvector extension and ORM tables exist before serving traffic."""
    init_db()
    yield


app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation over user-uploaded documents, powered by Groq.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui_index() -> FileResponse:
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")
