"""FastAPI entrypoint for Medical Multi-Agent CDSS."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.qdrant_store import get_qdrant_store
from app.db.sqlite import init_db
from app.ingestion.pipeline import get_ingestion_pipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings.ensure_dirs()
    init_db()
    try:
        get_qdrant_store()
    except Exception as e:
        logger.warning("Qdrant init deferred/failed: %s", e)
    # Seed sample KB on first boot if empty
    try:
        result = get_ingestion_pipeline().seed_sample_knowledge()
        logger.info("KB seed: %s", result)
    except Exception as e:
        logger.warning("KB seed skipped: %s", e)

    from app.core.llm import get_llm_client

    llm = get_llm_client()
    logger.info(
        "CDSS API ready | provider=%s | openrouter_active=%s | model=%s | qdrant=%s",
        llm.provider,
        llm.status().get("openrouter_configured"),
        settings.gemini_model,
        settings.qdrant_path,
    )
    if llm.provider == "stub":
        logger.error(
            "No working LLM provider. Set GEMINI_API_KEY in backend/.env and restart."
        )
    yield


app = FastAPI(
    title="Medical Multi-Agent CDSS",
    description=(
        "Local-first multi-agent clinical decision support: "
        "Superior router → one specialist → hybrid RAG → feedback loop."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "Medical Multi-Agent CDSS",
        "docs": "/docs",
        "health": "/api/health",
    }


def run() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    run()
