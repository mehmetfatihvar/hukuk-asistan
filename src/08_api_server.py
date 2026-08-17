"""
08_api_server.py
================
FastAPI service that exposes the RAG search engine over HTTP.

Endpoints:
  POST /search   → {"query": "...", "top_k": 5} → {"query": ..., "results": [...]}
  GET  /health   → {"status": "ok"} (or "unavailable" if the index isn't loaded)
  GET  /stats    → {"total_chunks": X, "model": "...", "vector_dim": 384}

Run:
    uvicorn 08_api_server:app --reload --host 0.0.0.0 --port 8000
  or:
    python src/08_api_server.py

Test:
    curl -X POST "http://localhost:8000/search" \
      -H "Content-Type: application/json" \
      -d '{"query": "elektrik kaçağında ceza", "top_k": 5}'
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config

# ---- logging -------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hukuk-asistan")

# ---- import the sibling module whose filename starts with a digit --------- #
_spec = importlib.util.spec_from_file_location(
    "search_engine", Path(__file__).with_name("07_search_engine.py")
)
_search_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_search_module)  # type: ignore[union-attr]
SearchEngine = _search_module.SearchEngine


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Hukuki konu / soru")
    top_k: int = Field(default=config.TOP_K, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: list


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Hukuk Asistanı — Turkish Legal RAG",
    description="Yargıtay kararları üzerinde anlamsal arama (semantic search).",
    version="1.0.0",
)

search_engine: SearchEngine | None = None


@app.on_event("startup")
def _startup() -> None:
    global search_engine
    try:
        logger.info("Loading search engine…")
        search_engine = SearchEngine()  # loads index + mapping + model
        logger.info("Search engine ready (%s chunks).",
                    search_engine.get_stats()["total_chunks"])
    except Exception as exc:  # noqa: BLE001 - surfaced via /health
        logger.error("Failed to load search engine: %s", exc)
        search_engine = None


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not loaded.")
    try:
        results = search_engine.search(request.query, request.top_k)
        logger.info("search q=%r top_k=%d -> %d results",
                    request.query, request.top_k, len(results))
        return {"query": request.query, "results": results}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("search failed")
        raise HTTPException(status_code=500, detail=f"Search error: {exc}")


@app.get("/health")
async def health():
    if search_engine is None:
        return {"status": "unavailable", "index_loaded": False}
    return {"status": "ok", "index_loaded": True}


@app.get("/stats")
async def stats():
    if search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not loaded.")
    return search_engine.get_stats()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
