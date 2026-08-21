"""
07_search_engine.py
===================
The retrieval core of the RAG system. Loads the FAISS index, the chunk mapping
and the embedding model, then answers natural-language queries with the most
relevant chunks, reranked by section importance.

Exposes a reusable `SearchEngine` class (imported by 08_api_server.py and
09_benchmark_test.py) and a small CLI / self-test.

    Inputs : data/processed/index.faiss
             data/processed/chunk_mapping.pkl

Reranking (section weights):
    KARAR   (ruling)   x2.0
    KANUN   (law)      x1.3
    GEREKÇE (reasoning)x1.0
    OYAL    (facts)    x0.8
    final_score = similarity_score * section_weight

Run:
    python src/07_search_engine.py            # runs 5 built-in test queries
    python src/07_search_engine.py "kira feshi"
"""

from __future__ import annotations

import pickle
import sys

import numpy as np

import config


class SearchEngine:
    """Load the index + model once, then serve many queries."""

    def __init__(self, autoload: bool = True) -> None:
        self._index = None
        self._mapping: dict[int, dict] = {}
        self._model = None
        self._loaded = False
        if autoload:
            self.load()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> "SearchEngine":
        if self._loaded:
            return self

        import faiss
        from sentence_transformers import SentenceTransformer

        if not config.FAISS_INDEX.exists():
            raise FileNotFoundError(
                f"FAISS index missing at {config.FAISS_INDEX}. "
                "Run 06_faiss_index.py first."
            )
        if not config.CHUNK_MAPPING_PKL.exists():
            raise FileNotFoundError(
                f"Chunk mapping missing at {config.CHUNK_MAPPING_PKL}. "
                "Run 06_faiss_index.py first."
            )

        self._index = faiss.read_index(str(config.FAISS_INDEX))
        with open(config.CHUNK_MAPPING_PKL, "rb") as f:
            self._mapping = pickle.load(f)
        self._model = SentenceTransformer(config.ST_MODEL_NAME)
        self._loaded = True
        return self

    # ------------------------------------------------------------------ #
    # Query embedding
    # ------------------------------------------------------------------ #
    def embed_query(self, query: str) -> np.ndarray:
        """Return the 384-dim (normalised) embedding for `query`."""
        if not self._loaded:
            self.load()
        # E5 models need the "query: " prefix (empty for other models) so the
        # query is embedded in the same space as the "passage: "-prefixed chunks.
        text = config.QUERY_PREFIX + query
        vec = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
        ).astype("float32")
        return vec

    # ------------------------------------------------------------------ #
    # Search + rerank
    # ------------------------------------------------------------------ #
    @staticmethod
    def _l2_to_cosine(sq_dist: float) -> float:
        """
        Convert FAISS squared-L2 distance to cosine similarity.

        For unit vectors: ||a-b||^2 = 2 - 2cos(a, b)  ->  cos = 1 - d/2.
        Clamped to [0, 1].
        """
        return max(0.0, min(1.0, 1.0 - sq_dist / 2.0))

    def rerank(self, results: list[dict], apply_weights: bool = True) -> list[dict]:
        """
        Sort results by final score. With `apply_weights` (default) the score is
        similarity × section weight; without it the section weight is 1.0, i.e.
        pure semantic ranking — used by the benchmark to measure whether
        section-weighted reranking actually helps.
        """
        for r in results:
            weight = config.section_weight(r["section"]) if apply_weights else 1.0
            r["section_weight"] = weight
            r["final_score"] = round(r["similarity_score"] * weight, 4)
        results.sort(key=lambda r: r["final_score"], reverse=True)
        for rank, r in enumerate(results, 1):
            r["rank"] = rank
        return results

    def search(self, query: str, top_k: int | None = None,
               rerank: bool | None = None) -> list[dict]:
        """
        Embed the query, retrieve candidates from FAISS, (optionally) rerank by
        section weight and return the top `top_k` results. `rerank` defaults to
        config.USE_SECTION_RERANK (off — pure semantic ranking, which the
        benchmark showed is more accurate); pass True/False to override.
        """
        if rerank is None:
            rerank = config.USE_SECTION_RERANK
        if not self._loaded:
            self.load()
        top_k = top_k or config.TOP_K
        candidates = max(config.SEARCH_CANDIDATES, top_k)

        qvec = self.embed_query(query)
        distances, indices = self._index.search(qvec, candidates)

        results: list[dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            info = self._mapping.get(int(idx))
            if info is None:
                continue
            results.append(
                {
                    "chunk_id": int(idx),
                    "text": info["text"],
                    "section": info["section"],
                    "original_doc_id": info.get("original_doc_id", ""),
                    "similarity_score": round(self._l2_to_cosine(float(dist)), 4),
                }
            )

        results = self.rerank(results, apply_weights=rerank)
        return results[:top_k]

    # ------------------------------------------------------------------ #
    # Stats (for the API /stats endpoint)
    # ------------------------------------------------------------------ #
    def get_stats(self) -> dict:
        if not self._loaded:
            self.load()
        return {
            "total_chunks": int(self._index.ntotal),
            "model": config.ST_MODEL_NAME,
            "vector_dim": config.EMBEDDING_DIM,
        }


# --------------------------------------------------------------------------- #
# CLI / self-test
# --------------------------------------------------------------------------- #
_TEST_QUERIES = [
    "elektrik kaçağında ceza",
    "kira sözleşmesinin feshi ve tahliye",
    "işçinin kıdem tazminatı hakkı",
    "boşanmada manevi tazminat",
    "taşınmaz satış vaadi sözleşmesi",
]


def _print_results(query: str, results: list[dict], limit: int = 3) -> None:
    print(f"\nQuery: {query}\n" + "=" * 60)
    if not results:
        print("  (no results)")
        return
    for r in results[:limit]:
        print(f"  #{r['rank']}  chunk_id={r['chunk_id']}  [{r['section']}]")
        print(f"      similarity={r['similarity_score']:.3f}  "
              f"weight={r['section_weight']}  final={r['final_score']:.3f}")
        print(f"      {r['text'][:120]}…")


def _cli() -> None:
    engine = SearchEngine()
    if len(sys.argv) >= 2:
        query = " ".join(sys.argv[1:])
        _print_results(query, engine.search(query))
    else:
        print("Running 5 built-in test queries…")
        for q in _TEST_QUERIES:
            _print_results(q, engine.search(q))


if __name__ == "__main__":
    _cli()
