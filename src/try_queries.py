"""
try_queries.py
=============
Run a batch of realistic, lawyer-style queries against the search engine and
print the results grouped by DECISION (distinct Yargıtay kararı), with a long
enough snippet to judge relevance by eye.

Use it to eyeball real retrieval quality together:
    python src/try_queries.py                 # runs the built-in query set
    python src/try_queries.py "kendi sorgum"  # runs your own query/queries
    python src/try_queries.py --n 3           # show 3 decisions per query

Needs the built index (06_faiss_index.py) — no re-embedding.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import config

# ---- load the digit-prefixed search engine module ------------------------ #
_spec = importlib.util.spec_from_file_location(
    "search_engine", Path(__file__).with_name("07_search_engine.py")
)
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)  # type: ignore[union-attr]
SearchEngine = _se.SearchEngine


# Realistic queries a lawyer might type (a described situation, not a keyword).
DEFAULT_QUERIES = [
    "kira sözleşmesinin haklı nedenle feshi ve kiracının tahliyesi",
    "işçinin haklı nedenle feshinde kıdem ve ihbar tazminatı",
    "trafik kazasında destekten yoksun kalma tazminatının hesabı",
    "boşanmada kusur oranına göre manevi tazminat",
    "muvazaalı taşınmaz satışında tapu iptali ve tescil",
    "kat karşılığı inşaat sözleşmesinin feshi ve tazminat",
    "senede dayalı icra takibine itirazın iptali",
    "tüketicinin ayıplı mal nedeniyle iade ve bedel talebi",
    "miras bırakanın tasarruflarına karşı tenkis davası",
    "sigortacının ödediği bedel için sorumluya rücu",
    "işçinin fazla mesai alacağının tanıkla ispatı",
    "haksız fiil sonucu oluşan zararda kusur ve illiyet bağı",
]


def run_query(engine, query: str, n_decisions: int, snippet: int) -> None:
    # Pull more chunks than needed, then collapse to distinct decisions.
    chunks = engine.search(query, top_k=max(40, n_decisions * 8))
    by_decision: dict[str, dict] = {}
    for r in chunks:
        d = r["original_doc_id"]
        if d not in by_decision:
            by_decision[d] = r
    top = list(by_decision.values())[:n_decisions]

    print("\n" + "=" * 78)
    print(f"SORGU: {query}")
    print("=" * 78)
    if not top:
        print("  (sonuç yok)")
        return
    for i, r in enumerate(top, 1):
        text = " ".join(str(r["text"]).split())
        print(f"\n#{i}  benzerlik={r['similarity_score']:.3f}  "
              f"[{r['section']}]  doc_id={r['original_doc_id']}")
        print(f"    {text[:snippet]}…")


def main() -> None:
    ap = argparse.ArgumentParser(description="Try real queries against the index.")
    ap.add_argument("query", nargs="*", help="one or more queries (default: built-in set)")
    ap.add_argument("--n", type=int, default=5, help="distinct decisions per query")
    ap.add_argument("--snippet", type=int, default=320, help="snippet length (chars)")
    args = ap.parse_args()

    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES
    print(f"Loading search engine ({config.ST_MODEL_NAME}) — rerank="
          f"{config.USE_SECTION_RERANK} …")
    engine = SearchEngine()
    for q in queries:
        run_query(engine, q, args.n, args.snippet)

    print("\n" + "=" * 78)
    print(f"{len(queries)} sorgu çalıştırıldı. Sonuçların ilgili olup olmadığını "
          "gözle değerlendirin.")


if __name__ == "__main__":
    main()
