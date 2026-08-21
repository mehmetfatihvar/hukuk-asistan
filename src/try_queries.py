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


# A domain-balanced diagnostic set: situations a lawyer would describe, spanning
# civil, labour, family, commercial, criminal and procedural law, plus a few
# abstract-concept queries (the known weak spot). Kept stable so it doubles as a
# before/after baseline when the embedding model is changed.
DEFAULT_QUERIES = [
    # --- civil / property ---
    "kira sözleşmesinin haklı nedenle feshi ve kiracının tahliyesi",
    "muvazaalı taşınmaz satışında tapu iptali ve tescil",
    "kat karşılığı inşaat sözleşmesinin feshi ve tazminat",
    # --- labour ---
    "işçinin haklı nedenle feshinde kıdem ve ihbar tazminatı",
    "işçinin fazla mesai alacağının tanıkla ispatı",
    # --- tort / compensation ---
    "trafik kazasında destekten yoksun kalma tazminatının hesabı",
    "haksız fiil sonucu oluşan zararda kusur ve illiyet bağı",
    # --- family ---
    "boşanmada kusur oranına göre manevi tazminat",
    "velayetin değiştirilmesi ve çocuğun üstün yararı",
    # --- inheritance / consumer / insurance ---
    "miras bırakanın tasarruflarına karşı tenkis davası",
    "tüketicinin ayıplı mal nedeniyle iade ve bedel talebi",
    "sigortacının ödediği bedel için sorumluya rücu",
    # --- commercial / enforcement ---
    "senede dayalı icra takibine itirazın iptali",
    "limited şirkette ortağın haklı nedenle ortaklıktan çıkarılması",
    # --- criminal ---
    "kasten yaralama suçunda haksız tahrik indirimi",
    "silahlı terör örgütüne yardım ile örgüt üyeliği arasındaki ayrım",
    # --- procedural ---
    "istinaf süresinin kaçırılması ve eski hale getirme talebi",
    # --- abstract concepts (known weak spot) ---
    "sözleşmedeki cezai şartın fahiş olması nedeniyle indirilmesi",
    "zamanaşımının kesilmesi ve durması halleri",
]


def _tokens(text: str, n: int = 50) -> set[str]:
    return set(str(text).lower().split()[:n])


def _near_duplicate(a: dict, b: dict, thresh: float = 0.6) -> bool:
    """Token-Jaccard on the first ~50 words — catches the same doctrine quoted
    verbatim in different decisions."""
    ta, tb = _tokens(a["text"]), _tokens(b["text"])
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= thresh


def run_query(engine, query: str, n_decisions: int, snippet: int) -> None:
    # Pull more chunks than needed, then collapse to distinct decisions and
    # drop near-duplicate texts (same doctrine repeated across decisions).
    chunks = engine.search(query, top_k=max(60, n_decisions * 12))
    seen_docs: set[str] = set()
    top: list[dict] = []
    for r in chunks:
        d = r["original_doc_id"]
        if d in seen_docs:
            continue
        if any(_near_duplicate(r, kept) for kept in top):
            continue
        seen_docs.add(d)
        top.append(r)
        if len(top) >= n_decisions:
            break

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
