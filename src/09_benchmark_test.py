"""
09_benchmark_test.py
====================
Run the 50 benchmark queries through the search engine and measure retrieval
quality and latency.

    Input : data/queries/benchmark_50.txt
            the built FAISS index + mapping
    Output: data/processed/benchmark_results.json + printed summary

Relevance labelling
-------------------
Manual Y/N labelling isn't possible inside an automated script, so relevance
is approximated: a top-k result counts as relevant when
    similarity_score >= BENCHMARK_REL_THRESHOLD
and, if the query line supplies expected keywords ("query | kw1, kw2"),
the result text contains at least one of them.

Metrics
-------
    accuracy            : mean over queries of (relevant results / results)
    false_positive_rate : mean over queries of (irrelevant results / results)
    avg_similarity      : mean similarity over all returned results

Run (after the API/engine artifacts exist):
    python src/09_benchmark_test.py
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import config

# ---- import the digit-prefixed search engine module --------------------- #
_spec = importlib.util.spec_from_file_location(
    "search_engine", Path(__file__).with_name("07_search_engine.py")
)
_search_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_search_module)  # type: ignore[union-attr]
SearchEngine = _search_module.SearchEngine


def load_queries(path: Path) -> list[dict]:
    """Parse benchmark_50.txt into [{query, keywords}] entries."""
    queries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            q, kw = line.split("|", 1)
            keywords = [k.strip().lower() for k in kw.split(",") if k.strip()]
        else:
            q, keywords = line, []
        queries.append({"query": q.strip(), "keywords": keywords})
    return queries


def is_relevant(result: dict, keywords: list[str]) -> bool:
    """Auto-label a single result as relevant (see module docstring)."""
    if result["similarity_score"] < config.BENCHMARK_REL_THRESHOLD:
        return False
    if not keywords:
        return True
    text = result["text"].lower()
    return any(kw in text for kw in keywords)


def main() -> None:
    if not config.BENCHMARK_QUERIES_TXT.exists():
        sys.exit(f"Benchmark queries not found at {config.BENCHMARK_QUERIES_TXT}.")

    queries = load_queries(config.BENCHMARK_QUERIES_TXT)
    print(f"Loaded {len(queries)} benchmark queries.")
    print(f"Loading search engine ({config.ST_MODEL_NAME})…")
    engine = SearchEngine()

    per_query: list[dict] = []
    all_sims: list[float] = []
    accuracies: list[float] = []
    fp_rates: list[float] = []
    latencies: list[float] = []

    print("\n" + "=" * 72)
    print(f"{'#':>3}  {'query':<45}{'ms':>6}{'acc':>7}")
    print("=" * 72)

    plain_accuracies: list[float] = []  # accuracy WITHOUT section reranking

    for i, item in enumerate(queries, 1):
        # Ablation: measure WITH and WITHOUT section-weighted reranking
        # explicitly, independent of the config default.
        t0 = time.perf_counter()
        results = engine.search(item["query"], config.TOP_K, rerank=True)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        plain = engine.search(item["query"], config.TOP_K, rerank=False)
        plain_rel = sum(1 for r in plain if is_relevant(r, item["keywords"]))
        plain_accuracies.append(plain_rel / (len(plain) or 1))

        labelled = []
        for r in results:
            rel = is_relevant(r, item["keywords"])
            all_sims.append(r["similarity_score"])
            labelled.append(
                {
                    "rank": r["rank"],
                    "chunk_id": r["chunk_id"],
                    "text": r["text"][:200],
                    "section": r["section"],
                    "similarity": r["similarity_score"],
                    "final_score": r["final_score"],
                    "relevant": rel,
                }
            )

        n = len(labelled) or 1
        n_rel = sum(1 for r in labelled if r["relevant"])
        q_acc = n_rel / n
        accuracies.append(q_acc)
        fp_rates.append(1 - q_acc)

        per_query.append(
            {
                "query": item["query"],
                "num_results": len(labelled),
                "accuracy": round(q_acc, 3),
                "results": labelled,
            }
        )
        print(f"{i:>3}  {item['query'][:44]:<45}{latency_ms:>6.0f}{q_acc:>7.2f}")

    # ---- Aggregate ------------------------------------------------------- #
    acc_rr = statistics.mean(accuracies) if accuracies else 0.0
    acc_plain = statistics.mean(plain_accuracies) if plain_accuracies else 0.0
    report = {
        "total_queries": len(queries),
        "accuracy": round(acc_rr, 3),
        "accuracy_no_rerank": round(acc_plain, 3),
        "rerank_delta": round(acc_rr - acc_plain, 3),
        "false_positive_rate": round(statistics.mean(fp_rates), 3) if fp_rates else 0.0,
        "avg_similarity": round(statistics.mean(all_sims), 3) if all_sims else 0.0,
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "queries": per_query,
    }

    print("=" * 72)
    print("AGGREGATE")
    print(f"  total queries        : {report['total_queries']}")
    print(f"  accuracy (rerank)    : {report['accuracy']:.2%}")
    print(f"  accuracy (no rerank) : {report['accuracy_no_rerank']:.2%}")
    print(f"  rerank delta         : {report['rerank_delta']:+.2%}  "
          f"({'reranking helps' if report['rerank_delta'] > 0 else 'reranking neutral/hurts'})")
    print(f"  false positive rate  : {report['false_positive_rate']:.2%}")
    print(f"  avg similarity       : {report['avg_similarity']:.3f}")
    print(f"  latency mean (ms)    : {report['latency_ms_mean']}")
    print("=" * 72)

    with open(config.BENCHMARK_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved report → {config.BENCHMARK_RESULTS_JSON}")

    if report["accuracy"] >= 0.70:
        print("✅ Success criterion met (accuracy >= 70%).")
    else:
        print("⚠️  Accuracy below 70% — consider a Turkish-tuned embedding model "
              "or tuning BENCHMARK_REL_THRESHOLD / SIMILARITY handling.")


if __name__ == "__main__":
    main()
