"""
06_faiss_index.py
=================
Build a searchable FAISS index from the saved embeddings and a chunk mapping
so search results can be resolved back to their text and section.

    Input : data/processed/embeddings.npy
            data/processed/chunks.csv
    Output: data/processed/index.faiss        (IndexFlatL2)
            data/processed/chunk_mapping.pkl   (chunk_id -> {text, section, doc_id})

We use IndexFlatL2 (exact, L2 distance). Embeddings are unit-normalised in
step 05, so L2 distance is monotonic with cosine similarity.

Run:
    python src/06_faiss_index.py
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

import config


def build_mapping(df: pd.DataFrame) -> dict[int, dict]:
    """chunk_id -> {text, section, original_doc_id, is_atomic}."""
    mapping: dict[int, dict] = {}
    for _, r in df.iterrows():
        mapping[int(r["chunk_id"])] = {
            "text": str(r["text"]),
            "section": str(r["section"]),
            "original_doc_id": str(r["original_doc_id"]),
            "is_atomic": bool(r["is_atomic"]),
        }
    return mapping


def main() -> None:
    if not config.EMBEDDINGS_NPY.exists():
        sys.exit(
            f"Embeddings not found at {config.EMBEDDINGS_NPY}.\n"
            "Run 05_embedding.py first."
        )
    if not config.CHUNKS_CSV.exists():
        sys.exit(
            f"Chunks not found at {config.CHUNKS_CSV}.\n"
            "Run 04_smart_chunking.py first."
        )

    try:
        import faiss
    except ImportError:
        sys.exit("faiss is not installed. Run: pip install faiss-cpu")

    config.ensure_dirs()
    embeddings = np.load(config.EMBEDDINGS_NPY).astype("float32")
    df = pd.read_csv(config.CHUNKS_CSV)

    if len(embeddings) != len(df):
        sys.exit(
            f"Row mismatch: {len(embeddings)} embeddings vs {len(df)} chunks. "
            "Re-run 05_embedding.py after chunking."
        )

    # ---- Build index ----------------------------------------------------- #
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(config.FAISS_INDEX))

    # ---- Build & save chunk mapping ------------------------------------- #
    mapping = build_mapping(df)
    with open(config.CHUNK_MAPPING_PKL, "wb") as f:
        pickle.dump(mapping, f)

    size_mb = config.FAISS_INDEX.stat().st_size / 1024**2

    # ---- Metrics --------------------------------------------------------- #
    print("\n===== FAISS index metrics =====")
    print(f"Index created  : {index.ntotal:,} vectors")
    print(f"Dimensionality : {dim}")
    print(f"Index size     : {size_mb:.2f} MB")
    print(f"💾 Saved index   → {config.FAISS_INDEX}")
    print(f"💾 Saved mapping → {config.CHUNK_MAPPING_PKL}")

    # ---- Sample search test --------------------------------------------- #
    print("\n----- Sample search test (query = first chunk) -----")
    query = embeddings[:1]
    distances, indices = index.search(query, 5)
    print("Top-5 nearest to chunk 0:")
    for rank, (d, idx) in enumerate(zip(distances[0], indices[0]), 1):
        info = mapping.get(int(idx), {})
        snippet = info.get("text", "")[:60].replace("\n", " ")
        print(f"  #{rank}  id={idx}  L2={d:.4f}  [{info.get('section', '?')}]  {snippet}…")
    print("Sample search test: OK")


if __name__ == "__main__":
    main()
