"""
05_embedding.py
===============
Encode every chunk into a dense vector with sentence-transformers.

    Input : data/processed/chunks.csv
    Output: data/processed/embeddings.npy          (float32, [N, 384])
            data/processed/embedding_metadata.json (model, dim, count, date)

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384-dim, Turkish-capable). Batch size 32; uses GPU (CUDA) when available.

The .npy rows are written in the exact order of chunks.csv, so
`embeddings[i]` corresponds to the row whose `chunk_id == i`.

Run:
    python src/05_embedding.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time

import numpy as np
import pandas as pd

import config


def main() -> None:
    if not config.CHUNKS_CSV.exists():
        sys.exit(
            f"Chunks not found at {config.CHUNKS_CSV}.\n"
            "Run 04_smart_chunking.py first."
        )

    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        sys.exit(
            "sentence-transformers / torch not installed.\n"
            "Install dependencies:  pip install -r requirements.txt"
        )

    config.ensure_dirs()
    df = pd.read_csv(config.CHUNKS_CSV)
    texts = df["text"].fillna("").astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded    : {config.ST_MODEL_NAME}")
    print(f"GPU available   : {'Yes (' + torch.cuda.get_device_name(0) + ')' if device == 'cuda' else 'No'}")
    print(f"Total chunks    : {len(texts):,}")

    model = SentenceTransformer(config.ST_MODEL_NAME, device=device)

    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0

    np.save(config.EMBEDDINGS_NPY, embeddings)

    metadata = {
        "total_chunks": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "model": config.ST_MODEL_NAME,
        "normalized": config.NORMALIZE_EMBEDDINGS,
        "date": dt.date.today().isoformat(),
    }
    with open(config.EMBEDDING_META_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ---- Metrics --------------------------------------------------------- #
    print("\n===== Embedding metrics =====")
    print(f"Total chunks embedded : {embeddings.shape[0]:,}")
    print(f"Embedding shape       : {embeddings.shape}")
    print(f"dtype                 : {embeddings.dtype}")
    print(f"Time taken            : {elapsed:.1f} seconds "
          f"({embeddings.shape[0] / max(elapsed, 1e-6):.0f} chunks/s)")
    print(f"\n💾 Saved embeddings → {config.EMBEDDINGS_NPY}")
    print(f"💾 Saved metadata   → {config.EMBEDDING_META_JSON}")


if __name__ == "__main__":
    main()
