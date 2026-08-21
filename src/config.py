"""
config.py
=========
Central configuration for the Turkish Legal Decisions RAG system
("Hukuk Asistanı").

All paths, model settings, chunking parameters, search parameters and API
settings live here so every pipeline step (01..09) reads from a single
source of truth.

Environment variables (loaded from a `.env` file in the project root) can
override the most important knobs without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; the pipeline still works without it
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# BASE_DIR = .../hukuk-asistan  (parent of this src/ folder)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load environment variables from <BASE_DIR>/.env if present.
load_dotenv(BASE_DIR / ".env")

DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
QUERIES_DIR: Path = DATA_DIR / "queries"

# Pipeline artifacts (files produced by each step)
RAW_CSV: Path = RAW_DIR / "decisions_100k.csv"
CLEAN_CSV: Path = PROCESSED_DIR / "decisions_clean.csv"
CHUNKS_CSV: Path = PROCESSED_DIR / "chunks.csv"
EMBEDDINGS_NPY: Path = PROCESSED_DIR / "embeddings.npy"
EMBEDDING_META_JSON: Path = PROCESSED_DIR / "embedding_metadata.json"
FAISS_INDEX: Path = PROCESSED_DIR / "index.faiss"
CHUNK_MAPPING_PKL: Path = PROCESSED_DIR / "chunk_mapping.pkl"
BENCHMARK_QUERIES_TXT: Path = QUERIES_DIR / "benchmark_50.txt"
BENCHMARK_RESULTS_JSON: Path = PROCESSED_DIR / "benchmark_results.json"
TEST_QUERIES_JSON: Path = QUERIES_DIR / "test_queries.json"


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
DATASET_NAME: str = os.getenv(
    "DATASET_NAME", "erdem-erdem/Turkish-Law-Documents-700k-clustered"
)
# Download a sample instead of the full 702K decisions (1.75 GB) for speed.
SAMPLE_SIZE: int = int(os.getenv("SAMPLE_SIZE", "100000"))
DATASET_SPLIT: str = os.getenv("DATASET_SPLIT", "train")
HF_TOKEN: str | None = os.getenv("HF_TOKEN") or None

# Expected columns in the source dataset. If the upstream schema differs, the
# loader in 01_setup_dataset.py maps whatever it finds onto these names.
EXPECTED_COLUMNS: list[str] = [
    "text",
    "source",
    "id",
    "esasNo",
    "kararNo",
    "kararTarihi",
    "cluster_ids",
]


# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #
# Turkish-capable multilingual model (384-dim, fast). This replaces the
# English-centric all-MiniLM-L6-v2 from the original spec for much better
# Turkish retrieval quality; it is dimension-compatible (384) so the rest of
# the pipeline is unchanged. To fall back to the English model set
#   ST_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ST_MODEL_NAME: str = os.getenv(
    "ST_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
# Normalize embeddings so L2 distance is monotonic with cosine similarity and
# can be converted to a cosine score (see 07_search_engine.py).
NORMALIZE_EMBEDDINGS: bool = os.getenv("NORMALIZE_EMBEDDINGS", "1") == "1"


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
# Chunk size / overlap are measured in characters.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
# Minimum length of a raw decision to keep during preprocessing.
MIN_TEXT_CHARS: int = int(os.getenv("MIN_TEXT_CHARS", "100"))

# Drop split chunks shorter than this (kills 4-char junk fragments). Atomic
# chunks (KARAR/KANUN) are exempt so a short but meaningful ruling survives.
MIN_CHUNK_CHARS: int = int(os.getenv("MIN_CHUNK_CHARS", "30"))
# Safety cap for ATOMIC sections: an atomic section longer than this is split
# anyway (a 1.25M-char "atomic" chunk is useless — the embedder only reads the
# first few hundred tokens). Keeps KANUN/KARAR whole in the normal case.
MAX_ATOMIC_CHARS: int = int(os.getenv("MAX_ATOMIC_CHARS", "4000"))
# Drop exact-duplicate chunk texts across the corpus (cuts boilerplate
# redundancy and embedding cost). Set to "0" to keep duplicates.
DEDUP_CHUNKS: bool = os.getenv("DEDUP_CHUNKS", "1") == "1"

# Whether 03_preprocess.py auto-applies the FREQUENCY-MINED patterns from
# identify_boilerplate.py. OFF by default: frequency mining also surfaces
# repeated legal *doctrine* (established case-law quoted across many
# decisions), which is content, not boilerplate — stripping it would gut the
# corpus. Only hand-curated seed patterns and human-confirmed review items are
# applied unless you set this to "1" after reviewing the mined list.
APPLY_MINED_PATTERNS: bool = os.getenv("APPLY_MINED_PATTERNS", "0") == "1"

# Canonical decision sections.
SECTION_OYAL = "OYAL"        # facts (olaylar)
SECTION_KANUN = "KANUN"      # legal references
SECTION_KARAR = "KARAR"      # ruling / decision
SECTION_GEREKCE = "GEREKÇE"  # reasoning
SECTION_GENERAL = "GENEL"    # fallback when no marker is found

# Sections kept ATOMIC (never split): legal references and the ruling lose
# meaning if fragmented.
ATOMIC_SECTIONS: set[str] = {SECTION_KANUN, SECTION_KARAR}


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
TOP_K: int = int(os.getenv("TOP_K", "5"))
# Candidates pulled from FAISS before reranking (larger than TOP_K so the
# section-weight reranker has room to reorder).
SEARCH_CANDIDATES: int = int(os.getenv("SEARCH_CANDIDATES", "30"))

# Section-weighted reranking. DISABLED by default: the 50-query benchmark
# showed it *lowers* accuracy (66.4% with vs 69.6% without) because Turkish
# legal chunks are homogeneous and boosting KARAR surfaces less-similar
# rulings. Set USE_SECTION_RERANK=1 to re-enable (e.g. after retuning weights).
USE_SECTION_RERANK: bool = os.getenv("USE_SECTION_RERANK", "0") == "1"

# Reranking weights applied to the cosine similarity per section (only used
# when USE_SECTION_RERANK is on).
SECTION_WEIGHTS: dict[str, float] = {
    SECTION_KARAR: 2.0,
    SECTION_KANUN: 1.3,
    SECTION_GEREKCE: 1.0,
    SECTION_OYAL: 0.8,
    SECTION_GENERAL: 1.0,
}
DEFAULT_SECTION_WEIGHT: float = 1.0

# Benchmark: a result counts as "relevant" if its cosine similarity is at
# least this, and (when keywords are provided) it contains an expected keyword.
BENCHMARK_REL_THRESHOLD: float = float(os.getenv("BENCHMARK_REL_THRESHOLD", "0.35"))


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def section_weight(section: str) -> float:
    """Return the reranking weight for a section label."""
    return SECTION_WEIGHTS.get(section, DEFAULT_SECTION_WEIGHT)


def ensure_dirs() -> None:
    """Create every data directory the pipeline needs (idempotent)."""
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, QUERIES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    """Return a human-readable snapshot of the active configuration."""
    lines = [
        "===== Hukuk Asistanı — Configuration =====",
        f"BASE_DIR             : {BASE_DIR}",
        f"DATASET_NAME         : {DATASET_NAME}",
        f"SAMPLE_SIZE          : {SAMPLE_SIZE:,}",
        f"EMBEDDING_MODEL      : {ST_MODEL_NAME}",
        f"EMBEDDING_DIM        : {EMBEDDING_DIM}",
        f"EMBEDDING_BATCH_SIZE : {EMBEDDING_BATCH_SIZE}",
        f"CHUNK_SIZE/OVERLAP   : {CHUNK_SIZE} / {CHUNK_OVERLAP}",
        f"ATOMIC_SECTIONS      : {sorted(ATOMIC_SECTIONS)}",
        f"SECTION_WEIGHTS      : {SECTION_WEIGHTS}",
        f"TOP_K                : {TOP_K}",
        f"API                  : {API_HOST}:{API_PORT}",
        "==========================================",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ensure_dirs()
    print(summary())
    print("\nData directories ready:")
    for d in (RAW_DIR, PROCESSED_DIR, QUERIES_DIR):
        print(f"  ✓ {d}")
