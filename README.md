# Hukuk Asistanı — Turkish Legal Decisions RAG System

A retrieval system for Turkish court decisions (Yargıtay kararları). An
attorney describes a legal topic in natural language, and the system returns
the most relevant decision chunks using semantic (vector) search with
section-aware reranking.

- **Dataset:** [`erdem-erdem/Turkish-Law-Documents-700k-clustered`](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered) — 702,296 decisions (1997–2025). The pipeline samples 100K by default.
- **Stack:** Hugging Face `datasets` → section-aware chunking → `paraphrase-multilingual-MiniLM-L12-v2` (Turkish-capable) embeddings → FAISS `IndexFlatL2` → section-weighted reranking → FastAPI → 50-query benchmark.

---

## Architecture

```
Query ─► embed ─► FAISS (L2) ─► L2→cosine ─► section-weight rerank ─► top-k chunks
                                     ▲
 decisions.csv ─► clean ─► sections ─► chunks ─► embeddings ─► index.faiss
```

## Section-aware chunking

Yargıtay decisions are organised into sections; each is handled differently:

| Section | Meaning | Policy | Rerank weight |
|---|---|---|---|
| `OYAL` | facts (olaylar) | **split** on sentences | 0.8 |
| `KANUN` | legal references | **atomic** (never split) | 1.3 |
| `KARAR` | ruling / decision | **atomic** (never split) | 2.0 |
| `GEREKÇE` | reasoning | **split** on sentences | 1.0 |

Legal references and the ruling lose meaning when fragmented, so they are kept
as a single chunk regardless of length. `extract_sections()` recognises header
variants (`OYAL:`, `OLAYLAR`, `HÜKÜM`, `SONUÇ`, `DEĞERLENDİRME`, …).
`chunk_size = 1000` chars, `overlap = 100` chars.

## Project layout

```
hukuk-asistan/
├─ data/
│  ├─ raw/          # downloaded dataset (decisions_100k.csv)
│  ├─ processed/    # clean text, chunks.csv, embeddings.npy, index.faiss, ...
│  └─ queries/      # benchmark_50.txt, test_queries.json
├─ src/
│  ├─ config.py            # single source of truth for all settings
│  ├─ 01_setup_dataset.py  # download sample → data/raw/decisions_100k.csv
│  ├─ 02_explore_data.py   # EDA report
│  ├─ 03_preprocess.py     # clean + normalise + filter
│  ├─ 04_smart_chunking.py # section-aware chunking → chunks.csv
│  ├─ 05_embedding.py      # all-MiniLM-L6-v2 → embeddings.npy + metadata
│  ├─ 06_faiss_index.py    # IndexFlatL2 → index.faiss + chunk_mapping.pkl
│  ├─ 07_search_engine.py  # SearchEngine: embed → search → rerank
│  ├─ 08_api_server.py     # FastAPI: /search, /health, /stats
│  └─ 09_benchmark_test.py # 50-query benchmark → benchmark_results.json
├─ requirements.txt
├─ run_pipeline.sh
└─ .env.example
```

## Setup

```bash
cd hukuk-asistan
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional overrides
```

## Run the pipeline

```bash
python src/config.py            # verify config + create data dirs
python src/01_setup_dataset.py  # download 100K sample
python src/02_explore_data.py   # exploratory report
python src/03_preprocess.py     # clean text → decisions_clean.csv
python src/04_smart_chunking.py # section-aware chunks → chunks.csv
python src/05_embedding.py      # embed chunks → embeddings.npy
python src/06_faiss_index.py    # build index.faiss + chunk_mapping.pkl
python src/07_search_engine.py  # 5 built-in test queries
python src/09_benchmark_test.py # 50-query benchmark
```

…or all at once: `bash run_pipeline.sh`

## Search from the CLI

```bash
python src/07_search_engine.py "elektrik kaçağında ceza"
python src/07_search_engine.py            # runs 5 built-in test queries
```

## Run the API

```bash
cd src
uvicorn 08_api_server:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

```bash
# POST /search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "elektrik kaçağında ceza", "top_k": 5}'

# GET /health
curl http://localhost:8000/health          # {"status": "ok", ...}

# GET /stats
curl http://localhost:8000/stats           # {"total_chunks": X, "model": ..., "vector_dim": 384}
```

Each search result:

```json
{
  "rank": 1,
  "chunk_id": 123,
  "text": "...",
  "section": "KARAR",
  "similarity_score": 0.92,
  "section_weight": 2.0,
  "final_score": 1.84
}
```

Interactive docs at `http://localhost:8000/docs`.

## Configuration

All knobs live in `src/config.py`, overridable via env / `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `SAMPLE_SIZE` | `100000` | How many decisions to download |
| `ST_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (see note below) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `100` | Chunk sizing (chars) |
| `TOP_K` | `5` | Results returned per query |
| `SEARCH_CANDIDATES` | `30` | Candidates pulled before reranking |
| `BENCHMARK_REL_THRESHOLD` | `0.35` | Cosine floor for the benchmark's relevance label |

## Notes

- **Embedding model:** the pipeline uses
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim,
  Turkish-capable) for good Turkish retrieval quality. To fall back to the
  faster English-centric model from the original spec set
  `ST_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2` (also 384-dim, so
  nothing else changes).
- Embeddings are unit-normalised, so FAISS squared-L2 distance converts to
  cosine similarity via `cos = 1 − d/2`.
- Section-weight reranking lets a strong ruling (`KARAR`, ×2.0) outrank a
  slightly-more-similar facts chunk (`OYAL`, ×0.8).
- Generated data artifacts (CSV/npy/faiss/pkl/json) are git-ignored — rerun the
  pipeline to regenerate them.
