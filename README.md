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
python src/validate_chunks.py   # validate chunking (fixtures + data health)
python src/05_embedding.py      # embed chunks → embeddings.npy
python src/06_faiss_index.py    # build index.faiss + chunk_mapping.pkl
python src/07_search_engine.py  # 5 built-in test queries
python src/09_benchmark_test.py # 50-query benchmark
```

…or all at once: `bash run_pipeline.sh`

## Validating the chunks

Chunking has no single ground-truth label, so correctness is checked on two
levels with `src/validate_chunks.py`:

```bash
python src/validate_chunks.py            # fixtures (+ data health if chunks.csv exists)
python src/validate_chunks.py --sample 5 # also dump 5 real decisions' section splits
```

1. **Fixture tests** (no dataset needed) — synthetic decisions with known
   sections assert that: all four sections are detected, header *variants*
   (`OLAYLAR`, `HÜKÜM`, `DEĞERLENDİRME`…) map to the right canonical section,
   a lowercase word like `karar` does **not** start a section, a long
   `GEREKÇE` splits **with overlap**, and a long `KANUN`/`KARAR` stays a single
   **atomic** chunk.
2. **Data health report** (once `chunks.csv` exists) — structural invariants
   every chunk must satisfy (size ≤ `CHUNK_SIZE` for split chunks, valid
   section, `is_atomic` consistency, `length` column matches, unique ids) plus
   quality signals: the **GENEL-only rate** (share of decisions where no header
   was detected — high means detection is failing on the real corpus) and
   **word coverage** vs `decisions_clean.csv` (must be ≈1.0 — proves no content
   was dropped during chunking).

The script exits non-zero on failure, so it doubles as a CI gate.

### Semantic validation

Structural checks confirm the chunks are *well-formed*; `src/validate_semantic.py`
checks whether they are *good semantic units* (needs `chunks.csv`, and reuses
`embeddings.npy` when present):

```bash
python src/validate_semantic.py               # all three metrics
python src/validate_semantic.py --skip-intra  # B & C only (no model load)
```

- **A. Intra-chunk coherence** — mean pairwise cosine of the sentences inside a
  chunk (sampled). Low ⇒ a chunk blends unrelated content (bad boundary).
- **B. Section separability** — within-section vs cross-section similarity; a
  positive margin means the OYAL/KANUN/KARAR/GEREKÇE labels carry real semantic
  signal (what the reranker relies on).
- **C. Redundancy** — share of chunks with a near-duplicate from a *different*
  decision; high ⇒ boilerplate that preprocessing should strip.

These are diagnostic signals (⚠️/✅ hints), not hard pass/fail gates.

### Boilerplate discovery

`src/identify_boilerplate.py` finds procedural / template text (e.g.
"Taraflar arasındaki davanın…", "…dosya incelendi gereği düşünüldü…") so it can
be stripped in `03_preprocess.py`, leaving only content-bearing legal text:

```bash
python src/identify_boilerplate.py            # auto-mine + print seed patterns
python src/identify_boilerplate.py --review    # interactive 30-sample Y/N review
python src/identify_boilerplate.py --show      # print a stratified sample only
```

- **Automatic mining** normalises away case-specific tokens (dates, case
  numbers `<NO>`, article numbers `<MADDE>`, digits `<NUM>`) then ranks
  sentences by how many **distinct decisions** they appear in — text shared by
  many decisions is boilerplate.
- **Manual review** draws a stratified sample (even across sections) and asks
  "Is this boilerplate? (y/N)" plus a category (procedural / transition /
  repetitive / content).

Results (seed + mined + reviewed patterns) are written to
`data/processed/boilerplate_patterns.json`.

**Closed loop:** on its next run `03_preprocess.py` automatically loads that
JSON and strips the confirmed patterns (placeholder tokens like `<NO>`/`<DATE>`
expand back to a value regex, so one mined pattern matches every case), then
re-chunk with `04`. The full cycle:

```
03_preprocess → 04_smart_chunking → identify_boilerplate (writes JSON)
             ↑                                              │
             └──────────── re-run strips patterns ──────────┘
```

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
| `MAX_ATOMIC_CHARS` | `4000` | Atomic KANUN/KARAR longer than this is split anyway |
| `MIN_CHUNK_CHARS` | `30` | Drop split chunks shorter than this (kills junk) |
| `DEDUP_CHUNKS` | `1` | Drop exact-duplicate chunk texts across the corpus |
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
