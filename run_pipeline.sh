#!/usr/bin/env bash
# Run the full RAG pipeline end-to-end.
# Usage:  bash run_pipeline.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "==> [0/9] Configuration"
python src/config.py

echo "==> [1/9] Download dataset sample"
python src/01_setup_dataset.py

echo "==> [2/9] Explore data"
python src/02_explore_data.py

echo "==> [3/9] Preprocess"
python src/03_preprocess.py

echo "==> [4/9] Smart chunking"
python src/04_smart_chunking.py

echo "==> [5/9] Embedding"
python src/05_embedding.py

echo "==> [6/9] Build FAISS index"
python src/06_faiss_index.py

echo "==> [7/9] Search smoke test (5 built-in queries)"
python src/07_search_engine.py

echo "==> [9/9] Benchmark (50 queries)"
python src/09_benchmark_test.py

echo
echo "Pipeline complete. Start the API with:"
echo "    cd src && uvicorn 08_api_server:app --host 0.0.0.0 --port 8000"
