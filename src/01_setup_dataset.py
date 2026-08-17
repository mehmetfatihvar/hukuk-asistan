"""
01_setup_dataset.py
===================
Download a sample of the Turkish Law Documents dataset from Hugging Face and
persist it as a CSV for the rest of the pipeline.

    Input : erdem-erdem/Turkish-Law-Documents-700k-clustered  (Hugging Face)
    Output: data/raw/decisions_100k.csv

We stream the dataset and take the first SAMPLE_SIZE rows instead of
materialising the full 702K decisions (~1.75 GB), which keeps the download
fast and memory-friendly.

Run:
    python src/01_setup_dataset.py
"""

from __future__ import annotations

import sys

import pandas as pd
from tqdm import tqdm

import config


def _map_columns(row: dict) -> dict:
    """
    Normalise an upstream row onto the schema declared in config.EXPECTED_COLUMNS.

    The upstream schema occasionally uses slightly different key names; we map
    the common variants and fall back to empty strings so downstream steps can
    always rely on the expected columns being present.
    """
    aliases = {
        "text": ["text", "content", "karar_metni", "decision_text"],
        "source": ["source", "kaynak", "court"],
        "id": ["id", "doc_id", "_id"],
        "esasNo": ["esasNo", "esas_no", "esasNumarasi"],
        "kararNo": ["kararNo", "karar_no", "kararNumarasi"],
        "kararTarihi": ["kararTarihi", "karar_tarihi", "date"],
        "cluster_ids": ["cluster_ids", "cluster_id", "clusters"],
    }
    out: dict = {}
    for target, candidates in aliases.items():
        value = ""
        for key in candidates:
            if key in row and row[key] is not None:
                value = row[key]
                break
        out[target] = value
    return out


def download_sample() -> pd.DataFrame:
    """Stream `SAMPLE_SIZE` rows from the dataset and return them as a DataFrame."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "The 'datasets' library is not installed.\n"
            "Install dependencies first:  pip install -r requirements.txt"
        )

    print(config.summary())
    print(f"\n⬇  Loading dataset '{config.DATASET_NAME}' (streaming)…")

    stream = load_dataset(
        config.DATASET_NAME,
        split=config.DATASET_SPLIT,
        streaming=True,
        token=config.HF_TOKEN,
    )

    rows: list[dict] = []
    for i, row in enumerate(
        tqdm(stream, total=config.SAMPLE_SIZE, desc="Downloading", unit="doc")
    ):
        if i >= config.SAMPLE_SIZE:
            break
        rows.append(_map_columns(row))

    df = pd.DataFrame(rows, columns=config.EXPECTED_COLUMNS)
    return df


def main() -> None:
    config.ensure_dirs()

    if config.RAW_CSV.exists():
        print(f"✓ Raw CSV already exists at {config.RAW_CSV} — skipping download.")
        print("  (delete it to force a re-download)")
        df = pd.read_csv(config.RAW_CSV)
    else:
        df = download_sample()
        df.to_csv(config.RAW_CSV, index=False)
        print(f"\n💾 Saved {len(df):,} decisions → {config.RAW_CSV}")

    # ---- Metrics ---------------------------------------------------------- #
    size_gb = config.RAW_CSV.stat().st_size / 1024**3
    print("\n===== Metrics =====")
    print(f"Dataset loaded : {len(df):,} decisions")
    print(f"Data size      : {size_gb:.3f} GB  ({config.RAW_CSV})")
    print(f"Columns        : {list(df.columns)}")

    if len(df):
        preview = str(df.iloc[0]["text"])[:500]
        print("\n----- Sample decision (first 500 chars) -----")
        print(preview)
        print("---------------------------------------------")


if __name__ == "__main__":
    main()
