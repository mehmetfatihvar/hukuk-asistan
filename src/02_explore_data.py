"""
02_explore_data.py
==================
Exploratory data analysis on the downloaded decisions. Prints a comprehensive,
terminal-friendly report — no external plotting needed (histograms are drawn
with text bars).

    Input : data/raw/decisions_100k.csv
    Output: printed statistics

Run:
    python src/02_explore_data.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config


def _text_histogram(values: np.ndarray, bins: int = 10, width: int = 50) -> None:
    """Draw a simple ASCII histogram of `values`."""
    if len(values) == 0:
        print("  (no data)")
        return
    counts, edges = np.histogram(values, bins=bins)
    peak = counts.max() or 1
    for i in range(bins):
        bar = "█" * int(width * counts[i] / peak)
        lo, hi = int(edges[i]), int(edges[i + 1])
        print(f"  {lo:>7,} – {hi:>7,} | {bar} {counts[i]:,}")


def main() -> None:
    if not config.RAW_CSV.exists():
        sys.exit(
            f"Raw CSV not found at {config.RAW_CSV}.\n"
            "Run 01_setup_dataset.py first."
        )

    df = pd.read_csv(config.RAW_CSV)
    df["text"] = df["text"].fillna("").astype(str)
    n = len(df)

    print("=" * 60)
    print(f"DATA EXPLORATION — {n:,} decisions")
    print("=" * 60)

    # ---- 1. Text length distribution ------------------------------------- #
    lengths = df["text"].str.len().to_numpy()
    print("\n[1] TEXT LENGTH (characters)")
    print(f"  count : {len(lengths):,}")
    print(f"  min   : {lengths.min():,}")
    print(f"  max   : {lengths.max():,}")
    print(f"  mean  : {lengths.mean():,.1f}")
    print(f"  median: {np.median(lengths):,.1f}")
    print(f"  std   : {lengths.std():,.1f}")
    print("\n  Histogram:")
    _text_histogram(lengths)

    # ---- 2. Source distribution ------------------------------------------ #
    print("\n[2] SOURCE DISTRIBUTION")
    if "source" in df.columns:
        src_counts = df["source"].fillna("<null>").value_counts()
        for src, cnt in src_counts.items():
            pct = 100 * cnt / n
            print(f"  {str(src):<12}: {cnt:>8,}  ({pct:5.1f}%)")
    else:
        print("  (no 'source' column)")

    # ---- 3. Metadata quality --------------------------------------------- #
    print("\n[3] METADATA QUALITY")
    print("  NULL / empty values per column:")
    for col in df.columns:
        col_series = df[col].astype(str)
        nulls = df[col].isna().sum()
        empties = (col_series.str.strip() == "").sum()
        print(f"    {col:<14}: null={nulls:>7,}  empty={empties:>7,}")

    # Duplicate detection (by exact text)
    dup_text = df["text"].duplicated().sum()
    print(f"\n  Duplicate decisions (identical text): {dup_text:,}")

    # Encoding check — flag rows containing the Unicode replacement char.
    bad_encoding = df["text"].str.contains("�", na=False).sum()
    print(f"  Rows with encoding artefacts (\\ufffd): {bad_encoding:,}")

    # ---- 4. Sample analysis ---------------------------------------------- #
    print("\n[4] SAMPLE DECISIONS (5 random)")
    sample = df.sample(min(5, n), random_state=42)
    for i, (_, r) in enumerate(sample.iterrows(), 1):
        print(f"\n  --- Sample {i} ---")
        print(f"  id        : {r.get('id', '')}")
        print(f"  source    : {r.get('source', '')}")
        print(f"  esasNo    : {r.get('esasNo', '')}")
        print(f"  kararNo   : {r.get('kararNo', '')}")
        print(f"  tarih     : {r.get('kararTarihi', '')}")
        text = str(r["text"])[:500].replace("\n", " ")
        print(f"  text[:500]: {text}")

    print("\n" + "=" * 60)
    print("Exploration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
