"""
03_preprocess.py
================
Clean and normalise the raw decision text.

    Input : data/raw/decisions_100k.csv
    Output: data/processed/decisions_clean.csv

Steps:
  1. Remove boilerplate header lines (MAHKEMESİ, TARİHİ, ...).
  2. Normalise whitespace (\\n, \\t, multiple spaces → single space).
  3. Quality filtering (drop text shorter than MIN_TEXT_CHARS, drop empties).
  4. Save the cleaned CSV and print a before/after comparison.

Run:
    python src/03_preprocess.py
"""

from __future__ import annotations

import re
import sys

import pandas as pd
from tqdm import tqdm

import config


# Boilerplate labels that commonly appear at the top of Yargıtay decisions.
# We remove the whole line whenever it starts with one of these labels.
_BOILERPLATE_LABELS = [
    "MAHKEMESİ",
    "MAHKEMESI",
    "DAVA TÜRÜ",
    "DAVA TURU",
    "TARİHİ",
    "TARIHI",
    "ESAS NO",
    "KARAR NO",
    "NUMARASI",
    "İTHAL",
    "TAtarafLAR",
    "TARAFLAR",
    "İNCELENEN KARARIN",
    "INCELENEN KARARIN",
]

_BOILERPLATE_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(lbl) for lbl in _BOILERPLATE_LABELS) + r")\b.*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Apply boilerplate removal and whitespace normalisation to one decision."""
    if not isinstance(text, str):
        return ""
    # 1. Remove boilerplate header lines.
    text = _BOILERPLATE_RE.sub(" ", text)
    # 2. Normalise all whitespace (newlines, tabs, repeated spaces) to a space.
    text = _WHITESPACE_RE.sub(" ", text)
    # 3. Trim.
    return text.strip()


def main() -> None:
    if not config.RAW_CSV.exists():
        sys.exit(
            f"Raw CSV not found at {config.RAW_CSV}.\n"
            "Run 01_setup_dataset.py first."
        )

    config.ensure_dirs()
    df = pd.read_csv(config.RAW_CSV)
    n_before = len(df)
    df["text"] = df["text"].fillna("").astype(str)

    len_before = df["text"].str.len()

    tqdm.pandas(desc="Cleaning")
    df["text"] = df["text"].progress_apply(clean_text)

    # Quality filtering.
    before_filter = len(df)
    df = df[df["text"].str.len() >= config.MIN_TEXT_CHARS].copy()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    n_after = len(df)

    len_after = df["text"].str.len()

    df.to_csv(config.CLEAN_CSV, index=False)

    # ---- Report ---------------------------------------------------------- #
    print("\n===== Preprocessing report =====")
    print(f"Rows in       : {n_before:,}")
    print(f"Rows dropped  : {before_filter - n_after:,} "
          f"(too short / empty / duplicate)")
    print(f"Rows out      : {n_after:,}")
    print("\nText length (characters):")
    print(f"  mean  before → after : {len_before.mean():,.0f} → {len_after.mean():,.0f}")
    print(f"  median before → after: {len_before.median():,.0f} → {len_after.median():,.0f}")
    reduction = 1 - (len_after.sum() / max(len_before.sum(), 1))
    print(f"  total char reduction : {reduction * 100:.1f}%")
    print(f"\n💾 Saved cleaned data → {config.CLEAN_CSV}")

    if n_after:
        print("\n----- Before / after (first decision) -----")
        print("BEFORE:", str(pd.read_csv(config.RAW_CSV).iloc[0]['text'])[:300])
        print("AFTER :", str(df.iloc[0]['text'])[:300])


if __name__ == "__main__":
    main()
