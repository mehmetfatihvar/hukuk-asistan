"""
04_smart_chunking.py
====================
Split each cleaned decision into meaningful chunks, respecting the legal
structure of Yargıtay decisions.

    Input : data/processed/decisions_clean.csv
    Output: data/processed/chunks.csv
            columns: chunk_id, text, section, original_doc_id, length, is_atomic

Sections and chunking policy
----------------------------
    OYAL    (facts / olaylar)       -> SPLIT on sentence boundaries
    KANUN   (legal references)      -> KEEP ATOMIC (never split)
    KARAR   (ruling / decision)     -> KEEP ATOMIC (never split)
    GEREKÇE (reasoning)             -> SPLIT on sentence boundaries

Legal references (KANUN) and the ruling (KARAR) lose their meaning when
fragmented, so they are stored as a single chunk regardless of length. The
narrative sections (OYAL, GEREKÇE) are split into <= CHUNK_SIZE character
chunks with CHUNK_OVERLAP characters of overlap to preserve context.

Run:
    python src/04_smart_chunking.py
"""

from __future__ import annotations

import re
import sys

import pandas as pd
from tqdm import tqdm

import config

# --------------------------------------------------------------------------- #
# Section detection
# --------------------------------------------------------------------------- #
# Each canonical section maps to the header variants that introduce it in the
# corpus. Matching is CASE-SENSITIVE (uppercase): real headers are upper case,
# while the same words in running text are lower case, so this avoids splitting
# mid-sentence on a common word like "karar".
_SECTION_VARIANTS: dict[str, list[str]] = {
    config.SECTION_OYAL: ["OLAYLAR", "OLAY", "OYAL", "MADDİ OLAY", "MADDI OLAY", "VAKIALAR"],
    config.SECTION_KANUN: [
        "KANUN", "İLGİLİ KANUN", "ILGILI KANUN", "YASAL DAYANAK",
        "İLGİLİ MEVZUAT", "ILGILI MEVZUAT", "KANUN MADDESİ", "MEVZUAT",
    ],
    config.SECTION_KARAR: ["KARAR", "HÜKÜM", "HUKUM", "SONUÇ", "SONUC", "HÜKÜM VE SONUÇ"],
    config.SECTION_GEREKCE: ["GEREKÇE", "GEREKCE", "GEREKÇESİ", "DEĞERLENDİRME", "DEGERLENDIRME", "İNCELEME"],
}

# Map every variant back to its canonical section, longest-first so that
# "HÜKÜM VE SONUÇ" is preferred over "HÜKÜM".
_VARIANT_TO_SECTION: dict[str, str] = {}
for _canon, _variants in _SECTION_VARIANTS.items():
    for _v in _variants:
        _VARIANT_TO_SECTION[_v] = _canon
_ALL_VARIANTS = sorted(_VARIANT_TO_SECTION, key=len, reverse=True)

# A marker = one of the variants followed by an optional ":" / "-" and a space.
# The lookahead keeps the marker attached to the section that follows it.
_MARKER_RE = re.compile(
    r"(?=\b(?:" + "|".join(re.escape(v) for v in _ALL_VARIANTS) + r")\s*[:\-]?\s)"
)

_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def _canonical_for(fragment: str) -> str | None:
    """Return the canonical section if `fragment` starts with a known marker."""
    head = fragment[:40].upper()
    for variant in _ALL_VARIANTS:
        if head.startswith(variant):
            return _VARIANT_TO_SECTION[variant]
    return None


def extract_sections(text: str) -> dict[str, str]:
    """
    Split a decision into a dict {section: text}, in order of appearance.

    Handles header variations (OYAL:, OYAL , OLAYLAR, ...). If a section
    appears more than once its parts are concatenated. When no known marker is
    present the whole text is returned under the GENEL (general) section.
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    fragments = [f.strip() for f in _MARKER_RE.split(text) if f and f.strip()]

    sections: dict[str, str] = {}
    matched_any = False
    for frag in fragments:
        canon = _canonical_for(frag)
        if canon is None:
            # Leading text before the first recognised header -> facts (OYAL).
            canon = config.SECTION_OYAL if not sections else config.SECTION_GENERAL
        else:
            matched_any = True
        sections[canon] = (sections.get(canon, "") + " " + frag).strip()

    if not matched_any:
        return {config.SECTION_GENERAL: text.strip()}
    return sections


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_section(text: str, size: int, overlap: int) -> list[str]:
    """
    Split `text` into <= `size`-char chunks on sentence boundaries, carrying
    `overlap` characters of context from the previous chunk into the next.
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= size:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sent}".strip()
            # A single sentence longer than `size` is hard-split.
            while len(current) > size:
                chunks.append(current[:size])
                current = current[size - overlap:]
    if current:
        chunks.append(current)
    return chunks


def chunk_decision(text: str) -> list[tuple[str, str, bool]]:
    """
    Chunk one decision.

    Returns a list of (section, chunk_text, is_atomic) tuples in document
    order. Atomic sections (KANUN, KARAR) yield exactly one chunk.
    """
    out: list[tuple[str, str, bool]] = []
    for section, sec_text in extract_sections(text).items():
        sec_text = sec_text.strip()
        if not sec_text:
            continue
        if section in config.ATOMIC_SECTIONS:
            out.append((section, sec_text, True))
        else:
            for piece in chunk_section(sec_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                if piece.strip():
                    out.append((section, piece.strip(), False))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if not config.CLEAN_CSV.exists():
        sys.exit(
            f"Cleaned CSV not found at {config.CLEAN_CSV}.\n"
            "Run 03_preprocess.py first."
        )

    config.ensure_dirs()
    df = pd.read_csv(config.CLEAN_CSV)
    df["text"] = df["text"].fillna("").astype(str)

    rows: list[dict] = []
    chunk_id = 0
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Chunking", unit="doc"):
        doc_id = r.get("id", "")
        for section, piece, is_atomic in chunk_decision(r["text"]):
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "text": piece,
                    "section": section,
                    "original_doc_id": doc_id,
                    "length": len(piece),
                    "is_atomic": is_atomic,
                }
            )
            chunk_id += 1

    chunks = pd.DataFrame(
        rows,
        columns=["chunk_id", "text", "section", "original_doc_id", "length", "is_atomic"],
    )
    chunks.to_csv(config.CHUNKS_CSV, index=False)

    # ---- Metrics --------------------------------------------------------- #
    n_docs = len(df)
    n_chunks = len(chunks)
    print("\n===== Chunking metrics =====")
    print(f"Total decisions        : {n_docs:,}")
    print(f"Total chunks           : {n_chunks:,}")
    print(f"Avg chunks per decision: {n_chunks / max(n_docs, 1):.2f}")

    print("\nSection distribution:")
    for sec, cnt in chunks["section"].value_counts().items():
        print(f"  {sec:<10}: {cnt:>8,}  ({100 * cnt / max(n_chunks, 1):5.1f}%)")

    print("\nChunk length (characters):")
    lengths = chunks["length"]
    if n_chunks:
        print(f"  min  : {lengths.min():,}")
        print(f"  max  : {lengths.max():,}")
        print(f"  mean : {lengths.mean():,.1f}")
        print(f"  std  : {lengths.std():,.1f}")

    atomic = int(chunks["is_atomic"].sum())
    print(f"\nAtomic vs split ratio  : {atomic:,} atomic / "
          f"{n_chunks - atomic:,} split "
          f"({100 * atomic / max(n_chunks, 1):.1f}% atomic)")
    print(f"\n💾 Saved chunks → {config.CHUNKS_CSV}")


if __name__ == "__main__":
    main()
