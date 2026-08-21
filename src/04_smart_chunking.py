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

import config

# NOTE: pandas / tqdm are imported inside main() so the pure chunking functions
# (extract_sections, chunk_section, chunk_decision) can be imported and unit
# tested without those heavy dependencies installed (see validate_chunks.py).

# --------------------------------------------------------------------------- #
# Section detection
# --------------------------------------------------------------------------- #
# Each canonical section maps to the header variants that introduce it in the
# corpus. Matching is CASE-SENSITIVE (uppercase): real headers are upper case,
# while the same words in running text are lower case, so this avoids splitting
# mid-sentence on a common word like "karar".
# Variants were expanded after inspecting the real corpus (inspect_sections.py):
# the dataset mostly uses SONUÇ/HÜKÜM(LER)/DAVA and the transition phrase
# "GEREĞİ GÖRÜŞÜLÜP DÜŞÜNÜLDÜ" rather than OYAL/KANUN/KARAR/GEREKÇE.
_SECTION_VARIANTS: dict[str, list[str]] = {
    config.SECTION_OYAL: [
        "OLAYLAR", "OLAY", "OYAL", "MADDİ OLAY", "MADDI OLAY", "VAKIALAR",
        "DAVA", "DAVA TÜRÜ", "İDDİA", "İSTEM", "DAVA VE KARAR",
    ],
    config.SECTION_KANUN: [
        "KANUN", "İLGİLİ KANUN", "ILGILI KANUN", "YASAL DAYANAK",
        "İLGİLİ MEVZUAT", "ILGILI MEVZUAT", "KANUN MADDESİ", "MEVZUAT",
    ],
    config.SECTION_KARAR: [
        "KARAR", "HÜKÜM", "HÜKÜMLER", "HUKUM", "SONUÇ", "SONUC",
        "HÜKÜM VE SONUÇ", "SONUÇ VE KARAR",
    ],
    config.SECTION_GEREKCE: [
        "GEREKÇE", "GEREKCE", "GEREKÇESİ", "DEĞERLENDİRME", "DEGERLENDIRME",
        "İNCELEME", "GEREĞİ GÖRÜŞÜLÜP DÜŞÜNÜLDÜ", "GEREĞİ DÜŞÜNÜLDÜ",
        "GEREĞİ GÖRÜŞÜLDÜ",
    ],
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

# --- Cue / positional detection (for decisions that carry no header) ------- #
# The transition phrase that separates the procedural preamble + facts from the
# court's own reasoning. Case-INSENSITIVE, because it is often written in mixed
# case ("Gereği görüşülüp düşünüldü;").
_TRANSITION_RE = re.compile(
    r"gere[ğg]i\s+(?:g[öo]r[üu][şs][üu]l[üu]p\s+)?(?:d[üu][şs][üu]n[üu]ld[üu]|g[öo]r[üu][şs][üu]ld[üu])",
    re.IGNORECASE,
)
# STRONG decree cues that mark the final ruling. Deliberately excludes bare
# "karar veril…", which appears in ~99.7% of decisions (including narrative) and
# so cannot discriminate the decree — using it would mislabel facts as KARAR.
_RULING_TAIL_RE = re.compile(
    r"(onanmasına|bozulmasına|bozularak|kabul[üu]ne|reddine|tesciline|"
    r"tahliyesine|iptaline|oybirli[ğg]iyle|oy\s*ço[ğg]unlu[ğg]uyla|"
    r"esastan\s+red|hükm[üu]n\s+(?:onan|bozul))",
    re.IGNORECASE,
)


def _canonical_for(fragment: str) -> str | None:
    """
    Return the canonical section if `fragment` starts with a known marker.

    The marker must end on a word boundary, so "DAVACI" (plaintiff) does NOT
    match the "DAVA" (claim) header, and "KARARI" does not match "KARAR".
    """
    head = fragment[:48].upper()
    for variant in _ALL_VARIANTS:
        if head.startswith(variant):
            after = head[len(variant):len(variant) + 1]
            if not after or not after.isalpha():
                return _VARIANT_TO_SECTION[variant]
    return None


def _header_sections(text: str) -> dict[str, str]:
    """Header-based split. Returns {} (no non-GENEL header found) or the split."""
    fragments = [f.strip() for f in _MARKER_RE.split(text) if f and f.strip()]
    sections: dict[str, str] = {}
    matched_any = False
    for frag in fragments:
        canon = _canonical_for(frag)
        if canon is None:
            canon = config.SECTION_OYAL if not sections else config.SECTION_GENERAL
        else:
            matched_any = True
        sections[canon] = (sections.get(canon, "") + " " + frag).strip()
    return sections if matched_any else {}


def _find_ruling_tail_start(text: str) -> int | None:
    """Index where the final ruling begins (start of that sentence), or None."""
    half = len(text) // 2
    pos = None
    for m in _RULING_TAIL_RE.finditer(text):
        if m.start() >= half:
            pos = m.start()
            break
    if pos is None:
        matches = list(_RULING_TAIL_RE.finditer(text))
        if matches:
            pos = matches[-1].start()
    if pos is None:
        return None
    prev = text.rfind(".", 0, pos)
    return prev + 1 if prev != -1 else 0


def _cue_positional_sections(text: str) -> dict[str, str]:
    """
    Section a header-less decision by cue + position:
      - with a "gereği görüşülüp düşünüldü" transition: before -> facts (OYAL),
        after -> reasoning (GEREKÇE) (the ruling is carved out afterwards);
      - without a transition: only structure the text when a STRONG decree cue
        marks a ruling tail (front -> OYAL, tail -> KARAR);
      - if neither is present, no structure was found -> GENEL (honest).
    """
    m = _TRANSITION_RE.search(text)
    if m:
        out: dict[str, str] = {}
        facts = text[:m.start()].strip()
        rest = text[m.end():].strip()
        if facts:
            out[config.SECTION_OYAL] = facts
        if rest:
            out[config.SECTION_GEREKCE] = rest
        return out or {config.SECTION_GENERAL: text.strip()}

    rs = _find_ruling_tail_start(text)
    if rs and rs > 0:
        return {
            config.SECTION_OYAL: text[:rs].strip(),
            config.SECTION_KARAR: text[rs:].strip(),
        }
    return {config.SECTION_GENERAL: text.strip()}


def _carve_ruling(sections: dict[str, str]) -> dict[str, str]:
    """
    If no explicit KARAR section exists, try to split the final decree out of
    the last narrative section so the ruling gets its (high) rerank weight.
    """
    if config.SECTION_KARAR in sections or not sections:
        return sections
    last = list(sections)[-1]
    if last == config.SECTION_KANUN:
        return sections
    txt = sections[last]
    rs = _find_ruling_tail_start(txt)
    if rs and rs > 0:
        head, ruling = txt[:rs].strip(), txt[rs:].strip()
        if ruling:
            if head:
                sections[last] = head
            else:
                del sections[last]
            sections[config.SECTION_KARAR] = ruling
    elif len(txt) < 400 and _RULING_TAIL_RE.search(txt[-160:]):
        # Terse single-sentence decree ("...hükmün ONANMASINA ... karar verildi").
        del sections[last]
        sections[config.SECTION_KARAR] = txt
    return sections


def extract_sections(text: str) -> dict[str, str]:
    """
    Split a decision into {section: text}.

    Hybrid strategy:
      1. If explicit uppercase headers are present, split on them.
      2. Otherwise fall back to cue/positional detection (transition phrase +
         ruling-tail cues), so header-less decisions are still sectioned
         instead of collapsing into GENEL.
      3. Either way, carve the final ruling into KARAR when it isn't already a
         section, so the decree keeps its rerank weight.
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    headers = _header_sections(text)
    sections = headers if headers else _cue_positional_sections(text)
    return _carve_ruling(sections)


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
    order. Atomic sections (KANUN, KARAR) yield one chunk unless they exceed
    MAX_ATOMIC_CHARS, in which case they are split (a huge "atomic" chunk is
    useless for embedding). Split chunks shorter than MIN_CHUNK_CHARS are
    dropped; atomic chunks are exempt so a short ruling still survives.
    """
    out: list[tuple[str, str, bool]] = []
    for section, sec_text in extract_sections(text).items():
        sec_text = sec_text.strip()
        if not sec_text:
            continue
        atomic = section in config.ATOMIC_SECTIONS
        if atomic and len(sec_text) <= config.MAX_ATOMIC_CHARS:
            out.append((section, sec_text, True))
        else:
            # Non-atomic, or an oversize atomic section that must be split.
            for piece in chunk_section(sec_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                piece = piece.strip()
                if len(piece) >= config.MIN_CHUNK_CHARS:
                    out.append((section, piece, False))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    import pandas as pd
    from tqdm import tqdm

    if not config.CLEAN_CSV.exists():
        sys.exit(
            f"Cleaned CSV not found at {config.CLEAN_CSV}.\n"
            "Run 03_preprocess.py first."
        )

    config.ensure_dirs()
    df = pd.read_csv(config.CLEAN_CSV)
    df["text"] = df["text"].fillna("").astype(str)

    rows: list[dict] = []
    seen: set[str] = set()
    n_dupes = 0
    chunk_id = 0
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Chunking", unit="doc"):
        doc_id = r.get("id", "")
        for section, piece, is_atomic in chunk_decision(r["text"]):
            # Drop exact-duplicate chunk text across the corpus (boilerplate
            # that survived preprocessing repeats verbatim in many decisions).
            if config.DEDUP_CHUNKS:
                key = piece.casefold()
                if key in seen:
                    n_dupes += 1
                    continue
                seen.add(key)
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
    if config.DEDUP_CHUNKS:
        print(f"Deduplication: dropped {n_dupes:,} exact-duplicate chunks.")

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
