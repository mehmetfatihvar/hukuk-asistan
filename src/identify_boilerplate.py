"""
identify_boilerplate.py
=======================
Find boilerplate (procedural / template) text in the chunks so it can be
stripped in 03_preprocess.py, leaving only content-bearing legal text.

Two modes:

  1. AUTOMATIC MINING (default, scales to 604K chunks)
     Boilerplate is text that repeats across *different* decisions. We split
     every chunk into sentences, normalise away the parts that vary between
     cases (dates, case numbers, article numbers, plain digits), then count
     how many DISTINCT decisions each normalised sentence appears in. Sentences
     shared by many decisions are boilerplate candidates, ranked by document
     coverage.

  2. MANUAL REVIEW (--review, interactive)
     Draw a stratified sample (even across OYAL/KANUN/KARAR/GEREKÇE/GENEL),
     print each chunk with its context, and ask "Is this boilerplate? (y/N)"
     plus a category (procedural / transition / repetitive / content). Labels
     are saved so the sample can be revisited.

Output (both modes contribute):
    data/processed/boilerplate_patterns.json
      { seed_patterns: [...], mined_patterns: [...], reviewed: [...] }

Run:
    python src/identify_boilerplate.py                 # mine + print seeds
    python src/identify_boilerplate.py --review         # interactive 30-sample
    python src/identify_boilerplate.py --show           # print sample, no prompt
    python src/identify_boilerplate.py --min-docs 50 --top 40
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import config

# ---- reuse the chunker's sentence splitter -------------------------------- #
_spec = importlib.util.spec_from_file_location(
    "chunking", Path(__file__).with_name("04_smart_chunking.py")
)
_chunking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chunking)  # type: ignore[union-attr]
_SENTENCE_RE = _chunking._SENTENCE_RE

OUTPUT_JSON = config.PROCESSED_DIR / "boilerplate_patterns.json"

SECTIONS = [
    config.SECTION_OYAL,
    config.SECTION_KANUN,
    config.SECTION_KARAR,
    config.SECTION_GEREKCE,
    config.SECTION_GENERAL,
]

CATEGORIES = {
    "p": "procedural",   # procedural / template text
    "t": "transition",   # transition phrases
    "r": "repetitive",   # repetitive sentences
    "c": "content",      # content-bearing (NOT boilerplate)
}


# --------------------------------------------------------------------------- #
# Seed patterns — well-known Turkish legal boilerplate (regex, case-insensitive)
# These are used both as a starting point and to feed 03_preprocess.py.
# --------------------------------------------------------------------------- #
SEED_PATTERNS: list[dict] = [
    {"category": "procedural", "regex": r"taraflar arasında(ki)?\s+.*?dava",
     "example": "Taraflar arasındaki davanın..."},
    {"category": "procedural", "regex": r"dosya(sı)?\s+incelen(di|erek)\s+gereği\s+düşünüldü",
     "example": "...dosya incelendi, gereği düşünüldü..."},
    {"category": "procedural", "regex": r"temyiz\s+edilmesi\s+üzerine",
     "example": "...temyiz edilmesi üzerine..."},
    {"category": "procedural", "regex": r"mahkemesince\s+verilen\s+.*?karar",
     "example": "...mahkemesince verilen ... karar..."},
    {"category": "procedural", "regex": r"süresi\s+içinde\s+temyiz\s+ed",
     "example": "...süresi içinde temyiz eden..."},
    {"category": "procedural", "regex": r"hükmün\s+.*?(bozulması|onanması)\s+.*?talep",
     "example": "...hükmün bozulması talep edilmiştir..."},
    {"category": "transition", "regex": r"yapılan\s+yargılama\s+sonunda",
     "example": "Yapılan yargılama sonunda..."},
    {"category": "transition", "regex": r"tüm\s+dosya\s+kapsamı\s+(birlikte\s+)?değerlendiril",
     "example": "...tüm dosya kapsamı değerlendirildiğinde..."},
    {"category": "procedural", "regex": r"gerekçeli\s+karar(ın)?\s+.*?tebliğ",
     "example": "...gerekçeli kararın tebliği..."},
    {"category": "procedural", "regex": r"usul\s+ve\s+yasaya\s+uygun",
     "example": "...usul ve yasaya uygun..."},
]


# --------------------------------------------------------------------------- #
# Normalisation for mining
# --------------------------------------------------------------------------- #
_NORM_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}"), "<DATE>"),   # 12/03/2019
    (re.compile(r"\b\d{4}\s*/\s*\d+\b"), "<NO>"),                   # 2018/1234 (esas/karar)
    (re.compile(r"\bm(?:adde|\.)\s*\d+\b", re.IGNORECASE), "<MADDE>"),  # m.141 / madde 141
    (re.compile(r"\b\d+\b"), "<NUM>"),                             # any other number
    (re.compile(r"\s+"), " "),
]


def normalize_sentence(sent: str) -> str:
    """Collapse case-specific tokens so template sentences match across cases."""
    s = sent.strip().casefold()
    for pat, repl in _NORM_RULES:
        s = pat.sub(repl, s)
    return s.strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(str(text)) if s.strip()]


# --------------------------------------------------------------------------- #
# Automatic mining
# --------------------------------------------------------------------------- #
def mine_patterns(
    records,               # iterable of (text, doc_id)
    min_docs: int,
    top: int,
    min_words: int = 4,
) -> list[dict]:
    """
    Return boilerplate candidates: normalised sentences that appear in at least
    `min_docs` distinct decisions, ranked by number of distinct decisions.

    `records` is an iterable of (chunk_text, original_doc_id) so this function
    is testable without pandas.
    """
    docs_per_sentence: dict[str, set] = defaultdict(set)
    example_of: dict[str, str] = {}
    total_docs: set = set()

    for text, doc_id in records:
        total_docs.add(doc_id)
        seen_in_this_doc: set[str] = set()
        for sent in _sentences(text):
            if len(sent.split()) < min_words:
                continue
            norm = normalize_sentence(sent)
            if len(norm.split()) < min_words:
                continue
            if norm not in seen_in_this_doc:
                docs_per_sentence[norm].add(doc_id)
                seen_in_this_doc.add(norm)
                example_of.setdefault(norm, sent)

    n_docs = max(len(total_docs), 1)
    candidates = [
        {
            "pattern": norm,
            "example": example_of[norm],
            "doc_count": len(docs),
            "doc_coverage": round(len(docs) / n_docs, 4),
            "category": "repetitive",
        }
        for norm, docs in docs_per_sentence.items()
        if len(docs) >= min_docs
    ]
    candidates.sort(key=lambda c: c["doc_count"], reverse=True)
    return candidates[:top]


# --------------------------------------------------------------------------- #
# Sampling / review
# --------------------------------------------------------------------------- #
def stratified_sample(df, n: int, seed: int = 42):
    """Sample ~evenly across sections (falls back gracefully if a section is small)."""
    random.seed(seed)
    per = max(1, n // len(SECTIONS))
    picked = []
    for sec in SECTIONS:
        pool = df[df["section"] == sec]
        if len(pool):
            picked.append(pool.sample(min(per, len(pool)), random_state=seed))
    import pandas as pd
    out = pd.concat(picked) if picked else df.head(0)
    if len(out) > n:
        out = out.sample(n, random_state=seed)
    return out.reset_index(drop=True)


def print_chunk(row, idx: int, total: int) -> None:
    print("\n" + "─" * 72)
    print(f"[{idx}/{total}]  chunk_id={row['chunk_id']}  "
          f"section={row['section']}  doc_id={row['original_doc_id']}  "
          f"len={row['length']}")
    print("─" * 72)
    print(str(row["text"]))


def interactive_review(df, n: int) -> list[dict]:
    sample = stratified_sample(df, n)
    total = len(sample)
    print(f"\nManual review — {total} chunks (stratified by section).")
    print("For each: y = boilerplate, N = content (default). Then a category:")
    print("  p=procedural  t=transition  r=repetitive  c=content   (q to quit early)\n")

    reviewed: list[dict] = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print_chunk(row, i, total)
        ans = input("Is this boilerplate? [y/N/q] ").strip().lower()
        if ans == "q":
            break
        is_bp = ans == "y"
        cat = "content"
        if is_bp:
            craw = input("  category [p/t/r] (default r): ").strip().lower()
            cat = CATEGORIES.get(craw, "repetitive")
        reviewed.append(
            {
                "chunk_id": int(row["chunk_id"]),
                "section": row["section"],
                "original_doc_id": str(row["original_doc_id"]),
                "is_boilerplate": is_bp,
                "category": cat,
                "first_sentence": _sentences(row["text"])[:1],
            }
        )

    bp = [r for r in reviewed if r["is_boilerplate"]]
    print("\n" + "=" * 72)
    print(f"Reviewed {len(reviewed)} chunks — {len(bp)} marked boilerplate.")
    by_cat = Counter(r["category"] for r in bp)
    for cat, cnt in by_cat.items():
        print(f"  {cat:<12}: {cnt}")
    return reviewed


def show_sample(df, n: int) -> None:
    sample = stratified_sample(df, n)
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print_chunk(row, i, len(sample))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Identify boilerplate in chunks.csv.")
    ap.add_argument("--review", action="store_true", help="interactive Y/N review")
    ap.add_argument("--show", action="store_true", help="print stratified sample, no prompt")
    ap.add_argument("--sample", type=int, default=30, help="sample size for review/show")
    ap.add_argument("--min-docs", type=int, default=25,
                    help="min distinct decisions a sentence must appear in to be boilerplate")
    ap.add_argument("--top", type=int, default=30, help="max mined patterns to output")
    args = ap.parse_args()

    import pandas as pd

    if not config.CHUNKS_CSV.exists():
        sys.exit(f"chunks.csv not found at {config.CHUNKS_CSV}. Run 04_smart_chunking.py first.")
    df = pd.read_csv(config.CHUNKS_CSV)
    df["text"] = df["text"].fillna("").astype(str)
    print(f"Loaded {len(df):,} chunks from {config.CHUNKS_CSV}")

    reviewed: list[dict] = []
    if args.show:
        show_sample(df, args.sample)
    if args.review:
        reviewed = interactive_review(df, args.sample)

    # Automatic mining (always runs unless only --show/--review requested).
    print("\n" + "=" * 72)
    print(f"AUTOMATIC MINING (sentences shared by >= {args.min_docs} decisions)")
    print("=" * 72)
    records = zip(df["text"].tolist(), df["original_doc_id"].astype(str).tolist())
    mined = mine_patterns(records, min_docs=args.min_docs, top=args.top)
    if not mined:
        print("  No sentence met the threshold — try a lower --min-docs.")
    for c in mined:
        print(f"  [{c['doc_count']:>6} docs | {c['doc_coverage']:6.2%}]  {c['example'][:90]}")

    # Consolidated output.
    out = {
        "seed_patterns": SEED_PATTERNS,
        "mined_patterns": mined,
        "reviewed": reviewed,
        "params": {"min_docs": args.min_docs, "top": args.top},
    }
    config.ensure_dirs()
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved patterns → {OUTPUT_JSON}")
    print(f"   seeds={len(SEED_PATTERNS)}  mined={len(mined)}  reviewed={len(reviewed)}")
    print("\nNext: feed the confirmed patterns into 03_preprocess.py to strip them.")


if __name__ == "__main__":
    main()
