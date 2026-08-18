"""
inspect_sections.py
===================
Diagnose *why* section detection misses (44% GENEL) by looking at the real
cleaned text — before we tune _SECTION_VARIANTS in 04_smart_chunking.py.

It reports three things:

  1. HEADER CANDIDATES — the most frequent UPPERCASE-word(s)-before-a-colon
     patterns in the corpus (e.g. "SUÇ :", "DAVA :", "SONUÇ :"). These are the
     real section markers the dataset actually uses, which may differ from our
     assumed OYAL/KANUN/KARAR/GEREKÇE.

  2. RULING-CUE COVERAGE — what fraction of decisions contain common ruling
     verbs ("bozulmasına", "onanmasına", "kabulüne", "karar verilmiştir", …).
     High coverage means a cue/positional detector could recover KARAR labels
     even when no header exists.

  3. RAW SAMPLES — the first ~900 chars of a few decisions that currently fall
     into GENEL (no header detected), for eyeballing their structure.

    Input : data/processed/decisions_clean.csv  (run 03_preprocess.py first)

Run:
    python src/inspect_sections.py               # report + 5 GENEL samples
    python src/inspect_sections.py --dump 10 --sample 5000
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

import config

# ---- reuse extract_sections from the chunker ----------------------------- #
_spec = importlib.util.spec_from_file_location(
    "chunking", Path(__file__).with_name("04_smart_chunking.py")
)
_chunking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chunking)  # type: ignore[union-attr]
extract_sections = _chunking.extract_sections

# UPPERCASE word(s) (Turkish letters) immediately before a colon = a candidate
# section header. 2–4 words, each >= 2 upper-case letters.
_UPPER = r"[A-ZÇĞİÖŞÜ]{2,}"
_HEADER_CANDIDATE_RE = re.compile(rf"\b({_UPPER}(?:\s+{_UPPER}){{0,3}})\s*:")

# Common Turkish ruling cues (lower-cased match).
_RULING_CUES = [
    "karar veril", "hükmün", "bozulmasına", "onanmasına", "kabulüne",
    "reddine", "mahkumiyet", "beraat", "tazminat", "hüküm kurul",
    "temyiz", "istinaf",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect real section structure.")
    ap.add_argument("--sample", type=int, default=5000,
                    help="decisions to scan for stats")
    ap.add_argument("--dump", type=int, default=5,
                    help="GENEL-only decisions to print in full-ish")
    ap.add_argument("--top", type=int, default=30, help="header candidates to list")
    args = ap.parse_args()

    import pandas as pd

    if not config.CLEAN_CSV.exists():
        sys.exit(f"{config.CLEAN_CSV} not found. Run 03_preprocess.py first.")
    df = pd.read_csv(config.CLEAN_CSV)
    df["text"] = df["text"].fillna("").astype(str)
    scan = df.sample(min(args.sample, len(df)), random_state=42)

    header_counter: Counter = Counter()
    cue_docs: Counter = Counter()
    section_counter: Counter = Counter()
    genel_only_ids: list[str] = []

    for _, r in scan.iterrows():
        text = r["text"]
        low = text.lower()

        for m in _HEADER_CANDIDATE_RE.findall(text):
            header_counter[m.strip()] += 1
        for cue in _RULING_CUES:
            if cue in low:
                cue_docs[cue] += 1

        secs = set(extract_sections(text).keys())
        for s in secs:
            section_counter[s] += 1
        if secs == {config.SECTION_GENERAL}:
            genel_only_ids.append(str(r.get("id", "")))

    n = len(scan)
    print("=" * 72)
    print(f"SECTION STRUCTURE INSPECTION  (sampled {n:,} decisions)")
    print("=" * 72)

    print("\n[1] TOP HEADER CANDIDATES  (UPPERCASE ... : )")
    print("    count   header")
    for header, cnt in header_counter.most_common(args.top):
        print(f"    {cnt:>6}  {header}")

    print("\n[2] CURRENT SECTION DETECTION (share of sampled decisions)")
    for s in ["OYAL", "KANUN", "KARAR", "GEREKÇE", "GENEL"]:
        c = section_counter.get(s, 0)
        print(f"    {s:<10}: {c:>6}  ({100*c/max(n,1):5.1f}%)")

    print("\n[3] RULING-CUE COVERAGE (share of decisions containing the cue)")
    for cue, c in cue_docs.most_common():
        print(f"    {c:>6}  ({100*c/max(n,1):5.1f}%)  \"{cue}\"")

    print(f"\n[4] RAW SAMPLES — {args.dump} GENEL-only decisions (first ~900 chars)")
    genel_df = df[df["id"].astype(str).isin(set(genel_only_ids))]
    for _, r in genel_df.sample(min(args.dump, len(genel_df)), random_state=7).iterrows():
        print("\n" + "-" * 72)
        print(f"id={r.get('id','')}")
        print(str(r["text"])[:900])

    print("\n" + "=" * 72)
    print("Paste sections [1], [3] and a couple of [4] samples back to calibrate "
          "_SECTION_VARIANTS / a cue detector.")


if __name__ == "__main__":
    main()
