"""
validate_chunks.py
==================
Validate that the smart chunker (04_smart_chunking.py) behaves correctly.

Two layers of validation, neither of which needs the 1.75 GB dataset for the
first layer:

  1. FIXTURE TESTS (always runnable)
     Synthetic-but-realistic Yargıtay decisions whose sections are known in
     advance. We assert that extract_sections() finds the right sections and
     that chunk_decision() applies the atomic/split policy, size limit and
     overlap correctly.

  2. DATA HEALTH REPORT (runs when data/processed/chunks.csv exists)
     Structural invariants that every produced chunk MUST satisfy
     (size limit, valid section, is_atomic correctness, length column,
     unique ids) plus statistics that reveal *quality* problems:
       - GENEL-only rate  : % of decisions where NO section header was
                            detected (high == section detection is failing)
       - over-size split chunks (must be 0)
       - word-coverage    : fraction of each decision's words preserved in its
                            chunks (must be ~1.0 == no content lost),
                            checked against decisions_clean.csv when present.

Run:
    python src/validate_chunks.py            # fixtures (+ data if present)
    python src/validate_chunks.py --data     # force data checks
    python src/validate_chunks.py --sample 5 # dump 5 real decisions' splits
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import re
import sys
from pathlib import Path

import config

# ---- import the digit-prefixed chunking module (pure functions only) ------ #
_spec = importlib.util.spec_from_file_location(
    "chunking", Path(__file__).with_name("04_smart_chunking.py")
)
_chunking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chunking)  # type: ignore[union-attr]
extract_sections = _chunking.extract_sections
chunk_decision = _chunking.chunk_decision

VALID_SECTIONS = {
    config.SECTION_OYAL,
    config.SECTION_KANUN,
    config.SECTION_KARAR,
    config.SECTION_GEREKCE,
    config.SECTION_GENERAL,
}

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _has_overlap(a: str, b: str, min_len: int = 15) -> bool:
    """
    True if the end of chunk `a` and the start of chunk `b` share a substring
    of at least `min_len` characters (evidence that CHUNK_OVERLAP worked).
    """
    tail = a[-(2 * config.CHUNK_OVERLAP):]
    head = b[: 2 * config.CHUNK_OVERLAP]
    # Slide the head's prefixes against the tail.
    for start in range(0, len(head) - min_len + 1):
        probe = head[start:start + min_len]
        if probe in tail:
            return True
    return False


# --------------------------------------------------------------------------- #
# Layer 1 — fixture tests
# --------------------------------------------------------------------------- #
# Each fixture: text, the sections we expect extract_sections to return.
_LONG = (
    "Mahkemece toplanan deliller değerlendirilmiştir. " * 40
)  # ~2000 chars, forces splitting

_FIXTURES = [
    {
        "name": "full decision (all 4 sections)",
        "text": (
            "OYAL: Davacı taşınmazı 2020 yılında satın aldı. "
            "Davalı elektrik kaçağı yaptığını iddia etti. "
            "KANUN: TCK m.141 karşılıksız yararlanma suçunu düzenler. "
            "KARAR: Mahkeme sanığın mahkumiyetine karar verdi. "
            "GEREKÇE: Kanun hükmüne göre ceza tayin edilmiştir."
        ),
        "expect": {"OYAL", "KANUN", "KARAR", "GEREKÇE"},
    },
    {
        "name": "header variants (OLAYLAR / HÜKÜM / DEĞERLENDİRME)",
        "text": (
            "OLAYLAR: Taraflar arasında kira ilişkisi vardır. "
            "DEĞERLENDİRME: Kiracının temerrüdü sabittir. "
            "HÜKÜM: Tahliyeye karar verilmiştir."
        ),
        # OLAYLAR->OYAL, DEĞERLENDİRME->GEREKÇE, HÜKÜM->KARAR
        "expect": {"OYAL", "GEREKÇE", "KARAR"},
    },
    {
        "name": "no markers -> GENEL",
        "text": "Bu metinde hiçbir bölüm başlığı yoktur, sadece düz metin bulunur.",
        "expect": {"GENEL"},
    },
    {
        "name": "lowercase 'karar' must NOT start a section",
        "text": "OYAL: Mahkeme gerekçesini açıkladı ve karar verildi böylece dava bitti.",
        "expect": {"OYAL"},
    },
    {
        "name": "long GEREKÇE forces split with overlap",
        "text": "GEREKÇE: " + _LONG,
        "expect": {"GEREKÇE"},
    },
    {
        "name": "long KANUN stays atomic (never split)",
        "text": "KANUN: " + _LONG,
        "expect": {"KANUN"},
    },
]


def run_fixtures() -> bool:
    print("=" * 72)
    print("LAYER 1 — FIXTURE TESTS")
    print("=" * 72)
    all_ok = True

    for fx in _FIXTURES:
        problems: list[str] = []
        secs = extract_sections(fx["text"])
        got = set(secs.keys())
        if got != fx["expect"]:
            problems.append(f"sections {sorted(got)} != expected {sorted(fx['expect'])}")

        chunks = chunk_decision(fx["text"])

        # is_atomic must equal (section in ATOMIC_SECTIONS)
        for section, text, is_atomic in chunks:
            if section not in VALID_SECTIONS:
                problems.append(f"invalid section '{section}'")
            expected_atomic = section in config.ATOMIC_SECTIONS
            if is_atomic != expected_atomic:
                problems.append(f"is_atomic={is_atomic} wrong for section {section}")
            if not is_atomic and len(text) > config.CHUNK_SIZE:
                problems.append(f"split chunk exceeds CHUNK_SIZE ({len(text)} chars)")

        # Atomic sections must be exactly ONE chunk.
        for atomic_sec in config.ATOMIC_SECTIONS:
            cnt = sum(1 for s, _, _ in chunks if s == atomic_sec)
            if atomic_sec in got and cnt != 1:
                problems.append(f"{atomic_sec} produced {cnt} chunks (must be 1)")

        # Long-split fixture: expect >1 chunk AND detectable overlap.
        if fx["name"].startswith("long GEREKÇE"):
            gk = [t for s, t, _ in chunks if s == "GEREKÇE"]
            if len(gk) < 2:
                problems.append(f"expected multiple chunks, got {len(gk)}")
            elif not any(_has_overlap(gk[i], gk[i + 1]) for i in range(len(gk) - 1)):
                problems.append("no overlap detected between consecutive chunks")

        # Long-atomic fixture: expect exactly ONE chunk despite length.
        if fx["name"].startswith("long KANUN"):
            kn = [t for s, t, _ in chunks if s == "KANUN"]
            if len(kn) != 1:
                problems.append(f"KANUN split into {len(kn)} chunks (must stay atomic)")

        status = "PASS" if not problems else "FAIL"
        all_ok = all_ok and not problems
        print(f"  [{status}] {fx['name']}")
        for p in problems:
            print(f"         ✗ {p}")

    print(f"\nFixtures: {'ALL PASSED ✅' if all_ok else 'FAILURES ❌'}")
    return all_ok


# --------------------------------------------------------------------------- #
# Layer 2 — structural invariants on produced chunks
# --------------------------------------------------------------------------- #
def check_invariants(rows: list[dict]) -> list[str]:
    """Return a list of invariant-violation messages (empty == all good)."""
    problems: list[str] = []
    seen_ids: set[int] = set()

    for r in rows:
        cid = r["chunk_id"]
        text = str(r["text"])
        section = r["section"]
        length = int(r["length"])
        is_atomic = bool(r["is_atomic"])

        if cid in seen_ids:
            problems.append(f"duplicate chunk_id {cid}")
        seen_ids.add(cid)

        if not text.strip():
            problems.append(f"chunk {cid}: empty text")
        if section not in VALID_SECTIONS:
            problems.append(f"chunk {cid}: invalid section '{section}'")
        if length != len(text):
            problems.append(f"chunk {cid}: length column {length} != len(text) {len(text)}")
        if is_atomic != (section in config.ATOMIC_SECTIONS):
            problems.append(f"chunk {cid}: is_atomic={is_atomic} inconsistent with section {section}")
        if not is_atomic and length > config.CHUNK_SIZE:
            problems.append(f"chunk {cid}: split chunk over CHUNK_SIZE ({length})")

    return problems


# --------------------------------------------------------------------------- #
# Layer 2 — data health report
# --------------------------------------------------------------------------- #
def validate_data(sample: int = 0) -> bool:
    import pandas as pd

    if not config.CHUNKS_CSV.exists():
        print(f"\n(no chunks.csv at {config.CHUNKS_CSV} — run 04_smart_chunking.py first)")
        return True

    print("\n" + "=" * 72)
    print("LAYER 2 — DATA HEALTH REPORT")
    print("=" * 72)

    df = pd.read_csv(config.CHUNKS_CSV)
    rows = df.to_dict("records")
    n = len(rows)
    print(f"chunks.csv rows: {n:,}")

    # --- invariants ---
    problems = check_invariants(rows)
    if problems:
        print(f"\n❌ {len(problems)} invariant violation(s) (showing first 10):")
        for p in problems[:10]:
            print(f"   ✗ {p}")
    else:
        print("✅ Structural invariants: all chunks valid "
              "(size, section, is_atomic, length, unique ids).")

    # --- GENEL-only rate (section-detection miss rate) ---
    by_doc: dict[str, set[str]] = {}
    for r in rows:
        by_doc.setdefault(str(r["original_doc_id"]), set()).add(r["section"])
    n_docs = len(by_doc)
    genel_only = sum(1 for secs in by_doc.values() if secs == {config.SECTION_GENERAL})
    rate = 100 * genel_only / max(n_docs, 1)
    print(f"\nSection detection:")
    print(f"  decisions                 : {n_docs:,}")
    print(f"  GENEL-only (no header)    : {genel_only:,}  ({rate:.1f}%)")
    if rate > 50:
        print("  ⚠️  >50% of decisions have NO detected section — the corpus may "
              "not use these headers; review _SECTION_VARIANTS.")

    # --- word coverage vs cleaned source (content-loss check) ---
    if config.CLEAN_CSV.exists():
        clean = pd.read_csv(config.CLEAN_CSV)
        clean["id"] = clean["id"].astype(str)
        clean["text"] = clean["text"].fillna("").astype(str)
        ids = list(by_doc.keys())
        random.seed(42)
        pick = random.sample(ids, min(len(ids), 200))
        chunk_words_by_doc: dict[str, set[str]] = {}
        for r in rows:
            d = str(r["original_doc_id"])
            if d in set(pick):
                chunk_words_by_doc.setdefault(d, set()).update(_words(str(r["text"])))
        clean_by_id = dict(zip(clean["id"], clean["text"]))
        coverages = []
        for d in pick:
            orig = _words(clean_by_id.get(d, ""))
            if not orig:
                continue
            covered = len(orig & chunk_words_by_doc.get(d, set())) / len(orig)
            coverages.append(covered)
        if coverages:
            avg_cov = sum(coverages) / len(coverages)
            worst = min(coverages)
            print(f"\nContent preservation (sampled {len(coverages)} decisions):")
            print(f"  mean word coverage        : {avg_cov:.3f}")
            print(f"  worst word coverage       : {worst:.3f}")
            if avg_cov < 0.98:
                print("  ⚠️  Some content appears to be lost during chunking.")
            else:
                print("  ✅ Effectively no content lost.")

    # --- optional manual-inspection dump ---
    if sample > 0 and config.CLEAN_CSV.exists():
        print("\n" + "-" * 72)
        print(f"MANUAL INSPECTION — {sample} random decisions")
        print("-" * 72)
        clean = pd.read_csv(config.CLEAN_CSV)
        clean["text"] = clean["text"].fillna("").astype(str)
        for _, row in clean.sample(min(sample, len(clean)), random_state=7).iterrows():
            print(f"\n### doc id={row.get('id', '')}")
            for section, text, is_atomic in chunk_decision(row["text"]):
                tag = "ATOMIC" if is_atomic else "split"
                print(f"  [{section:<8} {tag}] ({len(text)} chars) {text[:90]}…")

    ok = not problems and rate <= 90
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the smart chunker.")
    ap.add_argument("--data", action="store_true", help="run data-health checks")
    ap.add_argument("--sample", type=int, default=0,
                    help="dump N real decisions' section splits for inspection")
    args = ap.parse_args()

    fixtures_ok = run_fixtures()
    data_ok = validate_data(sample=args.sample) if (args.data or args.sample or config.CHUNKS_CSV.exists()) else True

    print("\n" + "=" * 72)
    if fixtures_ok and data_ok:
        print("VALIDATION RESULT: PASS ✅")
        sys.exit(0)
    else:
        print("VALIDATION RESULT: FAIL ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()
