"""
03_preprocess.py
================
Clean and normalise the raw decision text.

    Input : data/raw/decisions_100k.csv
    Output: data/processed/decisions_clean.csv

Steps:
  1. Remove boilerplate header lines (MAHKEMESİ, TARİHİ, ...).
  2. Normalise whitespace (\\n, \\t, multiple spaces → single space).
  3. Remove learned boilerplate patterns discovered by identify_boilerplate.py
     (data/processed/boilerplate_patterns.json), if that file exists.
  4. Quality filtering (drop text shorter than MIN_TEXT_CHARS, drop empties).
  5. Save the cleaned CSV and print a before/after comparison.

The discovery → cleaning loop:
    03_preprocess → 04_smart_chunking → identify_boilerplate (writes JSON)
    → re-run 03_preprocess (now strips the confirmed patterns) → 04 …

Run:
    python src/03_preprocess.py
"""

from __future__ import annotations

import json
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

# Always-on formatting-artifact cleanup, independent of the learned patterns.
# The source dataset leaks markdown emphasis (**), a source label
# ("İçtihat Metni") and stray decision-number metadata into the text — pure
# noise that is safe to strip everywhere.
_ARTIFACT_RES: list[re.Pattern] = [
    re.compile(r"\*+"),                                    # markdown ** bold
    re.compile(r'"?\s*İçtihat\s+Metni\s*"?', re.IGNORECASE),  # source label
]


# --------------------------------------------------------------------------- #
# Learned boilerplate patterns (from identify_boilerplate.py)
# --------------------------------------------------------------------------- #
BOILERPLATE_JSON = config.PROCESSED_DIR / "boilerplate_patterns.json"

# Forward normalisation (value -> placeholder), mirror of identify_boilerplate,
# used to turn a human-confirmed raw review sentence into a matching regex.
_FORWARD_NORM: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}"), "<DATE>"),
    (re.compile(r"\b\d{4}\s*/\s*\d+\b"), "<NO>"),
    (re.compile(r"\bm(?:adde|\.)\s*\d+\b", re.IGNORECASE), "<MADDE>"),
    (re.compile(r"\b\d+\b"), "<NUM>"),
]


def _normalize_raw(sent: str) -> str:
    s = sent.strip().casefold()
    for pat, repl in _FORWARD_NORM:
        s = pat.sub(repl, s)
    return re.sub(r"\s+", " ", s).strip()

# Reverse of the normalisation used in identify_boilerplate.py: turn the
# placeholder tokens back into the regex that matches the case-specific values.
_PLACEHOLDER_REGEX = {
    "<DATE>": r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}",
    "<NO>": r"\d{4}\s*/\s*\d+",
    "<MADDE>": r"m(?:adde|\.)\s*\d+",
    "<NUM>": r"\d+",
}
_PLACEHOLDER_SPLIT = re.compile(r"(<DATE>|<NO>|<MADDE>|<NUM>)")


def _word_to_regex(word: str) -> str:
    """Regex for a single whitespace-delimited token, expanding placeholders."""
    out: list[str] = []
    for piece in _PLACEHOLDER_SPLIT.split(word):
        if not piece:
            continue
        out.append(_PLACEHOLDER_REGEX.get(piece) or re.escape(piece))
    return "".join(out)


def _pattern_to_regex(normalized: str) -> re.Pattern | None:
    """
    Convert a mined (normalised) sentence into a removal regex. Words are joined
    with ``\\s+`` so whitespace matches flexibly, and placeholder tokens
    (<DATE>, <NO>, <MADDE>, <NUM>) expand back to their value regex.
    """
    words = normalized.split()
    if not words:
        return None
    try:
        return re.compile(r"\s+".join(_word_to_regex(w) for w in words), re.IGNORECASE)
    except re.error:
        return None


def load_boilerplate_patterns(path=BOILERPLATE_JSON) -> list[re.Pattern]:
    """
    Build the list of compiled boilerplate-removal regexes from the JSON that
    identify_boilerplate.py produces. Returns [] when the file is absent, so
    the first pass (before discovery) behaves exactly as before.
    """
    patterns: list[re.Pattern] = []
    if not path.exists():
        return patterns
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return patterns

    # 1. Hand-curated seed patterns — always safe to apply.
    for seed in data.get("seed_patterns", []):
        rx = seed.get("regex")
        if rx:
            try:
                patterns.append(re.compile(rx, re.IGNORECASE))
            except re.error:
                pass

    # 2. Human-confirmed review items (is_boilerplate == True) — always applied.
    for item in data.get("reviewed", []):
        if item.get("is_boilerplate") and item.get("first_sentence"):
            raw = " ".join(item["first_sentence"])
            rx = _pattern_to_regex(_normalize_raw(raw))
            if rx is not None:
                patterns.append(rx)

    # 3. Frequency-mined candidates — applied ONLY when explicitly enabled,
    #    because mining also surfaces repeated legal doctrine (content, not
    #    boilerplate). See config.APPLY_MINED_PATTERNS.
    if config.APPLY_MINED_PATTERNS:
        for mined in data.get("mined_patterns", []):
            norm = mined.get("pattern")
            if norm:
                rx = _pattern_to_regex(norm)
                if rx is not None:
                    patterns.append(rx)
    return patterns


# Compiled once at import; main() reports how many were loaded.
_BP_PATTERNS: list[re.Pattern] = load_boilerplate_patterns()


def clean_text(text: str) -> str:
    """Apply boilerplate removal and whitespace normalisation to one decision."""
    if not isinstance(text, str):
        return ""
    # 1. Remove boilerplate header lines (needs original line breaks).
    text = _BOILERPLATE_RE.sub(" ", text)
    # 2. Strip formatting artefacts (markdown, source label) everywhere.
    for pat in _ARTIFACT_RES:
        text = pat.sub(" ", text)
    # 3. Normalise whitespace first so learned (single-space) patterns match.
    text = _WHITESPACE_RE.sub(" ", text)
    # 4. Remove learned boilerplate sentences/phrases (curated + confirmed).
    for pat in _BP_PATTERNS:
        text = pat.sub(" ", text)
    # 4. Tidy orphan sentence terminators left where a sentence was removed
    #    (e.g. "...sabittir. . Tahliyeye..." -> "...sabittir. Tahliyeye...").
    text = re.sub(r"\.(?:\s*\.)+", ".", text)
    # 5. Collapse any whitespace the removals left behind, then trim.
    return _WHITESPACE_RE.sub(" ", text).strip()


def main() -> None:
    if not config.RAW_CSV.exists():
        sys.exit(
            f"Raw CSV not found at {config.RAW_CSV}.\n"
            "Run 01_setup_dataset.py first."
        )

    config.ensure_dirs()
    if _BP_PATTERNS:
        print(f"Loaded {len(_BP_PATTERNS)} learned boilerplate pattern(s) "
              f"from {BOILERPLATE_JSON.name}.")
    else:
        print("No learned boilerplate patterns yet "
              "(run identify_boilerplate.py, then re-run this step to strip them).")

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
    print(f"Learned patterns applied : {len(_BP_PATTERNS)}")
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
