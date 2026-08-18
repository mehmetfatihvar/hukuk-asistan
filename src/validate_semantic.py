"""
validate_semantic.py
====================
Semantic (meaning-level) validation of the chunks, complementing the
structural checks in validate_chunks.py.

Structural validation answers "are the chunks well-formed?". This answers
"are the chunks *good semantic units*?" with three metrics:

  A. INTRA-CHUNK COHERENCE  (needs the embedding model)
     For a sample of split chunks, embed each sentence and measure the mean
     pairwise cosine similarity of the sentences inside the chunk. High == the
     chunk is on one topic; low == the chunk blends unrelated content, i.e. a
     boundary was placed badly. Reported as mean/median plus the share of
     "incoherent" chunks (below --incoherent-threshold).

  B. SECTION SEPARABILITY   (reuses embeddings.npy if present)
     Mean cosine similarity of chunk pairs *within* the same section vs
     *across* sections. within > across (by a margin) means the OYAL / KANUN /
     KARAR / GEREKÇE labels carry real semantic signal — which is what the
     section-weighted reranker relies on.

  C. REDUNDANCY / NEAR-DUPLICATES (reuses embeddings.npy if present)
     Share of sampled chunks that have a near-duplicate (cosine >=
     --dup-threshold) coming from a *different* decision. High == boilerplate
     that preprocessing did not strip, inflating the index.

Metrics B and C reuse the vectors already produced by 05_embedding.py, so they
are cheap. Metric A re-embeds sentences for a small sample, so it loads the
model; skip it with --skip-intra.

Run:
    python src/validate_semantic.py                  # all three (samples)
    python src/validate_semantic.py --skip-intra     # B & C only (no model)
    python src/validate_semantic.py --sample 300
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

import config

# ---- reuse the chunker's sentence splitter -------------------------------- #
_spec = importlib.util.spec_from_file_location(
    "chunking", Path(__file__).with_name("04_smart_chunking.py")
)
_chunking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chunking)  # type: ignore[union-attr]
_SENTENCE_RE = _chunking._SENTENCE_RE


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-8, None)


def _mean_pairwise_cosine(vectors: np.ndarray) -> float:
    """Mean cosine over all distinct pairs of rows (upper triangle)."""
    if len(vectors) < 2:
        return float("nan")
    v = _normalize(vectors)
    sim = v @ v.T
    n = len(v)
    triu = sim[np.triu_indices(n, k=1)]
    return float(triu.mean())


# --------------------------------------------------------------------------- #
# Metric A — intra-chunk coherence (sentence level)
# --------------------------------------------------------------------------- #
def metric_intra_chunk(df, sample: int, incoherent_threshold: float) -> None:
    from sentence_transformers import SentenceTransformer

    # Only split (non-atomic) chunks with >= 2 sentences are meaningful here.
    candidates = []
    for _, r in df.iterrows():
        if bool(r["is_atomic"]):
            continue
        sents = [s.strip() for s in _SENTENCE_RE.split(str(r["text"])) if s.strip()]
        if len(sents) >= 2:
            candidates.append(sents)
    if not candidates:
        print("  (no multi-sentence split chunks to evaluate)")
        return

    random.seed(42)
    picked = random.sample(candidates, min(sample, len(candidates)))
    print(f"  Loading model {config.ST_MODEL_NAME} …")
    model = SentenceTransformer(config.ST_MODEL_NAME)

    coherences = []
    for sents in picked:
        emb = model.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
        coherences.append(_mean_pairwise_cosine(emb))
    coherences = [c for c in coherences if not np.isnan(c)]
    if not coherences:
        print("  (nothing measurable)")
        return

    arr = np.array(coherences)
    incoherent = float((arr < incoherent_threshold).mean())
    print(f"  chunks evaluated       : {len(arr):,}")
    print(f"  mean coherence         : {arr.mean():.3f}")
    print(f"  median coherence       : {np.median(arr):.3f}")
    print(f"  10th percentile        : {np.percentile(arr, 10):.3f}")
    print(f"  incoherent (< {incoherent_threshold:.2f})   : {incoherent:.1%}")
    if incoherent > 0.20:
        print("  ⚠️  >20% of chunks look incoherent — consider a smaller CHUNK_SIZE "
              "or splitting on paragraph boundaries.")
    else:
        print("  ✅ Chunks are internally coherent.")


# --------------------------------------------------------------------------- #
# Metric B — section separability
# --------------------------------------------------------------------------- #
def metric_section_separability(df, emb: np.ndarray, per_section: int) -> None:
    sections = [s for s in df["section"].unique()]
    idx_by_section: dict[str, list[int]] = {s: [] for s in sections}
    for i, s in enumerate(df["section"].tolist()):
        idx_by_section[s].append(i)

    random.seed(1)
    sample_idx: dict[str, np.ndarray] = {}
    for s, idxs in idx_by_section.items():
        if len(idxs) >= 2:
            take = random.sample(idxs, min(per_section, len(idxs)))
            sample_idx[s] = np.array(take)

    if len(sample_idx) < 2:
        print("  (need >= 2 populated sections)")
        return

    # Build one sampled, normalised matrix with section labels so within- and
    # cross-section similarities are measured the SAME way (both pairwise) —
    # comparing pairwise-within against centroid-cross (the old approach) was
    # apples-to-oranges and understated separability.
    labels: list[str] = []
    vecs: list[np.ndarray] = []
    for s, idxs in sample_idx.items():
        labels.extend([s] * len(idxs))
        vecs.append(emb[idxs])
    mat = _normalize(np.vstack(vecs))
    lab = np.array(labels)
    sim = mat @ mat.T
    n = len(lab)
    iu, ju = np.triu_indices(n, k=1)
    same = lab[iu] == lab[ju]
    pair_sims = sim[iu, ju]

    # Per-section within similarity (for the breakdown table).
    print("  within-section mean cosine:")
    for s in sample_idx:
        m = same & (lab[iu] == s)
        if m.any():
            print(f"    {s:<10}: {pair_sims[m].mean():.3f}")

    mean_within = float(pair_sims[same].mean()) if same.any() else float("nan")
    mean_cross = float(pair_sims[~same].mean()) if (~same).any() else float("nan")
    margin = mean_within - mean_cross
    print(f"  mean within-section    : {mean_within:.3f}")
    print(f"  mean cross-section     : {mean_cross:.3f}")
    print(f"  separability margin    : {margin:+.3f}  (pairwise, both sides)")
    if margin > 0.03:
        print("  ✅ Sections are semantically separable (labels carry signal).")
    else:
        print("  ⚠️  Sections barely separable — section-weighted reranking may add "
              "little; likely because boilerplate dominates (see metric C) or "
              "section detection is weak.")


# --------------------------------------------------------------------------- #
# Metric C — redundancy / near-duplicates
# --------------------------------------------------------------------------- #
def metric_redundancy(df, emb: np.ndarray, sample: int, dup_threshold: float) -> None:
    n = len(emb)
    random.seed(7)
    probe = np.array(random.sample(range(n), min(sample, n)))
    normed = _normalize(emb)
    doc_ids = df["original_doc_id"].astype(str).tolist()

    dup_count = 0
    for i in probe:
        sims = normed @ normed[i]
        sims[i] = -1.0  # exclude self
        j = int(np.argmax(sims))
        if sims[j] >= dup_threshold and doc_ids[j] != doc_ids[i]:
            dup_count += 1
    rate = dup_count / len(probe)
    print(f"  chunks probed          : {len(probe):,}")
    print(f"  cross-decision near-dups (>= {dup_threshold:.2f}): {rate:.1%}")
    if rate > 0.15:
        print("  ⚠️  Many near-duplicate chunks across decisions — likely "
              "boilerplate; tighten 03_preprocess.py.")
    else:
        print("  ✅ Low redundancy.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic validation of chunks.")
    ap.add_argument("--sample", type=int, default=200,
                    help="chunks to sample for intra-chunk coherence")
    ap.add_argument("--per-section", type=int, default=300,
                    help="chunks per section for separability")
    ap.add_argument("--dup-sample", type=int, default=500,
                    help="chunks to probe for near-duplicates")
    ap.add_argument("--incoherent-threshold", type=float, default=0.20)
    ap.add_argument("--dup-threshold", type=float, default=0.95)
    ap.add_argument("--skip-intra", action="store_true",
                    help="skip metric A (avoids loading the model)")
    args = ap.parse_args()

    import pandas as pd

    if not config.CHUNKS_CSV.exists():
        sys.exit(f"chunks.csv not found at {config.CHUNKS_CSV}. Run 04_smart_chunking.py first.")
    df = pd.read_csv(config.CHUNKS_CSV)
    df["text"] = df["text"].fillna("").astype(str)

    # Chunk embeddings for B & C: reuse embeddings.npy when it lines up.
    emb = None
    if config.EMBEDDINGS_NPY.exists():
        emb = np.load(config.EMBEDDINGS_NPY).astype("float32")
        if len(emb) != len(df):
            print(f"⚠️  embeddings.npy ({len(emb)}) != chunks.csv ({len(df)}); "
                  "re-run 05_embedding.py. Skipping metrics B & C.")
            emb = None

    print("=" * 72)
    print("SEMANTIC CHUNK VALIDATION")
    print("=" * 72)

    print("\n[A] INTRA-CHUNK COHERENCE")
    if args.skip_intra:
        print("  (skipped)")
    else:
        metric_intra_chunk(df, args.sample, args.incoherent_threshold)

    print("\n[B] SECTION SEPARABILITY")
    if emb is None:
        print("  (needs embeddings.npy from 05_embedding.py — skipped)")
    else:
        metric_section_separability(df, emb, args.per_section)

    print("\n[C] REDUNDANCY / NEAR-DUPLICATES")
    if emb is None:
        print("  (needs embeddings.npy from 05_embedding.py — skipped)")
    else:
        metric_redundancy(df, emb, args.dup_sample, args.dup_threshold)

    print("\n" + "=" * 72)
    print("Done. These are diagnostic signals, not pass/fail gates — read the "
          "⚠️/✅ hints above.")


if __name__ == "__main__":
    main()
