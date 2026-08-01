"""Post-retrieval quality: specialty scoring, dedupe, relevance filtering."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Sequence, Set

from app.models.schemas import RetrievalHit

# Specialty keyword profiles for soft filtering / boost (no re-index required)
SPECIALTY_KEYWORDS: Dict[str, Set[str]] = {
    "cardiology": {
        "acs", "nstemi", "stemi", "troponin", "ecg", "ekg", "ischemi", "angina",
        "myocardial", "infarction", "chest pain", "heart", "cardiac", "coronary",
        "timi", "heart score", "st depression", "st elevation", "pci", "cabg",
        "aspirin", "heparin", "antiplatelet", "arrhythmia", "hf", "failure",
        "hypertension", "cardiology", "diaphoresis", "dyspnea",
    },
    "dermatology": {
        "rash", "dermat", "lesion", "melanoma", "psoriasis", "eczema", "skin",
        "pruritus", "urticaria", "biopsy", "cellulitis",
    },
    "neurology": {
        "stroke", "seizure", "migraine", "neurolog", "headache", "hemiparesis",
        "aphasia", "cns", "mening", "parkinson", "neuropathy", "fast",
    },
    "clinical_pharmacology": {
        "drug interaction", "dosing", "pharmacolog", "contraindicat", "adverse",
        "warfarin", "polypharmacy", "side effect", "qt prolong", "inr",
        "renal dosing", "serotonin",
    },
    "general_internal_medicine": {
        "sepsis", "hyponatremia", "diabetes", "metabolic", "fever", "infection",
        "multi-system", "undifferentiated",
    },
}

# Content that should be down-weighted for cardiology ACS workups
CARDIOLOGY_IRRELEVANT = {
    "migraine", "seizure first aid", "psoriasis", "melanoma abcde", "sepsis:",
    "hyponatremia", "sertraline", "penicillin allergy", "qt-prolonging drugs",
    "drug list (sample)", "cellulitis vs erysipelas",
}


def _norm_text(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _token_set(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", _norm_text(text)))


def _fingerprint(text: str) -> str:
    """Coarse near-duplicate fingerprint (normalized whitespace + lowercase)."""
    t = _norm_text(text)
    # Collapse to first 400 chars for near-dup of same PDF page chunks
    return hashlib.md5(t[:400].encode("utf-8")).hexdigest()


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def specialty_relevance(text: str, specialty: str) -> float:
    """Score 0–1 how well chunk matches specialty lexicon."""
    t = _norm_text(text)
    keys = SPECIALTY_KEYWORDS.get(specialty, set())
    if not keys:
        return 0.5
    hits = sum(1 for k in keys if k in t)
    # Penalize strong off-specialty markers (except GIM which is broad)
    penalty = 0.0
    if specialty != "general_internal_medicine":
        for other, okeys in SPECIALTY_KEYWORDS.items():
            if other == specialty:
                continue
            # Only penalize if other specialty hits AND own specialty is weak
            oh = sum(1 for k in okeys if k in t)
            if oh >= 2 and hits == 0:
                penalty += 0.15 * min(oh, 4)
    if specialty == "cardiology":
        for bad in CARDIOLOGY_IRRELEVANT:
            if bad in t:
                penalty += 0.25
    score = min(1.0, hits / 4.0) - min(0.8, penalty)
    return max(0.0, score)


def query_overlap(query: str, text: str) -> float:
    q = _token_set(query)
    d = _token_set(text)
    return jaccard(q, d)


def dedupe_hits(
    hits: Sequence[RetrievalHit],
    *,
    jaccard_threshold: float = 0.72,
) -> List[RetrievalHit]:
    """Drop near-identical chunks (same PDF page rephrased / overlapping windows)."""
    kept: List[RetrievalHit] = []
    fps: Set[str] = set()
    token_sets: List[Set[str]] = []
    for h in hits:
        fp = _fingerprint(h.text)
        if fp in fps:
            continue
        toks = _token_set(h.text)
        if any(jaccard(toks, prev) >= jaccard_threshold for prev in token_sets):
            continue
        fps.add(fp)
        token_sets.append(toks)
        kept.append(h)
    return kept


def filter_and_rank_hits(
    hits: Sequence[RetrievalHit],
    *,
    query: str,
    specialty: Optional[str] = None,
    min_relevance: float = 0.18,
    max_hits: int = 5,
    prefer_case_attachments: bool = True,
) -> List[RetrievalHit]:
    """
    Specialty-aware re-rank + threshold + dedupe + top-N.

    Combined score blends:
      - original retrieval score (normalized-ish)
      - query token overlap
      - specialty keyword relevance
      - source preference (case attachment / specialty tags)
    """
    if not hits:
        return []

    scored: List[RetrievalHit] = []
    for h in hits:
        meta = h.metadata or {}
        spec_rel = specialty_relevance(h.text, specialty) if specialty else 0.4
        q_rel = query_overlap(query, h.text)
        # RRF scores are small; dense often 0–1. Use max of dense and scaled rrf.
        base = max(float(h.dense_score or 0.0), float(h.score or 0.0) * 8.0)
        base = max(0.0, min(1.0, base))

        source_bonus = 0.0
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        title = str(meta.get("title") or meta.get("filename") or "").lower()
        source_type = str(meta.get("source_type") or "").lower()

        if prefer_case_attachments and (
            source_type == "case_attachment"
            or "case" in tags
            or meta.get("case_id")
        ):
            source_bonus += 0.12
        if specialty and specialty.replace("_", " ") in title:
            source_bonus += 0.1
        if specialty and specialty in {str(t).lower() for t in tags}:
            source_bonus += 0.15
        meta_spec = str(meta.get("specialty") or "").lower()
        if specialty and meta_spec == specialty:
            source_bonus += 0.2
        elif specialty and meta_spec and meta_spec != specialty:
            source_bonus -= 0.25
        # Prefer sample specialty sections for cardiology
        if specialty == "cardiology" and any(
            k in _norm_text(h.text)[:200]
            for k in ("## cardiology", "stemi", "acs ", "timi", "heart score")
        ):
            source_bonus += 0.1
        # Down-rank generic sample KB when specialty is narrow
        if specialty and specialty != "general_internal_medicine":
            if "clinical_snippets" in title or source_type == "sample":
                if spec_rel < 0.25:
                    source_bonus -= 0.35

        combined = (
            0.35 * base
            + 0.25 * q_rel
            + 0.30 * spec_rel
            + source_bonus
        )
        # Store combined on score for downstream display
        new_hit = h.model_copy(deep=True)
        new_hit.score = round(float(combined), 4)
        new_hit.metadata = {
            **meta,
            "specialty_relevance": round(spec_rel, 3),
            "query_overlap": round(q_rel, 3),
            "combined_score": round(float(combined), 4),
        }
        if combined >= min_relevance or source_bonus >= 0.12:
            scored.append(new_hit)

    scored.sort(key=lambda x: x.score, reverse=True)
    deduped = dedupe_hits(scored)
    # If threshold was too aggressive, fall back to top deduped originals
    if not deduped and hits:
        fallback = dedupe_hits(list(hits))
        for h in fallback:
            h.score = float(h.score or 0.0)
        return fallback[:max_hits]
    return deduped[:max_hits]
