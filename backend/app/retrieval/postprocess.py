"""Post-retrieval quality: specialty hard-drop, case isolation, dedupe, ranking."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.models.schemas import RetrievalHit

logger = logging.getLogger(__name__)

# Specialty keyword profiles for scoring + mismatch detection
SPECIALTY_KEYWORDS: Dict[str, Set[str]] = {
    "cardiology": {
        "acs", "nstemi", "stemi", "troponin", "ecg", "ekg", "ischemi", "angina",
        "myocardial", "infarction", "chest pain", "heart", "cardiac", "coronary",
        "timi", "heart score", "st depression", "st elevation", "pci", "cabg",
        "aspirin", "heparin", "antiplatelet", "arrhythmia", "cardiology",
        "diaphoresis", "cardiogenic", "unstable angina",
    },
    "dermatology": {
        "rash", "dermat", "lesion", "melanoma", "psoriasis", "eczema", "skin",
        "pruritus", "urticaria", "biopsy", "cellulitis", "dermatology",
        "silvery scale", "abcde",
    },
    "neurology": {
        "stroke", "seizure", "migraine", "neurolog", "headache", "hemiparesis",
        "aphasia", "cns", "mening", "parkinson", "neuropathy", "fast",
        "thunderclap", "nihss", "tpa", "alteplase", "cerebr", "ischemic stroke",
        "hemorrhagic", "focal deficit",
    },
    "clinical_pharmacology": {
        "drug interaction", "dosing", "pharmacolog", "contraindicat", "adverse",
        "warfarin", "polypharmacy", "side effect", "qt prolong", "inr",
        "renal dosing", "serotonin", "drug list", "medication",
    },
    "general_internal_medicine": {
        "sepsis", "hyponatremia", "diabetes", "metabolic", "fever", "infection",
        "multi-system", "undifferentiated", "internal medicine",
    },
}

# Exclusive section headers that mark sample KB specialty slices
SECTION_MARKERS: Dict[str, Tuple[str, ...]] = {
    "cardiology": ("## cardiology", "cardiology"),
    "dermatology": ("## dermatology", "dermatology", "melanoma abcde"),
    "neurology": ("## neurology", "neurology", "stroke fast"),
    "clinical_pharmacology": (
        "## clinical pharmacology",
        "drug list",
        "drug safety",
    ),
    "general_internal_medicine": (
        "## general internal medicine",
        "internal medicine",
    ),
}


def _norm_text(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _token_set(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", _norm_text(text)))


def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    t = _norm_text(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def _fingerprint(text: str) -> str:
    t = _norm_text(text)
    return hashlib.md5(t[:500].encode("utf-8")).hexdigest()


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def text_similarity(a: str, b: str) -> float:
    """Near-duplicate score in [0,1] using char-trigram Jaccard."""
    return jaccard(_char_ngrams(a), _char_ngrams(b))


def _meta_tags(meta: Dict[str, Any]) -> Set[str]:
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return {str(t).lower() for t in tags}


def _declared_specialty(meta: Dict[str, Any], text: str) -> Optional[str]:
    """Best-effort specialty label from metadata or section headers."""
    meta_spec = str(meta.get("specialty") or "").lower().strip()
    if meta_spec in SPECIALTY_KEYWORDS:
        return meta_spec
    tags = _meta_tags(meta)
    for spec in SPECIALTY_KEYWORDS:
        if spec in tags:
            return spec
    title = str(meta.get("title") or meta.get("filename") or "").lower()
    head = _norm_text(text)[:120]
    for spec, markers in SECTION_MARKERS.items():
        for m in markers:
            if m in title or head.startswith(m) or f"## {spec.replace('_', ' ')}" in head:
                return spec
            if m.startswith("##") and m in head:
                return spec
    # Title patterns like "clinical_snippets — Neurology"
    for spec in SPECIALTY_KEYWORDS:
        pretty = spec.replace("_", " ")
        if pretty in title or spec in title:
            return spec
    return None


CASE_SOURCE_TYPES = {"case_attachment", "case_text"}
SHARED_CASE_IDS = {"", "shared", "null", "none"}


def _is_case_scoped(meta: Dict[str, Any]) -> bool:
    """True if this chunk belongs to some patient case (not shared KB).

    IMPORTANT: case_attachment / case_text are always case-scoped, even when
    case_id is missing (legacy indexing bug that caused cross-case leakage).
    """
    cid = str(meta.get("case_id") or "").strip().lower()
    st = str(meta.get("source_type") or "").lower()
    scope = str(meta.get("scope") or "").lower()
    tags = _meta_tags(meta)
    title = str(meta.get("title") or "")
    if st in CASE_SOURCE_TYPES:
        return True
    if scope == "case":
        return True
    if cid and cid not in SHARED_CASE_IDS:
        return True
    if "case" in tags:
        return True
    # Title pattern from ingestion: "Case {uuid8} — filename"
    if re.match(r"^Case\s+[0-9a-fA-F\-]{6,}", title.strip()):
        return True
    # Upload path under data/uploads/<case_id>/
    path = str(meta.get("path") or "").replace("\\", "/")
    if "/uploads/" in path:
        return True
    return False


def _is_current_case(meta: Dict[str, Any], case_id: Optional[str]) -> bool:
    if not case_id:
        return False
    cid = str(meta.get("case_id") or "").strip()
    if cid == str(case_id):
        return True
    # Path-based ownership: .../uploads/<case_id>/...
    path = str(meta.get("path") or "").replace("\\", "/")
    if f"/uploads/{case_id}/" in path or path.endswith(f"/uploads/{case_id}"):
        return True
    # Title prefix Case <first8 of uuid>
    title = str(meta.get("title") or "")
    prefix = str(case_id)[:8]
    if title.startswith(f"Case {prefix}"):
        return True
    # Tags may include the full case uuid
    if str(case_id).lower() in _meta_tags(meta):
        return True
    return False


def _foreign_case_id(meta: Dict[str, Any], case_id: Optional[str]) -> Optional[str]:
    """Return the foreign case_id if this chunk belongs to another case."""
    cid = str(meta.get("case_id") or "").strip()
    if cid and cid.lower() not in SHARED_CASE_IDS:
        if not case_id or cid != str(case_id):
            return cid
    path = str(meta.get("path") or "").replace("\\", "/")
    m = re.search(r"/uploads/([0-9a-fA-F\-]{36})/", path)
    if m:
        other = m.group(1)
        if not case_id or other != str(case_id):
            return other
    title = str(meta.get("title") or "")
    m2 = re.match(r"^Case\s+([0-9a-fA-F]{6,8})\b", title.strip())
    if m2 and case_id:
        if not str(case_id).startswith(m2.group(1)):
            return m2.group(1)
    return None


def _is_image_or_attachment(meta: Dict[str, Any], text: str) -> bool:
    mod = str(meta.get("modality") or "").lower()
    st = str(meta.get("source_type") or "").lower()
    fn = str(meta.get("filename") or meta.get("title") or "").lower()
    if st in {"case_attachment", "case_text"}:
        return True
    if mod in {"image", "pdf", "audio", "table"}:
        return True
    if fn.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf")):
        return True
    if "uploaded image" in _norm_text(text)[:80] or "ecg image" in _norm_text(text)[:80]:
        return True
    return False


def specialty_keyword_hits(text: str, specialty: str) -> int:
    keys = SPECIALTY_KEYWORDS.get(specialty, set())
    t = _norm_text(text)
    return sum(1 for k in keys if k in t)


def specialty_relevance(text: str, specialty: str) -> float:
    """Score 0–1 how well chunk matches specialty lexicon."""
    keys = SPECIALTY_KEYWORDS.get(specialty, set())
    if not keys:
        return 0.5
    hits = specialty_keyword_hits(text, specialty)
    penalty = 0.0
    if specialty != "general_internal_medicine":
        for other, okeys in SPECIALTY_KEYWORDS.items():
            if other == specialty:
                continue
            oh = sum(1 for k in okeys if k in _norm_text(text))
            if oh >= 2:
                penalty += 0.12 * min(oh, 5)
            if oh > hits + 1:
                penalty += 0.25
    score = min(1.0, hits / 3.5) - min(0.9, penalty)
    return max(0.0, score)


def is_specialty_mismatch(
    text: str,
    meta: Dict[str, Any],
    specialty: str,
) -> bool:
    """True when chunk clearly belongs to a different specialty."""
    if specialty == "general_internal_medicine":
        return False
    declared = _declared_specialty(meta, text)
    if declared and declared != specialty:
        return True
    own = specialty_keyword_hits(text, specialty)
    best_other = 0
    best_name = None
    for other in SPECIALTY_KEYWORDS:
        if other == specialty:
            continue
        oh = specialty_keyword_hits(text, other)
        if oh > best_other:
            best_other = oh
            best_name = other
    # Strong off-specialty signal with weak own specialty
    if best_other >= 2 and own == 0:
        logger.debug(
            "mismatch keywords: specialty=%s other=%s own=%s other_hits=%s title=%s",
            specialty,
            best_name,
            own,
            best_other,
            meta.get("title"),
        )
        return True
    if best_other >= 3 and own <= 1 and best_other >= own + 2:
        return True
    return False


def query_overlap(query: str, text: str) -> float:
    return jaccard(_token_set(query), _token_set(text))


def dedupe_hits(
    hits: Sequence[RetrievalHit],
    *,
    jaccard_threshold: float = 0.72,
    similarity_threshold: float = 0.92,
) -> List[RetrievalHit]:
    """Keep highest-scoring version of near-identical chunks."""
    ordered = sorted(hits, key=lambda h: float(h.score or 0.0), reverse=True)
    kept: List[RetrievalHit] = []
    fps: Set[str] = set()
    kept_texts: List[str] = []
    dropped = 0
    for h in ordered:
        fp = _fingerprint(h.text)
        if fp in fps:
            dropped += 1
            continue
        # Near-dup by char n-grams (≈ cosine-equivalent for short clinical chunks)
        if any(text_similarity(h.text, prev) >= similarity_threshold for prev in kept_texts):
            dropped += 1
            continue
        toks = _token_set(h.text)
        if any(
            jaccard(toks, _token_set(prev)) >= jaccard_threshold for prev in kept_texts
        ):
            dropped += 1
            continue
        fps.add(fp)
        kept_texts.append(h.text)
        kept.append(h)
    if dropped:
        logger.info("dedupe: dropped %s near-duplicate chunk(s), kept %s", dropped, len(kept))
    return kept


def isolate_case_hits(
    hits: Sequence[RetrievalHit],
    *,
    case_id: Optional[str],
) -> List[RetrievalHit]:
    """Drop other cases' materials; keep shared KB + current case only.

    Rules:
    - case_attachment / case_text / scope=case → must belong to current case_id
    - missing case_id on case-scoped types → DROP (never treat as shared)
    - shared KB (sample/manual/guidelines) → keep when not case-scoped
    """
    kept: List[RetrievalHit] = []
    dropped = 0
    for h in hits:
        meta = h.metadata or {}
        title = str(meta.get("title") or meta.get("filename") or "")[:100]
        foreign = _foreign_case_id(meta, case_id)
        if foreign:
            dropped += 1
            logger.info(
                "case_isolation: DROP foreign case_id=%s current=%s title=%s",
                foreign,
                case_id,
                title,
            )
            continue
        if _is_case_scoped(meta):
            if _is_current_case(meta, case_id):
                kept.append(h)
            else:
                # Case-scoped but not current (incl. missing case_id legacy)
                dropped += 1
                logger.info(
                    "case_isolation: DROP case-scoped non-current "
                    "meta_case_id=%r source_type=%r current=%s title=%s",
                    meta.get("case_id"),
                    meta.get("source_type"),
                    case_id,
                    title,
                )
        else:
            kept.append(h)
    if dropped:
        logger.info(
            "case_isolation: dropped %s chunk(s); kept %s for case_id=%s",
            dropped,
            len(kept),
            case_id,
        )
    return kept


def filter_and_rank_hits(
    hits: Sequence[RetrievalHit],
    *,
    query: str,
    specialty: Optional[str] = None,
    case_id: Optional[str] = None,
    min_relevance: float = 0.22,
    max_hits: int = 5,
    prefer_case_attachments: bool = True,
    hard_specialty_filter: bool = True,
) -> List[RetrievalHit]:
    """
    Case isolation → specialty hard-drop → score → dedupe → top-N.

    Preference order:
      1. Current case attachments / images
      2. Specialty-matched KB
      3. Generic untagged but keyword-aligned content
    """
    if not hits:
        return []

    n_in = len(hits)
    isolated = isolate_case_hits(hits, case_id=case_id)

    scored: List[RetrievalHit] = []
    dropped_mismatch = 0
    dropped_low = 0

    for h in isolated:
        meta = dict(h.metadata or {})
        text = h.text or ""
        is_current = _is_current_case(meta, case_id)
        is_attach = prefer_case_attachments and (
            is_current or _is_image_or_attachment(meta, text)
        )

        # --- Hard specialty filter (never drop current-case attachments) ---
        if (
            hard_specialty_filter
            and specialty
            and specialty != "general_internal_medicine"
            and not is_current
            and is_specialty_mismatch(text, meta, specialty)
        ):
            dropped_mismatch += 1
            logger.info(
                "specialty_filter: DROP mismatch specialty=%s title=%s snippet=%.60s",
                specialty,
                (meta.get("title") or meta.get("filename") or "?")[:60],
                text.replace("\n", " ")[:60],
            )
            continue

        declared = _declared_specialty(meta, text)
        spec_rel = specialty_relevance(text, specialty) if specialty else 0.4
        if declared == specialty:
            spec_rel = max(spec_rel, 0.85)
        q_rel = query_overlap(query, text)

        base = max(float(h.dense_score or 0.0), float(h.score or 0.0) * 8.0)
        base = max(0.0, min(1.0, base))

        source_bonus = 0.0
        st = str(meta.get("source_type") or "").lower()
        mod = str(meta.get("modality") or "").lower()
        title = str(meta.get("title") or meta.get("filename") or "").lower()

        # (1) Current case attachments / images — highest priority
        if is_current:
            source_bonus += 0.45
            if mod == "image" or title.endswith((".png", ".jpg", ".jpeg", ".webp")):
                source_bonus += 0.20
            elif mod == "pdf" or title.endswith(".pdf"):
                source_bonus += 0.12
        elif is_attach and case_id is None:
            source_bonus += 0.1

        # (2) Specialty-matched KB
        if specialty and declared == specialty:
            source_bonus += 0.35
        elif specialty and declared and declared != specialty:
            source_bonus -= 0.55  # should rarely reach here if hard filter on

        tags = _meta_tags(meta)
        if specialty and specialty in tags:
            source_bonus += 0.15

        # (3) Penalize untagged generic sample when weak specialty match
        if specialty and specialty != "general_internal_medicine":
            if st == "sample" or "clinical_snippets" in title:
                if declared is None and spec_rel < 0.35:
                    source_bonus -= 0.4
                if declared == specialty:
                    source_bonus += 0.1

        combined = (
            0.25 * base
            + 0.20 * q_rel
            + 0.35 * spec_rel
            + source_bonus
        )
        # Cap
        combined = max(0.0, min(1.5, combined))

        new_hit = h.model_copy(deep=True)
        new_hit.score = round(float(combined), 4)
        new_hit.metadata = {
            **meta,
            "specialty_relevance": round(spec_rel, 3),
            "query_overlap": round(q_rel, 3),
            "combined_score": round(float(combined), 4),
            "declared_specialty": declared,
            "is_current_case": is_current,
        }

        # Keep current-case attachments even if base relevance is modest
        if is_current or combined >= min_relevance:
            scored.append(new_hit)
        else:
            dropped_low += 1
            logger.debug(
                "relevance_filter: drop low score=%.3f title=%s",
                combined,
                title[:60],
            )

    scored.sort(key=lambda x: x.score, reverse=True)
    deduped = dedupe_hits(scored, similarity_threshold=0.92, jaccard_threshold=0.75)

    # Prefer a balanced set: current-case first, then specialty KB
    current = [h for h in deduped if (h.metadata or {}).get("is_current_case")]
    others = [h for h in deduped if not (h.metadata or {}).get("is_current_case")]
    ordered = current + others
    final = ordered[:max_hits]

    logger.info(
        "retrieval_postprocess: in=%s isolated=%s mismatch_drop=%s low_drop=%s "
        "after_dedupe=%s out=%s specialty=%s case_id=%s top=%s",
        n_in,
        len(isolated),
        dropped_mismatch,
        dropped_low,
        len(deduped),
        len(final),
        specialty,
        case_id,
        [
            (
                (h.metadata or {}).get("title")
                or (h.metadata or {}).get("filename")
                or h.id
            )[:40]
            for h in final
        ],
    )

    # Fallback: if hard filter emptied everything, return current-case only or
    # specialty-keyword filtered originals (never foreign cases).
    if not final:
        logger.warning(
            "retrieval_postprocess: empty after filters; falling back carefully"
        )
        fallback = isolate_case_hits(list(hits), case_id=case_id)
        if specialty and specialty != "general_internal_medicine":
            fb2 = [
                h
                for h in fallback
                if _is_current_case(h.metadata or {}, case_id)
                or not is_specialty_mismatch(h.text, h.metadata or {}, specialty)
            ]
            fallback = fb2 or [
                h for h in fallback if _is_current_case(h.metadata or {}, case_id)
            ]
        fallback = dedupe_hits(fallback)
        return fallback[:max_hits]

    return final


def inject_case_attachments(
    hits: List[RetrievalHit],
    *,
    case: Dict[str, Any],
    max_attachments: int = 4,
) -> List[RetrievalHit]:
    """Ensure current case attachments appear as high-priority synthetic hits."""
    case_id = str(case.get("id") or "")
    if not case_id:
        return hits

    existing_titles = {
        str((h.metadata or {}).get("filename") or (h.metadata or {}).get("title") or "")
        .lower()
        for h in hits
    }
    injected: List[RetrievalHit] = []
    for i, att in enumerate(case.get("attachments") or []):
        if i >= max_attachments:
            break
        filename = att.get("filename") or f"attachment-{i}"
        text = (att.get("extracted_text") or "").strip()
        if not text:
            continue
        if filename.lower() in existing_titles:
            # Boost existing hit instead of duplicating
            for h in hits:
                meta = h.metadata or {}
                if str(meta.get("filename") or meta.get("title") or "").lower() == filename.lower():
                    h.score = max(float(h.score or 0.0), 1.2)
                    h.metadata = {**meta, "is_current_case": True, "case_id": case_id}
            continue
        modality = att.get("modality") or "file"
        label = "ECG/image analysis" if modality == "image" else f"Case {modality}"
        injected.append(
            RetrievalHit(
                id=f"case-att-{case_id[:8]}-{i}",
                text=text[:2000],
                score=1.25 if modality == "image" else 1.1,
                dense_score=1.0,
                sparse_score=1.0,
                metadata={
                    "title": f"{label}: {filename}",
                    "filename": filename,
                    "modality": modality,
                    "source_type": "case_attachment",
                    "case_id": case_id,
                    "is_current_case": True,
                    "specialty": None,
                },
            )
        )
        logger.info(
            "inject_case_attachment: %s modality=%s chars=%s",
            filename,
            modality,
            len(text),
        )

    if not injected:
        return hits
    return injected + hits
