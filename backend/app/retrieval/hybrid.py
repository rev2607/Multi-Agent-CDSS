"""Pipeline Hybrid RAG: dense + sparse → RRF → specialty/case filters → dedupe → top-N."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.llm import get_llm_client
from app.db.qdrant_store import get_qdrant_store
from app.models.schemas import EvidenceItem, RetrievalHit
from app.retrieval.filters import (
    case_isolation_filter,
    merge_filters,
    specialty_prefer_filter,
)
from app.retrieval.postprocess import filter_and_rank_hits
from app.retrieval.sparse import SparseEncoder

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE_K = 5


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    *,
    k: int = 60,
    scores_map: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[tuple[str, float]]:
    fused: Dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    if scores_map:
        for doc_id, parts in scores_map.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + 0.05 * sum(parts.values())
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Default retrieval substrate used by all specialized agents."""

    def __init__(self) -> None:
        self.store = get_qdrant_store()
        self.llm = get_llm_client()
        self.sparse = SparseEncoder()

    def index_texts(
        self,
        texts: List[str],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not texts:
            return []
        dense = self.llm.embed(texts)
        sparse = self.sparse.encode_batch(texts)
        return self.store.upsert_chunks(
            texts=texts,
            dense_vectors=dense,
            sparse_vectors=sparse,
            payloads=payloads,
            ids=ids,
        )

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filters: Any = None,
        use_rerank: Optional[bool] = None,
        specialty: Optional[str] = None,
        case_id: Optional[str] = None,
        evidence_k: Optional[int] = None,
        min_relevance: Optional[float] = None,
        apply_postprocess: bool = True,
    ) -> List[RetrievalHit]:
        top_k = top_k or settings.hybrid_top_k
        evidence_k = evidence_k or getattr(settings, "evidence_top_k", DEFAULT_EVIDENCE_K)
        min_relevance = (
            min_relevance
            if min_relevance is not None
            else getattr(settings, "retrieval_min_relevance", 0.22)
        )
        fetch_k = max(top_k * 4, evidence_k * 8, 32)

        search_query = query
        if specialty:
            search_query = f"{query} {specialty.replace('_', ' ')}"

        # Case isolation + specialty preference at Qdrant layer
        q_filter = merge_filters(
            filters,
            case_isolation_filter(case_id),
            specialty_prefer_filter(specialty),
        )

        logger.info(
            "hybrid_search: specialty=%s case_id=%s fetch_k=%s query=%.80s",
            specialty,
            case_id,
            fetch_k,
            query.replace("\n", " "),
        )

        dense_q = self.llm.embed_query(search_query)
        sparse_q = self.sparse.encode(search_query)

        # Always enforce case isolation at the Qdrant layer when possible.
        # On filter failure: fall back to case-isolation-only, NEVER fully unfiltered
        # without a second isolation pass (postprocess still runs).
        case_only = merge_filters(filters, case_isolation_filter(case_id))
        try:
            dense_hits = self.store.search_dense(
                dense_q, limit=fetch_k, filters=q_filter
            )
            sparse_hits = self.store.search_sparse(
                sparse_q, limit=fetch_k, filters=q_filter
            )
        except Exception as e:
            logger.warning(
                "hybrid_search: combined filter failed (%s); retrying case-isolation only",
                e,
            )
            try:
                dense_hits = self.store.search_dense(
                    dense_q, limit=fetch_k, filters=case_only
                )
                sparse_hits = self.store.search_sparse(
                    sparse_q, limit=fetch_k, filters=case_only
                )
            except Exception as e2:
                logger.warning(
                    "hybrid_search: case filter failed (%s); "
                    "unfiltered fetch + STRICT postprocess isolation",
                    e2,
                )
                dense_hits = self.store.search_dense(
                    dense_q, limit=fetch_k, filters=None
                )
                sparse_hits = self.store.search_sparse(
                    sparse_q, limit=fetch_k, filters=None
                )

        # If specialty preference over-constrained, retry with case isolation only
        if len(dense_hits) + len(sparse_hits) < 3 and specialty:
            logger.info(
                "hybrid_search: sparse results with specialty filter; "
                "retrying case-isolation only"
            )
            try:
                dense_hits = self.store.search_dense(
                    dense_q, limit=fetch_k, filters=case_only
                )
                sparse_hits = self.store.search_sparse(
                    sparse_q, limit=fetch_k, filters=case_only
                )
            except Exception as e:
                logger.warning("case-only filter failed: %s", e)

        dense_ids = [str(h.id) for h in dense_hits]
        sparse_ids = [str(h.id) for h in sparse_hits]

        by_id: Dict[str, Dict[str, Any]] = {}
        scores_map: Dict[str, Dict[str, float]] = {}
        for h in dense_hits:
            did = str(h.id)
            by_id[did] = h.payload or {}
            scores_map.setdefault(did, {})["dense"] = float(h.score or 0.0)
        for h in sparse_hits:
            did = str(h.id)
            by_id.setdefault(did, h.payload or {})
            scores_map.setdefault(did, {})["sparse"] = float(h.score or 0.0)

        fused = reciprocal_rank_fusion(
            [dense_ids, sparse_ids],
            k=settings.rrf_k,
            scores_map=scores_map,
        )

        hits: List[RetrievalHit] = []
        for doc_id, rrf_score in fused[:fetch_k]:
            payload = by_id.get(doc_id, {})
            hits.append(
                RetrievalHit(
                    id=doc_id,
                    text=str(payload.get("text", "")),
                    score=float(rrf_score),
                    dense_score=float(scores_map.get(doc_id, {}).get("dense", 0.0)),
                    sparse_score=float(scores_map.get(doc_id, {}).get("sparse", 0.0)),
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )

        if use_rerank is None:
            use_rerank = settings.enable_rerank
        if use_rerank and hits:
            hits = self._rerank(query, hits)

        if apply_postprocess:
            hits = filter_and_rank_hits(
                hits,
                query=query,
                specialty=specialty,
                case_id=case_id,
                min_relevance=min_relevance,
                max_hits=evidence_k,
                hard_specialty_filter=True,
            )
        else:
            hits = hits[:evidence_k]

        logger.info(
            "hybrid_search: returning %s hit(s) specialty=%s case_id=%s",
            len(hits),
            specialty,
            case_id,
        )
        return hits

    def _rerank(self, query: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, h.text) for h in hits[:40]]
            scores = model.predict(pairs)
            for h, s in zip(hits[:40], scores):
                h.score = float(s)
            return sorted(hits[:40], key=lambda x: x.score, reverse=True)
        except Exception as e:
            logger.debug("Cross-encoder rerank skipped: %s", e)
            return hits

    def to_evidence(
        self,
        hits: List[RetrievalHit],
        *,
        max_items: int = DEFAULT_EVIDENCE_K,
    ) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        for h in hits[:max_items]:
            meta = h.metadata or {}
            evidence.append(
                EvidenceItem(
                    source_id=h.id,
                    title=str(meta.get("title") or meta.get("filename") or "KB chunk"),
                    snippet=h.text[:500],
                    score=h.score,
                    modality=str(meta.get("modality") or "text"),
                    metadata=meta,
                )
            )
        return evidence

    def best_score(self, hits: List[RetrievalHit]) -> float:
        if not hits:
            return 0.0
        return max(h.score for h in hits)
