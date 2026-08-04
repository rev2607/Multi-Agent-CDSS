"""Bounded Agentic RAG — opt-in secondary path with hard budgets."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.llm import get_llm_client
from app.models.schemas import RetrievalHit
from app.retrieval.hybrid import DEFAULT_EVIDENCE_K, HybridRetriever
from app.retrieval.postprocess import (
    dedupe_hits,
    filter_and_rank_hits,
    inject_case_attachments,
)

logger = logging.getLogger(__name__)


class BoundedAgenticRAG:
    def __init__(self, retriever: Optional[HybridRetriever] = None) -> None:
        self.retriever = retriever or HybridRetriever()
        self.llm = get_llm_client()
        self.max_steps = settings.agentic_max_steps
        self.wall_clock = settings.agentic_wall_clock_sec
        self.weak_threshold = settings.agentic_weak_hit_threshold

    def should_use_agentic(
        self,
        *,
        query: str,
        first_hits: List[RetrievalHit],
        force: bool = False,
    ) -> Tuple[bool, str]:
        if force:
            return True, "forced"
        if not first_hits:
            return True, "empty_first_retrieve"
        best_dense = max((h.dense_score for h in first_hits), default=0.0)
        best_combined = self.retriever.best_score(first_hits)
        if best_dense < self.weak_threshold and best_combined < 0.2:
            return True, f"weak_first_hit dense={best_dense:.4f} combined={best_combined:.4f}"
        q = query.lower()
        multi_hop_markers = [
            "compare",
            "versus",
            "vs ",
            "interaction",
            "multi-hop",
            "literature",
            "guideline series",
            "synthesize",
            "across studies",
            "differential with evidence",
        ]
        if any(m in q for m in multi_hop_markers):
            return True, "complex_research_intent"
        return False, "pipeline_sufficient"

    def run(
        self,
        query: str,
        *,
        case_context: str = "",
        force: bool = False,
        specialty: Optional[str] = None,
        case_id: Optional[str] = None,
        case: Optional[Dict[str, Any]] = None,
        evidence_k: int = DEFAULT_EVIDENCE_K,
    ) -> Dict[str, Any]:
        t0 = time.monotonic()
        case = case or {}
        case_id = case_id or case.get("id")

        first = self.retriever.search(
            query,
            specialty=specialty,
            case_id=case_id,
            evidence_k=max(evidence_k, 8),
            apply_postprocess=True,
        )
        # Ensure current attachments are visible
        if case:
            first = inject_case_attachments(first, case=case)
            first = filter_and_rank_hits(
                first,
                query=query,
                specialty=specialty,
                case_id=case_id,
                max_hits=evidence_k,
            )

        use, reason = self.should_use_agentic(
            query=query, first_hits=first, force=force
        )
        logger.info(
            "retrieval_path_decision: use_agentic=%s reason=%s specialty=%s case_id=%s hits=%s",
            use,
            reason,
            specialty,
            case_id,
            len(first),
        )

        if not use:
            return {
                "path": "pipeline_hybrid",
                "reason": reason,
                "hits": first[:evidence_k],
                "steps": 0,
                "queries": [query],
                "specialty": specialty,
                "case_id": case_id,
            }

        all_hits: Dict[str, RetrievalHit] = {h.id: h for h in first}
        queries = [query]
        steps = 0

        while steps < self.max_steps:
            if time.monotonic() - t0 > self.wall_clock:
                logger.info("Agentic RAG wall-clock budget hit after %s steps", steps)
                break
            steps += 1
            next_q = self._plan_next_query(
                original=query,
                case_context=case_context,
                prior_queries=queries,
                evidence_snippets=[h.text[:300] for h in list(all_hits.values())[:8]],
                specialty=specialty,
            )
            if not next_q or next_q.strip().lower() in {q.lower() for q in queries}:
                break
            queries.append(next_q)
            new_hits = self.retriever.search(
                next_q,
                specialty=specialty,
                case_id=case_id,
                evidence_k=max(evidence_k, 8),
            )
            for h in new_hits:
                if h.id not in all_hits or h.score > all_hits[h.id].score:
                    all_hits[h.id] = h

            grade = self._grade_sufficiency(query, list(all_hits.values())[:12])
            if grade >= 0.7:
                break

        merged = list(all_hits.values())
        if case:
            merged = inject_case_attachments(merged, case=case)
        top = filter_and_rank_hits(
            dedupe_hits(merged),
            query=query,
            specialty=specialty,
            case_id=case_id,
            max_hits=evidence_k,
        )
        return {
            "path": "bounded_agentic",
            "reason": reason,
            "hits": top,
            "steps": steps,
            "queries": queries,
            "elapsed_sec": round(time.monotonic() - t0, 3),
            "specialty": specialty,
            "case_id": case_id,
        }

    def _plan_next_query(
        self,
        *,
        original: str,
        case_context: str,
        prior_queries: List[str],
        evidence_snippets: List[str],
        specialty: Optional[str] = None,
    ) -> str:
        focus = specialty.replace("_", " ") if specialty else "clinical"
        system = (
            f"You are a medical retrieval planner focused on {focus} only. "
            "Propose ONE short follow-up search query for THIS specialty. "
            "Do not pull other specialties. Return query text only, or STOP."
        )
        user = (
            f"Original question:\n{original}\n\n"
            f"Case context (excerpt):\n{case_context[:1500]}\n\n"
            f"Prior queries:\n- " + "\n- ".join(prior_queries) + "\n\n"
            f"Evidence snippets:\n- " + "\n- ".join(evidence_snippets[:6]) + "\n\n"
            "Next query or STOP:"
        )
        out = self.llm.complete(system, user, temperature=0.1, max_tokens=128).strip()
        if out.upper().startswith("STOP"):
            return ""
        return out.splitlines()[0].strip().strip('"')

    def _grade_sufficiency(self, query: str, hits: List[RetrievalHit]) -> float:
        if not hits:
            return 0.0
        joined = " ".join(h.text.lower() for h in hits[:5])
        tokens = set(w for w in query.lower().split() if len(w) > 3)
        if not tokens:
            return 0.5
        covered = sum(1 for t in tokens if t in joined)
        return min(1.0, covered / max(len(tokens), 1) + 0.15 * min(len(hits), 5) / 5)
