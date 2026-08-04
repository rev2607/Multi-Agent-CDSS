"""LangGraph: load → route → specialize.

Routing is deterministic. Cardiology only when ACS hard rules match.
specialize_guard uses the same build_case_blob as SuperiorAgent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.case_patterns import (
    build_case_blob,
    hard_route_specialty,
    is_acs_presentation,
    is_primary_acs_evaluation,
    resolve_specialist,
)
from app.agents.registry import get_specialist
from app.agents.superior import SPECIALIST_DISPLAY, SuperiorAgent, format_routed_to
from app.db import sqlite as db
from app.models.schemas import SpecialistType

logger = logging.getLogger(__name__)


class CaseState(TypedDict, total=False):
    case_id: str
    case: Dict[str, Any]
    specialist: str
    routing_rationale: str
    report: Dict[str, Any]
    feedback_id: str
    feedback_text: str
    error: str


def _node_load_case(state: CaseState) -> CaseState:
    case = db.get_case(state["case_id"])
    if not case:
        return {**state, "error": f"Case {state['case_id']} not found"}
    db.update_case(state["case_id"], status="routing")
    return {**state, "case": case}


def _persist_specialist(
    case_id: str, specialist_key: str, rationale: str
) -> Dict[str, Any]:
    db.update_case(
        case_id,
        assigned_specialist=specialist_key,
        routing_rationale=rationale,
    )
    return db.get_case(case_id) or {}


def _node_route(state: CaseState) -> CaseState:
    if state.get("error"):
        return state

    superior = SuperiorAgent()
    specialist, rationale = superior.route(state["case"])

    # Absolute resolve with full blob
    final_key, final_reason = resolve_specialist(state["case"], specialist.value)
    if final_key != specialist.value:
        logger.warning(
            "route_resolve superior=%s → final=%s (%s)",
            specialist.value,
            final_key,
            final_reason,
        )
        specialist = SpecialistType(final_key)
        rationale = format_routed_to(specialist, final_reason)

    if not (rationale or "").lower().startswith("routed to:"):
        rationale = format_routed_to(specialist, rationale)

    logger.info(
        "case_routed case_id=%s specialist=%s rationale=%s",
        state["case_id"],
        specialist.value,
        rationale,
    )
    case = _persist_specialist(state["case_id"], specialist.value, rationale)
    return {
        **state,
        "case": case or state["case"],
        "specialist": specialist.value,
        "routing_rationale": rationale,
    }


def _strip_acs_scores(report: Dict[str, Any]) -> Dict[str, Any]:
    scores = report.get("risk_scores") or []
    cleaned = []
    for s in scores:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or "").lower()
        if any(x in name for x in ("heart", "timi", "grace")):
            continue
        cleaned.append(s)
    report["risk_scores"] = cleaned
    return report


def _sanitize_cardio_phrases(report: Dict[str, Any], display: str) -> Dict[str, Any]:
    import re

    def fix(text: str) -> str:
        if not text:
            return text
        out = text
        out = re.sub(
            r"(?i)\bthe\s+cardiology\s+agent(?:'s)?\s+role\b[^.!?]*[.!]?",
            f"The {display} provides specialty-appropriate decision support.",
            out,
        )
        out = re.sub(r"(?i)\bcardiology\s+agent\b", display, out)
        out = re.sub(r"(?i)\bcardiology\s+assessment\b", f"{display} assessment", out)
        out = re.sub(
            r"(?i)\bfrom\s+a\s+cardiology\s+perspective\b",
            f"from a {display} perspective",
            out,
        )
        out = re.sub(
            r"(?i)\b(?:the\s+)?HEART\s+score\b[^.!?]*[.!]?",
            "",
            out,
        )
        out = re.sub(
            r"(?i)\b(?:the\s+)?TIMI(?:\s+score|\s*\(UA/NSTEMI\))?\b[^.!?]*[.!]?",
            "",
            out,
        )
        return re.sub(r"[ \t]{2,}", " ", out).strip()

    for key in ("assessment", "case_summary", "reasoning", "chief_complaint"):
        if report.get(key):
            report[key] = fix(str(report[key]))
    return report


def _node_specialize(state: CaseState) -> CaseState:
    if state.get("error"):
        return state
    try:
        case = state["case"]
        before = (state.get("specialist") or "").strip()
        blob = build_case_blob(case)  # SAME as Superior
        hard = hard_route_specialty(blob)

        specialist_key = before
        hard_label = hard[0] if hard else None

        if hard is not None:
            specialist_key = hard[0]
            rationale = format_routed_to(SpecialistType(specialist_key), hard[1])
            case = _persist_specialist(state["case_id"], specialist_key, rationale) or case
            state = {
                **state,
                "case": case,
                "specialist": specialist_key,
                "routing_rationale": rationale,
            }

        # Absolute resolve before agent construction
        final_key, final_reason = resolve_specialist(case, specialist_key)
        if final_key != specialist_key:
            specialist_key = final_key
            rationale = format_routed_to(SpecialistType(specialist_key), final_reason)
            case = _persist_specialist(state["case_id"], specialist_key, rationale) or case
            state = {
                **state,
                "case": case,
                "specialist": specialist_key,
                "routing_rationale": rationale,
            }

        # Refuse Cardiology without ACS
        if specialist_key == "cardiology" and not is_acs_presentation(blob)[0]:
            logger.error(
                "specialize_guard REFUSE Cardiology on non-ACS case_id=%s",
                state["case_id"],
            )
            hard2 = hard_route_specialty(blob)
            specialist_key = hard2[0] if hard2 else "general_internal_medicine"
            rationale = format_routed_to(
                SpecialistType(specialist_key),
                "Absolute block: Cardiology refused without ACS hard match",
            )
            case = _persist_specialist(state["case_id"], specialist_key, rationale) or case
            state = {
                **state,
                "case": case,
                "specialist": specialist_key,
                "routing_rationale": rationale,
            }

        logger.info(
            "specialize_guard before=%s after=%s hard=%s case_id=%s",
            before,
            specialist_key,
            hard_label,
            state["case_id"],
        )

        agent = get_specialist(specialist_key)
        logger.info(
            "generate_report agent=%s specialist_key=%s case_id=%s",
            agent.__class__.__name__,
            specialist_key,
            state["case_id"],
        )

        # Double-check agent identity matches key
        if agent.specialist.value != specialist_key:
            logger.error(
                "AGENT MISMATCH key=%s agent.specialist=%s class=%s — refusing",
                specialist_key,
                agent.specialist.value,
                agent.__class__.__name__,
            )
            raise RuntimeError(
                f"Specialist registry mismatch: key={specialist_key} "
                f"agent={agent.specialist.value}"
            )

        report = agent.generate_report(case)
        report_dict = report.model_dump()

        try:
            st = SpecialistType(specialist_key)
            label = SPECIALIST_DISPLAY.get(st, specialist_key)
            display = agent.display_name
        except Exception:
            label = specialist_key
            display = specialist_key

        report_dict["routed_to"] = f"Routed to: {label}"
        report_dict["specialist"] = specialist_key

        # Strip ACS scores unless Cardiology + primary ACS
        if specialist_key != "cardiology" or not is_primary_acs_evaluation(case):
            n0 = len(report_dict.get("risk_scores") or [])
            report_dict = _strip_acs_scores(report_dict)
            n1 = len(report_dict.get("risk_scores") or [])
            if n0 != n1:
                logger.info(
                    "stripped ACS scores specialist=%s %s→%s",
                    specialist_key,
                    n0,
                    n1,
                )
            report_dict = _sanitize_cardio_phrases(report_dict, display)

        raw = report_dict.get("raw")
        if not isinstance(raw, dict):
            raw = {}
        raw["agent_class"] = agent.__class__.__name__
        raw["specialist_key"] = specialist_key
        report_dict["raw"] = raw

        db.update_case(
            state["case_id"],
            status="completed",
            report=report_dict,
            error=None,
            assigned_specialist=specialist_key,
        )
        case = db.get_case(state["case_id"])
        logger.info(
            "specialize_done case_id=%s specialist=%s agent=%s routed_to=%s",
            state["case_id"],
            specialist_key,
            agent.__class__.__name__,
            report_dict.get("routed_to"),
        )
        return {
            **state,
            "report": report_dict,
            "case": case or state["case"],
            "specialist": specialist_key,
        }
    except Exception as e:
        logger.exception("Specialist failed")
        db.update_case(state["case_id"], status="failed", error=str(e))
        return {**state, "error": str(e)}


def _node_apply_feedback(state: CaseState) -> CaseState:
    if state.get("error"):
        return state
    case = db.get_case(state["case_id"])
    if not case:
        return {**state, "error": "Case not found"}

    specialist = case.get("assigned_specialist")
    if not specialist:
        superior = SuperiorAgent()
        spec, rationale = superior.route(case)
        specialist = spec.value
        _persist_specialist(state["case_id"], specialist, rationale)
        case = db.get_case(state["case_id"]) or case

    final_key, final_reason = resolve_specialist(case, specialist)
    if final_key != specialist:
        logger.warning("feedback_guard %s → %s (%s)", specialist, final_key, final_reason)
        specialist = final_key
        _persist_specialist(
            state["case_id"],
            specialist,
            format_routed_to(SpecialistType(specialist), final_reason),
        )

    prev_report = case.get("report") or {}
    try:
        agent = get_specialist(specialist)
        logger.info(
            "feedback_report agent=%s specialist_key=%s",
            agent.__class__.__name__,
            specialist,
        )
        corrected = agent.apply_feedback(
            case, prev_report, state.get("feedback_text") or ""
        )
        report_dict = corrected.model_dump()
        try:
            st = SpecialistType(specialist)
            label = SPECIALIST_DISPLAY.get(st, specialist)
            display = agent.display_name
        except Exception:
            label = specialist
            display = specialist
        report_dict["routed_to"] = f"Routed to: {label}"
        report_dict["specialist"] = specialist
        if specialist != "cardiology" or not is_primary_acs_evaluation(case):
            report_dict = _strip_acs_scores(report_dict)
            report_dict = _sanitize_cardio_phrases(report_dict, display)

        knowledge_written = bool((report_dict.get("raw") or {}).get("knowledge_written"))
        db.update_case(state["case_id"], status="corrected", report=report_dict)
        if state.get("feedback_id"):
            db.mark_feedback(
                state["feedback_id"],
                applied=True,
                knowledge_written=knowledge_written,
            )
        case = db.get_case(state["case_id"])
        return {**state, "report": report_dict, "case": case or case}
    except Exception as e:
        logger.exception("Feedback correction failed")
        db.update_case(state["case_id"], status="failed", error=str(e))
        return {**state, "error": str(e)}


def build_process_graph():
    g = StateGraph(CaseState)
    g.add_node("load_case", _node_load_case)
    g.add_node("route", _node_route)
    g.add_node("specialize", _node_specialize)
    g.set_entry_point("load_case")
    g.add_edge("load_case", "route")
    g.add_edge("route", "specialize")
    g.add_edge("specialize", END)
    return g.compile()


def build_feedback_graph():
    g = StateGraph(CaseState)
    g.add_node("apply_feedback", _node_apply_feedback)
    g.set_entry_point("apply_feedback")
    g.add_edge("apply_feedback", END)
    return g.compile()


class CaseGraphRunner:
    def __init__(self) -> None:
        self.process_graph = build_process_graph()
        self.feedback_graph = build_feedback_graph()

    def process(self, case_id: str) -> Dict[str, Any]:
        return self.process_graph.invoke({"case_id": case_id})

    def feedback(
        self, case_id: str, feedback_id: str, feedback_text: str
    ) -> Dict[str, Any]:
        return self.feedback_graph.invoke(
            {
                "case_id": case_id,
                "feedback_id": feedback_id,
                "feedback_text": feedback_text,
            }
        )


_runner: Optional[CaseGraphRunner] = None


def get_runner() -> CaseGraphRunner:
    """Return process runner. Always rebuild graph so code reloads take effect."""
    global _runner
    # Rebuild each call in case module was reloaded without process restart
    # (still recommend full server restart in production)
    _runner = CaseGraphRunner()
    return _runner


def run_case_pipeline(case_id: str) -> Dict[str, Any]:
    return get_runner().process(case_id)


def run_feedback_pipeline(
    case_id: str, feedback_id: str, feedback_text: str
) -> Dict[str, Any]:
    return get_runner().feedback(case_id, feedback_id, feedback_text)
