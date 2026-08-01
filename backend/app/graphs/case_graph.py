"""LangGraph orchestration: route → specialize → report; feedback → correct.

Strictly one specialized agent per case. Superior never writes the final report.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.registry import get_specialist
from app.agents.superior import SuperiorAgent
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


def _node_route(state: CaseState) -> CaseState:
    if state.get("error"):
        return state
    superior = SuperiorAgent()
    specialist, rationale = superior.route(state["case"])
    db.update_case(
        state["case_id"],
        status="processing",
        assigned_specialist=specialist.value,
        routing_rationale=rationale,
    )
    case = db.get_case(state["case_id"]) or state["case"]
    return {
        **state,
        "case": case,
        "specialist": specialist.value,
        "routing_rationale": rationale,
    }


def _node_specialize(state: CaseState) -> CaseState:
    if state.get("error"):
        return state
    try:
        agent = get_specialist(state["specialist"])
        report = agent.generate_report(state["case"])
        report_dict = report.model_dump()
        db.update_case(
            state["case_id"],
            status="completed",
            report=report_dict,
            error=None,
        )
        case = db.get_case(state["case_id"])
        return {**state, "report": report_dict, "case": case or state["case"]}
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
        # Should not happen; re-route if missing
        superior = SuperiorAgent()
        spec, rationale = superior.route(case)
        specialist = spec.value
        db.update_case(
            state["case_id"],
            assigned_specialist=specialist,
            routing_rationale=rationale,
        )
        case = db.get_case(state["case_id"]) or case

    prev_report = case.get("report") or {}
    try:
        agent = get_specialist(specialist)
        corrected = agent.apply_feedback(case, prev_report, state.get("feedback_text") or "")
        report_dict = corrected.model_dump()
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
        result = self.process_graph.invoke({"case_id": case_id})
        return result

    def feedback(self, case_id: str, feedback_id: str, feedback_text: str) -> Dict[str, Any]:
        result = self.feedback_graph.invoke(
            {
                "case_id": case_id,
                "feedback_id": feedback_id,
                "feedback_text": feedback_text,
            }
        )
        return result


_runner: Optional[CaseGraphRunner] = None


def get_runner() -> CaseGraphRunner:
    global _runner
    if _runner is None:
        _runner = CaseGraphRunner()
    return _runner


def run_case_pipeline(case_id: str) -> Dict[str, Any]:
    return get_runner().process(case_id)


def run_feedback_pipeline(case_id: str, feedback_id: str, feedback_text: str) -> Dict[str, Any]:
    return get_runner().feedback(case_id, feedback_id, feedback_text)
