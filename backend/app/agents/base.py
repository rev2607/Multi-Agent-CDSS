"""Base specialized agent: retrieval → structured report → targeted feedback correction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from app.core.config import settings
from app.core.llm import get_llm_client
from app.ingestion.pipeline import get_ingestion_pipeline
from app.models.schemas import (
    ClinicalReport,
    DifferentialItem,
    EvidenceItem,
    ImageFindingItem,
    RiskScoreItem,
    SpecialistType,
)
from app.retrieval.agentic import BoundedAgenticRAG
from app.retrieval.hybrid import DEFAULT_EVIDENCE_K, HybridRetriever

logger = logging.getLogger(__name__)


class BaseSpecialistAgent:
    specialist: SpecialistType
    display_name: str
    system_focus: str

    def __init__(self) -> None:
        self.llm = get_llm_client()
        self.retriever = HybridRetriever()
        self.agentic = BoundedAgenticRAG(self.retriever)
        self.ingestion = get_ingestion_pipeline()

    def build_case_blob(self, case: Dict[str, Any]) -> str:
        parts = [
            f"Title: {case.get('title', '')}",
            f"Patient context: {case.get('patient_context', '')}",
            f"Clinical text: {case.get('clinical_text', '')}",
            f"Notes: {case.get('notes', '')}",
        ]
        for att in case.get("attachments") or []:
            modality = att.get("modality") or "file"
            filename = att.get("filename") or "attachment"
            # Explicit labels so the LLM can cite image/PDF sources
            if modality == "image" or str(filename).lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp", ".gif")
            ):
                header = f"UPLOADED IMAGE [{filename}] (cite as image analysis)"
            elif modality == "pdf":
                header = f"UPLOADED PDF [{filename}]"
            else:
                header = f"Attachment [{modality}] {filename}"
            parts.append(f"{header}:\n{(att.get('extracted_text') or '')[:4000]}")
        return "\n\n".join(parts)

    def retrieve(self, case: Dict[str, Any], *, force_agentic: bool = False) -> Dict[str, Any]:
        blob = self.build_case_blob(case)
        query = self._make_query(blob)
        evidence_k = getattr(settings, "evidence_top_k", DEFAULT_EVIDENCE_K)
        case_id = str(case.get("id") or "") or None
        logger.info(
            "agent_retrieve: specialist=%s case_id=%s evidence_k=%s query=%.100s",
            self.specialist.value,
            case_id,
            evidence_k,
            query.replace("\n", " "),
        )
        result = self.agentic.run(
            query,
            case_context=blob,
            force=force_agentic,
            specialty=self.specialist.value,
            case_id=case_id,
            case=case,
            evidence_k=evidence_k,
        )
        hits = result.get("hits") or []
        logger.info(
            "agent_retrieve_done: specialist=%s case_id=%s path=%s n_hits=%s titles=%s",
            self.specialist.value,
            case_id,
            result.get("path"),
            len(hits),
            [
                ((h.metadata or {}).get("title") or (h.metadata or {}).get("filename") or h.id)[
                    :50
                ]
                for h in hits
            ],
        )
        return result

    def _make_query(self, case_blob: str) -> str:
        try:
            q = self.llm.complete(
                "Compress the clinical case into a single search query (max 40 words) "
                f"focused on {self.display_name} knowledge needs. "
                "Include key diagnoses, biomarkers, and specialty terms. Return query only.",
                case_blob[:3500],
                temperature=0.0,
                max_tokens=120,
            ).strip()
            if q and len(q) < 400:
                return q
        except Exception:
            pass
        return case_blob[:500]

    def generate_report(self, case: Dict[str, Any]) -> ClinicalReport:
        logger.info(
            "generate_report: agent_class=%s specialist=%s case_id=%s",
            self.__class__.__name__,
            self.specialist.value,
            case.get("id"),
        )
        # Absolute safety: Cardiology agent must not precompute ACS scores on non-ACS cases
        if self.specialist == SpecialistType.cardiology:
            from app.agents.case_patterns import is_acs_presentation, case_text_blob

            if not is_acs_presentation(case_text_blob(case))[0]:
                logger.error(
                    "generate_report: CardiologyAgent invoked on non-ACS case_id=%s — "
                    "will omit HEART/TIMI and sanitize framing",
                    case.get("id"),
                )

        retrieval = self.retrieve(case)
        hits = retrieval["hits"]
        evidence = self.retriever.to_evidence(
            hits, max_items=getattr(settings, "evidence_top_k", DEFAULT_EVIDENCE_K)
        )
        path = retrieval.get("path", "pipeline_hybrid")

        extras = self._precompute_report_fields(case)
        system = self._report_system_prompt()
        user = self._report_user_prompt(case, evidence, path, extras=extras)
        raw_text = self.llm.complete(
            system, user, temperature=0.2, max_tokens=4096, json_mode=True
        )
        data = self._parse_json(raw_text)
        report = self._build_report(case, data, evidence, path, retrieval, extras)
        return report

    def _precompute_report_fields(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for specialist-specific deterministic fields (e.g. HEART/TIMI)."""
        return {}

    def _build_report(
        self,
        case: Dict[str, Any],
        data: Dict[str, Any],
        evidence: List[EvidenceItem],
        path: str,
        retrieval: Dict[str, Any],
        extras: Dict[str, Any],
    ) -> ClinicalReport:
        allow_acs = self._acs_scores_allowed_for_case(case, extras)
        risk_scores = self._parse_risk_scores(data, extras, allow_acs=allow_acs)
        image_findings = self._parse_image_findings(data, extras)
        differential = self._parse_differential(data)

        case_summary = str(data.get("case_summary") or data.get("summary") or "")
        assessment = str(data.get("assessment") or "")
        reasoning = str(data.get("reasoning") or "")
        if self.specialist != SpecialistType.cardiology:
            case_summary = self._sanitize_non_cardio_framing(case_summary)
            assessment = self._sanitize_non_cardio_framing(assessment)
            reasoning = self._sanitize_non_cardio_framing(reasoning)
        elif not allow_acs:
            # Cardiology specialty but non-ACS dominant problem — strip ACS score chatter
            case_summary = self._strip_acs_score_mentions(case_summary)
            assessment = self._strip_acs_score_mentions(assessment)
            reasoning = self._strip_acs_score_mentions(reasoning)

        routed_label = {
            SpecialistType.cardiology: "Cardiology",
            SpecialistType.dermatology: "Dermatology",
            SpecialistType.neurology: "Neurology",
            SpecialistType.general_internal_medicine: "General Internal Medicine",
            SpecialistType.clinical_pharmacology: "Clinical Pharmacology",
        }.get(self.specialist, self.specialist.value)
        routed_to = f"Routed to: {routed_label}"

        return ClinicalReport(
            specialist=self.specialist,
            routed_to=routed_to,
            chief_complaint=str(data.get("chief_complaint") or ""),
            case_summary=case_summary,
            key_findings=list(data.get("key_findings") or []),
            differential_diagnosis=differential,
            assessment=assessment,
            recommendations=list(data.get("recommendations") or []),
            red_flags=list(data.get("red_flags") or []),
            evidence=evidence,
            risk_scores=risk_scores,
            image_findings=image_findings,
            reasoning=reasoning,
            retrieval_path=path,
            raw={
                "llm": data,
                "retrieval_meta": {k: v for k, v in retrieval.items() if k != "hits"},
                "precomputed": extras,
                "acs_scores_allowed": allow_acs,
                "routed_specialist": self.specialist.value,
                "routed_to": routed_to,
            },
        )

    def apply_feedback(
        self,
        case: Dict[str, Any],
        report: Dict[str, Any],
        feedback_text: str,
    ) -> ClinicalReport:
        """Targeted correction — does not restart full reasoning from scratch."""
        system = (
            f"You are the {self.display_name}. You previously produced a structured clinical "
            "decision-support report. Apply the doctor's feedback as a TARGETED CORRECTION: "
            "revise only affected sections, preserve correct content, and keep the same JSON schema. "
            "Do not ignore safety-critical feedback. This is decision support, not a final diagnosis."
        )
        user = (
            f"Original case:\n{self.build_case_blob(case)[:5000]}\n\n"
            f"Previous report JSON:\n{json.dumps(report, default=str)[:6000]}\n\n"
            f"Doctor feedback:\n{feedback_text}\n\n"
            "Return the full updated report as JSON with keys: chief_complaint, case_summary, "
            "key_findings, differential_diagnosis (array of {diagnosis, likelihood, rationale}), "
            "assessment, recommendations, red_flags, reasoning, risk_scores, image_findings, "
            "knowledge_to_store (string or empty)."
        )
        raw_text = self.llm.complete(
            system, user, temperature=0.2, max_tokens=4096, json_mode=True
        )
        data = self._parse_json(raw_text)

        prev_evidence = []
        if isinstance(report, dict):
            prev_evidence = report.get("evidence") or []

        evidence_items: List[EvidenceItem] = []
        for e in prev_evidence:
            if isinstance(e, dict):
                evidence_items.append(EvidenceItem(**e))
            elif isinstance(e, EvidenceItem):
                evidence_items.append(e)

        knowledge = str(data.get("knowledge_to_store") or "").strip()
        knowledge_written = False
        if knowledge and len(knowledge) > 20:
            try:
                self.ingestion.ingest_text(
                    knowledge,
                    title=f"Doctor feedback knowledge — {self.specialist.value}",
                    source_type="doctor_feedback",
                    tags=["feedback", self.specialist.value, str(case.get("id", ""))],
                    extra_payload={
                        "modality": "text",
                        "case_id": case.get("id"),
                        "specialist": self.specialist.value,
                    },
                )
                knowledge_written = True
            except Exception as e:
                logger.warning("KB write-back failed: %s", e)

        extras = {
            "risk_scores": report.get("risk_scores") or [],
            "image_findings": report.get("image_findings") or [],
        }
        corrected = self._build_report(
            case,
            data,
            evidence_items,
            str(report.get("retrieval_path") or "pipeline_hybrid"),
            {"feedback_applied": True},
            extras,
        )
        corrected.raw["feedback_applied"] = True
        corrected.raw["knowledge_written"] = knowledge_written
        return corrected

    def _lane_discipline_block(self) -> str:
        """Hard framing rules injected into every specialist report prompt."""
        label = self.display_name
        if self.specialist == SpecialistType.cardiology:
            return (
                f"IDENTITY: You are the {label}. Frame the report as a cardiology evaluation "
                "only when the case is primarily cardiac. If the dominant problem is drug "
                "toxicity or SCAR, do not invent HEART/TIMI and do not force an ACS workup."
            )
        return (
            f"IDENTITY: You are the {label}. This case was routed to you — NOT to Cardiology.\n"
            f"- Open and center the report as a {label} assessment.\n"
            '- NEVER write "cardiology assessment", "cardiac risk stratification", or similar.\n'
            "- NEVER invent or discuss HEART score, TIMI score, or ACS chest-pain scores.\n"
            "- Secondary vital-sign effects (tachycardia, mild demand ischemia) may be mentioned "
            "briefly as supportive context only — they must not dominate the differential."
        )

    def _report_system_prompt(self) -> str:
        return (
            f"You are the {self.display_name} in a multi-agent clinical decision support system. "
            f"Focus: {self.system_focus}\n"
            f"{self._lane_discipline_block()}\n"
            "Produce structured decision support for a licensed clinician. "
            "Never claim to replace clinical judgment. "
            "Use only the retrieved evidence provided (already specialty-filtered). "
            "When uploaded images/PDFs are present, cite them explicitly. "
            "Rank differential diagnoses by likelihood; down-rank low-probability alternatives "
            "when high-risk objective data are present. "
            "Respond with JSON only."
        )

    def _report_user_prompt(
        self,
        case: Dict[str, Any],
        evidence: List[EvidenceItem],
        path: str,
        extras: Optional[Dict[str, Any]] = None,
    ) -> str:
        ev_lines = []
        for i, e in enumerate(evidence[:5], 1):
            ev_lines.append(f"[{i}] (score={e.score:.3f}) {e.title}: {e.snippet}")

        extra_block = ""
        if extras:
            extra_block = (
                "\nPrecomputed deterministic fields (use and display these; do not invent "
                "conflicting numeric scores):\n"
                f"{json.dumps(extras, default=str)[:3500]}\n"
            )

        return (
            f"{self._lane_discipline_block()}\n\n"
            f"Case:\n{self.build_case_blob(case)[:6500]}\n\n"
            f"Retrieval path: {path}\n"
            f"Evidence (top specialty-matched, deduplicated):\n"
            + ("\n".join(ev_lines) or "(no KB hits)")
            + "\n"
            + extra_block
            + "\nReturn JSON with keys:\n"
            "chief_complaint, case_summary, key_findings (array),\n"
            "differential_diagnosis (array of objects: "
            "{diagnosis, likelihood: leading|likely|possible|unlikely, rationale}),\n"
            "assessment, recommendations (array), red_flags (array), reasoning,\n"
            "image_findings (array of {source, label, summary}),\n"
            "risk_scores (array of {name, score, max_score, risk_band, detail}; "
            "empty unless specialty-appropriate — never HEART/TIMI unless primary ACS "
            "and you are the Cardiology Agent)."
        )

    # ACS scores only when Cardiology + primary ACS evaluation
    _ACS_SCORE_NAMES = frozenset(
        {
            "heart",
            "timi",
            "timi (ua/nstemi)",
            "timi ua/nstemi",
            "grace",
            "heart score",
            "timi score",
        }
    )

    def _is_acs_score_name(self, name: str) -> bool:
        n = (name or "").strip().lower()
        if n in self._ACS_SCORE_NAMES:
            return True
        return any(token in n for token in ("heart score", "timi", "grace score"))

    def _acs_scores_allowed_for_case(
        self, case: Dict[str, Any], extras: Optional[Dict[str, Any]] = None
    ) -> bool:
        """HEART/TIMI only for Cardiology on primary ACS workups."""
        if self.specialist != SpecialistType.cardiology:
            return False
        extras = extras or {}
        if "acs_scores_applicable" in extras:
            return bool(extras.get("acs_scores_applicable"))
        try:
            from app.agents.case_patterns import is_primary_acs_evaluation

            return is_primary_acs_evaluation(case)
        except Exception:
            return False

    def _sanitize_non_cardio_framing(self, text: str) -> str:
        """Remove mistaken cardiology framing from non-cardiology reports."""
        if not text:
            return text
        out = text
        out = re.sub(
            r"(?i)\bthe\s+cardiology\s+agent(?:'s|\s+s)?\s+role\b[^.!?]*[.!?]?",
            f"The {self.display_name}'s role is specialty-appropriate decision support.",
            out,
        )
        out = re.sub(
            r"(?i)\bfrom\s+a\s+cardiology\s+perspective\b",
            f"from a {self.display_name} perspective",
            out,
        )
        out = re.sub(
            r"(?i)\bcardiology\s+agent\b",
            self.display_name,
            out,
        )
        out = re.sub(
            r"(?i)\bcardiology\s+assessment\b",
            f"{self.display_name} assessment",
            out,
        )
        out = re.sub(
            r"(?i)\bcardiology\s+(?:evaluation|workup|consult|consultation)\b",
            f"{self.display_name} evaluation",
            out,
        )
        out = re.sub(
            r"(?i)\bas\s+a\s+cardiology\s+(?:assessment|evaluation|workup)\b",
            f"as a {self.display_name} assessment",
            out,
        )
        out = re.sub(
            r"(?i)\brule\s+out\s+ACS\b",
            "manage the primary toxicity/drug-safety problem",
            out,
        )
        out = re.sub(
            r"(?i)\bcardiac\s+risk\s+stratification\b",
            "specialty-appropriate risk assessment",
            out,
        )
        out = self._strip_acs_score_mentions(out)
        return out

    @staticmethod
    def _strip_acs_score_mentions(text: str) -> str:
        if not text:
            return text
        out = text
        out = re.sub(
            r"(?i)\b(?:the\s+)?HEART\s+score\b[^.!?]*[.!?]?",
            "",
            out,
        )
        out = re.sub(
            r"(?i)\b(?:the\s+)?TIMI(?:\s+score|\s*\(UA/NSTEMI\))?\b[^.!?]*[.!?]?",
            "",
            out,
        )
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()

    def _parse_risk_scores(
        self,
        data: Dict[str, Any],
        extras: Dict[str, Any],
        *,
        allow_acs: Optional[bool] = None,
    ) -> List[RiskScoreItem]:
        items: List[RiskScoreItem] = []
        if allow_acs is None:
            allow_acs = self.specialist == SpecialistType.cardiology

        def _append_from_dict(rs: Dict[str, Any]) -> None:
            name = str(rs.get("name") or "score")
            if not allow_acs and self._is_acs_score_name(name):
                logger.info(
                    "Stripping ACS risk score %r (specialist=%s allow_acs=%s)",
                    name,
                    self.specialist.value,
                    allow_acs,
                )
                return
            items.append(
                RiskScoreItem(
                    name=name,
                    score=rs.get("score"),
                    max_score=rs.get("max_score") or rs.get("max"),
                    risk_band=str(rs.get("risk_band") or ""),
                    detail=str(rs.get("detail") or rs.get("note") or ""),
                    components=dict(rs.get("components") or {}),
                )
            )

        # Prefer precomputed
        for rs in extras.get("risk_scores") or []:
            if isinstance(rs, RiskScoreItem):
                if not allow_acs and self._is_acs_score_name(rs.name):
                    continue
                items.append(rs)
            elif isinstance(rs, dict):
                _append_from_dict(rs)
        if items:
            return items
        for rs in data.get("risk_scores") or []:
            if isinstance(rs, dict):
                _append_from_dict(rs)
        return items

    def _parse_image_findings(
        self, data: Dict[str, Any], extras: Dict[str, Any]
    ) -> List[ImageFindingItem]:
        items: List[ImageFindingItem] = []
        for im in extras.get("image_findings") or []:
            if isinstance(im, ImageFindingItem):
                items.append(im)
            elif isinstance(im, dict):
                items.append(
                    ImageFindingItem(
                        source=str(im.get("source") or ""),
                        label=str(im.get("label") or "Uploaded image"),
                        summary=str(im.get("summary") or ""),
                        modality=str(im.get("modality") or "image"),
                    )
                )
        if items:
            return items
        for im in data.get("image_findings") or []:
            if isinstance(im, dict):
                items.append(
                    ImageFindingItem(
                        source=str(im.get("source") or ""),
                        label=str(im.get("label") or "Uploaded image"),
                        summary=str(im.get("summary") or ""),
                        modality=str(im.get("modality") or "image"),
                    )
                )
        return items

    def _parse_differential(
        self, data: Dict[str, Any]
    ) -> List[Union[str, Dict[str, Any]]]:
        raw = data.get("differential_diagnosis") or []
        out: List[Any] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(
                    {
                        "diagnosis": str(item.get("diagnosis") or item.get("name") or ""),
                        "likelihood": str(item.get("likelihood") or "possible"),
                        "rationale": str(item.get("rationale") or item.get("reason") or ""),
                    }
                )
            elif isinstance(item, DifferentialItem):
                out.append(item.model_dump())
        return out

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return {
            "case_summary": text[:2000],
            "assessment": text[:2000],
            "recommendations": [],
            "key_findings": [],
            "differential_diagnosis": [],
            "red_flags": [],
            "reasoning": "Model returned non-JSON; raw text captured in summary.",
        }
