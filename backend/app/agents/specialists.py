"""Five specialized clinical agents."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from app.agents.base import BaseSpecialistAgent
from app.agents.risk_scores import acs_risk_block
from app.models.schemas import RiskScoreItem, SpecialistType


class CardiologyAgent(BaseSpecialistAgent):
    specialist = SpecialistType.cardiology
    display_name = "Cardiology Agent"
    system_focus = (
        "Cardiovascular disease: ACS, heart failure, arrhythmias, valvular disease, "
        "hypertension emergencies, ECG interpretation support, risk stratification "
        "(HEART, TIMI), anti-ischemic therapy, disposition."
    )

    def _precompute_report_fields(self, case: Dict[str, Any]) -> Dict[str, Any]:
        block = acs_risk_block(case)
        heart = block["heart"]
        timi = block["timi"]
        risk_scores = [
            RiskScoreItem(
                name="HEART",
                score=heart.get("score"),
                max_score=heart.get("max", 10),
                risk_band=str(heart.get("risk_band") or ""),
                detail=str(heart.get("note") or ""),
                components=dict(heart.get("components") or {}),
            ).model_dump(),
            RiskScoreItem(
                name="TIMI (UA/NSTEMI)",
                score=timi.get("score"),
                max_score=timi.get("max", 7),
                risk_band=str(timi.get("risk_band") or ""),
                detail=str(timi.get("note") or ""),
                components={"factors": timi.get("factors") or []},
            ).model_dump(),
        ]
        return {
            "risk_scores": risk_scores,
            "image_findings": block.get("image_findings") or [],
            "high_risk_acs_pattern": block.get("high_risk_acs_pattern"),
            "heart_raw": heart,
            "timi_raw": timi,
        }

    def _report_system_prompt(self) -> str:
        return (
            "You are the Cardiology Agent in a multi-agent clinical decision support system. "
            "Focus: ACS/chest pain risk stratification, ECG + troponin integration, "
            "anti-ischemic management, and safe disposition.\n\n"
            "MANDATORY RULES:\n"
            "1) Always include the precomputed HEART and TIMI scores in risk_scores "
            "(do not invent different numeric totals).\n"
            "2) Explicitly cite uploaded images: use phrasing like "
            "'ECG image analysis shows…' when image findings are provided.\n"
            "3) For high-risk ACS patterns (elevated troponin + ischemic ECG), "
            "differential MUST rank NSTE-ACS/NSTEMI (or STEMI if ST elevation) as "
            "'leading' or 'likely'. Mark PE, aortic dissection, GERD, musculoskeletal pain, "
            "pericarditis as 'possible' or 'unlikely' with brief rationale — do not present "
            "them as equal likelihood.\n"
            "4) Prefer cardiology-matched evidence only; ignore off-topic snippets.\n"
            "5) Decision support only — not a final diagnosis.\n"
            "Respond with JSON only."
        )

    def _report_user_prompt(
        self,
        case: Dict[str, Any],
        evidence,
        path: str,
        extras=None,
    ) -> str:
        extras = extras or {}
        ev_lines: List[str] = []
        for i, e in enumerate(evidence[:5], 1):
            ev_lines.append(f"[{i}] (score={e.score:.3f}) {e.title}: {e.snippet}")

        img_lines = []
        for im in extras.get("image_findings") or []:
            img_lines.append(
                f"- {im.get('label')} (file: {im.get('source')}): {im.get('summary', '')[:500]}"
            )

        heart = extras.get("heart_raw") or {}
        timi = extras.get("timi_raw") or {}

        return (
            f"Case:\n{self.build_case_blob(case)[:6500]}\n\n"
            f"Retrieval path: {path}\n"
            "Evidence (top 3–5 cardiology-relevant, deduplicated sources):\n"
            + ("\n".join(ev_lines) or "(no KB hits)")
            + "\n\n"
            "UPLOADED IMAGE / DOCUMENT FINDINGS (cite explicitly in key_findings & assessment):\n"
            + ("\n".join(img_lines) or "(none)")
            + "\n\n"
            "PRECOMPUTED RISK SCORES (copy into risk_scores; explain briefly in assessment):\n"
            f"HEART: {heart.get('score')}/{heart.get('max')} ({heart.get('risk_band')}) "
            f"components={json.dumps(heart.get('components') or {}, default=str)[:1200]}\n"
            f"TIMI UA/NSTEMI: {timi.get('score')}/{timi.get('max')} ({timi.get('risk_band')}) "
            f"factors={json.dumps(timi.get('factors') or [], default=str)[:1200]}\n"
            f"High-risk ACS pattern (elevated troponin + ischemic ECG): "
            f"{extras.get('high_risk_acs_pattern')}\n\n"
            "Return JSON with keys:\n"
            "chief_complaint, case_summary, key_findings (array; include ECG image citation if any),\n"
            "differential_diagnosis (array of {diagnosis, likelihood: leading|likely|possible|unlikely, rationale}),\n"
            "assessment (must mention HEART and TIMI totals and risk bands),\n"
            "recommendations (array), red_flags (array), reasoning,\n"
            "image_findings (array of {source, label, summary}),\n"
            "risk_scores (array of {name, score, max_score, risk_band, detail})."
        )


class DermatologyAgent(BaseSpecialistAgent):
    specialist = SpecialistType.dermatology
    display_name = "Dermatology Agent"
    system_focus = (
        "Skin, hair, and nail conditions; lesion description; infection vs inflammatory "
        "vs neoplastic differentials; when to biopsy or refer urgently."
    )


class NeurologyAgent(BaseSpecialistAgent):
    specialist = SpecialistType.neurology
    display_name = "Neurology Agent"
    system_focus = (
        "Stroke, seizure, headache, neuromuscular disease, CNS infection red flags, "
        "localization and urgent neuroimaging considerations."
    )


class GeneralInternalMedicineAgent(BaseSpecialistAgent):
    specialist = SpecialistType.general_internal_medicine
    display_name = "General Internal Medicine Agent"
    system_focus = (
        "Broad multi-system evaluation, undifferentiated presentations, sepsis, "
        "metabolic disorders, comprehensive differential diagnosis, care coordination."
    )


class ClinicalPharmacologyAgent(BaseSpecialistAgent):
    specialist = SpecialistType.clinical_pharmacology
    display_name = "Clinical Pharmacology & Drug Safety Agent"
    system_focus = (
        "Medications, interactions, dosing (including renal/hepatic), adverse effects, "
        "contraindications, therapeutic drug monitoring, deprescribing."
    )
