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
        "(HEART, TIMI only when primarily ACS), anti-ischemic therapy, disposition."
    )

    def _precompute_report_fields(self, case: Dict[str, Any]) -> Dict[str, Any]:
        # HARD: never compute HEART/TIMI unless primary ACS evaluation
        block = acs_risk_block(case)
        images = block.get("image_findings") or []
        if not block.get("acs_scores_applicable"):
            return {
                "risk_scores": [],
                "image_findings": images,
                "high_risk_acs_pattern": False,
                "heart_raw": None,
                "timi_raw": None,
                "acs_scores_applicable": False,
                "acs_scores_omitted_reason": block.get("acs_scores_omitted_reason") or (
                    "HEART/TIMI omitted — not a primary ACS/chest-pain evaluation"
                ),
            }

        heart = block["heart"] or {}
        timi = block["timi"] or {}
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
            "image_findings": images,
            "high_risk_acs_pattern": block.get("high_risk_acs_pattern"),
            "heart_raw": heart,
            "timi_raw": timi,
            "acs_scores_applicable": True,
            "acs_scores_omitted_reason": "",
        }

    def _report_system_prompt(self) -> str:
        return (
            "You are the Cardiology Agent in a multi-agent clinical decision support system. "
            "Focus: ACS/chest pain risk stratification, ECG + troponin integration, "
            "anti-ischemic management, and safe disposition.\n\n"
            "MANDATORY RULES:\n"
            "1) Include precomputed HEART and TIMI scores ONLY when provided in "
            "precomputed fields (acs_scores_applicable=true). If ACS scores were omitted, "
            "do NOT invent HEART/TIMI and do not center the report on ACS scoring.\n"
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
        acs_ok = bool(extras.get("acs_scores_applicable"))

        if acs_ok:
            score_block = (
                "PRECOMPUTED RISK SCORES (copy into risk_scores; explain briefly in assessment):\n"
                f"HEART: {heart.get('score')}/{heart.get('max')} ({heart.get('risk_band')}) "
                f"components={json.dumps(heart.get('components') or {}, default=str)[:1200]}\n"
                f"TIMI UA/NSTEMI: {timi.get('score')}/{timi.get('max')} ({timi.get('risk_band')}) "
                f"factors={json.dumps(timi.get('factors') or [], default=str)[:1200]}\n"
                f"High-risk ACS pattern (elevated troponin + ischemic ECG): "
                f"{extras.get('high_risk_acs_pattern')}\n"
            )
            assessment_note = "assessment (must mention HEART and TIMI totals and risk bands),\n"
        else:
            score_block = (
                "ACS RISK SCORES: NOT APPLICABLE for this case "
                f"({extras.get('acs_scores_omitted_reason') or 'not primary ACS'}). "
                "Leave risk_scores empty. Do NOT invent HEART or TIMI.\n"
            )
            assessment_note = (
                "assessment (cardiology-relevant; do NOT invent HEART/TIMI),\n"
            )

        return (
            f"Case:\n{self.build_case_blob(case)[:6500]}\n\n"
            f"Retrieval path: {path}\n"
            "Evidence (top 3–5 cardiology-relevant, deduplicated sources):\n"
            + ("\n".join(ev_lines) or "(no KB hits)")
            + "\n\n"
            "UPLOADED IMAGE / DOCUMENT FINDINGS (cite explicitly in key_findings & assessment):\n"
            + ("\n".join(img_lines) or "(none)")
            + "\n\n"
            + score_block
            + "\nReturn JSON with keys:\n"
            "chief_complaint, case_summary, key_findings (array; include ECG image citation if any),\n"
            "differential_diagnosis (array of {diagnosis, likelihood: leading|likely|possible|unlikely, rationale}),\n"
            + assessment_note
            + "recommendations (array), red_flags (array), reasoning,\n"
            "image_findings (array of {source, label, summary}),\n"
            "risk_scores (array of {name, score, max_score, risk_band, detail})."
        )


class DermatologyAgent(BaseSpecialistAgent):
    specialist = SpecialistType.dermatology
    display_name = "Dermatology Agent"
    system_focus = (
        "Skin, hair, nail, and mucosal disease; lesion description; infection vs inflammatory "
        "vs neoplastic differentials; severe cutaneous adverse reactions (SJS/TEN, DRESS, AGEP); "
        "drug-induced rashes; when to biopsy or refer urgently (burn unit / derm emergency)."
    )

    def _report_system_prompt(self) -> str:
        return (
            "You are the Dermatology Agent in a multi-agent clinical decision support system. "
            "Focus: dermatologic diagnosis and management, including severe cutaneous adverse "
            "reactions (SJS/TEN, DRESS, AGEP), drug rashes with mucosal involvement, infection "
            "vs inflammatory vs neoplastic differentials, and urgent referral thresholds.\n\n"
            f"{self._lane_discipline_block()}\n\n"
            "MANDATORY RULES:\n"
            "1) Stay dermatology-focused. NEVER write 'cardiology assessment'. "
            "Do NOT calculate or discuss HEART, TIMI, or ACS risk scores.\n"
            "2) For suspected SCAR (SJS/TEN, DRESS, AGEP): culprit-drug stop, mucosal assessment, "
            "BSA/SCORTEN when TEN possible, urgent derm/burn-unit pathways. Rank SCAR leading.\n"
            "3) Integrate drug causality (carbamazepine, allopurinol, lamotrigine, sulfa) — "
            "do not reframe as ACS even if fever/tachycardia present.\n"
            "4) Cite rash/lesion images when present.\n"
            "5) Prefer dermatology evidence; ignore pure ACS snippets.\n"
            "6) Decision support only.\n"
            "Respond with JSON only. risk_scores empty unless SCORTEN-style derm detail; never HEART/TIMI."
        )


class NeurologyAgent(BaseSpecialistAgent):
    specialist = SpecialistType.neurology
    display_name = "Neurology Agent"
    system_focus = (
        "Stroke, seizure, headache, neuromuscular disease, CNS infection red flags, "
        "localization and urgent neuroimaging considerations."
    )

    def _report_system_prompt(self) -> str:
        return (
            "You are the Neurology Agent in a multi-agent clinical decision support system. "
            f"Focus: {self.system_focus}\n\n"
            "MANDATORY RULES:\n"
            "1) Stay neurology-focused. Do NOT invent HEART/TIMI or ACS chest-pain scores.\n"
            "2) For stroke/TIA: time last known well, NIHSS/focal deficits, imaging pathway, "
            "thrombolysis/thrombectomy eligibility framing as decision support only.\n"
            "3) Prefer neurology-matched evidence; ignore off-topic cardiology ACS snippets "
            "unless cardioembolic source is secondary context.\n"
            "4) Decision support only — not a final diagnosis.\n"
            "Respond with JSON only."
        )


class GeneralInternalMedicineAgent(BaseSpecialistAgent):
    specialist = SpecialistType.general_internal_medicine
    display_name = "General Internal Medicine Agent"
    system_focus = (
        "Broad multi-system evaluation, undifferentiated presentations, sepsis, "
        "metabolic disorders, comprehensive differential diagnosis, care coordination."
    )

    def _report_system_prompt(self) -> str:
        return (
            "You are the General Internal Medicine Agent in a multi-agent clinical decision "
            "support system. Focus: broad multi-system evaluation and care coordination.\n\n"
            "MANDATORY RULES:\n"
            "1) Do NOT automatically apply HEART/TIMI unless the case is clearly a primary "
            "ACS/chest-pain evaluation (and even then note scores are approximate).\n"
            "2) Prefer broad but ranked differentials; do not force ACS framing onto non-cardiac cases.\n"
            "3) Decision support only — not a final diagnosis.\n"
            "Respond with JSON only."
        )


class ClinicalPharmacologyAgent(BaseSpecialistAgent):
    specialist = SpecialistType.clinical_pharmacology
    display_name = "Clinical Pharmacology & Drug Safety Agent"
    system_focus = (
        "Medications, interactions, dosing (including renal/hepatic), adverse effects, "
        "drug toxicity (digoxin, lithium, serotonin syndrome), contraindications, TDM, deprescribing."
    )

    def _report_system_prompt(self) -> str:
        return (
            "You are the Clinical Pharmacology & Drug Safety Agent — NOT the Cardiology Agent. "
            "Focus: drug toxicity, major drug-drug interactions, toxidromes "
            "(serotonin syndrome, lithium toxicity, digoxin toxicity, anticholinergic toxicity), "
            "dosing, TDM, ADRs.\n\n"
            f"{self._lane_discipline_block()}\n\n"
            "MANDATORY RULES:\n"
            "1) Begin as a Clinical Pharmacology assessment. NEVER write 'Cardiology Agent', "
            "'cardiology assessment', or 'rule out ACS' as the center of the case.\n"
            "2) risk_scores MUST be [] — never HEART or TIMI.\n"
            "3) Lithium toxicity: hold lithium, check level/electrolytes/renal function, IV fluids, "
            "consider hemodialysis for severe toxicity; lead differential with lithium toxicity.\n"
            "4) Serotonin syndrome (SSRI + tramadol): stop serotonergics, benzos, cyproheptadine, cooling.\n"
            "5) Digoxin toxicity (+ macrolide): hold digoxin/interactor, K+/Mg2+, level, Fab if severe.\n"
            "6) Cardiac signs (bradycardia, tachycardia) are SECONDARY toxicity effects only.\n"
            "7) Decision support only.\n"
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

        return (
            f"{self._lane_discipline_block()}\n\n"
            f"Case:\n{self.build_case_blob(case)[:6500]}\n\n"
            f"Retrieval path: {path}\n"
            "Evidence (prefer drug safety / interaction / toxicity sources):\n"
            + ("\n".join(ev_lines) or "(no KB hits)")
            + "\n\n"
            "FRAMEING REQUIREMENTS:\n"
            "- This is a Clinical Pharmacology & Drug Safety report, NOT a cardiology assessment.\n"
            "- Lead differential with the toxicity/interaction (serotonin syndrome, digoxin toxicity, etc.).\n"
            "- Do NOT include HEART or TIMI scores under any circumstances.\n"
            "- Cardiac signs (tachycardia, bradycardia) only as secondary toxicity effects.\n\n"
            "Return JSON with keys:\n"
            "chief_complaint, case_summary, key_findings (array),\n"
            "differential_diagnosis (array of {diagnosis, likelihood: leading|likely|possible|unlikely, rationale}; "
            "lead with drug toxicity/interaction/toxidrome),\n"
            "assessment (toxicity/interaction first; never 'cardiology assessment'), "
            "recommendations (array), red_flags (array), reasoning,\n"
            "image_findings (array of {source, label, summary}),\n"
            "risk_scores ([] empty unless an explicit drug level is in the case)."
        )
