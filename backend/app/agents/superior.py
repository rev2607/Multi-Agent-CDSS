"""Superior Agent — routes each case to exactly one specialized agent."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Tuple

from app.core.llm import get_llm_client
from app.models.schemas import SpecialistType

logger = logging.getLogger(__name__)

VALID = {s.value for s in SpecialistType}

ROUTING_RUBRIC = """
Route to exactly ONE specialist:

- cardiology: heart, ACS, arrhythmia, heart failure, chest pain cardiac, ECG, BP cardiac workup
- dermatology: skin, rash, lesions, melanoma, dermatitis, psoriasis
- neurology: stroke, seizure, headache red flags, neuropathy, CNS, altered mental status neurologic
- clinical_pharmacology: drug interactions, dosing, adverse effects, polypharmacy, contraindications, TDM
- general_internal_medicine: multi-system, unclear specialty, sepsis, metabolic, broad differential, default when ambiguous

If multiple could apply, choose the single best primary owner. Prefer general_internal_medicine when unclear.
""".strip()


class SuperiorAgent:
    """Router only — does not generate the final clinical report."""

    def __init__(self) -> None:
        self.llm = get_llm_client()

    def route(self, case: Dict[str, Any]) -> Tuple[SpecialistType, str]:
        blob_parts = [
            case.get("title") or "",
            case.get("patient_context") or "",
            case.get("clinical_text") or "",
            case.get("notes") or "",
        ]
        for att in case.get("attachments") or []:
            blob_parts.append(
                f"{att.get('filename')}: {(att.get('extracted_text') or '')[:1500]}"
            )
        blob = "\n".join(blob_parts)[:6000]

        # Fast keyword pre-route for demo reliability without LLM
        pre = self._keyword_route(blob)
        if self.llm.provider == "stub":
            return pre, f"Keyword/stub routing → {pre.value}"

        system = (
            "You are the Superior Agent (router) in a medical multi-agent CDSS. "
            "Your ONLY job is to assign the case to exactly one specialized agent. "
            "Do not write a clinical report.\n\n"
            f"{ROUTING_RUBRIC}\n\n"
            "Return JSON: {\"specialist\": \"<one of: cardiology|dermatology|neurology|"
            "general_internal_medicine|clinical_pharmacology>\", \"rationale\": \"...\"}"
        )
        user = f"Case materials:\n{blob}"
        try:
            raw = self.llm.complete(system, user, temperature=0.0, max_tokens=400, json_mode=True)
            data = self._parse(raw)
            spec = str(data.get("specialist", "")).strip().lower()
            if spec not in VALID:
                # try fuzzy
                for v in VALID:
                    if v in spec or spec in v:
                        spec = v
                        break
            if spec not in VALID:
                logger.warning("Invalid specialist %r; using keyword route", spec)
                return pre, data.get("rationale") or f"Fallback keyword route → {pre.value}"
            return SpecialistType(spec), str(data.get("rationale") or "")
        except Exception as e:
            logger.warning("Superior routing LLM failed: %s", e)
            return pre, f"Fallback after error: {e} → {pre.value}"

    def _keyword_route(self, text: str) -> SpecialistType:
        t = text.lower()
        scores = {
            SpecialistType.cardiology: 0,
            SpecialistType.dermatology: 0,
            SpecialistType.neurology: 0,
            SpecialistType.clinical_pharmacology: 0,
            SpecialistType.general_internal_medicine: 0,
        }
        rules = [
            (SpecialistType.cardiology, [
                "chest pain", "myocardial", "stemi", "nstemi", "ecg", "ekg",
                "heart failure", "arrhythmia", "atrial fibrillation", "angina",
                "troponin", "cardiology", "hypertension crisis",
            ]),
            (SpecialistType.dermatology, [
                "rash", "dermat", "melanoma", "lesion", "psoriasis", "eczema",
                "pruritus", "skin", "mole", "urticaria",
            ]),
            (SpecialistType.neurology, [
                "stroke", "seizure", "migraine", "neurolog", "hemiparesis",
                "aphasia", "meningism", "parkinson", "neuropathy", "headache thunderclap",
            ]),
            (SpecialistType.clinical_pharmacology, [
                "drug interaction", "polypharmacy", "dosing", "adverse effect",
                "contraindication", "pharmacolog", "overdose", "inr", "qt prolongation",
                "side effect", "medication review",
            ]),
        ]
        for spec, kws in rules:
            for kw in kws:
                if kw in t:
                    scores[spec] += 1
        best = max(scores, key=lambda s: scores[s])
        if scores[best] == 0:
            return SpecialistType.general_internal_medicine
        return best

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
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
        return {}
