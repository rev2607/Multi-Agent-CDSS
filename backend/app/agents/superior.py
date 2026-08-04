"""Superior Agent — deterministic specialty router (first principles).

Flow:
  1. build_case_blob (full text + attachments)
  2. hard_route_specialty → if hit, return immediately (path=HARD)
  3. keyword_route with cardio suppression
  4. LLM last resort (temp 0), then hard override again
  5. Absolute: cardiology only if ACS hard match
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agents.case_patterns import (
    build_case_blob,
    forbids_cardiology,
    hard_route_specialty,
    is_acs_presentation,
    resolve_specialist,
)
from app.core.llm import get_llm_client
from app.models.schemas import SpecialistType

logger = logging.getLogger(__name__)

VALID = {s.value for s in SpecialistType}

SPECIALIST_DISPLAY = {
    SpecialistType.cardiology: "Cardiology",
    SpecialistType.dermatology: "Dermatology",
    SpecialistType.neurology: "Neurology",
    SpecialistType.general_internal_medicine: "General Internal Medicine",
    SpecialistType.clinical_pharmacology: "Clinical Pharmacology",
}

ROUTING_RUBRIC = """
HARD PRIORITY (first match wins):
1) clinical_pharmacology — drug toxicity, major DDI, toxidrome, overdose, high INR
2) dermatology — SJS/TEN/DRESS/AGEP, severe drug rash with mucosa
3) neurology — stroke/TIA/acute focal deficit
4) cardiology — ONLY ACS: chest pain/pressure + ischemic ECG/troponin, or STEMI/NSTEMI
5) general_internal_medicine — default (NEVER cardiology by default)

NEVER route lithium, serotonin syndrome, digoxin toxicity, warfarin high INR, or SCAR to cardiology.
""".strip()

FEW_SHOT = """
Examples:
"64M vomiting tremor ataxia on lithium + AKI" → clinical_pharmacology
"58F agitation clonus fever after tramadol + sertraline" → clinical_pharmacology
"72M yellow halos bradycardia digoxin + clarithromycin" → clinical_pharmacology
"68F warfarin + ciprofloxacin INR 5.8 epistaxis" → clinical_pharmacology
"45F painful rash fever oral erosions after carbamazepine" → dermatology
"62M chest pressure elevated troponin ST depression" → cardiology
"70M sudden right hemiparesis aphasia" → neurology
""".strip()


def format_routed_to(specialist: SpecialistType, rationale: str = "") -> str:
    label = SPECIALIST_DISPLAY.get(specialist, specialist.value)
    base = f"Routed to: {label}"
    r = (rationale or "").strip()
    if not r:
        return base
    if r.lower().startswith("routed to:"):
        return r
    return f"{base} — {r}"


class SuperiorAgent:
    def __init__(self) -> None:
        self.llm = get_llm_client()

    def route(self, case: Dict[str, Any]) -> Tuple[SpecialistType, str]:
        blob = build_case_blob(case)

        # 1) HARD first — always wins
        hard = hard_route_specialty(blob)
        if hard is not None:
            spec = SpecialistType(hard[0])
            rationale = format_routed_to(spec, hard[1])
            logger.info(
                "superior_route path=HARD specialist=%s rationale=%s",
                spec.value,
                rationale,
            )
            return spec, rationale

        # 2) Keywords
        kw = self._keyword_route(blob)
        path = "KEYWORD"

        if getattr(self.llm, "provider", None) == "stub":
            spec = self._block_false_cardio(blob, kw)
            logger.info("superior_route path=KEYWORD/STUB specialist=%s", spec.value)
            return spec, format_routed_to(spec, "Keyword/stub routing")

        # 3) LLM last resort
        try:
            system = (
                "You are a clinical specialty router. Return JSON only.\n"
                f"{ROUTING_RUBRIC}\n\n{FEW_SHOT}\n\n"
                'Format: {"specialist":"clinical_pharmacology|dermatology|neurology|'
                'cardiology|general_internal_medicine","rationale":"..."}'
            )
            raw = self.llm.complete(
                system,
                f"Case materials:\n{blob}",
                temperature=0.0,
                max_tokens=300,
                json_mode=True,
            )
            data = self._parse(raw)
            raw_spec = str(data.get("specialist") or "").strip().lower()
            if raw_spec not in VALID:
                for v in VALID:
                    if v in raw_spec:
                        raw_spec = v
                        break
            if raw_spec not in VALID:
                spec = self._block_false_cardio(blob, kw)
                logger.info(
                    "superior_route path=KEYWORD_FALLBACK specialist=%s",
                    spec.value,
                )
                return spec, format_routed_to(
                    spec, data.get("rationale") or "Fallback keyword"
                )

            spec = SpecialistType(raw_spec)
            reason = str(data.get("rationale") or "").strip()
            path = "LLM"

            # Hard override after LLM
            hard2 = hard_route_specialty(blob)
            if hard2 is not None:
                forced = SpecialistType(hard2[0])
                if forced != spec:
                    logger.warning(
                        "superior_route LLM=%s HARD_OVERRIDE→%s",
                        spec.value,
                        forced.value,
                    )
                    path = "HARD_OVERRIDE"
                spec = forced
                reason = hard2[1]

            spec = self._block_false_cardio(blob, spec)

            # Absolute resolve
            final, final_reason = resolve_specialist(case, spec.value)
            if final != spec.value:
                logger.warning(
                    "superior_route RESOLVE %s→%s (%s)",
                    spec.value,
                    final,
                    final_reason,
                )
                path = "RESOLVE"
                spec = SpecialistType(final)
                reason = final_reason

            logger.info("superior_route path=%s specialist=%s", path, spec.value)
            return spec, format_routed_to(spec, reason)
        except Exception as e:
            logger.warning("superior_route LLM failed: %s", e)
            spec = self._block_false_cardio(blob, kw)
            logger.info(
                "superior_route path=ERROR_FALLBACK specialist=%s",
                spec.value,
            )
            return spec, format_routed_to(spec, f"Fallback after error: {e}")

    def _block_false_cardio(
        self, blob: str, specialist: SpecialistType
    ) -> SpecialistType:
        hard = hard_route_specialty(blob)
        if hard is not None:
            return SpecialistType(hard[0])

        if specialist != SpecialistType.cardiology:
            return specialist

        acs_ok, _ = is_acs_presentation(blob)
        if acs_ok and not forbids_cardiology(blob):
            return SpecialistType.cardiology

        kw = self._keyword_route(blob)
        if kw != SpecialistType.cardiology:
            logger.warning("Blocked false cardiology → %s", kw.value)
            return kw
        logger.warning("Blocked false cardiology → GIM")
        return SpecialistType.general_internal_medicine

    @staticmethod
    def _case_blob(case: Dict[str, Any]) -> str:
        return build_case_blob(case)

    def _high_confidence_route(
        self, text: str
    ) -> Optional[Tuple[SpecialistType, str]]:
        hit = hard_route_specialty(text)
        if not hit:
            return None
        return SpecialistType(hit[0]), hit[1]

    def _keyword_route(self, text: str) -> SpecialistType:
        t = text.lower()
        hit = hard_route_specialty(t)
        if hit:
            return SpecialistType(hit[0])

        # Explicit scores — GIM is default, cardiology needs high bar
        scores = {
            "clinical_pharmacology": 0.0,
            "dermatology": 0.0,
            "neurology": 0.0,
            "cardiology": 0.0,
            "general_internal_medicine": 0.01,  # tiny floor so max never picks empty cardio
        }
        rules = [
            ("clinical_pharmacology", "lithium", 6.0),
            ("clinical_pharmacology", "serotonin", 6.0),
            ("clinical_pharmacology", "sertraline", 4.5),
            ("clinical_pharmacology", "tramadol", 4.5),
            ("clinical_pharmacology", "clonus", 5.0),
            ("clinical_pharmacology", "hyperreflex", 4.0),
            ("clinical_pharmacology", "digoxin", 5.0),
            ("clinical_pharmacology", "clarithromycin", 4.0),
            ("clinical_pharmacology", "macrolide", 3.5),
            ("clinical_pharmacology", "warfarin", 5.5),
            ("clinical_pharmacology", "coumadin", 5.5),
            ("clinical_pharmacology", "inr", 5.0),
            ("clinical_pharmacology", "ciprofloxacin", 4.5),
            ("clinical_pharmacology", "fluoroquinolone", 4.0),
            ("clinical_pharmacology", "epistaxis", 4.0),
            ("clinical_pharmacology", "bruising", 3.0),
            ("clinical_pharmacology", "bleeding", 3.0),
            ("clinical_pharmacology", "coagulopathy", 4.0),
            ("clinical_pharmacology", "toxicity", 4.5),
            ("clinical_pharmacology", "toxidrome", 5.5),
            ("clinical_pharmacology", "overdose", 5.0),
            ("clinical_pharmacology", "drug interaction", 5.0),
            ("clinical_pharmacology", "drug-drug", 5.0),
            ("clinical_pharmacology", "ataxia", 2.5),
            ("clinical_pharmacology", "tremor", 2.0),
            ("clinical_pharmacology", "ssri", 3.5),
            ("clinical_pharmacology", "supratherapeutic", 4.5),
            ("dermatology", "sjs", 6.0),
            ("dermatology", "stevens-johnson", 6.0),
            ("dermatology", "toxic epidermal", 6.0),
            ("dermatology", "dress", 5.0),
            ("dermatology", "agep", 5.0),
            ("dermatology", "carbamazepine", 4.0),
            ("dermatology", "lamotrigine", 3.5),
            ("dermatology", "allopurinol", 3.5),
            ("dermatology", "mucosal", 4.0),
            ("dermatology", "rash", 3.0),
            ("dermatology", "blister", 3.5),
            ("neurology", "stroke", 5.5),
            ("neurology", "hemiparesis", 5.5),
            ("neurology", "aphasia", 5.0),
            ("neurology", "nihss", 4.5),
            ("neurology", "tia", 4.5),
            ("cardiology", "chest pain", 5.0),
            ("cardiology", "chest pressure", 5.0),
            ("cardiology", "stemi", 6.0),
            ("cardiology", "nstemi", 6.0),
            ("cardiology", "acute coronary", 5.5),
            ("cardiology", "st depression", 4.0),
            ("cardiology", "st elevation", 4.0),
            ("cardiology", "angina", 4.5),
            ("cardiology", "myocardial", 4.0),
            ("cardiology", "troponin", 2.0),
            ("cardiology", "ecg", 0.15),
            ("cardiology", "ekg", 0.15),
            ("cardiology", "bradycardia", 0.2),
            ("cardiology", "atrial fibrillation", 0.5),
            ("cardiology", "arrhythmia", 0.4),
        ]
        for spec, kw, w in rules:
            if kw in t:
                scores[spec] += w

        if scores["clinical_pharmacology"] >= 2.0:
            scores["cardiology"] *= 0.05
        if scores["dermatology"] >= 2.0:
            scores["cardiology"] *= 0.05
        if scores["neurology"] >= 3.0:
            scores["cardiology"] *= 0.1

        best = max(scores, key=lambda k: scores[k])
        if best == "cardiology" and scores["cardiology"] < 4.5:
            best = "general_internal_medicine"
        if best == "cardiology" and forbids_cardiology(t):
            scores["cardiology"] = 0
            best = max(scores, key=lambda k: scores[k])

        return SpecialistType(best)

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
