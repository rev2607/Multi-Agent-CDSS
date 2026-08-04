"""Deterministic clinical routing — hard rules only.

Priority (first match wins):
  1. clinical_pharmacology
  2. dermatology (SCAR)
  3. neurology (stroke)
  4. cardiology (ACS only — never default)
  5. None → fall through

Cardiology is returned ONLY when ACS hard rules match.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

VALID_SPECIALTIES = frozenset({
    "clinical_pharmacology",
    "dermatology",
    "neurology",
    "cardiology",
    "general_internal_medicine",
})


# ---------------------------------------------------------------------------
# Canonical blob — Superior + specialize_guard MUST use this same function
# ---------------------------------------------------------------------------

def build_case_blob(
    case: Dict[str, Any],
    *,
    max_att_chars: int = 4000,
    max_total: int = 12000,
) -> str:
    parts: List[str] = [
        str(case.get("title") or ""),
        str(case.get("patient_context") or ""),
        str(case.get("clinical_text") or ""),
        str(case.get("notes") or ""),
    ]
    for att in case.get("attachments") or []:
        if not isinstance(att, dict):
            continue
        parts.append(str(att.get("filename") or ""))
        parts.append(str(att.get("extracted_text") or "")[:max_att_chars])
    return "\n".join(parts)[:max_total]


def case_text_blob(case: Dict[str, Any]) -> str:
    """Alias for build_case_blob (compat)."""
    return build_case_blob(case)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _low(text: str) -> str:
    return (text or "").lower()


def _has(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _re(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.I))


def _classic_acs(t: str) -> bool:
    """Strict ACS language — used as veto inside pharm generic branch only."""
    if _has(t, (
        "stemi", "nstemi", "nste-acs", "acute coronary",
        "myocardial infarction", "unstable angina",
    )):
        return True
    chest = _has(t, (
        "chest pain", "chest pressure", "crushing chest", "substernal", "angina",
    ))
    ischemic = _has(t, (
        "troponin", "st elevation", "st depression", "st-segment", "ischemic ecg",
    ))
    return chest and ischemic


# ---------------------------------------------------------------------------
# A) Clinical pharmacology
# ---------------------------------------------------------------------------

_SSRI = (
    "sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram",
    "fluvoxamine", "venlafaxine", "desvenlafaxine", "duloxetine", "ssri", "snri",
)
_SERO_CO = (
    "tramadol", "fentanyl", "meperidine", "pethidine", "linezolid", "maoi",
    "phenelzine", "tranylcypromine", "isocarboxazid", "methylene blue",
    "triptan", "sumatriptan", "mdma", "ecstasy", "st. john", "st john",
    "dextromethorphan",
)
_SERO_SIGNS = (
    "clonus", "hyperreflexia", "hyperreflex", "agitation", "fever",
    "diaphoresis", "rigidity", "hyperthermia", "myoclonus", "ocular clonus",
)
_DIG_INTERACT = (
    "clarithromycin", "azithromycin", "erythromycin", "macrolide",
    "amiodarone", "verapamil", "diltiazem", "quinidine",
)
_WARF_INTERACT = (
    "ciprofloxacin", "levofloxacin", "moxifloxacin", "fluoroquinolone",
    "metronidazole", "trimethoprim", "sulfamethoxazole", "co-trimoxazole",
    "bactrim", "clarithromycin", "erythromycin", "azithromycin", "macrolide",
    "fluconazole", "itraconazole", "ketoconazole", "amiodarone",
    "antibiotic", "antibiotics", "quinolone",
)
_BLEED = (
    "epistaxis", "nosebleed", "bruising", "ecchymosis", "hematoma", "bleeding",
    "haemorrhage", "hemorrhage", "melena", "melaena", "hematuria", "coagulopathy",
)


def match_clinical_pharmacology(text: str) -> Optional[str]:
    t = _low(text)

    if _has(t, ("serotonin syndrome", "serotonin toxicity", "serotonergic toxicity")):
        return "Serotonin syndrome / toxicity"
    if _has(t, (
        "lithium toxicity", "li toxicity", "toxic lithium", "elevated lithium",
        "supratherapeutic lithium", "high lithium level", "lithium level elevated",
    )):
        return "Lithium toxicity"
    if _has(t, (
        "digoxin toxicity", "digitalis toxicity", "dig toxicity",
        "elevated digoxin", "high digoxin", "supratherapeutic digoxin",
        "digoxin level", "serum digoxin",
    )):
        return "Digoxin toxicity"
    if _has(t, (
        "anticholinergic toxicity", "anticholinergic toxidrome", "anticholinergic syndrome",
    )):
        return "Anticholinergic toxicity"
    if _has(t, ("neuroleptic malignant", "malignant hyperthermia")):
        return "Drug-related hyperthermic toxidrome"
    if _has(t, (
        "overdose", "intentional overdose", "unintentional overdose",
        "drug overdose", "toxic ingestion",
    )) and not _classic_acs(t):
        return "Overdose / toxic ingestion"

    # Lithium + features
    if _re(t, r"\blithium\b") or _re(t, r"\bli\s+level\b"):
        if _has(t, (
            "tremor", "ataxia", "ataxic", "vomiting", "nausea", "diarrhea",
            "diarrhoea", "confusion", "aki", "acute kidney", "dehydration",
            "dehydrated", "nystagmus", "polyuria", "toxicity", "toxic",
            "level", "creatinine", "renal",
        )):
            return "Lithium with toxicity features"

    # Serotonin pattern
    if _has(t, _SSRI) and _has(t, _SERO_CO):
        if _has(t, _SERO_SIGNS):
            return "Serotonergic combination with toxicity signs (serotonin syndrome)"
        return "High-risk serotonergic drug combination"

    # Digoxin
    if _has(t, ("digoxin", "digitalis", "digitoxin")):
        if _has(t, _DIG_INTERACT) or _has(t, (
            "yellow halo", "yellow halos", "yellow vision", "xanthopsia",
            "visual halo", "visual change", "visual disturbance", "halos",
            "bradycardia", "hyperkalemia", "hyperkalaemia", "toxicity",
            "nausea", "vomiting", "confusion", "anorexia",
        )):
            return "Digoxin toxicity / precipitating interaction"

    # Warfarin
    on_warf = _has(t, ("warfarin", "coumadin", "vitamin k antagonist"))
    high_inr = (
        _has(t, (
            "high inr", "elevated inr", "supratherapeutic inr", "inr elevated",
            "inr high", "raised inr", "inr >", "inr>",
        ))
        or _re(t, r"\binr\b[^0-9]{0,15}([3-9](\.\d+)?|\d{2,}(\.\d+)?)")
        or _re(t, r"\binr\s*[:=]?\s*([3-9](\.\d+)?|\d{2,})")
    )
    bleed = _has(t, _BLEED)
    abx = _has(t, _WARF_INTERACT)
    if on_warf and (high_inr or bleed or abx):
        return "Warfarin interaction / high INR / bleeding risk"
    if high_inr and (on_warf or abx or bleed):
        return "Supratherapeutic anticoagulation / high INR"

    if _has(t, (
        "drug-drug interaction", "drug interaction", "drug–drug", "drug-drug",
        "major interaction", "interacting drug", "drug toxicity",
        "medication toxicity", "adverse drug reaction", "toxidrome",
        "supratherapeutic", "poisoning", "therapeutic drug monitoring",
    )) and not _classic_acs(t):
        return "Drug toxicity / major interaction"

    return None


# ---------------------------------------------------------------------------
# B) Dermatology / SCAR
# ---------------------------------------------------------------------------

def match_dermatology(text: str) -> Optional[str]:
    t = _low(text)

    if _has(t, (
        "sjs", "sjs/ten", "sjs-ten", "stevens-johnson", "stevens johnson",
        "toxic epidermal necrolysis", "toxic epidermal",
        "drug reaction with eosinophilia", "agep",
        "acute generalized exanthematous pustulosis", "severe cutaneous adverse",
    )) or _re(t, r"\bdress\b"):
        return "Named SCAR (SJS/TEN/DRESS/AGEP)"

    if _re(t, r"\bten\b") and _has(t, ("sjs", "epidermal", "necrolysis", "mucosal", "blister")):
        return "TEN / SCAR language"

    high_risk = _has(t, (
        "carbamazepine", "lamotrigine", "allopurinol", "phenytoin",
        "phenobarbital", "sulfonamide", "sulfa", "co-trimoxazole", "bactrim",
        "nevirapine", "abacavir",
    ))
    rash = _has(t, (
        "rash", "eruption", "blister", "bullae", "bullous", "desquamat",
        "painful skin", "painful rash", "spreading rash", "widespread erythema",
    ))
    mucosal = _has(t, (
        "mucosal", "mucosa", "oral eros", "oral ulcer", "oral lesion",
        "lip eros", "ocular", "conjunctiv", "genital ulcer", "mucous membrane",
    ))
    systemic = _has(t, ("fever", "febrile")) or mucosal

    if high_risk and rash and systemic:
        return "High-risk drug + rash + systemic/mucosal features (SCAR)"
    if rash and mucosal and (
        _has(t, ("fever", "drug", "days after", "weeks after", "started")) or high_risk
    ):
        return "Rash + mucosal involvement — SCAR pattern"
    return None


# ---------------------------------------------------------------------------
# C) Neurology
# ---------------------------------------------------------------------------

def match_neurology(text: str) -> Optional[str]:
    if match_clinical_pharmacology(text):
        return None
    t = _low(text)
    if _has(t, ("stroke", "cva", "tia", "cerebrovascular", "stroke code")):
        return "Stroke / TIA language"
    if _has(t, (
        "hemiparesis", "hemiplegia", "aphasia", "facial droop",
        "unilateral weakness", "nihss", "arm drift", "sudden weakness", "focal neuro",
    )) and _has(t, ("sudden", "acute", "minutes", "hours", "last known well")):
        return "Acute focal neurologic deficit"
    return None


# ---------------------------------------------------------------------------
# D) Cardiology / ACS — STRICT
# ---------------------------------------------------------------------------

def match_acs(text: str) -> Optional[str]:
    """ACS only. Never AF alone, bradycardia alone, ECG alone, troponin alone."""
    if match_clinical_pharmacology(text):
        return None
    if match_dermatology(text):
        return None
    if match_neurology(text):
        return None

    t = _low(text)
    if _has(t, (
        "stemi", "nstemi", "nste-acs", "acute coronary",
        "myocardial infarction", "unstable angina",
    )):
        return "Named ACS / MI syndrome"

    chest = _has(t, (
        "chest pain", "chest pressure", "crushing chest", "substernal",
        "angina", "retrosternal", "central chest pain",
    ))
    ischemic = _has(t, (
        "troponin", "st elevation", "st depression", "st-segment",
        "ischemic ecg", "t-wave inversion", "twave inversion",
    ))
    if chest and ischemic:
        return "Chest pain / pressure with ischemic markers"
    if chest and "acs" in t:
        return "Chest pain ACS workup language"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hard_route_specialty(text: str) -> Optional[Tuple[str, str]]:
    """Return (specialty, reason) or None. Cardiology ONLY if ACS hard-matches."""
    r = match_clinical_pharmacology(text)
    if r:
        return "clinical_pharmacology", r
    r = match_dermatology(text)
    if r:
        return "dermatology", r
    r = match_neurology(text)
    if r:
        return "neurology", r
    r = match_acs(text)
    if r:
        return "cardiology", r
    return None


def high_confidence_specialty(text: str) -> Optional[Tuple[str, str]]:
    return hard_route_specialty(text)


def is_acs_presentation(text: str) -> Tuple[bool, str]:
    r = match_acs(text)
    return (True, r) if r else (False, "")


def is_primary_acs_evaluation(case: Dict[str, Any]) -> bool:
    blob = build_case_blob(case)
    hard = hard_route_specialty(blob)
    if hard is not None and hard[0] != "cardiology":
        return False
    ok, _ = is_acs_presentation(blob)
    return ok


def forbids_cardiology(text: str) -> bool:
    hard = hard_route_specialty(text)
    if hard is not None:
        return hard[0] != "cardiology"
    # Soft forbid when tox/SCAR words present without ACS
    t = _low(text)
    if match_clinical_pharmacology(text) or match_dermatology(text):
        return True
    if _has(t, (
        "lithium", "serotonin", "tramadol", "sertraline", "digoxin", "warfarin",
        "ciprofloxacin", "toxicity", "toxidrome", "overdose", "clonus", "sjs",
        "inr", "epistaxis", "carbamazepine", "mucosal",
    )) and not _classic_acs(t):
        return True
    return False


def resolve_specialist(
    case: Dict[str, Any],
    proposed: Optional[str] = None,
) -> Tuple[str, str]:
    """Absolute final specialty. Never returns cardiology without ACS hard match."""
    blob = build_case_blob(case)
    hard = hard_route_specialty(blob)
    if hard is not None:
        return hard[0], f"HARD: {hard[1]}"

    prop = (proposed or "").strip().lower()
    if prop == "cardiology":
        return "general_internal_medicine", "BLOCK: cardiology without ACS → GIM"
    if prop in VALID_SPECIALTIES:
        return prop, "PROPOSED_OK"
    return "general_internal_medicine", "DEFAULT: GIM (never cardiology)"


# Compat alias used by older case_graph imports
def resolve_specialist_key(
    case: Dict[str, Any],
    current: Optional[str] = None,
) -> Tuple[str, str]:
    return resolve_specialist(case, current)
