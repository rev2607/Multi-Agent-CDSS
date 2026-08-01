"""Approximate ACS risk scores (HEART, TIMI) from structured/clinical text.

These are decision-support heuristics for the local demo — not validated calculators.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def _blob(case: Dict[str, Any]) -> str:
    parts = [
        case.get("title") or "",
        case.get("patient_context") or "",
        case.get("clinical_text") or "",
        case.get("notes") or "",
    ]
    for att in case.get("attachments") or []:
        parts.append(att.get("filename") or "")
        parts.append(att.get("extracted_text") or "")
    return "\n".join(parts)


def _age(text: str) -> Optional[int]:
    # 62M, 62-year-old, age 62, 62 yo
    patterns = [
        r"\b(\d{1,3})\s*(?:year(?:s)?\s*-?\s*old|y/?o|yo)\b",
        r"\bage[:\s]+(\d{1,3})\b",
        r"\b(\d{2,3})\s*[mMfF]\b",
        r"\b(\d{2,3})-?(?:year|yr)",
    ]
    t = text.lower()
    for p in patterns:
        m = re.search(p, t, re.I)
        if m:
            age = int(m.group(1))
            if 18 <= age <= 110:
                return age
    return None


def _sex(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\b(\d{2,3})\s*m\b", t) or re.search(r"\bmale\b", t):
        return "male"
    if re.search(r"\b(\d{2,3})\s*f\b", t) or re.search(r"\bfemale\b", t):
        return "female"
    return None


def _has_any(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def extract_image_findings(case: Dict[str, Any]) -> List[Dict[str, str]]:
    """Pull attachment-derived findings, especially ECG images."""
    out: List[Dict[str, str]] = []
    for att in case.get("attachments") or []:
        modality = (att.get("modality") or "").lower()
        filename = att.get("filename") or "attachment"
        text = (att.get("extracted_text") or "").strip()
        if not text:
            continue
        is_image = modality in {"image", "png", "jpg", "jpeg"} or filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".gif")
        )
        is_ecg = any(
            k in (filename + " " + text[:500]).lower()
            for k in ("ecg", "ekg", "electrocardi")
        )
        if is_image or is_ecg:
            label = "ECG image" if is_ecg else "Uploaded image"
            out.append(
                {
                    "source": filename,
                    "modality": modality or "image",
                    "label": label,
                    "summary": text[:800],
                }
            )
        elif modality == "pdf" and any(
            k in text.lower() for k in ("ecg", "ekg", "st depression", "troponin")
        ):
            out.append(
                {
                    "source": filename,
                    "modality": "pdf",
                    "label": "Uploaded PDF clinical document",
                    "summary": text[:800],
                }
            )
    return out


def compute_heart_score(case: Dict[str, Any]) -> Dict[str, Any]:
    """Approximate HEART score (0–10)."""
    text = _blob(case)
    t = text.lower()
    components: Dict[str, Any] = {}
    missing: List[str] = []

    # History (0–2)
    highly = _has_any(
        t,
        [
            "crushing",
            "pressure",
            "radiating to left arm",
            "diaphoresis",
            "exertional",
            "relieved by rest",
            "central chest",
        ],
    )
    moderate = _has_any(t, ["chest pain", "chest pressure", "dyspnea", "sob"])
    if highly:
        h_hist, h_note = 2, "History highly suspicious for ACS (pressure/radiation/diaphoresis)"
    elif moderate:
        h_hist, h_note = 1, "History moderately suspicious"
    else:
        h_hist, h_note = 0, "History slightly suspicious or limited detail"
        missing.append("detailed ischemic history features")
    components["history"] = {"points": h_hist, "note": h_note}

    # ECG (0–2)
    if _has_any(t, ["st elevation", "stemi", "new lbbb"]):
        h_ecg, e_note = 2, "Significant ST deviation (elevation) described"
    elif _has_any(t, ["st depression", "t-wave inversion", "twave inversion", "ischemic ecg"]):
        h_ecg, e_note = 2, "Significant ST depression / ischemic ECG changes"
    elif _has_any(t, ["nonspecific", "non-specific st", "sinus tachycardia"]) and "st" in t:
        h_ecg, e_note = 1, "Nonspecific repolarization changes / borderline ECG"
    elif _has_any(t, ["ecg normal", "normal ecg", "ecg: normal"]):
        h_ecg, e_note = 0, "Normal ECG described"
    elif _has_any(t, ["ecg", "ekg"]):
        h_ecg, e_note = 1, "ECG present; ischemic severity incompletely specified"
        missing.append("full ECG classification")
    else:
        h_ecg, e_note = 0, "No ECG data found"
        missing.append("ECG")
    components["ecg"] = {"points": h_ecg, "note": e_note}

    # Age (0–2)
    age = _age(text)
    if age is None:
        components["age"] = {"points": 0, "note": "Age not found", "value": None}
        missing.append("age")
        age_pts = 0
    elif age >= 65:
        age_pts = 2
        components["age"] = {"points": 2, "note": f"Age {age} (≥65)", "value": age}
    elif age >= 45:
        age_pts = 1
        components["age"] = {"points": 1, "note": f"Age {age} (45–64)", "value": age}
    else:
        age_pts = 0
        components["age"] = {"points": 0, "note": f"Age {age} (<45)", "value": age}

    # Risk factors (0–2): HTN, DM, smoking, obesity, hyperlipidemia, family hx, atherosclerosis
    rf_terms = [
        "hypertension",
        "htn",
        "diabetes",
        "dm2",
        "dm ",
        "smok",
        "hyperlipid",
        "dyslipid",
        "obesity",
        "cad",
        "prior mi",
        "atherosclero",
        "family history",
    ]
    rf_count = sum(1 for term in rf_terms if term in t)
    if "known cad" in t or "prior stent" in t or "cabg" in t:
        rf_pts, rf_note = 2, "Known atherosclerotic disease / multiple risk factors"
    elif rf_count >= 3:
        rf_pts, rf_note = 2, f"≥3 risk-factor markers detected (~{rf_count})"
    elif rf_count >= 1:
        rf_pts, rf_note = 1, f"1–2 risk-factor markers detected (~{rf_count})"
    else:
        rf_pts, rf_note = 0, "No clear risk factors extracted"
        missing.append("CAD risk factors")
    components["risk_factors"] = {"points": rf_pts, "note": rf_note}

    # Troponin (0–2)
    trop_m = re.search(
        r"troponin[^0-9]{0,20}([0-9]+\.?[0-9]*)",
        t,
    )
    if _has_any(t, ["troponin elevated", "positive troponin", "trop positive"]):
        trop_pts, trop_note = 2, "Troponin reported elevated/positive"
    elif trop_m:
        val = float(trop_m.group(1))
        # Heuristic without local assay ULN: any positive numeric with "elevated" nearby
        window = t[max(0, trop_m.start() - 40) : trop_m.end() + 40]
        if "elevat" in window or val >= 0.04:
            trop_pts, trop_note = 2, f"Troponin value {val} with elevation context"
        elif val > 0:
            trop_pts, trop_note = 1, f"Troponin value {val} (borderline without assay ULN)"
        else:
            trop_pts, trop_note = 0, f"Troponin value {val}"
    elif "troponin" in t and "normal" in t:
        trop_pts, trop_note = 0, "Troponin described as normal"
    elif "troponin" in t:
        trop_pts, trop_note = 1, "Troponin mentioned; elevation magnitude unclear"
        missing.append("troponin quantitative interpretation")
    else:
        trop_pts, trop_note = 0, "No troponin data"
        missing.append("troponin")
    components["troponin"] = {"points": trop_pts, "note": trop_note}

    total = h_hist + h_ecg + age_pts + rf_pts + trop_pts
    if total >= 7:
        risk_band = "high"
    elif total >= 4:
        risk_band = "moderate"
    else:
        risk_band = "low"

    return {
        "score": total,
        "max": 10,
        "risk_band": risk_band,
        "components": components,
        "missing_data": missing,
        "interpretable": len(missing) < 3,
        "note": (
            "Approximate HEART score for decision support only; "
            "confirm each domain with the clinical team and local assay cutoffs."
        ),
    }


def compute_timi_ua_nstemi(case: Dict[str, Any]) -> Dict[str, Any]:
    """Approximate TIMI risk score for UA/NSTEMI (0–7)."""
    text = _blob(case)
    t = text.lower()
    factors: List[Dict[str, Any]] = []
    score = 0

    def add(name: str, present: bool, detail: str) -> None:
        nonlocal score
        if present:
            score += 1
        factors.append({"factor": name, "present": present, "detail": detail})

    age = _age(text)
    add("Age ≥65", bool(age and age >= 65), f"age={age}" if age else "age unknown")

    rf_terms = ["hypertension", "htn", "diabetes", "dm", "smok", "hyperlipid", "family history"]
    rf_n = sum(1 for x in rf_terms if x in t)
    add("≥3 CAD risk factors", rf_n >= 3, f"markers≈{rf_n}")

    add(
        "Known CAD (stenosis ≥50% / prior MI / revasc)",
        _has_any(t, ["known cad", "prior mi", "stent", "cabg", "coronary disease"]),
        "from history text",
    )
    add(
        "ASA use in past 7 days",
        _has_any(t, ["aspirin", "asa ", "on asa", "taking asa"]),
        "only if explicitly documented",
    )
    add(
        "Severe angina (≥2 episodes / 24h)",
        _has_any(t, ["recurrent chest pain", "multiple episodes", "ongoing pain", "2 hours"]),
        "heuristic from symptom duration/recurrence wording",
    )
    add(
        "ST deviation ≥0.5 mm",
        _has_any(t, ["st depression", "st elevation", "st-segment", "1mm st", "1 mm st"]),
        "from ECG text/image extraction",
    )
    add(
        "Positive troponin",
        _has_any(t, ["troponin", "trop"])
        and (
            _has_any(t, ["elevat", "positive", "raised"])
            or bool(re.search(r"troponin[^0-9]{0,20}0\.\d*[1-9]", t))
        ),
        "biomarker",
    )

    if score >= 5:
        band = "high"
    elif score >= 3:
        band = "intermediate"
    else:
        band = "low"

    return {
        "score": score,
        "max": 7,
        "risk_band": band,
        "factors": factors,
        "interpretable": True,
        "note": (
            "Approximate TIMI UA/NSTEMI score for decision support only; "
            "missing documentation defaults to factor absent."
        ),
    }


def acs_risk_block(case: Dict[str, Any]) -> Dict[str, Any]:
    heart = compute_heart_score(case)
    timi = compute_timi_ua_nstemi(case)
    images = extract_image_findings(case)
    text = _blob(case).lower()
    high_risk_acs = (
        ("troponin" in text and any(x in text for x in ("elevat", "positive", "0.0")))
        and any(x in text for x in ("st depression", "st elevation", "ischemi"))
    )
    return {
        "heart": heart,
        "timi": timi,
        "image_findings": images,
        "high_risk_acs_pattern": high_risk_acs,
    }
