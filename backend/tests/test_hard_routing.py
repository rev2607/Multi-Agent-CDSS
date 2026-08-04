"""Hard-constraint routing tests — production success criteria."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.case_patterns import (
    case_text_blob,
    hard_route_specialty,
    is_primary_acs_evaluation,
    resolve_specialist_key,
)
from app.agents.risk_scores import acs_risk_block
from app.agents.superior import SuperiorAgent
from app.models.schemas import SpecialistType

SUCCESS = [
    (
        "lithium",
        "64M vomiting, tremor, ataxia on lithium + AKI",
        SpecialistType.clinical_pharmacology,
    ),
    (
        "serotonin",
        "58F agitation, clonus, fever after tramadol + sertraline",
        SpecialistType.clinical_pharmacology,
    ),
    (
        "digoxin",
        "72M confusion, yellow halos, bradycardia on digoxin + clarithromycin",
        SpecialistType.clinical_pharmacology,
    ),
    (
        "warfarin_inr",
        "68F on warfarin starts ciprofloxacin; INR 6.2 with epistaxis and bruising",
        SpecialistType.clinical_pharmacology,
    ),
    (
        "scar",
        "45F painful spreading rash + fever + oral erosions 12 days after carbamazepine",
        SpecialistType.dermatology,
    ),
    (
        "acs",
        "62M chest pressure + elevated troponin + ST depression",
        SpecialistType.cardiology,
    ),
    (
        "stroke",
        "70M sudden right hemiparesis and aphasia",
        SpecialistType.neurology,
    ),
]


def _case(text: str, attachments=None) -> dict:
    return {
        "title": "",
        "patient_context": "",
        "clinical_text": text,
        "notes": "",
        "attachments": attachments or [],
    }


def test_hard_route_table():
    for name, text, expected in SUCCESS:
        hit = hard_route_specialty(text)
        assert hit is not None, f"{name}: hard_route returned None for: {text}"
        assert hit[0] == expected.value, (
            f"{name}: hard_route={hit[0]} expected={expected.value} text={text}"
        )


def test_superior_route_table():
    agent = SuperiorAgent()
    for name, text, expected in SUCCESS:
        specialist, rationale = agent.route(_case(text))
        assert specialist == expected, (
            f"{name}: route={specialist.value} expected={expected.value}"
        )
        label = {
            SpecialistType.clinical_pharmacology: "Clinical Pharmacology",
            SpecialistType.dermatology: "Dermatology",
            SpecialistType.cardiology: "Cardiology",
            SpecialistType.neurology: "Neurology",
            SpecialistType.general_internal_medicine: "General Internal Medicine",
        }[expected]
        assert rationale.startswith(f"Routed to: {label}"), rationale
        if expected != SpecialistType.cardiology:
            assert "heart" not in rationale.lower()
            assert "timi" not in rationale.lower()


def test_attachment_only_lithium_routes_to_pharm():
    """Critical: drugs only in attachment extract must still hard-route."""
    case = _case(
        "",
        attachments=[
            {
                "filename": "note.pdf",
                "extracted_text": (
                    "64M with vomiting tremor ataxia on lithium, AKI, dehydrated. "
                    "Concern for lithium toxicity."
                ),
            }
        ],
    )
    agent = SuperiorAgent()
    specialist, rationale = agent.route(case)
    assert specialist == SpecialistType.clinical_pharmacology
    assert "Routed to: Clinical Pharmacology" in rationale
    # Blob used by specialize_guard must match
    blob = case_text_blob(case)
    hard = hard_route_specialty(blob)
    assert hard is not None
    assert hard[0] == "clinical_pharmacology"


def test_resolve_blocks_false_cardiology():
    case = _case("64M vomiting tremor ataxia on lithium + AKI")
    key, reason = resolve_specialist_key(case, "cardiology")
    assert key == "clinical_pharmacology"
    assert "HARD" in reason or "clinical" in reason.lower() or "lithium" in reason.lower()


def test_no_acs_scores_on_toxicity():
    for name, text, expected in SUCCESS:
        if expected == SpecialistType.cardiology:
            continue
        case = _case(text)
        assert is_primary_acs_evaluation(case) is False, name
        block = acs_risk_block(case)
        assert block["acs_scores_applicable"] is False, name
        assert block["heart"] is None, name
        assert block["timi"] is None, name


def test_acs_keeps_scores():
    case = _case("62M chest pressure + elevated troponin + ST depression")
    assert is_primary_acs_evaluation(case) is True
    block = acs_risk_block(case)
    assert block["acs_scores_applicable"] is True
    assert block["heart"] is not None


def test_af_alone_not_cardiology():
    text = "78M with atrial fibrillation rate 110, no chest pain, on digoxin and warfarin"
    # digoxin/warfarin may pull pharm; must NOT be pure ACS cardiology
    hit = hard_route_specialty(text)
    if hit:
        assert hit[0] != "cardiology"


def test_cardiology_agent_omits_scores_on_warfarin():
    from app.agents.specialists import CardiologyAgent

    cardio = CardiologyAgent.__new__(CardiologyAgent)
    cardio.specialist = SpecialistType.cardiology
    cardio.display_name = "Cardiology Agent"
    case = _case("warfarin + ciprofloxacin INR 5.8 epistaxis")
    extras = CardiologyAgent._precompute_report_fields(cardio, case)
    assert extras.get("acs_scores_applicable") is False
    assert extras.get("risk_scores") == []


if __name__ == "__main__":
    tests = [
        test_hard_route_table,
        test_superior_route_table,
        test_attachment_only_lithium_routes_to_pharm,
        test_resolve_blocks_false_cardiology,
        test_no_acs_scores_on_toxicity,
        test_acs_keeps_scores,
        test_af_alone_not_cardiology,
        test_cardiology_agent_omits_scores_on_warfarin,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    if failed:
        raise SystemExit(f"{failed} failed")
    print(f"\nAll {len(tests)} tests passed.")
