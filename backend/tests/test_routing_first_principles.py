"""First-principles routing tests — all production failure cases must pass."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.case_patterns import (
    build_case_blob,
    hard_route_specialty,
    is_primary_acs_evaluation,
    resolve_specialist,
)
from app.agents.registry import get_specialist
from app.agents.risk_scores import acs_risk_block
from app.agents.superior import SuperiorAgent
from app.models.schemas import SpecialistType

CASES = [
    (
        "lithium",
        "64M vomiting, tremor, ataxia on lithium + AKI",
        "clinical_pharmacology",
        "ClinicalPharmacologyAgent",
    ),
    (
        "serotonin",
        "58F agitation, clonus, fever after tramadol + sertraline",
        "clinical_pharmacology",
        "ClinicalPharmacologyAgent",
    ),
    (
        "digoxin",
        "72M confusion, yellow halos, bradycardia on digoxin + clarithromycin",
        "clinical_pharmacology",
        "ClinicalPharmacologyAgent",
    ),
    (
        "warfarin",
        "68F on warfarin starts ciprofloxacin; INR 5.8 with epistaxis and bruising",
        "clinical_pharmacology",
        "ClinicalPharmacologyAgent",
    ),
    (
        "scar",
        "45F painful spreading rash + fever + oral erosions 12 days after carbamazepine",
        "dermatology",
        "DermatologyAgent",
    ),
    (
        "acs",
        "62M chest pressure + elevated troponin + ST depression",
        "cardiology",
        "CardiologyAgent",
    ),
    (
        "stroke",
        "70M sudden right hemiparesis and aphasia",
        "neurology",
        "NeurologyAgent",
    ),
]


def _case(text: str, attachments=None) -> dict:
    return {
        "id": "test",
        "title": "",
        "patient_context": "",
        "clinical_text": text,
        "notes": "",
        "attachments": attachments or [],
    }


def test_hard_route_all_production_failures():
    for name, text, expected, _cls in CASES:
        hit = hard_route_specialty(text)
        assert hit is not None, f"{name}: hard_route None for {text!r}"
        assert hit[0] == expected, f"{name}: got {hit[0]} want {expected}"


def test_superior_route_all():
    agent = SuperiorAgent()
    for name, text, expected, _cls in CASES:
        spec, rationale = agent.route(_case(text))
        assert spec.value == expected, f"{name}: {spec.value} != {expected}"
        assert rationale.startswith("Routed to:"), rationale
        if expected != "cardiology":
            assert "heart" not in rationale.lower()
            assert "timi" not in rationale.lower()


def test_registry_maps_correct_agents():
    from app.agents.registry import _STRING_MAP, SPECIALIST_MAP

    for _name, _text, key, cls_name in CASES:
        cls = _STRING_MAP[key]
        assert cls.__name__ == cls_name
        # Instantiate without __init__ to avoid Qdrant lock in tests
        agent = cls.__new__(cls)
        assert agent.specialist.value == key
        assert SPECIALIST_MAP[SpecialistType(key)] is cls


def test_attachment_only_warfarin():
    case = _case(
        "",
        attachments=[
            {
                "filename": "labs.pdf",
                "extracted_text": (
                    "Patient on chronic warfarin. Started ciprofloxacin 3 days ago. "
                    "INR 5.8. Epistaxis and bruising."
                ),
            }
        ],
    )
    blob = build_case_blob(case)
    hard = hard_route_specialty(blob)
    assert hard is not None
    assert hard[0] == "clinical_pharmacology"
    agent = SuperiorAgent()
    spec, _ = agent.route(case)
    assert spec == SpecialistType.clinical_pharmacology


def test_attachment_only_lithium():
    case = _case(
        "",
        attachments=[
            {
                "filename": "note.txt",
                "extracted_text": "64M vomiting tremor ataxia on lithium + AKI dehydration",
            }
        ],
    )
    assert hard_route_specialty(build_case_blob(case))[0] == "clinical_pharmacology"
    assert SuperiorAgent().route(case)[0] == SpecialistType.clinical_pharmacology


def test_resolve_blocks_false_cardiology():
    for name, text, expected, _ in CASES:
        if expected == "cardiology":
            continue
        key, reason = resolve_specialist(_case(text), "cardiology")
        assert key == expected, f"{name}: resolve gave {key} want {expected} ({reason})"


def test_no_heart_timi_non_acs():
    for name, text, expected, _ in CASES:
        if expected == "cardiology":
            continue
        case = _case(text)
        assert is_primary_acs_evaluation(case) is False, name
        block = acs_risk_block(case)
        assert block["acs_scores_applicable"] is False, name
        assert block["heart"] is None
        assert block["timi"] is None


def test_acs_allows_scores():
    case = _case("62M chest pressure + elevated troponin + ST depression")
    assert is_primary_acs_evaluation(case) is True
    block = acs_risk_block(case)
    assert block["acs_scores_applicable"] is True
    assert block["heart"] is not None


def test_cardiology_precompute_empty_on_warfarin():
    from app.agents.specialists import CardiologyAgent

    cardio = CardiologyAgent.__new__(CardiologyAgent)
    cardio.specialist = SpecialistType.cardiology
    cardio.display_name = "Cardiology Agent"
    extras = CardiologyAgent._precompute_report_fields(
        cardio, _case("warfarin ciprofloxacin INR 5.8 epistaxis")
    )
    assert extras["acs_scores_applicable"] is False
    assert extras["risk_scores"] == []


def test_never_default_cardiology_empty_text():
    hit = hard_route_specialty("")
    assert hit is None
    key, _ = resolve_specialist(_case(""), None)
    assert key == "general_internal_medicine"
    key2, _ = resolve_specialist(_case(""), "cardiology")
    assert key2 != "cardiology"


if __name__ == "__main__":
    tests = [
        test_hard_route_all_production_failures,
        test_superior_route_all,
        test_registry_maps_correct_agents,
        test_attachment_only_warfarin,
        test_attachment_only_lithium,
        test_resolve_blocks_false_cardiology,
        test_no_heart_timi_non_acs,
        test_acs_allows_scores,
        test_cardiology_precompute_empty_on_warfarin,
        test_never_default_cardiology_empty_text,
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
