"""Unit tests for Superior Agent routing, ACS score gating, specialty retrieval bias."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python tests/test_superior_routing.py` from backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.risk_scores import is_primary_acs_evaluation, acs_risk_block
from app.agents.superior import SuperiorAgent, format_routed_to
from app.models.schemas import RetrievalHit, SpecialistType
from app.retrieval.postprocess import filter_and_rank_hits


def _case(text: str, title: str = "") -> dict:
    return {
        "title": title,
        "patient_context": "",
        "clinical_text": text,
        "notes": "",
        "attachments": [],
    }


def test_carbamazepine_scar_routes_to_dermatology():
    """SCAR case must never go to Cardiology."""
    case = _case(
        "45F with painful spreading rash and fever plus oral and ocular mucosal "
        "involvement 12 days after starting carbamazepine. Suspected SJS/TEN or DRESS.",
        title="Carbamazepine rash — possible SJS/TEN",
    )
    agent = SuperiorAgent()
    blob = agent._case_blob(case)
    override = agent._high_confidence_route(blob)
    assert override is not None, "SCAR case should match high-confidence override"
    specialist, reason = override
    assert specialist == SpecialistType.dermatology

    specialist2, rationale = agent.route(case)
    assert specialist2 == SpecialistType.dermatology
    assert rationale.lower().startswith("routed to: dermatology")
    assert "heart" not in rationale.lower()
    assert "timi" not in rationale.lower()


def test_digoxin_clarithromycin_routes_to_clinical_pharmacology():
    """Dominant digoxin toxicity / macrolide interaction → Clinical Pharmacology."""
    case = _case(
        "78M on chronic digoxin starts clarithromycin for pneumonia; develops nausea, "
        "vomiting, xanthopsia, bradycardia; digoxin level elevated. Concern for "
        "digoxin toxicity from drug interaction.",
        title="Digoxin + clarithromycin toxicity",
    )
    agent = SuperiorAgent()
    specialist, rationale = agent.route(case)
    assert specialist == SpecialistType.clinical_pharmacology
    assert "Routed to: Clinical Pharmacology" in rationale
    assert specialist != SpecialistType.cardiology


def test_serotonin_syndrome_routes_to_clinical_pharmacology():
    """Sertraline + tramadol toxidrome must never be a cardiology assessment."""
    case = _case(
        "Patient on sertraline started tramadol for pain; now with agitation, "
        "inducible clonus, diaphoresis, and fever. Concern for serotonin syndrome.",
        title="Sertraline + tramadol — serotonin syndrome",
    )
    agent = SuperiorAgent()
    specialist, rationale = agent.route(case)
    assert specialist == SpecialistType.clinical_pharmacology
    assert "Routed to: Clinical Pharmacology" in rationale
    assert "cardiology" not in rationale.lower() or "not" in rationale.lower()


def test_serotonin_syndrome_no_heart_timi():
    case = _case(
        "Sertraline + tramadol; agitation, clonus, fever, tachycardia. "
        "Serotonin syndrome suspected. No chest pain."
    )
    assert is_primary_acs_evaluation(case) is False
    block = acs_risk_block(case)
    assert block.get("acs_scores_applicable") is False
    assert block.get("heart") is None


def test_digoxin_toxicity_not_acs_scores():
    case = _case(
        "Elderly patient on digoxin with new clarithromycin; PVCs and bradycardia; "
        "suspected digitalis toxicity; digoxin level high."
    )
    assert is_primary_acs_evaluation(case) is False
    block = acs_risk_block(case)
    assert block.get("acs_scores_applicable") is False
    assert block.get("heart") is None
    assert block.get("timi") is None


def test_scar_not_acs_scores():
    case = _case(
        "45F painful spreading rash fever mucosal involvement after carbamazepine; SJS/TEN."
    )
    assert is_primary_acs_evaluation(case) is False
    block = acs_risk_block(case)
    assert block.get("acs_scores_applicable") is False


def test_true_acs_keeps_scores():
    case = _case(
        "62M crushing substernal chest pain radiating to left arm, diaphoresis, "
        "ST depression on ECG, troponin 0.42. Concern for NSTEMI."
    )
    assert is_primary_acs_evaluation(case) is True
    block = acs_risk_block(case)
    assert block.get("acs_scores_applicable") is True
    assert block.get("heart") is not None
    assert block.get("timi") is not None


def test_acs_routes_to_cardiology():
    case = _case(
        "62M crushing substernal chest pain radiating to left arm, diaphoresis, "
        "ST depression on ECG, troponin 0.42. Concern for NSTEMI."
    )
    agent = SuperiorAgent()
    specialist, rationale = agent.route(case)
    assert specialist == SpecialistType.cardiology
    assert "Routed to: Cardiology" in rationale


def test_stroke_routes_to_neurology():
    case = _case(
        "70M sudden right hemiparesis and aphasia for 45 minutes; NIHSS elevated. "
        "Stroke code activated."
    )
    agent = SuperiorAgent()
    specialist, rationale = agent.route(case)
    assert specialist == SpecialistType.neurology
    assert "Routed to: Neurology" in rationale


def test_fever_tachycardia_alone_not_cardiology_when_scar_present():
    case = _case(
        "Patient with fever and tachycardia after new anticonvulsant develops "
        "blistering rash involving lips and eyes. Concern for Stevens-Johnson syndrome."
    )
    agent = SuperiorAgent()
    specialist, rationale = agent.route(case)
    assert specialist == SpecialistType.dermatology
    assert "Routed to: Dermatology" in rationale


def test_keyword_fallback_scar():
    agent = SuperiorAgent()
    text = (
        "painful spreading rash mucosal involvement carbamazepine fever "
        "suspected dress syndrome"
    )
    assert agent._keyword_route(text) == SpecialistType.dermatology


def test_keyword_fallback_digoxin():
    agent = SuperiorAgent()
    text = "digoxin clarithromycin toxicity nausea bradycardia elevated digoxin level"
    assert agent._keyword_route(text) == SpecialistType.clinical_pharmacology


def test_format_routed_to():
    s = format_routed_to(SpecialistType.dermatology, "SCAR presentation")
    assert s == "Routed to: Dermatology — SCAR presentation"
    s2 = format_routed_to(
        SpecialistType.clinical_pharmacology, "digoxin toxicity"
    )
    assert s2 == "Routed to: Clinical Pharmacology — digoxin toxicity"


def _agent_shell(cls):
    """Construct specialist without __init__ (avoids Qdrant file lock in tests)."""
    obj = cls.__new__(cls)
    obj.specialist = cls.specialist
    obj.display_name = cls.display_name
    obj.system_focus = cls.system_focus
    return obj


def test_dermatology_strips_acs_scores():
    from app.agents.specialists import DermatologyAgent

    derm = _agent_shell(DermatologyAgent)
    extras = {
        "risk_scores": [
            {
                "name": "HEART",
                "score": 6,
                "max_score": 10,
                "risk_band": "moderate",
                "detail": "should be stripped",
            },
            {
                "name": "SCORTEN",
                "score": 2,
                "max_score": 7,
                "risk_band": "elevated",
                "detail": "TEN severity",
            },
        ]
    }
    items = derm._parse_risk_scores({}, extras, allow_acs=False)
    names = {i.name.upper() for i in items}
    assert "HEART" not in names
    assert "SCORTEN" in names


def test_pharmacology_strips_acs_scores():
    from app.agents.specialists import ClinicalPharmacologyAgent

    pharm = _agent_shell(ClinicalPharmacologyAgent)
    extras = {
        "risk_scores": [
            {
                "name": "TIMI (UA/NSTEMI)",
                "score": 4,
                "max_score": 7,
                "risk_band": "intermediate",
                "detail": "should be stripped",
            }
        ]
    }
    items = pharm._parse_risk_scores({}, extras, allow_acs=False)
    assert items == []


def test_non_cardio_framing_sanitizer():
    from app.agents.specialists import ClinicalPharmacologyAgent

    pharm = _agent_shell(ClinicalPharmacologyAgent)
    text = (
        "This cardiology assessment calculates a HEART score of 5. "
        "TIMI score is intermediate. Primary issue is serotonin syndrome."
    )
    cleaned = pharm._sanitize_non_cardio_framing(text)
    assert "cardiology assessment" not in cleaned.lower()
    assert "heart score" not in cleaned.lower()
    assert "timi" not in cleaned.lower()
    assert "serotonin" in cleaned.lower()


def test_cardiology_omits_heart_timi_for_toxicity_if_misrouted():
    """Even Cardiology agent must not auto-emit HEART/TIMI for digoxin toxicity."""
    from app.agents.specialists import CardiologyAgent

    case = _case(
        "78M digoxin + clarithromycin; digoxin toxicity with bradycardia; no chest pain."
    )
    cardio = _agent_shell(CardiologyAgent)
    extras = CardiologyAgent._precompute_report_fields(cardio, case)
    assert extras.get("acs_scores_applicable") is False
    assert extras.get("risk_scores") == []


def test_retrieval_prefers_pharmacology_over_cardiology_snippets():
    hits = [
        RetrievalHit(
            id="cardio-1",
            text="STEMI requires immediate reperfusion. ACS chest pain troponin ECG ischemia.",
            score=0.9,
            dense_score=0.9,
            sparse_score=0.5,
            metadata={
                "title": "clinical_snippets — Cardiology",
                "specialty": "cardiology",
                "source_type": "sample",
                "tags": ["sample", "cardiology"],
                "case_id": "shared",
            },
        ),
        RetrievalHit(
            id="pharm-1",
            text=(
                "Digoxin toxicity with clarithromycin: P-glycoprotein interaction raises "
                "digoxin levels; nausea xanthopsia bradycardia; hold digoxin; TDM."
            ),
            score=0.55,
            dense_score=0.55,
            sparse_score=0.4,
            metadata={
                "title": "clinical_snippets — Clinical Pharmacology & Drug Safety",
                "specialty": "clinical_pharmacology",
                "source_type": "sample",
                "tags": ["sample", "clinical_pharmacology"],
                "case_id": "shared",
            },
        ),
        RetrievalHit(
            id="generic-1",
            text="Sample clinical knowledge base mixed notes fever infection.",
            score=0.7,
            dense_score=0.7,
            sparse_score=0.3,
            metadata={
                "title": "clinical_snippets",
                "source_type": "sample",
                "tags": ["sample"],
                "case_id": "shared",
            },
        ),
    ]
    ranked = filter_and_rank_hits(
        hits,
        query="digoxin clarithromycin toxicity drug interaction elevated level",
        specialty="clinical_pharmacology",
        case_id="case-abc",
        min_relevance=0.05,
        max_hits=5,
        hard_specialty_filter=True,
    )
    assert ranked, "expected at least one hit"
    top_ids = [h.id for h in ranked]
    assert top_ids[0] == "pharm-1", f"expected pharmacology first, got {top_ids}"
    # Pure cardiology should be dropped or ranked last
    if "cardio-1" in top_ids:
        assert top_ids.index("pharm-1") < top_ids.index("cardio-1")


def test_retrieval_prefers_dermatology_over_cardiology_snippets():
    hits = [
        RetrievalHit(
            id="cardio-1",
            text="ACS NSTEMI troponin ST depression heart score TIMI antiplatelet.",
            score=0.85,
            dense_score=0.85,
            sparse_score=0.4,
            metadata={
                "title": "clinical_snippets — Cardiology",
                "specialty": "cardiology",
                "source_type": "sample",
                "case_id": "shared",
            },
        ),
        RetrievalHit(
            id="derm-1",
            text=(
                "SJS/TEN after carbamazepine: painful rash mucosal involvement fever; "
                "stop culprit drug; SCORTEN; burn unit dermatology emergency."
            ),
            score=0.5,
            dense_score=0.5,
            sparse_score=0.4,
            metadata={
                "title": "clinical_snippets — Dermatology",
                "specialty": "dermatology",
                "source_type": "sample",
                "case_id": "shared",
            },
        ),
    ]
    ranked = filter_and_rank_hits(
        hits,
        query="carbamazepine painful rash mucosal SJS TEN DRESS",
        specialty="dermatology",
        case_id="case-derm",
        min_relevance=0.05,
        max_hits=5,
    )
    assert ranked
    assert ranked[0].id == "derm-1"


if __name__ == "__main__":
    tests = [
        test_carbamazepine_scar_routes_to_dermatology,
        test_digoxin_clarithromycin_routes_to_clinical_pharmacology,
        test_serotonin_syndrome_routes_to_clinical_pharmacology,
        test_serotonin_syndrome_no_heart_timi,
        test_digoxin_toxicity_not_acs_scores,
        test_scar_not_acs_scores,
        test_true_acs_keeps_scores,
        test_acs_routes_to_cardiology,
        test_stroke_routes_to_neurology,
        test_fever_tachycardia_alone_not_cardiology_when_scar_present,
        test_keyword_fallback_scar,
        test_keyword_fallback_digoxin,
        test_format_routed_to,
        test_dermatology_strips_acs_scores,
        test_pharmacology_strips_acs_scores,
        test_non_cardio_framing_sanitizer,
        test_cardiology_omits_heart_timi_for_toxicity_if_misrouted,
        test_retrieval_prefers_pharmacology_over_cardiology_snippets,
        test_retrieval_prefers_dermatology_over_cardiology_snippets,
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
        raise SystemExit(f"{failed} test(s) failed")
    print(f"\nAll {len(tests)} tests passed.")
