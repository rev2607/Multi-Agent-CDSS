"""Specialist registry — map enum → agent instance factory."""

from __future__ import annotations

from typing import Dict, Type

from app.agents.base import BaseSpecialistAgent
from app.agents.specialists import (
    CardiologyAgent,
    ClinicalPharmacologyAgent,
    DermatologyAgent,
    GeneralInternalMedicineAgent,
    NeurologyAgent,
)
from app.models.schemas import SpecialistType

SPECIALIST_MAP: Dict[SpecialistType, Type[BaseSpecialistAgent]] = {
    SpecialistType.cardiology: CardiologyAgent,
    SpecialistType.dermatology: DermatologyAgent,
    SpecialistType.neurology: NeurologyAgent,
    SpecialistType.general_internal_medicine: GeneralInternalMedicineAgent,
    SpecialistType.clinical_pharmacology: ClinicalPharmacologyAgent,
}


def get_specialist(specialist: SpecialistType | str) -> BaseSpecialistAgent:
    if isinstance(specialist, str):
        specialist = SpecialistType(specialist)
    cls = SPECIALIST_MAP[specialist]
    return cls()
