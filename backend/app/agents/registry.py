"""Specialist registry — map specialty key → agent class. Explicit and fail-loud."""

from __future__ import annotations

import logging
from typing import Dict, Type, Union

from app.agents.base import BaseSpecialistAgent
from app.agents.specialists import (
    CardiologyAgent,
    ClinicalPharmacologyAgent,
    DermatologyAgent,
    GeneralInternalMedicineAgent,
    NeurologyAgent,
)
from app.models.schemas import SpecialistType

logger = logging.getLogger(__name__)

SPECIALIST_MAP: Dict[SpecialistType, Type[BaseSpecialistAgent]] = {
    SpecialistType.clinical_pharmacology: ClinicalPharmacologyAgent,
    SpecialistType.dermatology: DermatologyAgent,
    SpecialistType.neurology: NeurologyAgent,
    SpecialistType.cardiology: CardiologyAgent,
    SpecialistType.general_internal_medicine: GeneralInternalMedicineAgent,
}

# Explicit string map so typos never silently fall through
_STRING_MAP: Dict[str, Type[BaseSpecialistAgent]] = {
    "clinical_pharmacology": ClinicalPharmacologyAgent,
    "dermatology": DermatologyAgent,
    "neurology": NeurologyAgent,
    "cardiology": CardiologyAgent,
    "general_internal_medicine": GeneralInternalMedicineAgent,
}


def get_specialist(specialist: Union[SpecialistType, str]) -> BaseSpecialistAgent:
    if isinstance(specialist, SpecialistType):
        key = specialist.value
        cls = SPECIALIST_MAP.get(specialist)
    else:
        key = str(specialist or "").strip().lower()
        cls = _STRING_MAP.get(key)

    if cls is None:
        logger.error("Unknown specialist key %r — refusing Cardiology default", specialist)
        raise ValueError(
            f"Unknown specialist {specialist!r}. "
            f"Valid: {sorted(_STRING_MAP.keys())}"
        )

    agent = cls()
    if agent.specialist.value != key:
        raise RuntimeError(
            f"Registry identity mismatch: requested={key} agent={agent.specialist.value}"
        )
    logger.info(
        "get_specialist key=%s class=%s",
        key,
        agent.__class__.__name__,
    )
    return agent
