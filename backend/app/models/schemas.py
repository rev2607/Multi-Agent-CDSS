"""Pydantic API / domain schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CaseStatus(str, Enum):
    draft = "draft"
    ingested = "ingested"
    routing = "routing"
    processing = "processing"
    completed = "completed"
    feedback = "feedback"
    corrected = "corrected"
    failed = "failed"


class SpecialistType(str, Enum):
    cardiology = "cardiology"
    dermatology = "dermatology"
    neurology = "neurology"
    general_internal_medicine = "general_internal_medicine"
    clinical_pharmacology = "clinical_pharmacology"


class AgentType(str, Enum):
    superior = "superior"
    cardiology = "cardiology"
    dermatology = "dermatology"
    neurology = "neurology"
    general_internal_medicine = "general_internal_medicine"
    clinical_pharmacology = "clinical_pharmacology"


class EvidenceItem(BaseModel):
    source_id: str
    title: str = ""
    snippet: str
    score: float = 0.0
    modality: str = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskScoreItem(BaseModel):
    name: str
    score: Optional[float] = None
    max_score: Optional[float] = None
    risk_band: str = ""
    detail: str = ""
    components: Dict[str, Any] = Field(default_factory=dict)


class ImageFindingItem(BaseModel):
    source: str = ""
    label: str = ""
    summary: str = ""
    modality: str = "image"


class DifferentialItem(BaseModel):
    diagnosis: str
    likelihood: str = "possible"  # leading | likely | possible | unlikely
    rationale: str = ""


class ClinicalReport(BaseModel):
    """Structured clinical report produced by a specialized agent."""

    specialist: SpecialistType
    routed_to: str = ""  # e.g. "Routed to: Clinical Pharmacology" — set by pipeline
    chief_complaint: str = ""
    case_summary: str = ""
    key_findings: List[str] = Field(default_factory=list)
    # Prefer structured differential; string list still accepted for older agents
    differential_diagnosis: List[Any] = Field(default_factory=list)
    assessment: str = ""
    recommendations: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    risk_scores: List[RiskScoreItem] = Field(default_factory=list)
    image_findings: List[ImageFindingItem] = Field(default_factory=list)
    reasoning: str = ""
    limitations: str = (
        "Decision-support only. Not a substitute for clinical judgment or licensed care."
    )
    retrieval_path: str = "pipeline_hybrid"  # pipeline_hybrid | bounded_agentic
    raw: Dict[str, Any] = Field(default_factory=dict)


class CaseCreate(BaseModel):
    title: str = "Untitled case"
    patient_context: str = ""
    clinical_text: str = ""
    notes: str = ""


class CaseSummary(BaseModel):
    id: str
    title: str
    status: CaseStatus
    assigned_specialist: Optional[SpecialistType] = None
    created_at: datetime
    updated_at: datetime


class FeedbackRecord(BaseModel):
    id: str
    case_id: str
    text: str
    created_at: datetime
    applied: bool = False
    knowledge_written: bool = False


class CaseDetail(CaseSummary):
    patient_context: str = ""
    clinical_text: str = ""
    notes: str = ""
    modalities: List[str] = Field(default_factory=list)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    report: Optional[ClinicalReport] = None
    feedback: List[FeedbackRecord] = Field(default_factory=list)
    routing_rationale: str = ""
    error: Optional[str] = None


class FeedbackCreate(BaseModel):
    text: str = Field(..., min_length=1)


class ProcessResponse(BaseModel):
    case: CaseDetail
    message: str = "ok"


class IngestRequest(BaseModel):
    """Ingest free text or a path under data/ into the shared knowledge base."""

    title: str = "Untitled document"
    text: str = ""
    source_type: str = "manual"
    tags: List[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    message: str = "ok"


class RetrievalHit(BaseModel):
    id: str
    text: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
