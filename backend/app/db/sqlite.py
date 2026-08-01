"""SQLite metadata store: cases, attachments, feedback, document registry."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from app.core.config import settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class CaseRow(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    title: Mapped[str] = mapped_column(String(512), default="Untitled case")
    status: Mapped[str] = mapped_column(String(32), default="draft")
    patient_context: Mapped[str] = mapped_column(Text, default="")
    clinical_text: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    assigned_specialist: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    routing_rationale: Mapped[str] = mapped_column(Text, default="")
    modalities_json: Mapped[str] = mapped_column(Text, default="[]")
    report_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AttachmentRow(Base):
    __tablename__ = "attachments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(String(36), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128), default="application/octet-stream")
    modality: Mapped[str] = mapped_column(String(64), default="file")
    path: Mapped[str] = mapped_column(String(1024))
    extracted_text: Mapped[str] = mapped_column(Text, default="")
    meta_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class FeedbackRow(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(String(36), index=True)
    text: Mapped[str] = mapped_column(Text)
    applied: Mapped[bool] = mapped_column(Boolean, default=False)
    knowledge_written: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class DocumentRow(Base):
    """Registry of knowledge-base source documents (raw files stay on disk)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    title: Mapped[str] = mapped_column(String(512))
    source_type: Mapped[str] = mapped_column(String(64), default="manual")
    path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


_engine = None
_SessionLocal = None


def init_db() -> None:
    global _engine, _SessionLocal
    settings.ensure_dirs()
    url = f"sqlite:///{settings.sqlite_path.as_posix()}"
    _engine = create_engine(url, connect_args={"check_same_thread": False})
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=_engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ----------------------- helpers used by services -----------------------


def create_case(
    *,
    title: str,
    patient_context: str = "",
    clinical_text: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    with get_session() as s:
        row = CaseRow(
            id=_new_id(),
            title=title,
            patient_context=patient_context,
            clinical_text=clinical_text,
            notes=notes,
            status="draft",
        )
        s.add(row)
        s.flush()
        return _case_to_dict(s, row)


def list_cases(limit: int = 50) -> List[Dict[str, Any]]:
    with get_session() as s:
        rows = s.scalars(
            select(CaseRow).order_by(CaseRow.created_at.desc()).limit(limit)
        ).all()
        return [_case_to_dict(s, r, light=True) for r in rows]


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    with get_session() as s:
        row = s.get(CaseRow, case_id)
        if not row:
            return None
        return _case_to_dict(s, row)


def update_case(case_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    with get_session() as s:
        row = s.get(CaseRow, case_id)
        if not row:
            return None
        for k, v in fields.items():
            if k == "modalities":
                row.modalities_json = json.dumps(v)
            elif k == "report":
                row.report_json = json.dumps(v) if v is not None else None
            elif hasattr(row, k):
                setattr(row, k, v)
        row.updated_at = _utcnow()
        s.flush()
        return _case_to_dict(s, row)


def add_attachment(
    case_id: str,
    *,
    filename: str,
    content_type: str,
    modality: str,
    path: str,
    extracted_text: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    with get_session() as s:
        row = AttachmentRow(
            id=_new_id(),
            case_id=case_id,
            filename=filename,
            content_type=content_type,
            modality=modality,
            path=path,
            extracted_text=extracted_text,
            meta_json=json.dumps(meta or {}),
        )
        s.add(row)
        case = s.get(CaseRow, case_id)
        if case:
            mods = json.loads(case.modalities_json or "[]")
            if modality not in mods:
                mods.append(modality)
            case.modalities_json = json.dumps(mods)
            case.updated_at = _utcnow()
        s.flush()
        return {
            "id": row.id,
            "case_id": case_id,
            "filename": filename,
            "content_type": content_type,
            "modality": modality,
            "path": path,
            "extracted_text": extracted_text,
            "metadata": meta or {},
        }


def add_feedback(case_id: str, text: str) -> Dict[str, Any]:
    with get_session() as s:
        row = FeedbackRow(id=_new_id(), case_id=case_id, text=text)
        s.add(row)
        case = s.get(CaseRow, case_id)
        if case:
            case.status = "feedback"
            case.updated_at = _utcnow()
        s.flush()
        return {
            "id": row.id,
            "case_id": case_id,
            "text": text,
            "created_at": row.created_at,
            "applied": False,
            "knowledge_written": False,
        }


def mark_feedback(feedback_id: str, *, applied: bool = True, knowledge_written: bool = False) -> None:
    with get_session() as s:
        row = s.get(FeedbackRow, feedback_id)
        if row:
            row.applied = applied
            row.knowledge_written = knowledge_written


def register_document(
    *,
    title: str,
    source_type: str,
    path: Optional[str],
    tags: List[str],
    chunk_count: int,
) -> str:
    with get_session() as s:
        doc_id = _new_id()
        s.add(
            DocumentRow(
                id=doc_id,
                title=title,
                source_type=source_type,
                path=path,
                tags_json=json.dumps(tags),
                chunk_count=chunk_count,
            )
        )
        return doc_id


def _case_to_dict(session: Session, row: CaseRow, light: bool = False) -> Dict[str, Any]:
    base = {
        "id": row.id,
        "title": row.title,
        "status": row.status,
        "assigned_specialist": row.assigned_specialist,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
    if light:
        return base
    atts = session.scalars(
        select(AttachmentRow).where(AttachmentRow.case_id == row.id)
    ).all()
    fbs = session.scalars(
        select(FeedbackRow).where(FeedbackRow.case_id == row.id).order_by(FeedbackRow.created_at)
    ).all()
    report = json.loads(row.report_json) if row.report_json else None
    return {
        **base,
        "patient_context": row.patient_context or "",
        "clinical_text": row.clinical_text or "",
        "notes": row.notes or "",
        "modalities": json.loads(row.modalities_json or "[]"),
        "routing_rationale": row.routing_rationale or "",
        "report": report,
        "error": row.error,
        "attachments": [
            {
                "id": a.id,
                "filename": a.filename,
                "content_type": a.content_type,
                "modality": a.modality,
                "path": a.path,
                "extracted_text": a.extracted_text,
                "metadata": json.loads(a.meta_json or "{}"),
            }
            for a in atts
        ],
        "feedback": [
            {
                "id": f.id,
                "case_id": f.case_id,
                "text": f.text,
                "created_at": f.created_at,
                "applied": f.applied,
                "knowledge_written": f.knowledge_written,
            }
            for f in fbs
        ],
    }
