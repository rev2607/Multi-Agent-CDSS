"""FastAPI routes for cases, ingestion, health."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.core.config import settings
from app.core.llm import get_llm_client
from app.db import sqlite as db
from app.graphs.case_graph import run_case_pipeline, run_feedback_pipeline
from app.ingestion.pipeline import get_ingestion_pipeline
from app.models.schemas import (
    CaseCreate,
    CaseDetail,
    CaseSummary,
    FeedbackCreate,
    IngestRequest,
    IngestResponse,
    ProcessResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _kb_point_count() -> int:
    try:
        from app.db.qdrant_store import get_qdrant_store

        return get_qdrant_store().count()
    except Exception as e:
        logger.warning("KB count unavailable: %s", e)
        return -1


@router.get("/health")
def health():
    llm = get_llm_client()
    status = llm.status()
    return {
        "status": "ok",
        "llm_provider": llm.provider,
        "llm": status,
        "kb_points": _kb_point_count(),
        "version": "0.1.0",
        "hint": (
            None
            if llm.provider != "stub"
            else "No valid API keys in backend/.env — set GEMINI_API_KEY and/or OPENROUTER_API_KEY"
        ),
    }


@router.get("/cases", response_model=List[CaseSummary])
def list_cases(limit: int = 50):
    return db.list_cases(limit=limit)


@router.post("/cases", response_model=CaseDetail)
def create_case(body: CaseCreate):
    case = db.create_case(
        title=body.title,
        patient_context=body.patient_context,
        clinical_text=body.clinical_text,
        notes=body.notes,
    )
    # Best-effort index — case creation must not fail if Qdrant/embeddings are down.
    # Always stamp case_id for isolation from other cases at retrieval time.
    if body.clinical_text.strip():
        try:
            get_ingestion_pipeline().ingest_text(
                body.clinical_text,
                title=f"Case text — {body.title}",
                source_type="case_text",
                tags=["case", case["id"]],
                extra_payload={
                    "case_id": str(case["id"]),
                    "modality": "text",
                    "scope": "case",
                },
            )
        except Exception as e:
            logger.warning("Case text KB index skipped: %s", e)
    return db.get_case(case["id"])


@router.get("/cases/{case_id}", response_model=CaseDetail)
def get_case(case_id: str):
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    return case


def _is_upload(value: object) -> bool:
    return isinstance(value, (UploadFile, StarletteUploadFile)) or (
        hasattr(value, "filename") and hasattr(value, "read")
    )


@router.post("/cases/{case_id}/attachments")
async def upload_attachments(
    case_id: str,
    request: Request,
    files: Annotated[
        Optional[List[UploadFile]],
        File(description="One or more case attachments"),
    ] = None,
):
    """Accept any number of multimodal attachments for a case.

    Robust to multi-file FormData under field name ``files`` (or any upload fields).
    Disk + SQLite save always succeeds even if vector KB indexing fails.
    """
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    uploads: List[UploadFile] = list(files or [])

    # Fallback: parse raw multipart so multi-file / alternate field names still work
    if not uploads:
        form = await request.form()
        for _, value in form.multi_items():
            if _is_upload(value):
                uploads.append(value)  # type: ignore[arg-type]

    if not uploads:
        raise HTTPException(
            400,
            "No files received. Send multipart form field 'files' (one or more).",
        )

    case_dir = settings.upload_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    pipeline = get_ingestion_pipeline()
    saved = []
    errors: List[str] = []

    for f in uploads:
        filename = (getattr(f, "filename", None) or "").strip() or "upload.bin"
        try:
            content = await f.read()
            if not content and filename == "upload.bin":
                continue
            suffix = Path(filename).suffix
            dest = case_dir / f"{uuid.uuid4().hex}{suffix}"
            dest.write_bytes(content or b"")
            att = pipeline.ingest_case_attachment(
                case_id,
                dest,
                filename=filename,
                content_type=getattr(f, "content_type", None) or "",
            )
            saved.append(att)
        except Exception as e:
            logger.exception("Failed to save attachment %s", filename)
            errors.append(f"{filename}: {e}")

    if not saved:
        raise HTTPException(
            400,
            "No valid files were uploaded"
            + (f" ({'; '.join(errors)})" if errors else ""),
        )

    # Do not overwrite terminal statuses that already have a report
    if case.get("status") in ("draft", "ingested", None, ""):
        db.update_case(case_id, status="ingested")

    return {
        "attachments": saved,
        "count": len(saved),
        "errors": errors,
        "case": db.get_case(case_id),
    }


def _http_error_from_pipeline(error: str) -> HTTPException:
    low = (error or "").lower()
    if any(
        m in low
        for m in (
            "401",
            "403",
            "user not found",
            "authentication failed",
            "invalid api key",
            "llm authentication",
        )
    ):
        return HTTPException(status_code=401, detail=error)
    return HTTPException(status_code=500, detail=error)


@router.post("/cases/{case_id}/process", response_model=ProcessResponse)
def process_case(case_id: str):
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    result = run_case_pipeline(case_id)
    if result.get("error") and not result.get("report"):
        raise _http_error_from_pipeline(str(result["error"]))
    updated = db.get_case(case_id)
    return ProcessResponse(case=updated, message="processed")


@router.post("/cases/{case_id}/feedback", response_model=ProcessResponse)
def submit_feedback(case_id: str, body: FeedbackCreate):
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    if not case.get("report"):
        raise HTTPException(400, "Case has no report yet; process the case first")

    fb = db.add_feedback(case_id, body.text)
    result = run_feedback_pipeline(case_id, fb["id"], body.text)
    if result.get("error") and not result.get("report"):
        raise _http_error_from_pipeline(str(result["error"]))
    updated = db.get_case(case_id)
    return ProcessResponse(case=updated, message="feedback applied")


@router.post("/kb/ingest", response_model=IngestResponse)
def ingest_kb(body: IngestRequest):
    if not body.text.strip():
        raise HTTPException(400, "text is required")
    r = get_ingestion_pipeline().ingest_text(
        body.text,
        title=body.title,
        source_type=body.source_type,
        tags=body.tags,
    )
    return IngestResponse(
        document_id=r["document_id"],
        chunks_indexed=r["chunks_indexed"],
    )


@router.post("/kb/ingest-file")
async def ingest_kb_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    tags: str = Form(""),
):
    dest_dir = settings.data_dir / "kb_files"
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = file.filename or "upload.bin"
    dest = dest_dir / f"{uuid.uuid4().hex}_{name}"
    dest.write_bytes(await file.read())
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    r = get_ingestion_pipeline().ingest_file(
        dest,
        title=title or name,
        content_type=file.content_type or "",
        source_type="upload",
        tags=tag_list,
    )
    return {
        "document_id": r["document_id"],
        "chunks_indexed": r["chunks_indexed"],
        "modality": r.get("modality"),
        "path": r.get("path"),
    }


@router.get("/kb/search")
def kb_search(q: str, top_k: int = 8):
    if not q.strip():
        raise HTTPException(400, "q required")
    from app.retrieval.hybrid import HybridRetriever

    hits = HybridRetriever().search(q, top_k=top_k)
    return {"query": q, "hits": [h.model_dump() for h in hits]}


@router.post("/kb/seed")
def seed_kb():
    r = get_ingestion_pipeline().seed_sample_knowledge()
    return r
