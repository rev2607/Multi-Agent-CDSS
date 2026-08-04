"""Multimodal ingestion: extract → chunk → hybrid embed → Qdrant + SQLite registry."""

from __future__ import annotations

import logging
import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.db import sqlite as db
from app.ingestion.extractors import chunk_text, detect_modality, extract_file
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()

    def ingest_text(
        self,
        text: str,
        *,
        title: str = "Untitled",
        source_type: str = "manual",
        tags: Optional[List[str]] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        tags = tags or []
        chunks = chunk_text(text)
        if not chunks:
            doc_id = db.register_document(
                title=title, source_type=source_type, path=path, tags=tags, chunk_count=0
            )
            return {"document_id": doc_id, "chunks_indexed": 0}

        doc_id = str(uuid.uuid4())
        payloads = []
        for i, ch in enumerate(chunks):
            payload = {
                "document_id": doc_id,
                "title": title,
                "source_type": source_type,
                "chunk_index": i,
                "modality": (extra_payload or {}).get("modality", "text"),
                "tags": tags,
            }
            if extra_payload:
                payload.update({k: v for k, v in extra_payload.items() if k != "text"})
            if path:
                payload["path"] = path
            # Normalize ownership: case docs require case_id; shared KB stamped "shared"
            st = str(payload.get("source_type") or source_type).lower()
            cid = payload.get("case_id")
            if st in {"case_attachment", "case_text"} or str(payload.get("scope") or "") == "case":
                payload["scope"] = "case"
                if not cid:
                    logger.warning(
                        "ingest_text: case-scoped doc missing case_id title=%s",
                        title[:80],
                    )
            else:
                # Shared knowledge base
                if not cid:
                    payload["case_id"] = "shared"
                payload.setdefault("scope", "shared")
            payloads.append(payload)

        ids = self.retriever.index_texts(chunks, payloads)
        db.register_document(
            title=title,
            source_type=source_type,
            path=path,
            tags=tags,
            chunk_count=len(ids),
        )
        # Keep document_id consistent in return even if SQLite issues a new id —
        # payload uses doc_id for vector linkage.
        return {"document_id": doc_id, "chunks_indexed": len(ids), "chunk_ids": ids}

    def ingest_file(
        self,
        file_path: Path,
        *,
        title: Optional[str] = None,
        content_type: str = "",
        source_type: str = "file",
        tags: Optional[List[str]] = None,
        copy_to_data: bool = False,
    ) -> Dict[str, Any]:
        file_path = Path(file_path)
        title = title or file_path.name
        dest = file_path
        if copy_to_data:
            dest_dir = settings.data_dir / "kb_files"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{uuid.uuid4().hex}_{file_path.name}"
            shutil.copy2(file_path, dest)

        text, modality, meta = extract_file(dest, filename=title, content_type=content_type)
        result = self.ingest_text(
            text,
            title=title,
            source_type=source_type,
            tags=tags or [modality],
            extra_payload={"modality": modality, **meta},
            path=str(dest),
        )
        result["modality"] = modality
        result["extracted_chars"] = len(text)
        result["extracted_text"] = text
        result["path"] = str(dest)
        return result

    def ingest_case_attachment(
        self,
        case_id: str,
        file_path: Path,
        *,
        filename: str,
        content_type: str = "",
        index_to_kb: bool = True,
    ) -> Dict[str, Any]:
        """Save attachment metadata + extract text; optionally index into shared KB.

        File persistence and SQLite metadata always succeed even if vector indexing fails.
        """
        try:
            text, modality, meta = extract_file(
                file_path, filename=filename, content_type=content_type
            )
        except Exception as e:
            logger.exception("Extract failed for %s", filename)
            text = f"[Extraction failed for {filename}: {e}]"
            modality = "file"
            meta = {"filename": filename, "modality": modality, "error": str(e)}

        att = db.add_attachment(
            case_id,
            filename=filename,
            content_type=content_type or "application/octet-stream",
            modality=modality,
            path=str(file_path),
            extracted_text=text or "",
            meta=meta,
        )

        # Best-effort KB projection — never fail the upload if Qdrant/embeddings are down
        if index_to_kb and (text or "").strip():
            try:
                self.ingest_text(
                    text,
                    title=f"Case {case_id[:8]} — {filename}",
                    source_type="case_attachment",
                    tags=["case", case_id, modality],
                    extra_payload={
                        "modality": modality,
                        "case_id": str(case_id),
                        "filename": filename,
                        "attachment_id": att["id"],
                        "scope": "case",
                    },
                    path=str(file_path),
                )
                att["indexed"] = True
            except Exception as e:
                logger.warning(
                    "KB index skipped for attachment %s (%s): %s",
                    filename,
                    att.get("id"),
                    e,
                )
                att["indexed"] = False
                att["index_error"] = str(e)
        else:
            att["indexed"] = False

        return att

    def seed_sample_knowledge(self) -> Dict[str, Any]:
        """Load sample clinical snippets if KB is empty."""
        sample_dir = settings.sample_data_dir
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_file = sample_dir / "clinical_snippets.md"
        if not sample_file.exists():
            sample_file.write_text(_DEFAULT_SAMPLE_KB, encoding="utf-8")

        store_count = self.retriever.store.count()
        if store_count > 0:
            return {"skipped": True, "existing_points": store_count}

        total = 0
        for path in sample_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".md":
                total += self._ingest_sample_markdown(path)
            elif path.suffix.lower() in {".txt", ".pdf", ".csv"}:
                r = self.ingest_file(path, source_type="sample", tags=["sample"])
                total += r.get("chunks_indexed", 0)
        return {"skipped": False, "chunks_indexed": total}

    def _ingest_sample_markdown(self, path: Path) -> int:
        """Split sample KB by ## sections and tag specialty for cleaner retrieval."""
        text = path.read_text(encoding="utf-8", errors="replace")
        sections = _split_markdown_sections(text)
        total = 0
        if not sections:
            r = self.ingest_text(
                text,
                title=path.name,
                source_type="sample",
                tags=["sample"],
                extra_payload={"modality": "text", "filename": path.name},
                path=str(path),
            )
            return int(r.get("chunks_indexed") or 0)

        for title, body in sections:
            specialty = _section_specialty(title + "\n" + body)
            tags = ["sample"]
            if specialty:
                tags.append(specialty)
            r = self.ingest_text(
                f"## {title}\n{body}".strip(),
                title=f"{path.stem} — {title}"[:200],
                source_type="sample",
                tags=tags,
                extra_payload={
                    "modality": "text",
                    "filename": path.name,
                    "section": title,
                    "specialty": specialty or "",
                },
                path=str(path),
            )
            total += int(r.get("chunks_indexed") or 0)
        return total


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    import re

    parts = re.split(r"(?m)^##\s+", text)
    out: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if body:
            out.append((title, body))
    return out


def _section_specialty(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("cardiology", "stemi", "acs", "heart failure", "atrial")):
        return "cardiology"
    if any(k in t for k in ("dermatology", "melanoma", "psoriasis", "rash")):
        return "dermatology"
    if any(k in t for k in ("neurology", "stroke", "seizure", "migraine")):
        return "neurology"
    if any(
        k in t
        for k in (
            "pharmacology",
            "drug",
            "warfarin",
            "interaction",
            "dosing",
        )
    ):
        return "clinical_pharmacology"
    if any(k in t for k in ("internal medicine", "sepsis", "hyponatremia", "diabetes")):
        return "general_internal_medicine"
    return ""


_DEFAULT_SAMPLE_KB = """# Sample Clinical Knowledge Base (Demo)

## Cardiology
- STEMI: ST-elevation myocardial infarction requires immediate reperfusion.
- ACS symptoms: chest pain, diaphoresis, dyspnea, radiation to arm/jaw.
- Heart failure: evaluate EF, volume status, guideline-directed medical therapy (GDMT).
- Atrial fibrillation: rate vs rhythm control; anticoagulation based on CHA2DS2-VASc.

## Dermatology
- Melanoma ABCDE: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving.
- Cellulitis vs erysipelas: deeper dermal/subcutis vs superficial; systemic signs guide antibiotics.
- Psoriasis: well-demarcated plaques with silvery scale; consider topical steroids / biologics.

## Neurology
- Stroke FAST: Face droop, Arm weakness, Speech difficulty, Time to call emergency.
- Migraine vs secondary headache: red flags include thunderclap, fever, neuro deficit, age >50 new onset.
- Seizure first aid: protect airway, time the event, do not put objects in mouth.

## General Internal Medicine
- Sepsis: suspect infection + organ dysfunction; early fluids, cultures, antibiotics.
- Hyponatremia workup: serum osmolality, volume status, urine Na/osmolality.
- Diabetes type 2: metformin first-line if tolerated; screen for complications.

## Clinical Pharmacology & Drug Safety
- Warfarin + NSAIDs: increased bleeding risk.
- ACE inhibitors + potassium-sparing diuretics: hyperkalemia risk.
- QT-prolonging drugs: macrolides, fluoroquinolones, some antipsychotics — check interactions.
- Renal dosing: adjust metformin, DOACs, many antibiotics for reduced eGFR.
- Penicillin allergy: clarify reaction type; many reported allergies are not IgE-mediated.

## Drug list (sample)
| Drug | Class | Key caution |
|------|-------|-------------|
| Metformin | Biguanide | Lactic acidosis risk in severe renal impairment |
| Lisinopril | ACEI | Cough, hyperkalemia, angioedema |
| Atorvastatin | Statin | Myopathy; check interactions with strong CYP3A4 inhibitors |
| Warfarin | VKA | INR monitoring; many drug/food interactions |
| Sertraline | SSRI | Serotonin syndrome with MAOIs; bleeding risk with anticoagulants |
"""


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline()
