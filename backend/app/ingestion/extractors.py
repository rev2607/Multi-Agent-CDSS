"""Modality-specific extractors: PDF, image, audio, tables, text."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings
from app.core.llm import get_llm_client

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
TABLE_EXTS = {".csv", ".tsv", ".xlsx", ".xls"}
PDF_EXTS = {".pdf"}
TEXT_EXTS = {".txt", ".md", ".json", ".xml", ".html", ".htm", ".rtf"}
DOC_EXTS = {".docx"}


def detect_modality(filename: str, content_type: str = "") -> str:
    ext = Path(filename).suffix.lower()
    ct = (content_type or "").lower()
    if ext in PDF_EXTS or "pdf" in ct:
        return "pdf"
    if ext in IMAGE_EXTS or ct.startswith("image/"):
        return "image"
    if ext in AUDIO_EXTS or ct.startswith("audio/"):
        return "audio"
    if ext in TABLE_EXTS or "spreadsheet" in ct or "csv" in ct:
        return "table"
    if ext in DOC_EXTS:
        return "document"
    if ext in TEXT_EXTS or ct.startswith("text/"):
        return "text"
    return "file"


def extract_file(
    path: Path,
    *,
    filename: Optional[str] = None,
    content_type: str = "",
) -> Tuple[str, str, Dict[str, Any]]:
    """Return (extracted_text, modality, metadata)."""
    filename = filename or path.name
    modality = detect_modality(filename, content_type)
    meta: Dict[str, Any] = {"filename": filename, "modality": modality}

    try:
        if modality == "pdf":
            text = _extract_pdf(path)
        elif modality == "image":
            text = _extract_image(path, content_type or "image/png")
        elif modality == "audio":
            text = _extract_audio(path)
        elif modality == "table":
            text = _extract_table(path)
        elif modality == "document":
            text = _extract_docx(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            modality = "text"
    except Exception as e:
        logger.exception("Extraction failed for %s", path)
        text = f"[Extraction error for {filename}: {e}]"
        meta["error"] = str(e)

    meta["char_count"] = len(text)
    return text, modality, meta


def _extract_pdf(path: Path) -> str:
    # Prefer PyMuPDF; fall back to unstructured if available
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        parts = []
        for i, page in enumerate(doc):
            parts.append(page.get_text("text"))
        doc.close()
        text = "\n".join(parts).strip()
        if text:
            return text
    except Exception as e:
        logger.warning("PyMuPDF failed: %s", e)

    try:
        from unstructured.partition.auto import partition

        elements = partition(filename=str(path))
        return "\n".join(str(el) for el in elements)
    except Exception as e:
        logger.warning("unstructured PDF failed: %s", e)
        return f"[Could not extract PDF text from {path.name}]"


def _extract_image(path: Path, content_type: str) -> str:
    data = path.read_bytes()
    llm = get_llm_client()
    prompt = (
        "You are a clinical document assistant. Extract all readable text "
        "(including handwritten notes) and describe any clinically relevant "
        "visual findings. Structure as:\n"
        "TRANSCRIPTION:\n...\nFINDINGS:\n..."
    )
    return llm.describe_image(data, content_type, prompt)


def _extract_audio(path: Path) -> str:
    if not settings.whisper_enabled:
        return f"[Whisper disabled] Audio file: {path.name}"
    try:
        import whisper

        model = whisper.load_model(settings.whisper_model)
        result = model.transcribe(str(path))
        return (result.get("text") or "").strip()
    except Exception as e:
        logger.warning("Whisper transcription failed: %s", e)
        return f"[Audio transcription unavailable: {e}] File: {path.name}"


def _extract_table(path: Path) -> str:
    import pandas as pd

    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    # Compact textual representation for RAG
    preview = df.head(100)
    buf = io.StringIO()
    preview.to_csv(buf, index=False)
    summary = (
        f"Table: {path.name}\n"
        f"Columns: {list(df.columns)}\n"
        f"Rows: {len(df)}\n\n"
        f"Preview (first {len(preview)} rows):\n{buf.getvalue()}"
    )
    # Optional LLM structured extraction when keys available
    llm = get_llm_client()
    if llm.provider != "stub" and len(summary) < 12000:
        try:
            structured = llm.complete(
                "Extract clinically relevant facts from this table as bullet points.",
                summary[:8000],
                temperature=0.1,
                max_tokens=1024,
            )
            return summary + "\n\nStructured extraction:\n" + structured
        except Exception:
            pass
    return summary


def _extract_docx(path: Path) -> str:
    try:
        import docx

        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX extract failed: {e}]"


def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Simple character-window chunking with overlap (token-agnostic local demo)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Prefer break on paragraph/sentence boundary
        if end < n:
            window = text[start:end]
            br = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if br > chunk_size // 3:
                end = start + br + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
