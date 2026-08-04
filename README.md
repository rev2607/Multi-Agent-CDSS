# Medical Multi-Agent Clinical Decision Support System

Local-first multi-agent **clinical decision support** demo for doctors.

A **Superior Agent** routes each case to **exactly one** specialized agent. That specialist retrieves evidence (hybrid RAG, with bounded agentic RAG when needed), produces a structured clinical report, and applies targeted corrections from doctor feedback.

> **Decision-support only.** Not a substitute for clinical judgment or licensed care.

---

## Core workflow

1. Doctor submits a case (text + optional multimodal attachments).
2. **Superior Agent** routes to exactly one specialist (hard rules + LLM routing).
3. Specialist retrieves evidence, reasons, and returns a structured report.
4. Doctor can submit **text feedback**.
5. The **same** specialist applies a **targeted correction** and may write useful knowledge back into the shared KB.

**Principles**

- Strictly **one specialized agent per case**
- Multimodal input (text, PDF, images, audio, tables, CSV, etc.)
- Shared knowledge base that grows from sample data and useful feedback
- Local-first (CPU + cloud LLMs; no hosting required for the demo)

---

## Agents

| Role | Responsibility |
|------|----------------|
| **Superior Agent** | Understand the case and assign **exactly one** specialist. Does not write the final report. |
| **Cardiology** | Heart / ACS-oriented cases (hard rules only when ACS criteria match) |
| **Dermatology** | Skin / dermatology cases (including SCAR-style hard routes) |
| **Neurology** | Brain / neurological cases (e.g. stroke hard routes) |
| **General Internal Medicine** | Broad, multi-system, or unclear cases + differential diagnosis |
| **Clinical Pharmacology & Drug Safety** | Medications, interactions, dosing, side effects, contraindications |

Routing combines deterministic hard rules (priority: pharmacology → dermatology/SCAR → neurology/stroke → cardiology/ACS only) with LLM routing, then a final resolve step so cardiology is never a silent default.

---

## Architecture — components we use

| Component | What we use | Role in this system |
|-----------|-------------|---------------------|
| **Frontend / UI** | React 18 + Vite | Case create/list, multi-file upload, report view, text feedback, KB search helper |
| **API layer** | FastAPI + Uvicorn | REST under `/api`; CORS for local UI; OpenAPI at `/docs` |
| **Agent orchestration** | LangGraph | Stateful case pipeline: load → route → specialize; feedback correction loop |
| **Router agent** | Superior Agent (hard rules + LLM) | Assigns **exactly one** specialist; cardiology only when ACS hard rules match |
| **Specialist agents** | 5 domain agents (LangChain-style base + Gemini/OpenRouter) | Full clinical reasoning + structured report for one specialty per case |
| **LLM (chat / vision)** | Gemini primary (`gemini-2.5-flash` default); optional OpenRouter | Routing, report generation, vision/OCR labels, agentic query rewrites |
| **Embeddings (dense)** | Gemini `text-embedding-004` (or OpenRouter embeddings); hash fallback offline | Semantic vectors for hybrid search |
| **Embeddings (sparse)** | BM25 / rank-BM25-style sparse encoder | Exact medical terms, drug names, codes, guideline titles |
| **Pipeline Hybrid RAG** | Custom `HybridRetriever` on Qdrant | **Default path:** dense + sparse retrieve → RRF fusion → filters → postprocess → top‑N evidence |
| **Fusion** | Reciprocal Rank Fusion (RRF, `k=60`) | Merges dense and sparse rankings into one score list |
| **Rerank** | Optional cross-encoder (`sentence-transformers`); off by default | Re-order top candidates when `ENABLE_RERANK=true` |
| **Bounded Agentic RAG** | Custom `BoundedAgenticRAG` | **Secondary path only** when first retrieve is empty/weak or multi-hop intent; hard budgets (`max_steps`, wall-clock) |
| **Evidence postprocess** | Specialty keyword filter, case isolation, dedupe, min relevance | Keeps top 4–6 specialty-matched, case-scoped chunks; injects case attachments when needed |
| **Vector store** | Qdrant **local / embedded** (`path=` on disk) | Hybrid collection: named dense + sparse vectors; payload filters |
| **Metadata / cases DB** | SQLite (SQLAlchemy) | Cases, attachments metadata, reports, feedback history, document registry |
| **Object / file storage** | Local filesystem (`backend/data/uploads`, `kb_files`, `sample`) | Raw PDFs, images, audio; vectors only store projections + pointers |
| **Cache** | In-process `functools.lru_cache` singletons (settings, LLM client, Qdrant, ingestion) | No Redis/Memcached in the local demo; process-local reuse only |
| **Multimodal ingestion** | PyMuPDF, Unstructured, Pillow, Pandas, Whisper (CPU tiny/base), Gemini Vision | Detect modality → extract text/OCR/transcript → chunk → hybrid embed → upsert |
| **Knowledge base seed** | Sample MD + CSV under `data/sample/` | Clinical snippets + drug interactions indexed on first empty boot |
| **Clinical helpers** | Deterministic risk scores (HEART/TIMI heuristics), image-finding extract | Decision-support fields on reports (not validated calculators) |
| **Feedback write-back** | Same specialist + ingestion pipeline | Targeted report correction; useful feedback may be written into shared KB |
| **Config** | Pydantic Settings + `backend/.env` | API keys, model names, retrieval budgets, paths, CORS |
| **Logging** | Python logging | Provider status, routing decisions, retrieve path |

### Retrieval model (how RAG is chosen)

| Path | When | Behavior |
|------|------|----------|
| **Pipeline Hybrid RAG** | Default for almost all cases | Dense + sparse search → RRF → specialty/case filters → quality cut → evidence for the specialist |
| **Bounded Agentic RAG** | Empty first hit, weak scores, complex multi-hop / interaction-style queries, or forced | Re-plan / re-query over the **same** hybrid indexes with `max_steps` (default 3) and wall-clock limit (default 45s); stop when quality is good enough |

Design rule: **vectors are searchable projections, not the source of truth.** Raw files stay on disk; clinical case facts live in SQLite; Qdrant holds embeddings + metadata + pointers.

### High-level flow

```text
Doctor (React UI)
      │  text + files
      ▼
FastAPI  ──►  Ingestion (extract → chunk → dense+sparse embed)
      │              │
      │              ├── SQLite (cases, docs, feedback)
      │              └── Qdrant local (hybrid vectors)
      ▼
LangGraph: Superior route → one Specialist
      │
      ├─ Pipeline Hybrid RAG  (default)
      └─ Bounded Agentic RAG  (gated)
      ▼
Structured clinical report → optional text feedback → same specialist correction
```

---


## Project structure

```text
Multi-Agent-CDSS/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint
│   │   ├── api/                 # REST routes
│   │   ├── agents/              # Superior + specialists, routing, risk scores
│   │   ├── graphs/              # LangGraph case + feedback pipelines
│   │   ├── ingestion/           # Multimodal extract + index
│   │   ├── retrieval/           # Hybrid + bounded agentic RAG
│   │   ├── models/              # Pydantic schemas
│   │   ├── db/                  # Qdrant + SQLite
│   │   └── core/                # Config, LLM client, logging
│   ├── data/
│   │   ├── sample/              # Seed clinical snippets + drug interactions
│   │   ├── uploads/             # Per-case attachments
│   │   ├── qdrant/              # Local vector store
│   │   └── cdss.db              # SQLite (created at runtime)
│   ├── tests/                   # Routing / specialist resolution tests
│   ├── requirements.txt
│   └── .env.example
├── frontend/                    # React UI (cases, upload, report, feedback)
├── info.md                      # Extended design notes
└── README.md
```

---

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for the frontend)
- At least one LLM API key:
  - [Google AI Studio](https://aistudio.google.com/apikey) — `GEMINI_API_KEY` (recommended)
  - Optionally [OpenRouter](https://openrouter.ai/keys) — `OPENROUTER_API_KEY` (disabled by default unless enabled in config)

---

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Edit `backend/.env` and set a real key (do not leave placeholders):

```env
GEMINI_API_KEY=your_real_key
LLM_PROVIDER=auto
# Optional OpenRouter:
# OPENROUTER_ENABLED=true
# OPENROUTER_API_KEY=sk-or-v1-...
```

Start the API (from `backend/`):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API root: http://127.0.0.1:8000  
- Interactive docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/api/health  

On first boot the app initializes SQLite, Qdrant, and seeds sample knowledge if the KB is empty.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173  

Vite proxies `/api` to `http://127.0.0.1:8000`.

---

## Using the app

1. Create a case (title, patient context, clinical text, notes).
2. Optionally attach files (PDF, images, audio, CSV, text) via multi-file picker or drag-and-drop.
3. **Process case** — routes to a specialist and generates a structured report (summary, findings, differential, recommendations, red flags, evidence, optional risk scores / image findings).
4. Submit **text feedback** to refine the same report; useful feedback can be written into the shared KB.

---

## API overview

All routes are under `/api`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health, LLM provider, KB point count |
| `GET` | `/cases` | List cases |
| `POST` | `/cases` | Create case |
| `GET` | `/cases/{id}` | Case detail + report + feedback |
| `POST` | `/cases/{id}/attachments` | Upload multimodal files (`files` multipart field) |
| `POST` | `/cases/{id}/process` | Route + specialist report |
| `POST` | `/cases/{id}/feedback` | Apply text feedback |
| `POST` | `/kb/ingest` | Ingest free text into shared KB |
| `POST` | `/kb/ingest-file` | Ingest a file into shared KB |
| `GET` | `/kb/search?q=...` | Hybrid search over the KB |
| `POST` | `/kb/seed?force=false` | Seed / re-seed sample knowledge |

Full schemas and try-it-out: `/docs`.

---

## Configuration

Primary settings live in `backend/.env` (see `backend/.env.example`). Important variables:

| Variable | Meaning |
|----------|---------|
| `GEMINI_API_KEY` | Primary LLM + vision + embeddings |
| `GEMINI_MODEL` / `GEMINI_VISION_MODEL` | Chat / vision model names |
| `OPENROUTER_API_KEY` / `OPENROUTER_ENABLED` | Optional OpenRouter fallback |
| `LLM_PROVIDER` | `gemini` \| `openrouter` \| `auto` |
| `QDRANT_PATH` / `SQLITE_PATH` | Local data paths |
| `HYBRID_TOP_K` / `EVIDENCE_TOP_K` | Retrieval sizing |
| `AGENTIC_MAX_STEPS` / `AGENTIC_WALL_CLOCK_SEC` | Bounded agentic budgets |
| `WHISPER_ENABLED` / `WHISPER_MODEL` | CPU audio transcription |
| `CORS_ORIGINS` | Frontend origins (default Vite) |

---

## Tests

Routing and specialist-resolution tests (no live LLM required for hard-rule cases):

```bash
cd backend
pytest tests/ -q
```

Covers hard routing, first-principles resolve behavior, and superior routing guards.

---

## Sample knowledge

On first empty KB boot (or via `POST /api/kb/seed`), the pipeline can index files under `backend/data/sample/`, including:

- `clinical_snippets.md` — short clinical reference snippets  
- `drug_interactions.csv` — drug interaction rows for pharmacology demos  

---

## Disclaimer

This repository is a **local research / demo** system for clinical decision **support**. Outputs may be incomplete or incorrect. Always verify against primary sources and clinical judgment. Not for production clinical use without appropriate validation, governance, and regulatory review.
