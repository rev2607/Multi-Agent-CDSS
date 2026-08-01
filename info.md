```markdown
# Medical Multi-Agent Clinical Decision Support System
## Local Demo Implementation

> Local-first multi-agent system for doctors.  
> One Superior Agent routes the case → One Specialized Agent owns the full reasoning + feedback loop.

This document describes the **local demo** version.  
Retrieval architecture follows the production research principles in `rag_research.md` (Hybrid substrate as default + Bounded Agentic RAG only when needed).

---

## 1. Project Overview

This is a **local multi-agent medical AI system** designed to assist doctors with clinical cases.

**Core Workflow:**
1. Doctor submits a case (any modality: text, PDF, images, audio, tables, handwritten notes, etc.)
2. **Superior Agent** understands the case and routes it to **exactly one** specialized agent
3. The specialized agent uses the retrieval service to gather evidence, reasons, and produces a structured clinical report
4. Doctor can give **text feedback**
5. The same specialized agent performs **targeted correction** using the feedback + previous context
6. Useful knowledge from feedback is written back into the **shared knowledge base**

**Key Principles:**
- Strictly **one specialized agent per case**
- Fully multimodal input support
- Shared knowledge base that grows over time
- Local-first (CPU + cloud LLMs)
- No hosting required

---

## 2. Agents

### Superior Agent (Router)
- Only responsibility: understand the incoming case and assign it to **exactly one** specialized agent.
- Does not generate the final clinical report.

### Specialized Agents (5)

| # | Agent                                         | Responsibility                                              |
|---|-----------------------------------------------|-------------------------------------------------------------|
| 1 | **Cardiology Agent**                          | Heart-related cases                                         |
| 2 | **Dermatology Agent**                         | Skin / dermatology cases                                    |
| 3 | **Neurology Agent**                           | Brain / neurological cases                                  |
| 4 | **General Internal Medicine Agent**           | Broad, multi-system, or unclear cases + differential diagnosis |
| 5 | **Clinical Pharmacology & Drug Safety Agent** | Medications, interactions, dosing, side effects, contraindications |

---

## 3. Tech Stack (Local Demo)

| Layer                    | Technology                                      | Notes |
|--------------------------|-------------------------------------------------|-------|
| **Backend**              | FastAPI                                         | Async API layer |
| **Frontend**             | React (Vite)                                    | Basic clean UI |
| **Agent Orchestration**  | LangGraph                                       | Stateful agents + human-in-the-loop feedback |
| **RAG Framework**        | LlamaIndex (+ custom tools)                     | Multimodal + tool calling support |
| **Vector Database**      | **Qdrant** (local mode)                         | Hybrid dense + sparse search |
| **Metadata / Cases DB**  | SQLite (can later move to PostgreSQL)           | Cases, feedback history, preferences |
| **LLM**                  | Gemini (primary) + OpenRouter fallback          | Strong multimodal capabilities |
| **Embeddings**           | Gemini Embedding / text-embedding-3-large + BM25| Dense + Sparse |
| **PDF / Documents**      | Unstructured + PyMuPDF                          | Text + scanned documents |
| **Images / Handwritten** | Gemini Vision                                   | Excellent OCR + understanding |
| **Audio**                | Whisper (tiny / base) on CPU                    | Transcription |
| **Tables**               | Pandas + LLM structured extraction              | Lab results, drug lists, etc. |

---

## 4. Retrieval Architecture (Aligned with Research)

We follow the exact philosophy from the production research:

> **Hybrid pipeline is the engine; Bounded Agentic RAG is an optional driver for hard trips — same roads (indexes), different driving style.**

### 4.1 Foundation Layer → Hybrid Multimodal Search (Default Path)

This is the **primary retrieval substrate** for almost all requests.

- **Dense embeddings** → Semantic understanding
- **Sparse (BM25)** → Exact medical terms, drug names, lab codes, guideline titles
- **Fusion** → Reciprocal Rank Fusion (RRF) or weighted fusion
- **Optional rerank** → Cross-encoder on top candidates (when latency allows)
- Supports: text, tables, image captions, OCR output, audio transcripts

**Vector DB:** Qdrant (local) — native hybrid dense + sparse + rich payload filtering

### 4.2 Orchestration Layer → Bounded Agentic RAG (Secondary Path)

Agentic RAG is **not** always-on.

It is used only when the case needs it:
- Multi-hop reasoning
- Empty or weak first retrieve
- Complex research / literature synthesis
- Cases that require multiple tools (local knowledge + online + structured data)

**Rules of engagement (from research):**
1. Default path = **Pipeline Hybrid RAG**
2. Agentic path = **opt-in** by intent or empty-first-hit
3. Hard budgets: `max_steps` (typically 2–4), wall-clock limit
4. Same hybrid indexes and filters are used by every tool call
5. The specialized agent plans the retrieval, but the underlying store remains hybrid

```
                    ┌──────────────────────────────────────┐
                    │  Query classifier / router             │
                    │  (cheap, deterministic or small LLM) │
                    └───────────┬──────────────────────────┘
              simple / case-scoped │              complex / multi-hop / research
                                  ▼                              ▼
                    ┌─────────────────────────┐    ┌────────────────────────────┐
                    │ PIPELINE HYBRID RAG     │    │ BOUNDED AGENTIC RAG        │
                    │ hybrid + RRF + rerank   │    │ max_steps 2–4              │
                    │ filters → fast path     │    │ hard wall-clock budget     │
                    └─────────────────────────┘    │ tools = same hybrid + SQL  │
                                                   │ stop on grade ≥ threshold  │
                                                   └────────────────────────────┘
                                  │                              │
                                  └──────────┬───────────────────┘
                                             ▼
                              Same substrate: Qdrant hybrid + local files + metadata DB
```

---

## 5. Knowledge Sources

The system can use:

- Existing local materials (already available):
  - Drug lists / drug databases
  - PDFs & clinical guidelines
  - Documents
  - Images
  - Audio files
  - Tables
  - Text notes
- Online research (PubMed, guidelines, research papers) — controlled by the agent
- Doctor feedback (automatically written into shared knowledge base when useful)

All knowledge lives in a **shared** knowledge base (Qdrant + metadata store).

**Design rule (from research):**  
Vectors never own the source of truth.  
Vectors own **searchable projections**.  
Raw files stay on disk / local storage; clinical facts and cases live in the metadata DB; vectors store embeddings + metadata + pointers.

---

## 6. Feedback Loop

- Feedback type: **Text only** (for now)
- Behavior: **Targeted correction** (not full re-reasoning from scratch)
- Knowledge update: Useful feedback is automatically added to the shared knowledge base
- The same specialized agent that handled the case also handles the feedback

---

## 7. High-Level Architecture (Local Demo)

```
Doctor Input (any modality)
        ↓
[Multimodal Ingestion Pipeline]
  • Type detection
  • Extraction (OCR / Whisper / Table parsing / Vision)
  • Chunking + Hybrid Embedding (Dense + Sparse)
  • Store in Qdrant + Metadata DB
        ↓
Superior Agent (Router)
  → Understands case → Routes to exactly 1 specialist
        ↓
Specialized Agent (LangGraph)
  │
  ├─ Query classifier decides path
  │
  ├─ Simple path  → Pipeline Hybrid RAG (default)
  │                 (dense + sparse + RRF + optional rerank)
  │
  └─ Complex path → Bounded Agentic RAG
                    (plan → tools → re-retrieve → stop)
                    max_steps + wall-clock budget
        ↓
  Clinical reasoning + structured report generation
        ↓
Doctor views report
        ↓
Doctor gives text feedback
        ↓
Same Specialized Agent
  → Targeted correction using previous state + feedback
  → Writes useful knowledge back to shared KB
  → Returns improved report
```

---

## 8. Project Structure

```text
medical-multi-agent/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/                  # FastAPI routes
│   │   ├── agents/               # Superior + 5 specialized agents
│   │   ├── graphs/               # LangGraph definitions
│   │   ├── ingestion/            # Multimodal ingestion pipeline
│   │   ├── retrieval/            # Hybrid search + Bounded Agentic tools
│   │   ├── models/               # Pydantic schemas
│   │   ├── db/                   # Qdrant + SQLite clients
│   │   └── core/                 # Config, LLM clients, settings
│   ├── data/                     # Your existing PDFs, drug lists, etc.
│   └── requirements.txt
├── frontend/                     # React (Vite)
└── README.md
```

---

## 9. Implementation Phases

**Phase 1 – Foundation**
- Project scaffolding (FastAPI + React)
- Qdrant (local) + SQLite setup
- Multimodal ingestion pipeline
- Ingest existing drug lists, PDFs, images, audio, tables
- Basic hybrid (dense + sparse) search working

**Phase 2 – Core Agents + Pipeline Hybrid**
- Superior Router Agent
- One full specialized agent
- Pipeline Hybrid RAG as default path
- Structured report generation
- Feedback → targeted correction loop

**Phase 3 – Bounded Agentic + Expand**
- Add Bounded Agentic RAG path (gated)
- Implement remaining 4 specialized agents
- Shared knowledge write-back from feedback
- Better query classification (simple vs complex)

**Phase 4 – UI & Polish**
- React interface (upload, report view, feedback, history)
- Streaming responses
- Case history & export

---

## 10. Current Decisions Log

| Decision                        | Choice                                              | Date       |
|--------------------------------|-----------------------------------------------------|------------|
| Number of specialized agents   | 5                                                   | 2026-08-01 |
| Routing style                  | Strictly one agent per case                         | 2026-08-01 |
| Feedback type                  | Text only                                           | 2026-08-01 |
| Feedback handling              | Targeted correction                                 | 2026-08-01 |
| Knowledge base                 | Shared + auto-ingest useful feedback                | 2026-08-01 |
| Vector DB                      | Qdrant (local)                                      | 2026-08-01 |
| Retrieval substrate            | Hybrid (Dense + Sparse + RRF)                       | 2026-08-01 |
| Retrieval orchestration        | Bounded Agentic RAG (gated, not always-on)          | 2026-08-01 |
| Default path                   | Pipeline Hybrid RAG                                 | 2026-08-01 |
| UI                             | FastAPI + React                                     | 2026-08-01 |
| LLM                            | Gemini primary + OpenRouter fallback                | 2026-08-01 |
| Hardware target                | CPU only (local)                                    | 2026-08-01 |

---

## 11. Alignment with Production Research

This local demo intentionally follows the same architectural principles defined in the production research:

| Research Principle                          | Local Demo Implementation                     |
|--------------------------------------------|-----------------------------------------------|
| Hybrid dense + sparse as primary substrate | Yes – Qdrant hybrid                           |
| Agentic RAG only as bounded secondary path | Yes – gated by complexity / empty hit         |
| Same indexes used by both paths            | Yes                                           |
| Hard budgets on agentic loops              | Yes (`max_steps` + wall-clock)                |
| Vectors are projections, not source of truth | Yes                                         |
| Qdrant as primary hybrid engine            | Yes (local mode)                              |
| Clear separation of pipeline vs agentic    | Yes                                           |

The main differences from production are environmental (local CPU, no AWS VPC, no multi-tenant PHI isolation yet). The **retrieval philosophy and flow remain the same**.

---

*This document will be updated as the project evolves.*
```