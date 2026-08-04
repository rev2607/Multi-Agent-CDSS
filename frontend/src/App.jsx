import React, { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";

const EMPTY_FORM = {
  title: "",
  patient_context: "",
  clinical_text: "",
  notes: "",
};

function statusBadge(status) {
  if (["completed", "corrected"].includes(status)) return "ok";
  if (["failed"].includes(status)) return "danger";
  if (["processing", "routing", "feedback"].includes(status)) return "warn";
  return "";
}

function formatSpecialist(s) {
  if (!s) return "—";
  return s.replaceAll("_", " ");
}

function formatBytes(n) {
  if (n == null || Number.isNaN(n)) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Merge newly picked files into the existing list (dedupe by name+size+lastModified). */
function mergeFiles(prev, incoming) {
  const key = (f) => `${f.name}::${f.size}::${f.lastModified}`;
  const map = new Map(prev.map((f) => [key(f), f]));
  for (const f of incoming) {
    map.set(key(f), f);
  }
  return Array.from(map.values());
}

function MultiFilePicker({ files, onChange, disabled, id = "case-attachments" }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  function addFiles(fileList) {
    const picked = Array.from(fileList || []).filter(Boolean);
    if (picked.length) {
      onChange(mergeFiles(files, picked));
    }
  }

  function handlePick(e) {
    addFiles(e.target.files);
    e.target.value = "";
  }

  function removeAt(index) {
    onChange(files.filter((_, i) => i !== index));
  }

  function clearAll() {
    onChange([]);
    if (inputRef.current) inputRef.current.value = "";
  }

  function onDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (disabled) return;
    addFiles(e.dataTransfer.files);
  }

  return (
    <div
      className={`file-input ${dragOver ? "drag-over" : ""}`}
      onDragEnter={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={(e) => {
        e.preventDefault();
        setDragOver(false);
      }}
      onDrop={onDrop}
    >
      <label htmlFor={id} className="file-drop-label">
        <strong>Click to choose files</strong>
        <span>or drag &amp; drop here</span>
        <span className="file-hint-inline">
          Multiple files supported (PDF, images, audio, CSV, text…)
        </span>
      </label>
      <input
        ref={inputRef}
        id={id}
        type="file"
        multiple
        disabled={disabled}
        onChange={handlePick}
        className="file-native-input"
      />
      {files.length > 0 && (
        <div className="file-list">
          <div className="file-list-header">
            <strong>
              {files.length} file{files.length === 1 ? "" : "s"} selected
            </strong>
            <button type="button" className="linkish" onClick={clearAll} disabled={disabled}>
              Clear all
            </button>
          </div>
          <ul>
            {files.map((f, i) => (
              <li key={`${f.name}-${f.size}-${f.lastModified}-${i}`}>
                <span className="file-name" title={f.name}>
                  {f.name}
                </span>
                <span className="file-size">{formatBytes(f.size)}</span>
                <button
                  type="button"
                  className="linkish"
                  onClick={() => removeAt(i)}
                  disabled={disabled}
                  aria-label={`Remove ${f.name}`}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [cases, setCases] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [caseDetail, setCaseDetail] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [files, setFiles] = useState([]);
  const [extraFiles, setExtraFiles] = useState([]);
  const [feedback, setFeedback] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("new"); // new | view
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const refreshList = useCallback(async () => {
    const list = await api.listCases();
    setCases(list);
  }, []);

  const loadCase = useCallback(async (id) => {
    const c = await api.getCase(id);
    setCaseDetail(c);
    setSelectedId(id);
    setMode("view");
    setExtraFiles([]);
    setSidebarOpen(false);
  }, []);

  useEffect(() => {
    (async () => {
      try {
        setHealth(await api.health());
        await refreshList();
      } catch (e) {
        setError(e.message);
      }
    })();
  }, [refreshList]);

  async function handleCreateAndProcess(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    let createdId = null;
    try {
      const created = await api.createCase({
        title: form.title || "Untitled case",
        patient_context: form.patient_context,
        clinical_text: form.clinical_text,
        notes: form.notes,
      });
      createdId = created.id;
      if (files.length) {
        const up = await api.uploadFiles(created.id, files);
        if (up.errors?.length) {
          console.warn("Some attachment index warnings:", up.errors);
        }
        if (!up.count && files.length) {
          throw new Error("Upload returned 0 files — check API logs");
        }
      }
      const result = await api.processCase(created.id);
      setCaseDetail(result.case);
      setSelectedId(result.case.id);
      setMode("view");
      setForm(EMPTY_FORM);
      setFiles([]);
      await refreshList();
    } catch (err) {
      const msg = err?.message || String(err);
      setError(
        createdId
          ? `${msg} (case ${createdId} was created — open it from the sidebar to retry upload/process)`
          : msg
      );
      if (createdId) {
        try {
          await loadCase(createdId);
          await refreshList();
        } catch {
          /* ignore */
        }
      }
    } finally {
      setBusy(false);
    }
  }

  async function handleAddAttachments(e) {
    e.preventDefault();
    if (!selectedId || !extraFiles.length) return;
    setError("");
    setBusy(true);
    try {
      const result = await api.uploadFiles(selectedId, extraFiles);
      if (!result.count) {
        throw new Error("No files were saved on the server");
      }
      setCaseDetail(result.case || (await api.getCase(selectedId)));
      setExtraFiles([]);
      await refreshList();
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleFeedback(e) {
    e.preventDefault();
    if (!selectedId || !feedback.trim()) return;
    setError("");
    setBusy(true);
    try {
      const result = await api.feedback(selectedId, feedback.trim());
      setCaseDetail(result.case);
      setFeedback("");
      await refreshList();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const report = caseDetail?.report;

  return (
    <div className="app">
      <button
        type="button"
        className={`sidebar-overlay ${sidebarOpen ? "visible" : ""}`}
        aria-label="Close menu"
        onClick={() => setSidebarOpen(false)}
      />

      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">
            <span className="brand-logo">CD</span>
            <h1>Medical Multi-Agent CDSS</h1>
          </div>
          <p>Local demo · Superior router → one specialist</p>
        </div>

        <div className="health">
          <span className={`health-dot ${health?.status === "ok" ? "live" : ""}`} />
          {health ? (
            <>
              {health.status} · llm: {health.llm_provider} · kb: {health.kb_points}
            </>
          ) : (
            "connecting…"
          )}
        </div>

        <div className="sidebar-actions">
          <button
            className="primary"
            type="button"
            onClick={() => {
              setMode("new");
              setSelectedId(null);
              setCaseDetail(null);
              setSidebarOpen(false);
            }}
          >
            + New case
          </button>
        </div>

        <div className="sidebar-section-label">Cases</div>
        <div className="case-list">
          {cases.length === 0 && (
            <div className="empty-inline">No cases yet</div>
          )}
          {cases.map((c) => (
            <button
              key={c.id}
              type="button"
              className={`case-item ${selectedId === c.id ? "active" : ""}`}
              onClick={() => loadCase(c.id).catch((e) => setError(e.message))}
            >
              <div className="title">{c.title}</div>
              <div className="meta">
                <span className={`badge ${statusBadge(c.status)}`}>
                  {c.status}
                </span>
                <span>{formatSpecialist(c.assigned_specialist)}</span>
              </div>
            </button>
          ))}
        </div>
      </aside>

      <div className="main-wrap">
        <header className="topbar">
          <button
            type="button"
            className="icon-btn ghost"
            aria-label="Open menu"
            onClick={() => setSidebarOpen(true)}
          >
            ☰
          </button>
          <span className="topbar-title">CDSS</span>
        </header>

        <main className="main">
          {error && <div className="error">{error}</div>}

          {mode === "new" && (
            <form className="panel" onSubmit={handleCreateAndProcess}>
              <div className="panel-header">
                <div>
                  <h2>Submit clinical case</h2>
                  <p className="panel-subtitle">
                    Route to one specialist and generate a structured report
                  </p>
                </div>
              </div>

              <div className="field">
                <label>Title</label>
                <input
                  value={form.title}
                  onChange={(e) => setForm({ ...form, title: e.target.value })}
                  placeholder="e.g. 62M chest pain and dyspnea"
                />
              </div>
              <div className="grid-2">
                <div className="field">
                  <label>Patient context</label>
                  <textarea
                    value={form.patient_context}
                    onChange={(e) =>
                      setForm({ ...form, patient_context: e.target.value })
                    }
                    placeholder="Age, sex, PMH, meds, allergies…"
                  />
                </div>
                <div className="field">
                  <label>Notes</label>
                  <textarea
                    value={form.notes}
                    onChange={(e) => setForm({ ...form, notes: e.target.value })}
                    placeholder="Clinician notes / questions"
                  />
                </div>
              </div>
              <div className="field">
                <label>Clinical presentation</label>
                <textarea
                  className="clinical-area"
                  style={{ minHeight: 140 }}
                  value={form.clinical_text}
                  onChange={(e) =>
                    setForm({ ...form, clinical_text: e.target.value })
                  }
                  placeholder="HPI, exam, labs, imaging summary…"
                  required
                />
              </div>
              <div className="field">
                <label>Attachments (any number — PDF, images, audio, CSV…)</label>
                <MultiFilePicker
                  files={files}
                  onChange={setFiles}
                  disabled={busy}
                  id="new-case-attachments"
                />
              </div>
              <div className="row">
                <button className="primary" type="submit" disabled={busy}>
                  {busy && <span className="spinner" />}
                  Route & generate report
                  {files.length > 0
                    ? ` (${files.length} file${files.length === 1 ? "" : "s"})`
                    : ""}
                </button>
              </div>
            </form>
          )}

          {mode === "view" && caseDetail && (
            <>
              <div className="panel">
                <div className="panel-header">
                  <div>
                    <h2>{caseDetail.title}</h2>
                    <div className="badge-row" style={{ marginTop: "0.55rem" }}>
                      <span className={`badge ${statusBadge(caseDetail.status)}`}>
                        {caseDetail.status}
                      </span>
                      {caseDetail.assigned_specialist && (
                        <span className="badge ok">
                          Routed to: {formatSpecialist(caseDetail.assigned_specialist)}
                        </span>
                      )}
                      {report?.retrieval_path && (
                        <span className="badge">{report.retrieval_path}</span>
                      )}
                    </div>
                  </div>
                </div>

                {caseDetail.routing_rationale && (
                  <p className="routing-line">
                    <strong>Routed to:</strong>{" "}
                    {formatSpecialist(caseDetail.assigned_specialist) || "—"}
                    {caseDetail.routing_rationale && (
                      <>
                        {" — "}
                        {caseDetail.routing_rationale.replace(
                          /^Routed to:\s*[^—–-]+[—–-]\s*/i,
                          ""
                        )}
                      </>
                    )}
                  </p>
                )}

                {caseDetail.clinical_text && (
                  <div className="report-section">
                    <h3>Case text</h3>
                    <p className="prewrap">{caseDetail.clinical_text}</p>
                  </div>
                )}

                <div className="report-section">
                  <h3>
                    Attachments
                    {caseDetail.attachments?.length
                      ? ` (${caseDetail.attachments.length})`
                      : ""}
                  </h3>
                  {caseDetail.attachments?.length > 0 ? (
                    <ul className="attach-list">
                      {caseDetail.attachments.map((a) => (
                        <li key={a.id}>
                          <span>{a.filename}</span>
                          <span className="badge">{a.modality}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted-text">No attachments yet.</p>
                  )}
                </div>

                <form onSubmit={handleAddAttachments}>
                  <div className="field">
                    <label>Add more attachments</label>
                    <MultiFilePicker
                      files={extraFiles}
                      onChange={setExtraFiles}
                      disabled={busy}
                      id="existing-case-attachments"
                    />
                  </div>
                  <button
                    type="submit"
                    className="ghost"
                    disabled={busy || extraFiles.length === 0}
                  >
                    {busy && <span className="spinner" />}
                    Upload {extraFiles.length > 0 ? `${extraFiles.length} ` : ""}
                    file{extraFiles.length === 1 ? "" : "s"}
                  </button>
                </form>
              </div>

              {report && (
                <div className="panel">
                  <div className="panel-header">
                    <div>
                      <h2>Structured clinical report</h2>
                      <div className="badge-row" style={{ marginTop: "0.55rem" }}>
                        <span className="badge ok">
                          {report.routed_to ||
                            (caseDetail.assigned_specialist
                              ? `Routed to: ${formatSpecialist(caseDetail.assigned_specialist)}`
                              : "Routed to: —")}
                        </span>
                        {report.specialist && (
                          <span className="badge">
                            Agent: {formatSpecialist(report.specialist)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  <Section title="Chief complaint" body={report.chief_complaint} />
                  <Section title="Case summary" body={report.case_summary} />
                  <ListSection title="Key findings" items={report.key_findings} />

                  {report.risk_scores?.length > 0 && (
                    <div className="report-section">
                      <h3>Risk scores</h3>
                      <div className="risk-grid">
                        {report.risk_scores.map((rs, i) => (
                          <div className="risk-card" key={rs.name || i}>
                            <div className="risk-name">{rs.name}</div>
                            <div className="risk-value">
                              {rs.score != null ? rs.score : "—"}
                              {rs.max_score != null ? ` / ${rs.max_score}` : ""}
                            </div>
                            {rs.risk_band && (
                              <span className={`badge ${riskBandClass(rs.risk_band)}`}>
                                {rs.risk_band}
                              </span>
                            )}
                            {rs.detail && (
                              <p className="risk-detail">{rs.detail}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {report.image_findings?.length > 0 && (
                    <div className="report-section">
                      <h3>Image / document findings</h3>
                      {report.image_findings.map((im, i) => (
                        <div className="evidence" key={(im.source || "") + i}>
                          <div>
                            <strong>{im.label || "Uploaded image"}</strong>
                            {im.source ? (
                              <span className="score"> · {im.source}</span>
                            ) : null}
                          </div>
                          <div className="prewrap">{im.summary}</div>
                        </div>
                      ))}
                    </div>
                  )}

                  <DifferentialSection items={report.differential_diagnosis} />
                  <Section title="Assessment" body={report.assessment} />
                  <ListSection title="Recommendations" items={report.recommendations} />
                  <ListSection title="Red flags" items={report.red_flags} />
                  <Section title="Reasoning" body={report.reasoning} />
                  {report.evidence?.length > 0 && (
                    <div className="report-section">
                      <h3>Evidence (top specialty-matched)</h3>
                      {report.evidence.slice(0, 5).map((e) => (
                        <div
                          className="evidence"
                          key={e.source_id + e.snippet.slice(0, 20)}
                        >
                          <div>
                            <strong>{e.title || e.source_id}</strong>{" "}
                            <span className="score">
                              score {Number(e.score).toFixed(3)}
                            </span>
                          </div>
                          <div>{e.snippet}</div>
                        </div>
                      ))}
                    </div>
                  )}
                  <p className="limitations">{report.limitations}</p>
                </div>
              )}

              {caseDetail.report && (
                <form className="panel" onSubmit={handleFeedback}>
                  <div className="panel-header">
                    <div>
                      <h2>Doctor feedback</h2>
                      <p className="panel-subtitle">
                        Targeted correction by the same specialist agent
                      </p>
                    </div>
                  </div>
                  <div className="field">
                    <label>Text feedback</label>
                    <textarea
                      value={feedback}
                      onChange={(e) => setFeedback(e.target.value)}
                      placeholder="e.g. Please emphasize renal dosing and remove suggestion X…"
                      required
                    />
                  </div>
                  <button className="primary" type="submit" disabled={busy}>
                    {busy && <span className="spinner" />}
                    Apply feedback
                  </button>
                  {caseDetail.feedback?.length > 0 && (
                    <div className="report-section" style={{ marginTop: "1.15rem" }}>
                      <h3>Feedback history</h3>
                      <ul className="history-list">
                        {caseDetail.feedback.map((f) => (
                          <li key={f.id}>
                            {f.text}{" "}
                            {f.applied && <span className="badge ok">applied</span>}
                            {f.knowledge_written && (
                              <span className="badge">kb write-back</span>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </form>
              )}
            </>
          )}

          {mode === "view" && !caseDetail && (
            <div className="empty">Select a case from the sidebar</div>
          )}
        </main>
      </div>
    </div>
  );
}

function Section({ title, body }) {
  if (!body) return null;
  return (
    <div className="report-section">
      <h3>{title}</h3>
      <p className="prewrap">{body}</p>
    </div>
  );
}

function ListSection({ title, items }) {
  if (!items?.length) return null;
  return (
    <div className="report-section">
      <h3>{title}</h3>
      <ul>
        {items.map((item, i) => (
          <li key={i}>{typeof item === "string" ? item : JSON.stringify(item)}</li>
        ))}
      </ul>
    </div>
  );
}

function riskBandClass(band) {
  const b = String(band || "").toLowerCase();
  if (b.includes("high")) return "danger";
  if (b.includes("moderate") || b.includes("intermediate")) return "warn";
  if (b.includes("low")) return "ok";
  return "";
}

function DifferentialSection({ items }) {
  if (!items?.length) return null;
  return (
    <div className="report-section">
      <h3>Differential diagnosis</h3>
      <ul className="diff-list">
        {items.map((item, i) => {
          if (typeof item === "string") {
            return <li key={i}>{item}</li>;
          }
          const likelihood = item.likelihood || "possible";
          return (
            <li key={i}>
              <span className={`badge ${likelihoodBadge(likelihood)}`}>
                {likelihood}
              </span>{" "}
              <strong>{item.diagnosis}</strong>
              {item.rationale ? (
                <div className="diff-rationale">{item.rationale}</div>
              ) : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function likelihoodBadge(likelihood) {
  const l = String(likelihood || "").toLowerCase();
  if (l === "leading" || l === "likely") return "danger";
  if (l === "possible") return "warn";
  if (l === "unlikely") return "ok";
  return "";
}
