const BASE = import.meta.env.VITE_API_BASE || "";

function formatErrorDetail(detail) {
  if (detail == null) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        if (typeof d === "string") return d;
        if (d && typeof d === "object") {
          const loc = Array.isArray(d.loc) ? d.loc.join(".") : "";
          return [loc, d.msg || d.message || JSON.stringify(d)]
            .filter(Boolean)
            .join(": ");
        }
        return String(d);
      })
      .join("; ");
  }
  if (typeof detail === "object") {
    return detail.message || detail.error || JSON.stringify(detail);
  }
  return String(detail);
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch (err) {
    throw new Error(
      `Network error calling ${path}. Is the API running on port 8000? (${err.message})`
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = formatErrorDetail(body.detail) || JSON.stringify(body);
    } catch {
      try {
        detail = (await res.text()) || detail;
      } catch {
        /* ignore */
      }
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  // Some endpoints may return empty body
  const text = await res.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

export const api = {
  health: () => request("/api/health"),
  listCases: () => request("/api/cases"),
  getCase: (id) => request(`/api/cases/${id}`),
  createCase: (body) =>
    request("/api/cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  uploadFiles: async (caseId, files) => {
    const list = Array.from(files || []).filter(Boolean);
    if (!list.length) {
      throw new Error("No files to upload");
    }
    const fd = new FormData();
    // FastAPI expects repeated field name "files"
    for (const f of list) {
      const name = f.name || "upload.bin";
      fd.append("files", f, name);
    }
    return request(`/api/cases/${caseId}/attachments`, {
      method: "POST",
      body: fd,
      // Do NOT set Content-Type — browser sets multipart boundary
    });
  },
  processCase: (id) =>
    request(`/api/cases/${id}/process`, { method: "POST" }),
  feedback: (id, text) =>
    request(`/api/cases/${id}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }),
  searchKb: (q) => request(`/api/kb/search?q=${encodeURIComponent(q)}`),
};
