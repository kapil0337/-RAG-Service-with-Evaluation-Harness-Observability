// Simple client for the RAG Service API. All endpoints are same-origin.

const $ = (id) => document.getElementById(id);

function setStatus(el, message, kind) {
  el.textContent = message;
  el.classList.remove("hidden", "status-info", "status-error", "status-ok");
  el.classList.add(`status-${kind}`);
}

function hideStatus(el) {
  el.classList.add("hidden");
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ---- Tabs ----

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`${btn.dataset.tab}-panel`).classList.add("active");
  });
});

// ---- Health ----

async function checkHealth() {
  const badge = $("health-badge");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.status === "ok") {
      badge.textContent = `ok · ${data.embedding_model} · ${data.llm_model}`;
      badge.className = "badge badge-ok";
    } else {
      badge.textContent = "degraded (database unavailable)";
      badge.className = "badge badge-degraded";
    }
  } catch (err) {
    badge.textContent = "unreachable";
    badge.className = "badge badge-degraded";
  }
}

// ---- Documents ----

async function loadDocuments() {
  const tbody = $("documents-tbody");
  const select = $("document-filter");
  const statusEl = $("documents-status");

  try {
    const res = await fetch("/documents");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const documents = await res.json();

    select.innerHTML = '<option value="">All documents</option>';

    if (documents.length === 0) {
      tbody.innerHTML = '<tr class="empty-row"><td colspan="7">No documents ingested yet.</td></tr>';
      return;
    }

    tbody.innerHTML = "";
    for (const doc of documents) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${doc.id}</td>
        <td>${escapeHtml(doc.filename)}</td>
        <td>${doc.chunk_count}</td>
        <td>${doc.chunk_size}</td>
        <td>${doc.chunk_overlap}</td>
        <td>${new Date(doc.created_at).toLocaleString()}</td>
        <td><button class="reindex-btn" data-id="${doc.id}">Re-index</button></td>
      `;
      tbody.appendChild(tr);

      const option = document.createElement("option");
      option.value = doc.id;
      option.textContent = doc.filename;
      select.appendChild(option);
    }

    tbody.querySelectorAll(".reindex-btn").forEach((btn) => {
      btn.addEventListener("click", () => reindexDocument(btn.dataset.id));
    });
  } catch (err) {
    setStatus(statusEl, `Failed to load documents: ${err.message}`, "error");
  }
}

async function reindexDocument(documentId) {
  const statusEl = $("documents-status");
  setStatus(statusEl, `Re-indexing document ${documentId}...`, "info");
  try {
    const res = await fetch(`/documents/${documentId}/reindex`, { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
    const data = await res.json();
    setStatus(statusEl, `Re-indexed "${data.document.filename}" (${data.chunks_created} chunks).`, "ok");
    await loadDocuments();
  } catch (err) {
    setStatus(statusEl, `Re-index failed: ${err.message}`, "error");
  }
}

$("reindex-all-btn").addEventListener("click", async () => {
  const statusEl = $("documents-status");
  setStatus(statusEl, "Re-indexing all documents...", "info");
  try {
    const res = await fetch("/reindex", { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
    const data = await res.json();
    setStatus(statusEl, `Re-indexed ${data.length} document(s).`, "ok");
    await loadDocuments();
  } catch (err) {
    setStatus(statusEl, `Re-index failed: ${err.message}`, "error");
  }
});

// ---- Upload ----

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = $("upload-file");
  const statusEl = $("upload-status");
  const btn = $("upload-btn");

  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const params = new URLSearchParams();
  const chunkSize = $("chunk-size").value;
  const chunkOverlap = $("chunk-overlap").value;
  if (chunkSize) params.set("chunk_size", chunkSize);
  if (chunkOverlap) params.set("chunk_overlap", chunkOverlap);

  btn.disabled = true;
  setStatus(statusEl, "Uploading and ingesting...", "info");

  try {
    const res = await fetch(`/documents?${params.toString()}`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
    const data = await res.json();
    setStatus(
      statusEl,
      `Ingested "${data.document.filename}" (${data.chunks_created} chunks created).`,
      "ok"
    );
    fileInput.value = "";
    await loadDocuments();
  } catch (err) {
    setStatus(statusEl, `Upload failed: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
  }
});

// ---- Query ----

$("query-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const question = $("question").value.trim();
  if (!question) return;

  const statusEl = $("ask-status");
  const answerCard = $("answer-card");
  const askBtn = $("ask-btn");

  const body = { question };

  const topK = $("top-k").value;
  if (topK) body.top_k = Number(topK);

  body.rerank = $("rerank").checked;

  const documentId = $("document-filter").value;
  if (documentId) body.document_id = Number(documentId);

  askBtn.disabled = true;
  answerCard.classList.add("hidden");
  setStatus(statusEl, "Thinking...", "info");

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
    const data = await res.json();
    renderAnswer(data);
    hideStatus(statusEl);
  } catch (err) {
    setStatus(statusEl, `Query failed: ${err.message}`, "error");
  } finally {
    askBtn.disabled = false;
  }
});

function renderAnswer(data) {
  $("answer-text").textContent = data.answer;

  const citationsList = $("citations-list");
  citationsList.innerHTML = "";
  for (const [i, citation] of data.citations.entries()) {
    const div = document.createElement("div");
    div.className = "citation";
    div.innerHTML = `
      <div class="citation-header">
        <span>[${i + 1}] ${escapeHtml(citation.filename)} (chunk ${citation.chunk_index})</span>
        <span>score: ${citation.score.toFixed(3)}</span>
      </div>
      <div class="citation-content">${escapeHtml(citation.content)}</div>
    `;
    citationsList.appendChild(div);
  }

  $("meta-row").innerHTML = `
    <span>Model: ${escapeHtml(data.model)}</span>
    <span>Tokens: ${data.usage.total_tokens} (prompt ${data.usage.prompt_tokens} / completion ${data.usage.completion_tokens})</span>
    <span>Retrieval: ${data.retrieval_latency_ms} ms</span>
    <span>Generation: ${data.generation_latency_ms} ms</span>
    <span>Total: ${data.total_latency_ms} ms</span>
  `;

  $("answer-card").classList.remove("hidden");
}

// ---- Init ----

checkHealth();
loadDocuments();
