const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const uploadStatus = document.getElementById("upload-status");
const docList = document.getElementById("doc-list");
const messages = document.getElementById("messages");
const emptyState = document.getElementById("empty-state");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");

let uploading = false;

// ---------- helpers ----------

async function apiError(response) {
    try {
        const data = await response.json();
        if (typeof data.detail === "string") return data.detail;
        if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join(", ");
    } catch (_) {
        // response was not JSON
    }
    return `Request failed (${response.status})`;
}

function setStatus(text, kind = "info") {
    uploadStatus.textContent = text;
    uploadStatus.className = `status ${kind}`;
}

function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
}

function addMessage(className) {
    if (emptyState) emptyState.remove();
    const div = document.createElement("div");
    div.className = `message ${className}`;
    messages.appendChild(div);
    return div;
}

// Turn "[1]" into a styled badge. Uses text nodes only, never innerHTML,
// so text from documents or the LLM can never inject HTML or scripts.
function withCitations(text) {
    const fragment = document.createDocumentFragment();
    text.split(/(\[\d+\])/).forEach((part) => {
        if (/^\[\d+\]$/.test(part)) {
            const badge = document.createElement("span");
            badge.className = "cite";
            badge.textContent = part;
            fragment.appendChild(badge);
        } else if (part) {
            fragment.appendChild(document.createTextNode(part));
        }
    });
    return fragment;
}

// ---------- documents ----------

function renderDocuments(docs) {
    docList.replaceChildren();

    if (docs.length === 0) {
        const li = document.createElement("li");
        li.className = "doc-empty";
        li.textContent = "No documents yet.";
        docList.appendChild(li);
        return;
    }

    docs.forEach((doc) => {
        const li = document.createElement("li");
        li.className = "doc-item";

        const info = document.createElement("div");
        info.className = "doc-info";
        const name = document.createElement("span");
        name.className = "doc-name";
        name.textContent = doc.source;
        const meta = document.createElement("span");
        meta.className = "doc-meta";
        meta.textContent = `${doc.chunks} chunks`;
        info.append(name, meta);

        const del = document.createElement("button");
        del.type = "button";
        del.className = "delete-btn";
        del.textContent = "Delete";
        del.addEventListener("click", () => deleteDocument(doc));

        li.append(info, del);
        docList.appendChild(li);
    });
}

async function loadDocuments() {
    try {
        const res = await fetch("/documents");
        if (!res.ok) throw new Error(await apiError(res));
        renderDocuments(await res.json());
    } catch (err) {
        setStatus(err.message, "error");
    }
}

async function uploadFile(file) {
    if (!file || uploading) return;
    uploading = true;
    const formData = new FormData();
    formData.append("file", file);
    setStatus(`Uploading and indexing ${file.name}... (the first upload can be slow)`, "info");

    try {
        const res = await fetch("/documents/upload", { method: "POST", body: formData });
        if (!res.ok) throw new Error(await apiError(res));
        const data = await res.json();
        setStatus(`${data.filename}: ${data.pages} page(s), ${data.indexed_chunks} chunks indexed.`, "success");
        await loadDocuments();
    } catch (err) {
        setStatus(err.message, "error");
    } finally {
        uploading = false;
        fileInput.value = "";
    }
}

async function deleteDocument(doc) {
    if (!confirm(`Delete "${doc.source}"?`)) return;
    try {
        const res = await fetch(`/documents/${encodeURIComponent(doc.doc_id)}`, { method: "DELETE" });
        if (!res.ok) throw new Error(await apiError(res));
        setStatus(`Deleted ${doc.source}.`, "success");
        await loadDocuments();
    } catch (err) {
        setStatus(err.message, "error");
    }
}

fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add("dragging");
    })
);
["dragleave", "drop"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragging");
    })
);
dropZone.addEventListener("drop", (e) => uploadFile(e.dataTransfer.files[0]));

// ---------- chat ----------

function addUserMessage(text) {
    addMessage("user").textContent = text;
    scrollToBottom();
}

function addThinking() {
    const div = addMessage("bot thinking");
    div.textContent = "Searching your documents...";
    scrollToBottom();
    return div;
}

function addErrorMessage(text) {
    addMessage("bot error").textContent = text;
    scrollToBottom();
}

function addAnswer(data) {
    const div = addMessage(data.found ? "bot" : "bot not-found");

    const text = document.createElement("p");
    text.className = "answer-text";
    text.appendChild(withCitations(data.answer));
    div.appendChild(text);

    if (data.sources.length > 0) {
        const heading = document.createElement("div");
        heading.className = "sources-title";
        heading.textContent = "Sources";
        div.appendChild(heading);

        data.sources.forEach((src) => {
            const details = document.createElement("details");
            details.className = "source";
            const summary = document.createElement("summary");
            summary.textContent = `[${src.id}] ${src.source} · page ${src.page} · match ${src.score}`;
            const snippet = document.createElement("p");
            snippet.textContent = src.snippet + (src.snippet.length >= 200 ? "…" : "");
            details.append(summary, snippet);
            div.appendChild(details);
        });
    }
    scrollToBottom();
}

function setAsking(isAsking) {
    askBtn.disabled = isAsking;
    questionInput.disabled = isAsking;
}

askForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (question.length < 3) return;

    addUserMessage(question);
    questionInput.value = "";
    setAsking(true);
    const thinking = addThinking();

    try {
        const res = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });
        if (!res.ok) throw new Error(await apiError(res));
        const data = await res.json();
        thinking.remove();
        addAnswer(data);
    } catch (err) {
        thinking.remove();
        addErrorMessage(err.message);
    } finally {
        setAsking(false);
        questionInput.focus();
    }
});

// ---------- start ----------

loadDocuments();