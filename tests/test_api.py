import app.api.documents as documents_module
import app.services.rag as rag
from app.services.llm import LLMError

PLANT_TEXT = b"Photosynthesis lets green plants make glucose from sunlight."
PLANT_QUESTION = "How do green plants make glucose using sunlight?"


def upload(client, name="notes.txt", content=PLANT_TEXT, content_type="text/plain"):
    return client.post("/documents/upload", files={"file": (name, content, content_type)})


# ---------- basic pages ----------

def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "documind"}


def test_home_page_and_static_files_are_served(client):
    home = client.get("/")
    css = client.get("/static/style.css")

    assert home.status_code == 200
    assert "text/html" in home.headers["content-type"]
    assert "DocuMind" in home.text
    assert css.status_code == 200


# ---------- upload ----------

def test_upload_txt_indexes_and_lists_the_document(client):
    response = upload(client)

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "notes.txt"
    assert data["pages"] == 1
    assert data["total_chunks"] >= 1
    assert data["indexed_chunks"] == data["total_chunks"]

    listed = client.get("/documents").json()
    assert len(listed) == 1
    assert listed[0]["source"] == "notes.txt"
    assert listed[0]["doc_id"] == data["doc_id"]


def test_upload_rejects_unsupported_file_type(client):
    response = upload(client, name="image.png", content=b"fake", content_type="image/png")

    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_upload_rejects_empty_file(client):
    response = upload(client, content=b"")

    assert response.status_code == 400


def test_upload_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(documents_module, "MAX_FILE_SIZE_MB", 0)

    response = upload(client)

    assert response.status_code == 413


def test_upload_rejects_corrupt_docx_and_cleans_up(client, upload_dir):
    response = upload(client, name="bad.docx", content=b"this is not a real docx file")

    assert response.status_code == 422
    assert list(upload_dir.iterdir()) == []


def test_upload_rejects_file_without_text(client, upload_dir):
    response = upload(client, content=b"   \n\n   ")

    assert response.status_code == 422
    assert "No readable text" in response.json()["detail"]
    assert list(upload_dir.iterdir()) == []


# ---------- search ----------

def test_search_finds_uploaded_content(client):
    upload(client)

    response = client.get("/documents/search", params={"q": "green plants glucose sunlight"})

    assert response.status_code == 200
    assert response.json()["results"][0]["source"] == "notes.txt"


def test_search_requires_at_least_three_characters(client):
    response = client.get("/documents/search", params={"q": "ab"})

    assert response.status_code == 422


# ---------- delete ----------

def test_delete_removes_document_and_saved_file(client, upload_dir):
    doc_id = upload(client).json()["doc_id"]
    assert len(list(upload_dir.iterdir())) == 1

    response = client.delete(f"/documents/{doc_id}")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] >= 1
    assert list(upload_dir.iterdir()) == []
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_delete_unknown_document_returns_404(client):
    assert client.delete("/documents/unknown").status_code == 404


# ---------- ask ----------

def test_ask_returns_answer_with_sources(client, loaded_store, fake_llm):
    response = client.post("/ask", json={"question": PLANT_QUESTION})

    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert data["answer"] == fake_llm.reply
    assert data["sources"][0]["source"] == "science.txt"


def test_ask_returns_not_found_for_unrelated_question(client, loaded_store, fake_llm):
    response = client.post("/ask", json={"question": "Capital city of France"})

    assert response.status_code == 200
    assert response.json()["found"] is False
    assert fake_llm.calls == []


def test_ask_without_documents(client, fake_llm):
    response = client.post("/ask", json={"question": PLANT_QUESTION})

    assert response.status_code == 200
    assert response.json()["answer"] == rag.NO_DOCUMENTS_MESSAGE


def test_ask_validates_input(client):
    too_short = client.post("/ask", json={"question": "hi"})
    top_k_too_big = client.post("/ask", json={"question": "a valid question", "top_k": 50})

    assert too_short.status_code == 422
    assert top_k_too_big.status_code == 422


def test_ask_returns_502_when_llm_fails(client, loaded_store, monkeypatch):
    def broken_llm(system_prompt, user_prompt):
        raise LLMError("The model is unavailable")

    monkeypatch.setattr(rag, "generate_answer", broken_llm)

    response = client.post("/ask", json={"question": PLANT_QUESTION})

    assert response.status_code == 502
    assert response.json()["detail"] == "The model is unavailable"