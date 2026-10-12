import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.documents as documents_module
import app.services.rag as rag_module
import app.services.vector_store as vs_module
from app.main import app as fastapi_app

EMBEDDING_DIM = 1024


# ---------- fakes ----------

def fake_embed_texts(texts):
    """A fast, predictable stand-in for the real embedding model.

    Every word is hashed into one of 1024 slots, so two texts that share words
    get similar vectors and texts with no shared words get near-zero similarity.
    """
    vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype="float32")
    for row, text in enumerate(texts):
        for word in text.lower().split():
            word = word.strip(".,!?;:()[]\"'")
            if word:
                vectors[row, zlib.crc32(word.encode()) % EMBEDDING_DIM] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


class FakeLLM:
    """Records what it was asked and returns a fixed reply (no network)."""

    def __init__(self):
        self.reply = "Plants make glucose using sunlight [1]."
        self.calls = []

    def __call__(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.reply


# ---------- sample data ----------

SCIENCE_CHUNKS = [
    {
        "chunk_id": 0,
        "source": "science.txt",
        "page": 1,
        "text": "Photosynthesis is the process by which green plants make glucose using sunlight.",
    },
    {
        "chunk_id": 1,
        "source": "science.txt",
        "page": 2,
        "text": "Water boils at one hundred degrees Celsius at sea level.",
    },
]

TECH_CHUNKS = [
    {
        "chunk_id": 0,
        "source": "tech.txt",
        "page": 1,
        "text": "FastAPI is a Python web framework used to build APIs quickly.",
    },
]


# ---------- fixtures ----------

@pytest.fixture
def upload_dir(tmp_path):
    folder = tmp_path / "uploads"
    folder.mkdir()
    return folder


@pytest.fixture(autouse=True)
def isolated_app(tmp_path, upload_dir, monkeypatch):
    """Runs before EVERY test: empty index, temp folders, fake embeddings.

    monkeypatch puts everything back after each test, so your real index
    and uploaded files are never touched.
    """
    store = vs_module.vector_store
    monkeypatch.setattr(vs_module, "INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr(vs_module, "META_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(vs_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(store, "index", None)
    monkeypatch.setattr(store, "metadata", {})
    monkeypatch.setattr(store, "next_id", 0)
    monkeypatch.setattr(documents_module, "UPLOAD_DIR", upload_dir)
    # The real project default is 0.30. The fake embeddings behave differently,
    # so tests use their own value, independent of your .env file.
    monkeypatch.setattr(rag_module, "MIN_RELEVANCE_SCORE", 0.4)


@pytest.fixture
def loaded_store():
    """An index that already contains two documents."""
    store = vs_module.vector_store
    store.add_chunks("doc_a", SCIENCE_CHUNKS)
    store.add_chunks("doc_b", TECH_CHUNKS)
    return store


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(rag_module, "generate_answer", llm)
    return llm


@pytest.fixture
def client():
    return TestClient(fastapi_app)