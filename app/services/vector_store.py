import json
import logging
import threading

import faiss
import numpy as np

from app.core.config import INDEX_DIR, TOP_K
from app.services.embeddings import embed_texts

logger = logging.getLogger(__name__)

INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "metadata.json"


class VectorStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.index = None
        self.metadata: dict[int, dict] = {}
        self.next_id = 0
        self._load()

    def _load(self):
        if INDEX_PATH.exists() and META_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            raw = json.loads(META_PATH.read_text(encoding="utf-8"))
            self.metadata = {int(k): v for k, v in raw["chunks"].items()}
            self.next_id = raw["next_id"]
            logger.info("Loaded FAISS index with %d chunks", self.index.ntotal)

    def _save(self):
        faiss.write_index(self.index, str(INDEX_PATH))
        META_PATH.write_text(
            json.dumps({"next_id": self.next_id, "chunks": self.metadata}),
            encoding="utf-8",
        )

    def add_chunks(self, doc_id: str, chunks: list[dict]) -> int:
        # Embedding is slow, so do it before taking the lock.
        vectors = embed_texts([c["text"] for c in chunks])
        with self._lock:
            if self.index is None:
                self.index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
            ids = np.arange(self.next_id, self.next_id + len(chunks), dtype="int64")
            self.index.add_with_ids(vectors, ids)
            for faiss_id, chunk in zip(ids, chunks):
                self.metadata[int(faiss_id)] = {
                    "doc_id": doc_id,
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                }
            self.next_id += len(chunks)
            self._save()
        return len(chunks)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vector = embed_texts([query])
        with self._lock:
            scores, ids = self.index.search(query_vector, min(k, self.index.ntotal))
        results = []
        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id == -1:
                continue
            meta = self.metadata.get(int(faiss_id))
            if meta:
                results.append({**meta, "score": float(score)})
        return results

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for meta in self.metadata.values():
            doc = docs.setdefault(
                meta["doc_id"],
                {"doc_id": meta["doc_id"], "source": meta["source"], "chunks": 0},
            )
            doc["chunks"] += 1
        return list(docs.values())

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            ids = [i for i, m in self.metadata.items() if m["doc_id"] == doc_id]
            if not ids:
                return 0
            self.index.remove_ids(np.array(ids, dtype="int64"))
            for i in ids:
                del self.metadata[i]
            self._save()
        return len(ids)


# One shared store for the whole app.
vector_store = VectorStore()