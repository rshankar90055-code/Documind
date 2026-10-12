import app.services.vector_store as vs_module


def test_search_returns_most_similar_chunk_first(loaded_store):
    results = loaded_store.search("how do green plants make glucose using sunlight", k=3)

    assert results[0]["source"] == "science.txt"
    assert results[0]["score"] > results[-1]["score"]


def test_search_on_empty_store_returns_empty_list():
    assert vs_module.vector_store.search("anything at all") == []


def test_search_results_include_source_page_and_text(loaded_store):
    top = loaded_store.search("python web framework to build apis", k=1)[0]

    assert top["source"] == "tech.txt"
    assert top["page"] == 1
    assert "FastAPI" in top["text"]


def test_index_is_saved_and_reloaded_from_disk(loaded_store):
    reloaded = vs_module.VectorStore()

    assert reloaded.index is not None
    assert reloaded.index.ntotal == 3
    assert len(reloaded.metadata) == 3
    assert reloaded.next_id == 3


def test_list_documents_counts_chunks(loaded_store):
    docs = {doc["doc_id"]: doc for doc in loaded_store.list_documents()}

    assert docs["doc_a"]["chunks"] == 2
    assert docs["doc_a"]["source"] == "science.txt"
    assert docs["doc_b"]["chunks"] == 1


def test_delete_document_removes_only_its_chunks(loaded_store):
    removed = loaded_store.delete_document("doc_a")

    assert removed == 2
    assert loaded_store.index.ntotal == 1
    results = loaded_store.search("green plants glucose sunlight photosynthesis", k=5)
    assert all(result["doc_id"] == "doc_b" for result in results)


def test_delete_unknown_document_returns_zero(loaded_store):
    assert loaded_store.delete_document("does-not-exist") == 0