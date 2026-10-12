import pytest

import app.services.rag as rag
from app.services.llm import LLMError

PLANT_QUESTION = "How do green plants make glucose using sunlight?"


def test_no_documents_uploaded_never_calls_the_llm(fake_llm):
    result = rag.answer_question(PLANT_QUESTION)

    assert result["found"] is False
    assert result["answer"] == rag.NO_DOCUMENTS_MESSAGE
    assert fake_llm.calls == []


def test_relevant_question_returns_answer_with_sources(loaded_store, fake_llm):
    result = rag.answer_question(PLANT_QUESTION)

    assert result["found"] is True
    assert result["answer"] == fake_llm.reply
    assert result["sources"][0]["id"] == 1
    assert result["sources"][0]["source"] == "science.txt"
    assert result["sources"][0]["page"] == 1

    system_prompt, user_prompt = fake_llm.calls[0]
    assert "ONLY the provided context" in system_prompt
    assert "Photosynthesis is the process" in user_prompt
    assert f"Question: {PLANT_QUESTION}" in user_prompt


def test_unrelated_question_is_stopped_before_the_llm(loaded_store, fake_llm):
    result = rag.answer_question("Capital city of France")

    assert result["found"] is False
    assert result["answer"] == rag.NOT_FOUND_MESSAGE
    assert result["sources"] == []
    assert fake_llm.calls == []


def test_model_saying_not_found_returns_no_sources(loaded_store, fake_llm):
    fake_llm.reply = "NOT_FOUND"

    result = rag.answer_question(PLANT_QUESTION)

    assert result["found"] is False
    assert result["answer"] == rag.NOT_FOUND_MESSAGE
    assert result["sources"] == []
    assert len(fake_llm.calls) == 1


def test_llm_failure_is_not_swallowed(loaded_store, monkeypatch):
    def broken_llm(system_prompt, user_prompt):
        raise LLMError("boom")

    monkeypatch.setattr(rag, "generate_answer", broken_llm)

    with pytest.raises(LLMError):
        rag.answer_question(PLANT_QUESTION)


def test_build_context_numbers_each_source():
    hits = [
        {"source": "a.pdf", "page": 3, "text": "Alpha"},
        {"source": "b.txt", "page": 1, "text": "Beta"},
    ]

    context = rag.build_context(hits)

    assert "[1] (source: a.pdf, page 3)\nAlpha" in context
    assert "[2] (source: b.txt, page 1)\nBeta" in context