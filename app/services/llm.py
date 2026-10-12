import logging

import httpx

from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the language model cannot produce an answer."""


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    if LLM_PROVIDER == "gemini":
        return _call_gemini(system_prompt, user_prompt)
    if LLM_PROVIDER == "ollama":
        return _call_ollama(system_prompt, user_prompt)
    raise LLMError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'gemini' or 'ollama'.")


def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    try:
        response = httpx.post(
            url,
            json=payload,
            headers={"x-goog-api-key": GEMINI_API_KEY},
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("Gemini error %s: %s", e.response.status_code, e.response.text[:300])
        raise LLMError(
            f"Gemini returned an error ({e.response.status_code}). "
            "Check your API key, model name and rate limits."
        ) from e
    except httpx.HTTPError as e:
        logger.error("Could not reach Gemini: %s", e)
        raise LLMError("Could not reach the Gemini API.") from e

    try:
        parts = response.json()["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts).strip()
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError("Gemini returned an empty or blocked response.") from e


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("Ollama error %s: %s", e.response.status_code, e.response.text[:300])
        raise LLMError(
            f"Ollama returned an error ({e.response.status_code}). "
            f"Did you run 'ollama pull {OLLAMA_MODEL}'?"
        ) from e
    except httpx.HTTPError as e:
        logger.error("Could not reach Ollama: %s", e)
        raise LLMError("Could not reach Ollama. Is it running?") from e

    try:
        return response.json()["message"]["content"].strip()
    except (KeyError, TypeError) as e:
        raise LLMError("Ollama returned an unexpected response.") from e