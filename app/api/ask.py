import logging

from fastapi import APIRouter, HTTPException

from app.core.config import TOP_K
from app.models.schemas import AskRequest, AskResponse
from app.services.llm import LLMError
from app.services.rag import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ask"])


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = answer_question(request.question, request.top_k or TOP_K)
    except LLMError as e:
        logger.error("LLM failure: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
    return AskResponse(question=request.question, **result)