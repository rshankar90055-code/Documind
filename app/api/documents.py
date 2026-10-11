import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.core.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, TOP_K, UPLOAD_DIR
from app.models.schemas import (
    ChunkPreview,
    DocumentInfo,
    SearchResponse,
    SearchResult,
    UploadResponse,
)
from app.services.ingestion import chunk_pages, parse_document
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    safe_name = Path(file.filename or "").name
    ext = Path(safe_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
        )

    doc_id = uuid.uuid4().hex[:8]
    save_path = UPLOAD_DIR / f"{doc_id}_{safe_name}"
    save_path.write_bytes(contents)

    try:
        pages = parse_document(save_path)
    except Exception:
        logger.exception("Failed to parse %s", safe_name)
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="Could not read this file. It may be corrupted or password protected.",
        )

    if not pages:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="No readable text found. Scanned PDFs need OCR, which is not supported yet.",
        )

    chunks = chunk_pages(pages, source=safe_name)

    try:
        indexed = await run_in_threadpool(vector_store.add_chunks, doc_id, chunks)
    except Exception:
        logger.exception("Failed to index %s", safe_name)
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Indexing failed. Check the server logs (is the embedding model downloaded?).",
        )

    logger.info("Indexed %s: %d pages, %d chunks", safe_name, len(pages), indexed)

    return UploadResponse(
        doc_id=doc_id,
        filename=safe_name,
        pages=len(pages),
        total_chunks=len(chunks),
        indexed_chunks=indexed,
        preview=[
            ChunkPreview(chunk_id=c["chunk_id"], page=c["page"], text=c["text"])
            for c in chunks[:3]
        ],
    )


@router.get("", response_model=list[DocumentInfo])
def list_documents():
    return vector_store.list_documents()


@router.get("/search", response_model=SearchResponse)
def search_documents(
    q: str = Query(..., min_length=3, description="Your question or search text"),
    k: int = Query(TOP_K, ge=1, le=20),
):
    results = vector_store.search(q, k)
    return SearchResponse(query=q, results=[SearchResult(**r) for r in results])


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    removed = vector_store.delete_document(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    for saved_file in UPLOAD_DIR.glob(f"{doc_id}_*"):
        saved_file.unlink(missing_ok=True)
    return {"doc_id": doc_id, "deleted_chunks": removed}