import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, UPLOAD_DIR
from app.models.schemas import ChunkPreview, UploadResponse
from app.services.ingestion import chunk_pages, parse_document

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
    logger.info("Uploaded %s: %d pages, %d chunks", safe_name, len(pages), len(chunks))

    return UploadResponse(
        doc_id=doc_id,
        filename=safe_name,
        pages=len(pages),
        total_chunks=len(chunks),
        preview=[
            ChunkPreview(chunk_id=c["chunk_id"], page=c["page"], text=c["text"])
            for c in chunks[:3]
        ],
    )