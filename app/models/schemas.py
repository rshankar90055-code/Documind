from pydantic import BaseModel


class ChunkPreview(BaseModel):
    chunk_id: int
    page: int
    text: str


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    total_chunks: int
    preview: list[ChunkPreview]