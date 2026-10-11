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
    indexed_chunks: int
    preview: list[ChunkPreview]


class SearchResult(BaseModel):
    source: str
    page: int
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    chunks: int