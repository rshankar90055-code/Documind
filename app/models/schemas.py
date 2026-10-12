from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Source(BaseModel):
    id: int
    source: str
    page: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool
    sources: list[Source]
    
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