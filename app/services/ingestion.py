import logging
import re
from pathlib import Path

from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Remove junk characters and extra whitespace."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"page": page_number, "text": text})
    return pages


def parse_docx(path: Path) -> list[dict]:
    doc = Document(str(path))
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    text = clean_text(text)
    return [{"page": 1, "text": text}] if text else []


def parse_txt(path: Path) -> list[dict]:
    text = clean_text(Path(path).read_text(encoding="utf-8", errors="ignore"))
    return [{"page": 1, "text": text}] if text else []


PARSERS = {".pdf": parse_pdf, ".docx": parse_docx, ".txt": parse_txt}


def parse_document(path: Path) -> list[dict]:
    """Return a list of {"page": int, "text": str} for any supported file."""
    ext = Path(path).suffix.lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return parser(path)


def chunk_pages(pages: list[dict], source: str) -> list[dict]:
    """Split each page into overlapping chunks, keeping source and page info."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "source": source,
                    "page": page["page"],
                    "text": piece,
                }
            )
    return chunks