FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

# Run the app as a normal user instead of root
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# 1) CPU-only PyTorch first. The default Linux build bundles GPU libraries
#    and is about 2 GB bigger, which we don't need.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) The other dependencies. This layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) The application code (changes most often, so it comes last)
COPY app ./app
COPY frontend ./frontend

# 4) A folder for uploads and the FAISS index, owned by the normal user
RUN mkdir -p /app/data && chown -R appuser:appuser /app /home/appuser

USER appuser

# 5) Download the embedding model now, so containers start fast and work offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

# Docker calls your /health endpoint to check the app is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

# 0.0.0.0 means "accept connections from outside the container"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]