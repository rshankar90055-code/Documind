import logging

from fastapi import FastAPI

from app.api import ask,documents, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="DocuMind",
    description="RAG Document Q&A System",
    version="0.3.0",
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(ask.router)


@app.get("/")
def root():
    return {"message": "Welcome to DocuMind"}