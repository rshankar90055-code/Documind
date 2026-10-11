from fastapi import FastAPI

from app.api import health

app = FastAPI(
    title="DocuMind",
    description="RAG Document Q&A System",
    version="0.1.0",
)

app.include_router(health.router)


@app.get("/")
def root():
    return {"message": "Welcome to DocuMind"}