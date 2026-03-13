# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from src.rag_engine import get_response

app = FastAPI(
    title="PGDBA AI Assistant",
    description="Hybrid RAG API (BM25 + FAISS + Cross-Encoder Reranker) for PGDBA Queries",
    version="2.0.0",
)

# ─────────────────────────────────────────────────────────────
# CORS  (restrict origin in production)
# ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ← change to ["https://pgdba.ml"] on DigitalOcean
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────
class Query(BaseModel):
    question: str

class ResponseModel(BaseModel):
    answer: str
    sources: List[str]

# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "PGDBA AI Assistant is running 🚀", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}


# Using a standard (non-async) def here is intentional:
# LangChain's .invoke() is blocking, so FastAPI automatically offloads
# this to a thread-pool instead of blocking the event loop.
@app.post("/chat", response_model=ResponseModel)
def chat(query: Query):
    if not query.question or not query.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    try:
        result = get_response(query.question)
        return {"answer": result["answer"], "sources": result["sources"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))