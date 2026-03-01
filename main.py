# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from src.rag_engine import get_response

app = FastAPI(title="PGDBA AI Assistant", description="RAG API for PGDBA Queries")

# -----------------------------
# Enable CORS (for pgdba.ml / WordPress integration)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to "https://pgdba.ml" when deploying to DigitalOcean
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request & Response Schemas
# -----------------------------
class Query(BaseModel):
    question: str

class ResponseModel(BaseModel):
    answer: str
    sources: List[str]

# -----------------------------
# Chat Endpoint
# -----------------------------
# 🔥 FIX: Removed 'async' from def. Since Langchain's invoke() is blocking,
# a standard 'def' tells FastAPI to run this in a separate background thread!
@app.post("/chat", response_model=ResponseModel)
def chat(query: Query):
    try:
        result = get_response(query.question)
        return {"answer": result["answer"], "sources": result["sources"]}
    except Exception as e:
        # Return a proper 500 error instead of a 200 OK with an error message
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "PGDBA AI Assistant is running 🚀"}