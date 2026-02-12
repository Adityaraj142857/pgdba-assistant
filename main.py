# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.rag_engine import get_response


app = FastAPI(title="PGDBA AI Assistant")

# -----------------------------
# Enable CORS (for WordPress)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request Schema
# -----------------------------
class Query(BaseModel):
    question: str


# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/chat")
async def chat(query: Query):
    try:
        answer = get_response(query.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "PGDBA AI Assistant is running 🚀"}
