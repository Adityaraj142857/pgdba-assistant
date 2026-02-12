# PGDBA Assistant 🎓

AI-powered conversational assistant for PGDBA (Post Graduate Diploma in Business Analytics).

Built using:
- FastAPI
- LangChain (RAG Architecture)
- FAISS Vector Database
- HuggingFace Embeddings (MiniLM)
- Google Gemini 2.5 Flash

This assistant retrieves official PGDBA information and generates accurate, context-aware responses.

---

## 🚀 Architecture

Frontend (PGDBA.ml Website)
        ↓
FastAPI Backend (RAG API)
        ↓
Retriever (FAISS)
        ↓
HuggingFace Embeddings
        ↓
Gemini 2.5 Flash (LLM)

---

## 🧠 Key Features

- Retrieval Augmented Generation (RAG)
- Local embeddings (no embedding API limits)
- Gemini 2.5 Flash for fast responses
- FAISS for efficient similarity search
- Modular architecture
- Production-ready FastAPI backend

---

## 📂 Project Structure

# PGDBA Assistant 🎓

AI-powered conversational assistant for PGDBA (Post Graduate Diploma in Business Analytics).

Built using:
- FastAPI
- LangChain (RAG Architecture)
- FAISS Vector Database
- HuggingFace Embeddings (MiniLM)
- Google Gemini 2.5 Flash

This assistant retrieves official PGDBA information and generates accurate, context-aware responses.

---

## 🚀 Architecture

Frontend (PGDBA.ml Website)
        ↓
FastAPI Backend (RAG API)
        ↓
Retriever (FAISS)
        ↓
HuggingFace Embeddings
        ↓
Gemini 2.5 Flash (LLM)

---

## 🧠 Key Features

- Retrieval Augmented Generation (RAG)
- Local embeddings (no embedding API limits)
- Gemini 2.5 Flash for fast responses
- FAISS for efficient similarity search
- Modular architecture
- Production-ready FastAPI backend

---

## 📂 Project Structure

pgdba-assistant/
│
├── main.py # FastAPI app
├── config.py # Config variables
├── .env # Environment variables (not committed)
│
├── src/
│ ├── rag_engine.py # RAG pipeline logic
│ └── ingestion.py # Vector store builder
│
├── data/
│ ├── raw/ # Source text files
│ └── faiss_index/ # Saved vector store
│
└── requirements.txt



---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

git clone https://github.com/Adityaraj142857/pgdba-assistant.git
cd pgdba-assistant


---

### 2️⃣ Create Virtual Environment

python3 -m venv venv
source venv/bin/activate


---

### 3️⃣ Install Dependencies

pip install -r requirements.txt


---

### 4️⃣ Add Environment Variables

Create a `.env` file in root:

GOOGLE_API_KEY=your_gemini_api_key_here


---

### 5️⃣ Build Vector Store

Run ingestion script once:

python src/ingestion.py

This creates the FAISS index.

---

### 6️⃣ Start API Server

uvicorn main:app --reload

API will run at:

http://127.0.0.1:8000

Swagger docs available at:

http://127.0.0.1:8000/docs

---

## 🔌 API Usage

### POST `/chat`

**Request:**

```json
{
  "question": "What is PGDBA eligibility?"
}
Response:
{
  "answer": "..."
}