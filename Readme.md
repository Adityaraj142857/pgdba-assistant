# PGDBA Assistant 🎓

An AI-powered conversational assistant for the **Post Graduate Diploma in Business Analytics (PGDBA)** — jointly offered by IIT Kharagpur, IIM Calcutta, and ISI Kolkata.

Built with a production-grade **Hybrid RAG** pipeline (BM25 + FAISS + Cross-Encoder Reranker) and served via a FastAPI backend.

---

## 🚀 Architecture

```
Student Query
     │
     ▼
FastAPI  (/chat endpoint)
     │
     ▼
┌────────────────────────────────────────┐
│         Hybrid Retriever               │
│                                        │
│  BM25 (keyword, case-insensitive)      │
│       +                                │
│  FAISS + BGE-large (semantic/dense)    │
│       ↓                                │
│  EnsembleRetriever (40% BM25, 60% FAISS│
└────────────────────────────────────────┘
     │
     ▼
Cross-Encoder Reranker
(ms-marco-MiniLM-L-6-v2 → top 6 chunks)
     │
     ▼
Gemini 2.5 Flash (LLM)
     │
     ▼
Answer + Source URLs
```

---

## 🧠 Key Features

| Feature | Detail |
|---|---|
| **Hybrid Search** | BM25 (exact keyword) + FAISS (semantic) combined via EnsembleRetriever |
| **Case-insensitive BM25** | Custom preprocessor fixes proper-noun mismatches (e.g. `"aditya"` = `"Aditya"`) |
| **MMR Retrieval** | Maximal Marginal Relevance reduces redundant chunks from FAISS |
| **Cross-Encoder Reranking** | Re-scores hybrid candidates; keeps only the top 6 most relevant chunks |
| **Local Embeddings** | BAAI/bge-large-en-v1.5 — no embedding API quota consumed |
| **Gemini 2.5 Flash** | Fast, cost-effective LLM at temperature 0 for factual accuracy |
| **Web Crawler** | Async crawler (aiohttp) with PDF + OCR support for ingestion |
| **FastAPI Backend** | Production-ready REST API with CORS, error handling, and health endpoint |

---

## 📂 Project Structure

```
pgdba-assistant/
│
├── main.py               # FastAPI application (REST API)
├── config.py             # All configuration constants
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed to git)
│
├── src/
│   ├── rag_engine.py     # Full hybrid RAG pipeline
│   └── ingestion.py      # Crawler output → FAISS + BM25 index builder
│
├── data/
│   ├── website_data.json # Raw crawled content
│   ├── faiss_index/      # FAISS vector store
│   └── docs.pkl          # Serialised LangChain Documents for BM25
│
├── crawler.py            # Async web crawler (HTML + PDF + OCR)
├── evaluation.py         # BLEU + Ragas evaluation script
└── endpoints.txt         # Seed URLs for the crawler
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Adityaraj142857/pgdba-assistant.git
cd pgdba-assistant
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Crawl the website (optional — skip if you already have `data/website_data.json`)

```bash
python crawler.py
```

### 6. Build the vector index

Run once after crawling (or whenever new data is available):

```bash
python src/ingestion.py
```

This creates:
- `data/faiss_index/` — dense vector store for semantic search
- `data/docs.pkl` — serialised chunks for BM25 keyword search

### 7. Start the API server

```bash
uvicorn main:app --reload
```

- API → `http://127.0.0.1:8000`
- Swagger docs → `http://127.0.0.1:8000/docs`
- Health check → `http://127.0.0.1:8000/health`

---

## 🔌 API Reference

### `POST /chat`

**Request body:**
```json
{ "question": "What is the PGDBA eligibility criteria?" }
```

**Response:**
```json
{
  "answer": "To be eligible for PGDBA, candidates must ...",
  "sources": [
    "https://pgdba.iitkgp.ac.in/admissions",
    "https://www.iimcal.ac.in/programs/pgdba"
  ]
}
```

### `GET /health`

Returns `{ "status": "ok" }` — useful for uptime monitoring.

---

## 🐛 Known Fixes in v2

| Issue | Root Cause | Fix |
|---|---|---|
| `"aditya"` query returned no results, `"Aditya"` did | `ingestion.py` lowercased stored text; BM25 tokeniser was case-sensitive against original query casing | Removed lowercasing from ingestion; added `preprocess_func=lambda t: t.lower().split()` to `BM25Retriever` |
| Redundant retrieved chunks | FAISS similarity search returns near-duplicate passages | Switched FAISS retriever to **MMR** (`search_type="mmr"`) |
| Evaluation used wrong model | `evaluation.py` hardcoded `gemini-1.5-flash` while `config.py` specifies `gemini-2.5-flash` | `evaluation.py` now reads `config.LLM_MODEL` |

---

## 📊 Evaluation

Run the offline evaluation suite:

```bash
python evaluation.py
```

Produces `evaluation_results.csv` with per-question scores:

| Metric | Description |
|---|---|
| `bleu_score` | n-gram overlap with ground truth (fast, offline) |
| `faithfulness` | Is the answer grounded in the retrieved context? (Ragas) |
| `answer_relevancy` | Does the answer address the question? (Ragas) |

---

## 🚢 Production Deployment (DigitalOcean)

1. Replace `allow_origins=["*"]` in `main.py` with `["https://pgdba.ml"]`
2. Set `GOOGLE_API_KEY` as an environment variable on the server
3. Run with gunicorn + uvicorn workers:

```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker --workers 2 --bind 0.0.0.0:8000
```

---

## 🛠️ Tech Stack

- **LangChain** — RAG orchestration
- **FAISS** — Dense vector similarity search
- **BGE-large-en-v1.5** — Local sentence embeddings (HuggingFace)
- **BM25** — Keyword retrieval (`rank-bm25`)
- **ms-marco-MiniLM-L-6-v2** — Cross-encoder reranker
- **Gemini 2.5 Flash** — Response generation
- **FastAPI** — REST API server
- **trafilatura** — Web content extraction
- **aiohttp** — Async web crawling