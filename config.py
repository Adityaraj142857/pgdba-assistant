import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ─────────────────────────────────────────────────────────────
# Hardware
# ─────────────────────────────────────────────────────────────
# "mps" → Apple Silicon  |  "cuda" → NVIDIA GPU  |  "cpu" → fallback
if sys.platform == "darwin":
    COMPUTE_DEVICE = "mps"
elif os.system("nvidia-smi > /dev/null 2>&1") == 0:
    COMPUTE_DEVICE = "cuda"
else:
    COMPUTE_DEVICE = "cpu"

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_DATA_FILE   = os.path.join(DATA_DIR, "website_data.json")
VECTOR_DB_PATH  = os.path.join(DATA_DIR, "faiss_index")
DOCS_PATH       = os.path.join(DATA_DIR, "docs.pkl")   # serialised chunks for BM25

# ─────────────────────────────────────────────────────────────
# AI Models
# ─────────────────────────────────────────────────────────────
# Dense embedding model — strong multilingual + English performance
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Lightweight cross-encoder reranker (fast, accurate)
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Gemini LLM
LLM_MODEL   = "gemini-2.5-flash"
TEMPERATURE = 0.0   # 0.0 → maximum factual accuracy

# ─────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# ─────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────
BM25_TOP_K       = 20   # candidates from keyword search
FAISS_TOP_K      = 30   # candidates from semantic search
RERANKER_TOP_N   = 6    # final chunks fed to the LLM (increased from 5 for richer context)