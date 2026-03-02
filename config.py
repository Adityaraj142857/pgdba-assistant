import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_DATA_FILE = os.path.join(BASE_DIR, "data/website_data.json")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "faiss_index")
# NEW: Path to save our document chunks for BM25 Keyword Search
DOCS_PATH = os.path.join(DATA_DIR, "docs.pkl") 

# AI Models
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
# NEW: A lightweight, highly accurate reranker
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
LLM_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.0 # Dropped to 0.0 for maximum factual accuracy

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150