import os
from dotenv import load_dotenv

load_dotenv()

# API Keys (Still needed for Gemini LLM)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
DATA_DIR = "data"
RAW_DATA_FILE = os.path.join(DATA_DIR, "website_data.txt")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "faiss_index_hf") # Renamed to avoid conflict

# Model Settings
# "all-MiniLM-L6-v2" is the industry standard for fast, local CPU embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
LLM_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.3