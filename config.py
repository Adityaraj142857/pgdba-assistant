import os
from dotenv import load_dotenv

load_dotenv()

# --- PATH SETUP ---
# Get the absolute path of the project root (where config.py is)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define Data Directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# Input File (From Crawler)
RAW_DATA_FILE = os.path.join(DATA_DIR, "website_data.json") 

# Output Vector Database Path
VECTOR_DB_PATH = os.path.join(DATA_DIR, "faiss_index")

# --- API KEYS ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- MODEL SETTINGS ---
# "BAAI/bge-large-en-v1.5" is excellent for RAG
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5" 
LLM_MODEL = "gemini-2.5-flash" # Optimized for speed/cost. Use "gemini-1.5-pro" for complex reasoning.
TEMPERATURE = 0.3

# --- SPLITTER SETTINGS ---
# Optimized for retaining context
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 200