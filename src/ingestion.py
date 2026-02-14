import sys
import os
import json
from tqdm import tqdm  # ✅ Import for progress bar

# --- IMPORT FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def ingest_data():
    print(f"🚀 Starting Ingestion pipeline...")
    
    # 1. Load Data
    if not os.path.exists(config.RAW_DATA_FILE):
        raise FileNotFoundError(f"❌ File not found: {config.RAW_DATA_FILE}")

    print(f"📂 Loading data from {config.RAW_DATA_FILE}...")
    with open(config.RAW_DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for entry in data:
        if entry.get("content"):
            doc = Document(
                page_content=entry.get("content", ""),
                metadata={"source": entry.get("url", "Unknown")}
            )
            documents.append(doc)

    print(f"✅ Loaded {len(documents)} source documents.")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    split_docs = text_splitter.split_documents(documents)
    total_chunks = len(split_docs)
    print(f"✂️  Split into {total_chunks} chunks.")

    if total_chunks == 0:
        print("⚠️ No documents to ingest. Exiting.")
        return

    # 3. Initialize Embeddings
    print(f"🧠 Loading Embedding Model: {config.EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. Create Vector Store with Progress Bar
    print(f"💾 Building FAISS Index (This may take a while)...")
    
    vector_store = None
    batch_size = 32  # Process 32 chunks at a time to update the progress bar often

    # 🔄 The Loop: This adds the progress bar
    for i in tqdm(range(0, total_chunks, batch_size), desc="Embedding Chunks", unit="batch"):
        batch = split_docs[i : i + batch_size]
        
        if vector_store is None:
            # Create the store with the first batch
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            # Add subsequent batches to the existing store
            vector_store.add_documents(batch)

    # 5. Save
    print(f"\n💾 Saving to {config.VECTOR_DB_PATH}...")
    vector_store.save_local(config.VECTOR_DB_PATH)
    print(f"🎉 Success! Index saved.")

if __name__ == "__main__":
    ingest_data()