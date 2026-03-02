import sys
import os
import json
import pickle
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def ingest_data():
    print(f"🚀 Starting Advanced Ingestion pipeline...")

    with open(config.RAW_DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for entry in data:
        if entry.get("content"):
            clean_text = entry.get("content", "").lower()
            doc = Document(
                page_content=clean_text,
                metadata={"source": entry.get("url", "Unknown")}
            )
            documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    split_docs = text_splitter.split_documents(documents)
    
    # -----------------------------
    # Save chunks for Hybrid Search (BM25)
    # -----------------------------
    print(f"📦 Saving {len(split_docs)} chunks for Keyword Search...")
    with open(config.DOCS_PATH, 'wb') as f:
        pickle.dump(split_docs, f)

    # -----------------------------
    # Embeddings (Hardware Accelerated)
    # -----------------------------
    print(f"🧠 Loading Embedding Model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'mps'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # -----------------------------
    # Batched FAISS Build to prevent M-series memory freeze
    # -----------------------------
    print(f"💾 Building FAISS Index in batches...")
    vector_store = None
    batch_size = 32 # Process 32 chunks at a time

    for i in tqdm(range(0, len(split_docs), batch_size), desc="Embedding", unit="batch"):
        batch = split_docs[i: i + batch_size]
        
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

    vector_store.save_local(config.VECTOR_DB_PATH)
    print(f"\n🎉 Success! Index and Documents saved.")

if __name__ == "__main__":
    ingest_data()