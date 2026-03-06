import sys
import os
import json
import pickle
import re
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    """
    Normalize text for better retrieval
    """
    text = text.lower()  # case insensitive search

    # remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # remove weird characters
    text = re.sub(r"[^\w\s.,!?;:()\-]", "", text)

    return text.strip()


# -----------------------------
# Load JSON data
# -----------------------------
def load_documents():
    print("📂 Loading JSON data...")

    with open(config.RAW_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    for entry in tqdm(data, desc="Processing documents"):

        content = entry.get("content", "")
        url = entry.get("url", "unknown")

        if not content:
            continue

        cleaned = clean_text(content)

        doc = Document(
            page_content=cleaned,
            metadata={
                "source": url,
                "length": len(cleaned)
            }
        )

        documents.append(doc)

    print(f"✅ Loaded {len(documents)} documents")

    return documents


# -----------------------------
# Chunk documents
# -----------------------------
def split_documents(documents):

    print("✂️ Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,      # e.g. 700
        chunk_overlap=config.CHUNK_OVERLAP, # e.g. 120
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " "
        ]
    )

    split_docs = splitter.split_documents(documents)

    print(f"✅ Created {len(split_docs)} chunks")

    return split_docs


# -----------------------------
# Save documents for BM25
# -----------------------------
def save_chunks(split_docs):

    print("💾 Saving chunks for hybrid search (BM25)...")

    os.makedirs(os.path.dirname(config.DOCS_PATH), exist_ok=True)

    with open(config.DOCS_PATH, "wb") as f:
        pickle.dump(split_docs, f)

    print("✅ Chunks saved")


# -----------------------------
# Build FAISS
# -----------------------------
def build_faiss(split_docs):

    print("🧠 Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={
            "device": config.COMPUTE_DEVICE
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print("📦 Building FAISS index...")

    batch_size = 64
    vector_store = None

    for i in tqdm(range(0, len(split_docs), batch_size), desc="Embedding"):

        batch = split_docs[i:i + batch_size]

        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

    os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)

    vector_store.save_local(config.VECTOR_DB_PATH)

    print("✅ FAISS index saved")


# -----------------------------
# Main pipeline
# -----------------------------
def ingest_data():

    print("\n🚀 Starting Advanced Ingestion Pipeline\n")

    documents = load_documents()

    split_docs = split_documents(documents)

    save_chunks(split_docs)

    build_faiss(split_docs)

    print("\n🎉 Ingestion Complete!")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    ingest_data()