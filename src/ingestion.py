"""
ingestion.py — Build the FAISS vector index and BM25 docs.pkl from crawled data.

Run this once after crawler.py has produced data/website_data.json:
    python src/ingestion.py

Outputs:
    data/faiss_index/   — dense vector store for semantic search
    data/docs.pkl       — serialised LangChain Documents for BM25 keyword search
"""

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


# ─────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Light normalisation for better retrieval quality.

    What we do:
      • Collapse excessive blank lines and horizontal whitespace.
      • Strip non-printable / control characters.
      • Drop lines that are pure noise (only digits/punctuation, very short).

    What we deliberately do NOT do:
      • Lowercase — lowercasing stored content breaks BM25 for proper nouns
        like "Aditya", "PGDBA", "IIT".  Case-insensitivity is handled at
        query time inside rag_engine.py via a BM25 preprocessor function.
      • Aggressive punctuation removal — embeddings work better on natural text.
    """
    # Collapse 3+ consecutive newlines → double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse tabs and multiple spaces → single space
    text = re.sub(r"[ \t]+", " ", text)
    # Remove non-printable / control characters (preserve \t \n \r and printable range)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    # Drop lines that are pure noise: only digits/punctuation, or very short
    lines = [
        line.strip() for line in text.split("\n")
        if len(line.strip()) > 3 and not re.fullmatch(r"[\d\W]+", line.strip())
    ]
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────
# Load JSON data
# ─────────────────────────────────────────────────────────────
def load_documents() -> list[Document]:
    # ── Guard: check the raw data file exists ─────────────────
    if not os.path.exists(config.RAW_DATA_FILE):
        raise FileNotFoundError(
            f"\n❌ Raw data file not found: {config.RAW_DATA_FILE}\n\n"
            "Run the crawler first to generate it:\n"
            "    python crawler.py\n"
        )

    print("📂 Loading JSON data...")
    with open(config.RAW_DATA_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"\n❌ {config.RAW_DATA_FILE} is not valid JSON: {e}\n"
                "Re-run the crawler to regenerate it:\n"
                "    python crawler.py\n"
            )

    # ── Guard: check the file isn't empty ─────────────────────
    if not data:
        raise ValueError(
            f"\n❌ {config.RAW_DATA_FILE} is empty (0 documents).\n\n"
            "The crawler ran but extracted nothing.  Possible causes:\n"
            "  • The target websites blocked the crawler.\n"
            "  • All pages were shorter than the minimum content length.\n"
            "Re-run the crawler and check its output:\n"
            "    python crawler.py\n"
        )

    documents = []
    skipped   = 0

    for entry in tqdm(data, desc="Processing documents"):
        content  = entry.get("content", "").strip()
        url      = entry.get("url", "unknown")
        doc_type = entry.get("type", "html")

        if not content:
            skipped += 1
            continue

        cleaned = clean_text(content)

        if len(cleaned) < 100:
            skipped += 1
            continue

        documents.append(Document(
            page_content=cleaned,
            metadata={
                "source": url,
                "type":   doc_type,
                "length": len(cleaned),
            },
        ))

    print(f"✅ Loaded {len(documents)} documents  ({skipped} skipped — empty or too short)")

    # ── Guard: check we actually got documents ─────────────────
    if not documents:
        raise ValueError(
            "\n❌ No usable documents after cleaning.\n\n"
            "All entries in website_data.json were either empty or too short.\n"
            "Re-run the crawler with a lower MIN_CONTENT_LENGTH, or check\n"
            "that the target sites are returning content:\n"
            "    python crawler.py\n"
        )

    return documents


# ─────────────────────────────────────────────────────────────
# Chunk documents
# ─────────────────────────────────────────────────────────────
def split_documents(documents: list[Document]) -> list[Document]:
    print("✂️  Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )

    split_docs = splitter.split_documents(documents)

    # Drop chunks too short to carry useful signal
    split_docs = [d for d in split_docs if len(d.page_content.strip()) > 50]

    print(f"✅ Created {len(split_docs)} chunks from {len(documents)} documents")
    return split_docs


# ─────────────────────────────────────────────────────────────
# Save docs.pkl for BM25
# ─────────────────────────────────────────────────────────────
def save_chunks(split_docs: list[Document]) -> None:
    print("💾 Saving chunks for BM25 keyword search...")

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(config.DOCS_PATH), exist_ok=True)

    with open(config.DOCS_PATH, "wb") as f:
        pickle.dump(split_docs, f)

    size_mb = os.path.getsize(config.DOCS_PATH) / 1024 / 1024
    print(f"✅ Saved {len(split_docs)} chunks → {config.DOCS_PATH}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────
# Build FAISS vector index
# ─────────────────────────────────────────────────────────────
def build_faiss(split_docs: list[Document]) -> None:
    print("🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.COMPUTE_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"📦 Building FAISS index (batch size 64, device={config.COMPUTE_DEVICE})...")
    vector_store = None
    batch_size   = 64

    for i in tqdm(range(0, len(split_docs), batch_size), desc="Embedding batches"):
        batch = split_docs[i : i + batch_size]
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

    os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
    vector_store.save_local(config.VECTOR_DB_PATH)
    print(f"✅ FAISS index saved → {config.VECTOR_DB_PATH}")


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def ingest_data() -> None:
    print("\n🚀 PGDBA Ingestion Pipeline\n" + "─" * 40)

    try:
        documents  = load_documents()
        split_docs = split_documents(documents)
        save_chunks(split_docs)
        build_faiss(split_docs)
    except (FileNotFoundError, ValueError) as e:
        # Print the clean error message we wrote and exit — no traceback noise
        print(e)
        sys.exit(1)

    print("\n" + "─" * 40)
    print("🎉 Ingestion complete!")
    print(f"   Vector index : {config.VECTOR_DB_PATH}")
    print(f"   BM25 chunks  : {config.DOCS_PATH}")
    print("\nYou can now start the API server:")
    print("    uvicorn main:app --reload\n")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingest_data()