"""
rag_engine.py — Hybrid RAG pipeline (BM25 + FAISS + Cross-Encoder Reranker).

All heavy objects (embeddings, FAISS index, BM25, LLM) are built ONCE on the
first call to get_response() — NOT at import time.  This means:

  • Importing this module never crashes, even before ingestion has run.
  • main.py starts up instantly regardless of whether the index exists.
  • A clear, actionable error is raised only when a query is actually made.
"""

import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Lazy pipeline — built on first call, reused for all subsequent calls
# ─────────────────────────────────────────────────────────────
_pipeline = None   # holds the initialised qa_chain


def _check_assets():
    """Raise a clear, actionable error if required data files are missing."""
    missing = []
    if not os.path.exists(config.DOCS_PATH):
        missing.append(f"  • docs.pkl  not found at: {config.DOCS_PATH}")
    if not os.path.exists(config.VECTOR_DB_PATH):
        missing.append(f"  • faiss_index not found at: {config.VECTOR_DB_PATH}")
    if missing:
        raise FileNotFoundError(
            "\n\n❌ RAG index files are missing. Run ingestion first:\n\n"
            "    python src/ingestion.py\n\n"
            "Missing files:\n" + "\n".join(missing) + "\n"
        )


def _build_pipeline():
    """
    Build the full hybrid RAG pipeline and return the qa_chain.
    Called exactly once — result is cached in module-level _pipeline.
    """
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings

    print("🔧 Initialising RAG pipeline (first request only)...")

    # ── 1. Embeddings ──────────────────────────────────────────
    print("  [1/5] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.COMPUTE_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── 2. FAISS semantic retriever ────────────────────────────
    print("  [2/5] Loading FAISS index...")
    vector_store = FAISS.load_local(
        config.VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    faiss_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.FAISS_TOP_K,
            "fetch_k": config.FAISS_TOP_K * 2,
            "lambda_mult": 0.6,   # diversity vs relevance balance
        },
    )

    # ── 3. BM25 keyword retriever ──────────────────────────────
    # preprocess_func makes BM25 case-insensitive so "aditya" == "Aditya".
    # Stored documents are NOT lowercased — only the tokenisation step is.
    print("  [3/5] Loading BM25 retriever...")
    with open(config.DOCS_PATH, "rb") as f:
        raw_docs = pickle.load(f)

    if not raw_docs:
        raise ValueError(
            "docs.pkl is empty — ingestion produced no chunks.\n"
            "Check that data/website_data.json is populated and re-run:\n"
            "    python src/ingestion.py"
        )

    def _bm25_preprocess(text: str) -> list[str]:
        """Lowercase tokeniser — makes BM25 case-insensitive."""
        return text.lower().split()

    bm25_retriever = BM25Retriever.from_documents(
        raw_docs,
        preprocess_func=_bm25_preprocess,
    )
    bm25_retriever.k = config.BM25_TOP_K

    # ── 4. Ensemble → Cross-Encoder reranker ──────────────────
    print("  [4/5] Building hybrid retriever + reranker...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6],   # 40 % keyword, 60 % semantic
    )

    cross_encoder = HuggingFaceCrossEncoder(
        model_name=config.RERANKER_MODEL,
        model_kwargs={"device": config.COMPUTE_DEVICE},
    )
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=config.RERANKER_TOP_N)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    # ── 5. LLM + prompt chain ─────────────────────────────────
    print("  [5/5] Connecting to Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(compression_retriever, document_chain)

    print("✅ RAG pipeline ready.\n")
    return qa_chain


# ─────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are the official PGDBA Assistant — a helpful, accurate, and professional guide \
for the Post Graduate Diploma in Business Analytics program jointly offered by \
IIT Kharagpur, IIM Calcutta, and ISI Kolkata.

Answer the student's question using ONLY the information in the CONTEXT below. \
Do not rely on prior knowledge outside this context.

─────────────────────────────────────────────────────────────────
RULES
─────────────────────────────────────────────────────────────────
1. Answer exclusively from the CONTEXT. Be thorough and precise.
2. If the CONTEXT has related but not exact information, share it and note what is missing.
3. If the CONTEXT has NO relevant information at all, respond with:
   "I don't have that information. Please check the official website: https://pgdba.iitkgp.ac.in"
4. Never fabricate facts, dates, names, or figures.
5. Use bullet points for lists (eligibility, requirements, deadlines, steps).
6. Preserve exact spelling and casing of people's names from the context.
7. Keep your tone friendly and professional.
─────────────────────────────────────────────────────────────────

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:
"""


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def get_response(query: str) -> dict:
    """
    Run the full RAG pipeline for a given query.

    Initialises all heavy components on the very first call,
    then reuses them for every subsequent call (no repeated loading).

    Raises FileNotFoundError with a clear message if ingestion
    has not been run yet.
    """
    global _pipeline

    # Lazy init — runs only once
    if _pipeline is None:
        _check_assets()
        _pipeline = _build_pipeline()

    print(f"🔍 Query: '{query}'")

    result       = _pipeline.invoke({"input": query})
    context_docs = result.get("context", [])

    # Deduplicate sources while preserving order
    sources = list(dict.fromkeys(
        doc.metadata.get("source", "Unknown")
        for doc in context_docs
    ))

    print(f"✅ Used {len(context_docs)} chunks | {len(sources)} source(s)")

    return {
        "answer":  result.get("answer", "No answer found."),
        "sources": sources,
    }


# ─────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is the PGDBA selection process?",
        "Can you share the interview experience of Aditya?",
        "can you share the interview experience of aditya?",  # same — BM25 is case-insensitive
    ]
    for q in test_queries:
        try:
            resp = get_response(q)
            print(f"\n🤖 Answer:\n{resp['answer']}")
            print(f"📎 Sources: {resp['sources']}")
            print("─" * 60)
        except FileNotFoundError as e:
            print(e)
            break