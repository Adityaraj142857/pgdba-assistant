import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# NEW: Hybrid & Reranking Imports
from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

# 1️⃣ Load Embeddings & FAISS (Semantic Search)
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={'device': config.COMPUTE_DEVICE}, # Hardware acceleration
    encode_kwargs={'normalize_embeddings': True}
)
vector_store = FAISS.load_local(config.VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 30})

# 2️⃣ Load Saved Docs & Initialize BM25 (Keyword Search)
with open(config.DOCS_PATH, 'rb') as f:
    raw_docs = pickle.load(f)
bm25_retriever = BM25Retriever.from_documents(raw_docs)
bm25_retriever.k = 15

# 3️⃣ Hybrid Search (Blend 60% Semantic, 40% Keyword)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)

# 4️⃣ Cross-Encoder Reranker
# This model grades the 30 chunks from Hybrid Search and picks the top 5
cross_encoder = HuggingFaceCrossEncoder(
    model_name=config.RERANKER_MODEL, 
    model_kwargs={'device': config.COMPUTE_DEVICE}
)
compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=ensemble_retriever
)

# 5️⃣ LLM & Prompt
llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    google_api_key=config.GOOGLE_API_KEY
)

PROMPT_TEMPLATE = """
You are the official PGDBA Assistant. Answer the student's question accurately using ONLY the provided context.

INSTRUCTIONS:
- If the exact answer isn't in the context, but related information is, provide the related information and state what is missing.
- If the context does not contain the answer at all, state: "I don't have that information."
- Use bullet points for requirements, eligibility, or lists.

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(compression_retriever, document_chain) # Use the Reranker!

def get_response(query: str):
    normalized_query = query.lower()
    print(f"\n🔍 RUNNING HYBRID SEARCH & RERANKING FOR: '{normalized_query}'...")
    
    result = qa_chain.invoke({"input": normalized_query})
    context_docs = result.get("context", [])

    sources = list(set([doc.metadata.get("source", "Unknown") for doc in context_docs]))
    
    print(f"✅ Selected {len(context_docs)} highly relevant chunks for generation.")

    return {
        "answer": result.get("answer", "No answer found."),
        "sources": sources
    }

if __name__ == "__main__":
    response = get_response("What is the exact minimum CGPA required for an engineering graduate to apply?")
    print(f"\n🤖 Answer:\n{response['answer']}")