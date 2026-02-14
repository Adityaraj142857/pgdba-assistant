import sys
import os

# --- IMPORT FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ CORRECT IMPORTS FOR LANGCHAIN 1.x
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# 1️⃣ Load Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2️⃣ Load Vector DB
try:
    vector_store = FAISS.load_local(
        config.VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"❌ Error loading vector DB from {config.VECTOR_DB_PATH}")
    print("💡 Suggestion: Run 'python src/ingestion.py' first.")
    raise e

# 3️⃣ Retriever (MMR Optimized)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
)

# 4️⃣ LLM
llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    google_api_key=config.GOOGLE_API_KEY
)

# 5️⃣ Prompt Template
PROMPT_TEMPLATE = """
You are the PGDBA Assistant. Your goal is to answer student questions accurately using the context below.

INSTRUCTIONS:
- Use ONLY the provided context. If the answer is missing, state "I don't have that information."
- Be concise but professional.
- If the context lists steps or requirements, use bullet points.

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# 6️⃣ Create Modern Retrieval Chain (LangChain 1.x style)
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)


def get_response(query: str):
    result = qa_chain.invoke({"input": query})

    context_docs = result.get("context", [])

    if context_docs:
        sources = [context_docs[0].metadata.get("source", "Unknown")]
    else:
        sources = []

    return {
        "answer": result.get("answer", "No answer found."),
        "sources": sources
    }


if __name__ == "__main__":
    print("Testing RAG Engine...")
    response = get_response("What is the eligibility for PGDBA?")
    print(f"\n🤖 Answer: {response['answer']}")
    print(f"🔗 Sources: {response['sources']}")
