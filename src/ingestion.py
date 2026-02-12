# src/rag_engine.py

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config

# Load environment variables
load_dotenv()

# 1️⃣ Load Embeddings (MUST match ingestion model)
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL
)

# 2️⃣ Load Vector Store
vector_store = FAISS.load_local(
    config.VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# 3️⃣ Create Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 4️⃣ Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


def get_response(query: str) -> str:
    """
    Main RAG pipeline:
    1. Retrieve relevant documents
    2. Inject into prompt
    3. Ask Gemini to answer
    """

    # 🔎 Retrieve context
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    # 🧠 Create prompt
    prompt = f"""
You are PGDBA Assistant — an intelligent, professional student guidance bot.

Use ONLY the context below to answer the question.
If answer is not in context, say:
"I don't have that information. Please check the official PGDBA website."

Context:
{context}

Question:
{query}

Answer clearly and helpfully:
"""

    # 🤖 Generate answer
    response = llm.invoke(prompt)

    return response.content
