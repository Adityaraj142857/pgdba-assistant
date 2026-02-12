# src/rag_engine.py

import os
from dotenv import load_dotenv


load_dotenv()



from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config

# Load environment variables
load_dotenv()
# Load embeddings (must match ingestion model)
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL
)

# Load FAISS vector store
vector_store = FAISS.load_local(
    config.VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


def get_response(query: str) -> str:
    # Retrieve relevant docs
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are PGDBA Assistant — an intelligent and professional student guidance bot.

Use ONLY the context below to answer.
If the answer is not available, say:
"I don't have that information. Please check the official PGDBA website."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content
