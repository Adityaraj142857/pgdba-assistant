import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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
    raise e

# 3️⃣ Retriever (🔥 FIX: Switched to similarity and increased K)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}  # Grabbing 8 chunks gives Gemini massive context
)

# 4️⃣ LLM
llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    google_api_key=config.GOOGLE_API_KEY
)

# 5️⃣ Prompt Template
PROMPT_TEMPLATE = """
You are the PGDBA Assistant. Your goal is to answer student questions comprehensively using the context below.

INSTRUCTIONS:
- Use the provided context to answer the question. 
- If the exact answer isn't in the context, but related information is, provide the related information and state clearly what is missing.
- Be clear, professional, and well-structured.
- Use bullet points for lists, requirements, or steps.

CONTEXT:
{context}

QUESTION:
{input}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)


def get_response(query: str):
    # 🔥 FIX: Diagnostic Print - See exactly what chunks are being retrieved
    print(f"\n{'=' * 50}")
    print(f"🔍 RETRIEVING CONTEXT FOR: '{query}'")
    print(f"{'=' * 50}")

    docs = retriever.invoke(query)
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        # Print the first 250 characters of each chunk
        snippet = doc.page_content.replace("\n", " ")[:250]
        print(f"[{i + 1}] Source: {source}\nSnippet: {snippet}...\n")
    print(f"{'=' * 50}\n")

    # Generate Response
    result = qa_chain.invoke({"input": query})
    context_docs = result.get("context", [])

    # Extract unique sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in context_docs]))

    return {
        "answer": result.get("answer", "No answer found."),
        "sources": sources
    }


if __name__ == "__main__":
    print("Testing RAG Engine...")
    response = get_response("What is the eligibility criteria for PGDBA?")
    print(f"🤖 Answer:\n{response['answer']}")
    print(f"\n🔗 Sources: {response['sources']}")