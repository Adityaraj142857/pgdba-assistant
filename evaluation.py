"""
evaluation.py — Offline RAG evaluation using BLEU + Ragas metrics.

Metrics:
  • BLEU             — n-gram overlap between answer and ground truth
  • faithfulness     — Is the answer grounded in the retrieved context?
  • answer_relevancy — Does the answer actually address the question?

Run:
    python evaluation.py
"""

import os
import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    import evaluate                         # HuggingFace evaluate (BLEU)
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.run_config import RunConfig
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

try:
    import config
except ImportError:
    # Fallback minimal config when running standalone
    class config:
        GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
        VECTOR_DB_PATH  = os.path.join("data", "faiss_index")
        EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
        LLM_MODEL       = "gemini-2.5-flash"   # keep in sync with config.py
        TEMPERATURE     = 0.0
        COMPUTE_DEVICE  = "cpu"

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────
print("⚙️  Initialising evaluation components...")

embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},        # always CPU for eval
    encode_kwargs={'normalize_embeddings': True}
)

if not os.path.exists(config.VECTOR_DB_PATH):
    print(f"❌ Vector DB not found at '{config.VECTOR_DB_PATH}'. Run ingestion first.")
    sys.exit(1)

vector_store = FAISS.load_local(
    config.VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.6}
)

llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    google_api_key=config.GOOGLE_API_KEY,
    timeout=90,
    max_retries=3,
)

# ─────────────────────────────────────────────────────────────
# RAG helper (eval uses FAISS-only for speed; production uses hybrid)
# ─────────────────────────────────────────────────────────────
def local_get_response(query: str) -> dict:
    docs = retriever.invoke(query)
    context_text = "\n\n".join(d.page_content for d in docs)

    prompt = f"""You are the PGDBA Assistant. Answer using ONLY the context below.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

    response = llm.invoke(prompt)
    return {"result": response.content, "source_documents": docs}


# ─────────────────────────────────────────────────────────────
# Test dataset
# ─────────────────────────────────────────────────────────────
TEST_DATA = [
    {
        "question": "What is the PGDBA program?",
        "ground_truth": (
            "PGDBA is a Post Graduate Diploma in Business Analytics jointly offered "
            "by ISI Kolkata, IIT Kharagpur, and IIM Calcutta."
        ),
    },
    {
        "question": "What is the duration of the PGDBA program?",
        "ground_truth": "The PGDBA program is a 2-year full-time residential program.",
    },
    {
        "question": "Which institutes jointly offer the PGDBA program?",
        "ground_truth": "IIM Calcutta, ISI Kolkata, and IIT Kharagpur jointly offer PGDBA.",
    },
    {
        "question": "Is there an entrance exam for PGDBA?",
        "ground_truth": (
            "Yes, admission is based on a written test followed by a personal interview."
        ),
    },
    {
        "question": "What is the PGDBA selection process?",
        "ground_truth": (
            "The selection process consists of a computer-based written test "
            "followed by a personal interview."
        ),
    },
    {
        "question": "What is the eligibility criteria for PGDBA?",
        "ground_truth": (
            "Candidates must hold a Bachelor's degree with at least 60% marks "
            "or equivalent CGPA from a recognised university."
        ),
    },
]


# ─────────────────────────────────────────────────────────────
# BLEU
# ─────────────────────────────────────────────────────────────
def calculate_bleu(answers: list, ground_truths: list) -> list:
    try:
        bleu = evaluate.load("bleu")
        scores = []
        for ans, gt in zip(answers, ground_truths):
            if not ans or not ans.strip():
                scores.append(0.0)
                continue
            result = bleu.compute(predictions=[ans], references=[[gt]])
            scores.append(round(result["bleu"], 4))
        return scores
    except Exception as e:
        print(f"⚠️  BLEU error: {e}")
        return [0.0] * len(answers)


# ─────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────
def run_evaluation():
    print("\n🚀 Starting RAG Evaluation Pipeline\n")

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_DATA:
        q = item["question"]
        print(f"   ➤  {q}")
        try:
            res = local_get_response(q)
            questions.append(q)
            answers.append(res["result"])
            contexts.append([d.page_content for d in res["source_documents"]])
            ground_truths.append(item["ground_truth"])
        except Exception as e:
            print(f"   ⚠️  Skipped (error: {e})")

    if not questions:
        print("❌ No questions were processed successfully.")
        return

    df = pd.DataFrame({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # BLEU (fast, offline)
    print("\n📐 Calculating BLEU scores...")
    df["bleu_score"] = calculate_bleu(df["answer"].tolist(), df["ground_truth"].tolist())

    # Ragas (AI metrics, slower — throttled to avoid rate-limits)
    print("\n⚖️  Running Ragas metrics (faithfulness + answer_relevancy)...")
    try:
        dataset = Dataset.from_pandas(df)
        run_config = RunConfig(max_workers=1, timeout=120)

        ragas_results = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
        )

        ragas_df = ragas_results.to_pandas()
        for col in ("faithfulness", "answer_relevancy"):
            if col in ragas_df.columns:
                df[col] = ragas_df[col].round(4)

    except Exception as e:
        print(f"⚠️  Ragas failed (skipping AI metrics): {e}")

    # Save
    output_path = "evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to '{output_path}'")

    print("\n─────────────── 📊 Average Scores ───────────────")
    print(df.mean(numeric_only=True).to_string())
    print("─" * 50)


if __name__ == "__main__":
    run_evaluation()