import os
import sys
import pandas as pd
import warnings
import asyncio

# --- 1. SETUP & IMPORTS ---
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    import evaluate  # BLEU
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.run_config import RunConfig # Import config to control speed

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    sys.exit(1)

# Import Config
try:
    import config
except ImportError:
    class Config:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        DATA_DIR = "data"
        VECTOR_DB_PATH = os.path.join("data", "faiss_index")
        EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
        LLM_MODEL = "gemini-1.5-flash"
        TEMPERATURE = 0.3
    config = Config()

load_dotenv()

# --- 2. INITIALIZE COMPONENTS ---
print("⚙️  Initializing RAG components...")

embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

try:
    if os.path.exists(config.VECTOR_DB_PATH):
        vector_store = FAISS.load_local(
            config.VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
        )
    else:
        print(f"❌ Vector DB not found at {config.VECTOR_DB_PATH}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error loading FAISS: {e}")
    sys.exit(1)

llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    google_api_key=config.GOOGLE_API_KEY,
    timeout=60, # Increased timeout
    max_retries=3
)

# --- 3. MANUAL RAG FUNCTION ---
def local_get_response(query: str):
    try:
        docs = retriever.invoke(query)
    except AttributeError:
        docs = retriever.get_relevant_documents(query)

    context_text = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are the PGDBA Assistant. Answer using ONLY the context below.

CONTEXT:
{context_text}

QUESTION: 
{query}

ANSWER:
"""
    response_msg = llm.invoke(prompt)
    
    return {
        "result": response_msg.content,
        "source_documents": docs
    }

# --- 4. EVALUATION LOGIC ---
test_data = [
    {
        "question": "What is the PGDBA program?",
        "ground_truth": "PGDBA is a Post Graduate Diploma in Business Analytics jointly offered by ISI Kolkata, IIT Kharagpur, and IIM Calcutta."
    },
    {
        "question": "What is the duration of the program?",
        "ground_truth": "The PGDBA program is a 2-year full-time residential program."
    },
    {
        "question": "Which institutes offer PGDBA?",
        "ground_truth": "It is offered jointly by IIM Calcutta, ISI Kolkata, and IIT Kharagpur."
    },
    {
        "question": "Is there an entrance exam?",
        "ground_truth": "Yes, admission is based on a written test followed by a personal interview."
    },
    {
        "question": "What is the selection process?",
        "ground_truth": "The selection process consists of a computer-based written test and a personal interview."
    }
]

def calculate_bleu(answers, ground_truths):
    try:
        bleu = evaluate.load("bleu")
        scores = []
        for ans, gt in zip(answers, ground_truths):
            if not ans or not ans.strip():
                scores.append(0.0)
                continue
            res = bleu.compute(predictions=[ans], references=[[gt]])
            scores.append(res['bleu'])
        return scores
    except Exception as e:
        print(f"⚠️ BLEU Error: {e}")
        return [0.0] * len(answers)

def run_evaluation():
    print("🚀 Starting RAG Pipeline...")
    
    questions, answers, contexts, ground_truths = [], [], [], []

    # 1. Generate Answers
    for item in test_data:
        q = item["question"]
        print(f"   Processing: {q}")
        try:
            res = local_get_response(q)
            
            questions.append(q)
            answers.append(res["result"])
            # Ragas needs contexts as a list of strings
            contexts.append([d.page_content for d in res["source_documents"]])
            ground_truths.append(item["ground_truth"])
        except Exception as e:
            print(f"⚠️ Error on '{q}': {e}")

    if not questions:
        print("❌ No questions processed.")
        return

    # 2. Build Base Dataframe
    df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # 3. Calculate BLEU (Fast & Reliable)
    print("MATCHING: Calculating BLEU scores...")
    df["bleu_score"] = calculate_bleu(df["answer"].tolist(), df["ground_truth"].tolist())

    # 4. Run Ragas (Slow, with Fallback)
    print("\n⚖️  Running Ragas (AI Metrics)...")
    try:
        # Only using 2 essential metrics to save time/quota
        ragas_metrics = [faithfulness, answer_relevancy]
        
        # Convert DF to Dataset for Ragas
        dataset = Dataset.from_pandas(df)
        
        # Throttled Configuration to prevent Timeouts
        run_config = RunConfig(
            max_workers=1,  # 🛑 CRITICAL: Process 1 by 1 to avoid rate limits
            timeout=120     # Give each request more time
        )

        results = ragas_evaluate(
            dataset=dataset,
            metrics=ragas_metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config
        )
        
        # Merge Ragas results back into DataFrame
        ragas_df = results.to_pandas()
        # We only need the metric columns, merge on index or common columns
        if 'faithfulness' in ragas_df.columns:
            df['faithfulness'] = ragas_df['faithfulness']
        if 'answer_relevancy' in ragas_df.columns:
            df['answer_relevancy'] = ragas_df['answer_relevancy']
            
    except Exception as e:
        print(f"❌ Ragas Failed (Skipping AI metrics): {e}")
        # Proceed with just BLEU results

    # 5. Save Results
    df.to_csv("evaluation_results.csv", index=False)
    print(f"\n✅ Done! Results saved to evaluation_results.csv")
    
    # Print Average Scores
    print("\n--- 📊 Average Scores ---")
    print(df.mean(numeric_only=True))

if __name__ == "__main__":
    run_evaluation()