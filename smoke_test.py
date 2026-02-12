# smoke_test.py
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in environment.")
    exit(1)

print(f"✅ API Key found: {api_key[:5]}...{api_key[-4:]}")

# 2. Test Embedding
try:
    print("Testing Google Embeddings connection...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", # The new correct model
        google_api_key=api_key
    )
    
    vector = embeddings.embed_query("Hello, world!")
    print(f"✅ Success! Generated embedding with {len(vector)} dimensions.")
    print("You are ready to run ingestion.py")

except Exception as e:
    print(f"❌ Failed: {e}")