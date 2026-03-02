#!/bin/bash

echo "🚀 Starting Full PGDBA Database Rebuild..."

# 1. Enter the project and activate the virtual environment
cd /var/www/pgdba-assistant
source venv/bin/activate

# 2. Delete the old data entirely to prevent duplicates
echo "🗑️ Deleting old data..."
rm -f data/website_data.json
rm -f data/docs.pkl
rm -rf data/faiss_index

# 3. Scrape the fresh website data
echo "🕸️ Crawling the latest PGDBA websites..."
python crawler.py

# 4. Rebuild the FAISS index and BM25 docs
echo "🧠 Generating new embeddings..."
python src/ingestion.py

# 5. Restart the FastAPI background service so it loads the new database
echo "🔄 Restarting the live API..."
sudo systemctl restart pgdba.service

echo "✅ Update Complete! The chatbot is now using the newest data."