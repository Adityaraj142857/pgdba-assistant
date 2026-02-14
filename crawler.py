import asyncio
import aiohttp
import ssl
import json
import re
import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlparse

# -----------------------------
# CONFIG
# -----------------------------
KEYWORDS = [
    "PGDBA",
    "pgdba IIM Calcutta",
    "Post Graduate Diploma in Business Analytics",
    "PGDBA eligibility",
    "PGDBA admission process"
]

INPUT_FILE = "final_endpoints.txt"
OUTPUT_FILE = "scraped_data.json"

MAX_RESULTS_PER_KEYWORD = 30
MAX_CONCURRENT = 20
TIMEOUT = 15
RETRIES = 3

SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".pdf", ".zip", ".rar", ".exe",
    ".css", ".ico", ".woff", ".ttf"
)

results = []

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


# -----------------------------
# DUCKDUCKGO SEARCH (NO API)
# -----------------------------
def get_duckduckgo_links(query, max_results=20):
    links = []
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}

    response = requests.post(url, data=params, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.find_all("a", class_="result__a")

    for result in results:
        link = result.get("href")
        if link and link.startswith("http"):
            links.append(link)
        if len(links) >= max_results:
            break
    print(links)
    return links


# -----------------------------
# URL VALIDATION
# -----------------------------
def is_valid_url(url):
    parsed = urlparse(url)
    if not parsed.scheme.startswith("http"):
        return False
    if any(url.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    return True


# -----------------------------
# TEXT CLEANING
# -----------------------------
def extract_clean_text(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    # Remove very short pages (low value)
    if len(text) < 500:
        return None

    return text


# -----------------------------
# ASYNC FETCH
# -----------------------------
async def fetch(session, url):
    for attempt in range(RETRIES):
        try:
            async with session.get(url, timeout=TIMEOUT, ssl=False) as response:
                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" in content_type:
                        return await response.text()
        except Exception:
            await asyncio.sleep(1)
    return None


async def worker(semaphore, session, url):
    async with semaphore:
        if not is_valid_url(url):
            return

        html = await fetch(session, url)
        if html:
            text = extract_clean_text(html)
            if text:
                results.append({
                    "url": url,
                    "content": text
                })


# -----------------------------
# MERGE & DEDUP
# -----------------------------
def merge_and_deduplicate_links(new_links):
    existing_links = set()

    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, "r") as f:
            existing_links = set(line.strip() for line in f if line.strip())

    all_links = existing_links.union(set(new_links))

    with open(INPUT_FILE, "w") as f:
        for link in sorted(all_links):
            f.write(link + "\n")

    return list(all_links)


# -----------------------------
# MAIN
# -----------------------------
async def crawl_all(urls):
    connector = aiohttp.TCPConnector(ssl=False)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [worker(semaphore, session, url) for url in urls]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await task


async def main():
    print("🔎 Fetching DuckDuckGo links...\n")

    search_links = []
    for keyword in KEYWORDS:
        print(f"Searching: {keyword}")
        links = get_duckduckgo_links(keyword, MAX_RESULTS_PER_KEYWORD)
        search_links.extend(links)

    print(f"\n🌐 Total links fetched: {len(search_links)}")

    all_links = merge_and_deduplicate_links(search_links)
    print(f"📎 Total unique links after merge: {len(all_links)}")

    print("\n🚀 Starting async crawl...\n")
    await crawl_all(all_links)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Scraping completed. Saved to {OUTPUT_FILE}")
    print(f"Total pages scraped successfully: {len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
