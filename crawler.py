import asyncio
import aiohttp
import json
import os
import re
import hashlib
import requests
import pandas as pd
import trafilatura
import pytesseract

from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag
from io import BytesIO

from pdfminer.high_level import extract_text
from pdf2image import convert_from_bytes


# -----------------------------
# CONFIG
# -----------------------------

KEYWORDS = [
    "PGDBA IIT Kharagpur",
    "PGDBA IIM Calcutta",
    "PGDBA ISI Kolkata",
    "Post Graduate Diploma Business Analytics",
    "PGDBA eligibility",
    "PGDBA admission process"
]

TARGET_PREFIXES = [
    "https://pgdba.iitkgp.ac.in",
    "https://www.iimcal.ac.in/programs/pgdba",
    "https://www2.isical.ac.in/~pgdba",
    "https://www.isical.ac.in/~pgdba",
    "https://pgdba.ml"
]

INPUT_FILE = "endpoints.txt"
OUTPUT_FILE = "data/website_data.json"

MAX_CONCURRENT = 20
MAX_RESULTS_PER_KEYWORD = 20
TIMEOUT = 20
RETRIES = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

SKIP_EXTENSIONS = (
    ".jpg",".jpeg",".png",".gif",".svg",
    ".zip",".rar",".exe",".css",".ico",".woff",".ttf"
)

# -----------------------------
# GLOBALS
# -----------------------------

visited_urls = set()
seen_hashes = set()
results = []


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


def is_in_scope(url):

    return any(url.startswith(prefix) for prefix in TARGET_PREFIXES)


# -----------------------------
# DUCKDUCKGO SEARCH
# -----------------------------

def get_duckduckgo_links(query):

    links = []

    try:

        r = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=HEADERS
        )

        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.find_all("a", class_="result__a"):

            link = a.get("href")

            if link and link.startswith("http"):
                links.append(link)

            if len(links) >= MAX_RESULTS_PER_KEYWORD:
                break

    except:
        pass

    return links


# -----------------------------
# CONTACT EXTRACTION
# -----------------------------

def extract_contacts(text):

    emails = list(set(
        re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    ))

    phones = list(set(
        re.findall(r"\+?\d[\d\-\s]{8,}\d", text)
    ))

    return emails, phones


# -----------------------------
# TABLE EXTRACTION
# -----------------------------

def extract_tables(html):

    tables = []

    try:

        dfs = pd.read_html(html)

        for df in dfs:
            tables.append(df.to_dict())

    except:
        pass

    return tables


# -----------------------------
# HTML TEXT EXTRACTION
# -----------------------------

def extract_clean_text(html):

    text = trafilatura.extract(
        html,
        include_tables=True,
        include_links=False,
        include_images=False
    )

    if not text:
        return None

    if len(text) < 200:
        return None

    return text


# -----------------------------
# LINK EXTRACTION
# -----------------------------

def extract_links(html, base):

    soup = BeautifulSoup(html,"html.parser")

    links = set()

    for a in soup.find_all("a",href=True):

        url = urljoin(base,a["href"])

        url,_ = urldefrag(url)

        if is_valid_url(url) and is_in_scope(url):
            links.add(url)

    return links


# -----------------------------
# DEDUPLICATION
# -----------------------------

def is_duplicate(text):

    h = hashlib.md5(text.encode()).hexdigest()

    if h in seen_hashes:
        return True

    seen_hashes.add(h)

    return False


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------

def extract_pdf_text(data):

    try:

        text = extract_text(BytesIO(data))

        if text and len(text) > 200:
            return text

    except:
        pass

    return None


# -----------------------------
# OCR SCANNED PDF
# -----------------------------

def ocr_pdf(data):

    try:

        images = convert_from_bytes(data)

        text = ""

        for img in images:
            text += pytesseract.image_to_string(img)

        if len(text) > 200:
            return text

    except:
        pass

    return None


# -----------------------------
# FETCH
# -----------------------------

async def fetch(session,url):

    for _ in range(RETRIES):

        try:

            async with session.get(
                url,
                timeout=TIMEOUT,
                ssl=False,
                headers=HEADERS
            ) as r:

                if r.status != 200:
                    return None,None

                content_type = r.headers.get("Content-Type","")

                if "pdf" in content_type or url.endswith(".pdf"):

                    data = await r.read()
                    return "pdf",data

                if "text/html" in content_type:

                    html = await r.text()
                    return "html",html

        except:

            await asyncio.sleep(1)

    return None,None


# -----------------------------
# WORKER
# -----------------------------

async def worker(queue,session,pbar):

    while True:

        url = await queue.get()

        try:

            if url in visited_urls:
                continue

            visited_urls.add(url)

            content_type,content = await fetch(session,url)

            if not content:
                continue


            # -------- HTML

            if content_type=="html":

                text = extract_clean_text(content)

                if not text:
                    continue

                if is_duplicate(text):
                    continue

                emails,phones = extract_contacts(text)

                tables = extract_tables(content)

                results.append({
                    "url":url,
                    "type":"html",
                    "content":text,
                    "emails":emails,
                    "phones":phones,
                    "tables":tables
                })

                new_links = extract_links(content,url)

                for link in new_links:

                    if link not in visited_urls:
                        await queue.put(link)


            # -------- PDF

            if content_type=="pdf":

                text = extract_pdf_text(content)

                if not text:
                    text = ocr_pdf(content)

                if not text:
                    continue

                if is_duplicate(text):
                    continue

                emails,phones = extract_contacts(text)

                results.append({
                    "url":url,
                    "type":"pdf",
                    "content":text,
                    "emails":emails,
                    "phones":phones,
                    "tables":[]
                })


        finally:

            pbar.update(1)

            queue.task_done()


# -----------------------------
# INITIAL URL COLLECTION
# -----------------------------

def load_initial_links():

    urls = set()

    if os.path.exists(INPUT_FILE):

        with open(INPUT_FILE) as f:

            for line in f:

                link = line.strip()

                if is_valid_url(link):
                    urls.add(link)

    print("Loaded endpoints:",len(urls))

    print("Searching DuckDuckGo...")

    for keyword in KEYWORDS:

        links = get_duckduckgo_links(keyword)

        for link in links:

            if is_valid_url(link) and is_in_scope(link):
                urls.add(link)

    return urls


# -----------------------------
# MAIN
# -----------------------------

async def main():

    os.makedirs("data",exist_ok=True)

    initial_urls = load_initial_links()

    queue = asyncio.Queue()

    for u in initial_urls:
        queue.put_nowait(u)

    print("Initial queue size:",queue.qsize())

    connector = aiohttp.TCPConnector(ssl=False)

    pbar = tqdm(desc="Pages Crawled")

    async with aiohttp.ClientSession(connector=connector) as session:

        workers = [

            asyncio.create_task(
                worker(queue,session,pbar)
            )

            for _ in range(MAX_CONCURRENT)
        ]

        await queue.join()

        for w in workers:
            w.cancel()

        await asyncio.gather(*workers,return_exceptions=True)

    with open(OUTPUT_FILE,"w",encoding="utf-8") as f:

        json.dump(results,f,indent=2,ensure_ascii=False)

    print("\nCrawled URLs:",len(visited_urls))
    print("Documents extracted:",len(results))
    print("Saved to:",OUTPUT_FILE)


# -----------------------------
# RUN
# -----------------------------

if __name__=="__main__":

    asyncio.run(main())