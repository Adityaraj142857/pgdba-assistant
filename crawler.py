"""
crawler.py — Async web crawler for PGDBA website content.

Fetches HTML pages and PDFs from TARGET_PREFIXES, extracts clean text,
deduplicates by content hash, and writes data/website_data.json.

Quick start:
    pip install ddgs          # enables DuckDuckGo seed discovery
    python crawler.py
"""

import asyncio
import importlib
import aiohttp
import hashlib
import json
import logging
import os
import re
import time

import requests
import trafilatura
import pytesseract

from bs4 import BeautifulSoup
from io import BytesIO
from tqdm import tqdm
from urllib.parse import parse_qs, unquote, urldefrag, urljoin, urlparse

from pdf2image import convert_from_bytes
from pdfminer.high_level import extract_text as pdf_extract_text


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crawler")


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
KEYWORDS = [
    "PGDBA IIT Kharagpur",
    "PGDBA IIM Calcutta",
    "PGDBA ISI Kolkata",
    "Post Graduate Diploma Business Analytics",
    "PGDBA eligibility",
    "PGDBA admission process",
    "PGDBA interview experience",
    "PGDBA placements",
]

TARGET_PREFIXES = [
    "https://pgdba.iitkgp.ac.in",
    "https://www.iimcal.ac.in/programs/pgdba",
    "https://www2.isical.ac.in/~pgdba",
    "https://www.isical.ac.in/~pgdba",
    "https://pgdba.ml",
]

INPUT_FILE  = "endpoints.txt"
OUTPUT_FILE = "data/website_data.json"

MAX_CONCURRENT          = 20     # parallel async workers
MAX_PAGES               = 2000   # hard crawl cap — prevents infinite loops
MAX_RESULTS_PER_KEYWORD = 10     # DDG results per keyword
MAX_PDF_MB              = 20     # skip PDFs larger than this
TIMEOUT_SECONDS         = 20     # per-request timeout
RETRIES                 = 3      # retries for 5xx / network errors only
AUTOSAVE_EVERY          = 100    # checkpoint frequency (pages)
MIN_CONTENT_LENGTH      = 200    # ignore pages shorter than this

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".zip", ".rar", ".tar", ".gz", ".exe",
    ".css", ".js", ".ico",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp4", ".mp3", ".avi", ".mov",
    ".xls", ".xlsx", ".ppt", ".pptx",   # PDFs are handled separately
)


# ══════════════════════════════════════════════════════════════
# GLOBALS
#
# visited_urls  — URLs a worker has actually FETCHED (processed set)
# queued_urls   — URLs placed in the queue (enqueue dedup set)
#
# These MUST be separate sets. Merging them causes the critical bug where
# every seed URL gets pre-marked as "visited" before any worker runs,
# resulting in 0 documents extracted (the crawler finishes in milliseconds).
#
# Rule:
#   queued_urls  ← updated immediately when a URL enters the queue
#   visited_urls ← updated inside the worker, right before fetching
# ══════════════════════════════════════════════════════════════
visited_urls: set = set()
queued_urls:  set = set()
seen_hashes:  set = set()
results:      list = []


# ══════════════════════════════════════════════════════════════
# URL HELPERS
# ══════════════════════════════════════════════════════════════
def is_valid_url(url: str) -> bool:
    """Return True if url is a crawlable HTTP/HTTPS link."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    if not p.netloc:
        return False
    if any(url.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    # Reject mailto:, javascript:, tel:, data: etc.
    if re.match(r"^(mailto|javascript|tel|data):", url, re.IGNORECASE):
        return False
    return True


def is_in_scope(url: str) -> bool:
    """Return True if url belongs to one of our target sites."""
    return any(url.startswith(prefix) for prefix in TARGET_PREFIXES)


def normalise_url(url: str) -> str:
    """Strip fragment and trailing slash for consistent deduplication."""
    url, _ = urldefrag(url)
    return url.rstrip("/")


# ══════════════════════════════════════════════════════════════
# DUCKDUCKGO SEED DISCOVERY
#
# Problem history:
#   html.duckduckgo.com wraps every result in a redirect URL:
#       https://duckduckgo.com/l/?uddg=https%3A%2F%2Fpgdba.iitkgp...
#   is_in_scope() checked for "https://pgdba.iitkgp..." so it never matched.
#   DDG also rate-limits repeated POST requests from the same IP.
#
# Solution:
#   1. PRIMARY — ddgs / duckduckgo_search library (handles DDG internally,
#      returns real destination URLs, avoids rate-limits).
#      Tries both package names since the library was renamed:
#          old: pip install duckduckgo-search  (imports as duckduckgo_search)
#          new: pip install ddgs               (imports as ddgs)
#   2. FALLBACK — scrape html.duckduckgo.com directly, decode the uddg
#      redirect param, add a 1 s delay to stay under rate-limits.
# ══════════════════════════════════════════════════════════════
def _load_ddgs_class():
    """
    Return the DDGS class from whichever package is installed.
    Returns None if neither is available.
    Handles both the old (duckduckgo_search) and new (ddgs) package names,
    and both context-manager and plain-object usage styles.
    """
    for pkg in ("ddgs", "duckduckgo_search"):
        try:
            mod = importlib.import_module(pkg)
            cls = getattr(mod, "DDGS", None)
            if cls is not None:
                return cls
        except ImportError:
            continue
    return None


def _unwrap_ddg_redirect(href: str) -> str:
    """Decode https://duckduckgo.com/l/?uddg=<encoded> → real destination URL."""
    try:
        p = urlparse(href)
        if "duckduckgo.com" in p.netloc:
            params = parse_qs(p.query)
            if "uddg" in params:
                return unquote(params["uddg"][0])
    except Exception:
        pass
    return href


def _ddg_via_library(query: str, DDGS) -> list:
    """Use the ddgs/duckduckgo_search library to search. Returns real URLs."""
    urls = []
    try:
        # Support both context-manager style (newer) and plain instantiation
        try:
            client = DDGS()
            results_iter = client.text(query, max_results=MAX_RESULTS_PER_KEYWORD)
        except TypeError:
            # Some versions require no arguments or have different signatures
            client = DDGS.__new__(DDGS)
            results_iter = client.text(query, max_results=MAX_RESULTS_PER_KEYWORD)

        for r in results_iter:
            url = r.get("href", "") or r.get("url", "")
            if url.startswith("http"):
                urls.append(url)
    except Exception as e:
        log.debug(f"DDG library search failed for '{query}': {e}")
    return urls


def _ddg_via_html(query: str) -> list:
    """
    Fallback DDG search by scraping html.duckduckgo.com.
    Decodes uddg redirect params. Sleeps 1 s between calls to avoid bans.
    """
    time.sleep(1)
    links = []
    for attempt in range(2):
        try:
            r = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers=HEADERS,
                timeout=10,
            )
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # DDG occasionally changes CSS class names — try both
            anchors = soup.find_all("a", class_="result__a") or soup.find_all("a", href=True)

            for a in anchors:
                raw  = a.get("href", "")
                real = _unwrap_ddg_redirect(raw)
                if real.startswith("http"):
                    links.append(real)
                if len(links) >= MAX_RESULTS_PER_KEYWORD:
                    break
            break   # success — exit retry loop

        except Exception as e:
            log.debug(f"DDG HTML attempt {attempt + 1} failed for '{query}': {e}")
    return links


def get_duckduckgo_links(query: str) -> list:
    """
    Return real destination URLs from DDG for query.
    Uses library if installed, otherwise falls back to HTML scraping.
    """
    DDGS = _load_ddgs_class()
    if DDGS is not None:
        results_list = _ddg_via_library(query, DDGS)
        if results_list is not None:
            return results_list
    log.debug("DDG library unavailable — using HTML fallback (pip install ddgs to fix)")
    return _ddg_via_html(query)


# ══════════════════════════════════════════════════════════════
# EMAIL DE-OBFUSCATION
#
# Many academic sites mangle contact emails to foil scrapers:
#   name[at]domain[dot]ac[dot]in
#   name (at) domain (dot) ac (dot) in
#   name AT domain DOT ac DOT in
#   name_at_domain_dot_ac_dot_in       ← underscore style
#   name@domain[dot]ac[dot]in          ← mixed style
#
# deobfuscate_emails_only() is safe to run on full page text because:
#   • Bracketed [at]/[dot] markers are unambiguous — always replaced.
#   • Bare "at"/"dot" words are replaced ONLY when flanked by \w on
#     both sides (i.e., between two non-space identifier characters).
#     "look at this"  → unchanged  (space after "this")
#     "user at iimcal" → "user@iimcal"  (\w on both sides)
#   • We do NOT do re.sub(r"\s*\.\s*", ".", t) — that would corrupt
#     every sentence ending ("Hello. World" → "Hello.World").
#
# Usage:
#   • extract_contacts(text)  — extracts emails/phones from any text
#   • deobfuscate_emails_only(text) — clean the stored content so RAG
#     returns real email addresses to users instead of mangled strings
# ══════════════════════════════════════════════════════════════
def deobfuscate_emails_only(text: str) -> str:
    t = text

    # 1. Bracketed markers — completely unambiguous, always safe to replace
    t = re.sub(r"\s*[\(\[]at[\)\]]\s*",  "@", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*[\(\[]dot[\)\]]\s*", ".", t, flags=re.IGNORECASE)

    # 2. Underscore-obfuscated style: name_at_domain_dot_ac
    t = re.sub(r"(?<=\w)_at_(?=\w)",  "@", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<=\w)_dot_(?=\w)", ".", t, flags=re.IGNORECASE)

    # 3. Bare words flanked by word characters on both sides
    t = re.sub(r"(?<=\w)\s+at\s+(?=\w)",  "@", t, flags=re.IGNORECASE)
    t = re.sub(r"(?<=\w)\s+dot\s+(?=\w)", ".", t, flags=re.IGNORECASE)

    # 4. Tidy stray spaces around @ only (@ never appears in normal prose)
    t = re.sub(r"\s*@\s*", "@", t)

    return t


# ══════════════════════════════════════════════════════════════
# CONTACT EXTRACTION
# ══════════════════════════════════════════════════════════════
def extract_contacts(text: str) -> tuple[list, list]:
    """
    Returns (emails, phones) extracted from text.
    De-obfuscates email patterns before running the regex.
    Phones use the original text (no dot-related concerns).
    """
    clean = deobfuscate_emails_only(text)

    emails = sorted(set(re.findall(
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", clean
    )))

    phones = sorted(set(re.findall(
        r"\+?\d[\d\-\s]{8,}\d", text
    )))

    return emails, phones


# ══════════════════════════════════════════════════════════════
# TABLE EXTRACTION
# Stored as pipe-separated plain-text rows so table content
# gets chunked and embedded alongside normal text in RAG.
# (Raw DataFrame dicts are huge, won't serialise cleanly to JSON,
#  and are completely useless for embedding.)
# ══════════════════════════════════════════════════════════════
def extract_tables(html: str) -> list[str]:
    tables = []
    try:
        import pandas as pd
        for df in pd.read_html(html):
            rows = df.fillna("").astype(str).values.tolist()
            table_text = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
            if table_text.strip():
                tables.append(table_text)
    except Exception:
        pass    # no tables — perfectly normal
    return tables


# ══════════════════════════════════════════════════════════════
# HTML CONTENT EXTRACTION
# ══════════════════════════════════════════════════════════════
def extract_clean_text(html: str) -> str | None:
    """Extract main body text from raw HTML using trafilatura."""
    text = trafilatura.extract(
        html,
        include_tables=True,
        include_links=False,
        include_images=False,
        favor_precision=True,   # filters out nav/footer boilerplate
        deduplicate=True,
    )
    if not text or len(text.strip()) < MIN_CONTENT_LENGTH:
        return None
    return text.strip()


# ══════════════════════════════════════════════════════════════
# LINK EXTRACTION
# ══════════════════════════════════════════════════════════════
def extract_links(html: str, base_url: str) -> set[str]:
    """Extract all in-scope href links from an HTML page."""
    soup  = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        try:
            url = normalise_url(urljoin(base_url, a["href"].strip()))
            if is_valid_url(url) and is_in_scope(url):
                links.add(url)
        except Exception:
            continue
    return links


# ══════════════════════════════════════════════════════════════
# CONTENT DEDUPLICATION
# ══════════════════════════════════════════════════════════════
def is_duplicate(text: str) -> bool:
    """Return True if this exact content has been seen before (MD5 hash)."""
    h = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
    if h in seen_hashes:
        return True
    seen_hashes.add(h)
    return False


# ══════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ══════════════════════════════════════════════════════════════
def extract_pdf_text(data: bytes) -> str | None:
    """Extract text from a machine-readable PDF using pdfminer."""
    try:
        text = pdf_extract_text(BytesIO(data))
        if text and len(text.strip()) > MIN_CONTENT_LENGTH:
            return text.strip()
    except Exception as e:
        log.debug(f"pdfminer failed: {e}")
    return None


def ocr_pdf(data: bytes) -> str | None:
    """OCR a scanned PDF using pdf2image + pytesseract."""
    try:
        images = convert_from_bytes(data)
        text   = "\n".join(pytesseract.image_to_string(img) for img in images)
        if len(text.strip()) > MIN_CONTENT_LENGTH:
            return text.strip()
    except Exception as e:
        log.debug(f"OCR failed: {e}")
    return None


# ══════════════════════════════════════════════════════════════
# HTTP FETCH
#
# Uses aiohttp.ClientTimeout (not a bare int — bare int is deprecated).
# Does NOT retry 4xx errors — they will never recover.
# Retries 5xx and network errors up to RETRIES times with backoff.
# Guards against oversized PDFs before reading into memory.
# ══════════════════════════════════════════════════════════════
async def fetch(session: aiohttp.ClientSession, url: str) -> tuple:
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)

    for attempt in range(RETRIES):
        try:
            async with session.get(url, timeout=timeout, ssl=False, headers=HEADERS) as r:

                # 4xx — will never recover; stop immediately
                if 400 <= r.status < 500:
                    log.debug(f"HTTP {r.status} (no retry): {url}")
                    return None, None

                # 5xx — server error; worth retrying
                if r.status >= 500:
                    log.debug(f"HTTP {r.status} (attempt {attempt + 1}/{RETRIES}): {url}")
                    await asyncio.sleep(2 ** attempt)   # exponential backoff
                    continue

                if r.status != 200:
                    return None, None

                ct = r.headers.get("Content-Type", "").lower()

                # PDF
                if "pdf" in ct or url.lower().endswith(".pdf"):
                    cl = r.headers.get("Content-Length")
                    if cl and int(cl) > MAX_PDF_MB * 1024 * 1024:
                        log.info(f"Skipping large PDF ({int(cl) // 1024 // 1024} MB): {url}")
                        return None, None
                    return "pdf", await r.read()

                # HTML
                if "text/html" in ct:
                    return "html", await r.text(errors="replace")

                return None, None

        except asyncio.TimeoutError:
            log.debug(f"Timeout (attempt {attempt + 1}/{RETRIES}): {url}")
            await asyncio.sleep(2 ** attempt)
        except aiohttp.ClientError as e:
            log.debug(f"Network error (attempt {attempt + 1}/{RETRIES}): {url} — {e}")
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            log.debug(f"Unexpected fetch error: {url} — {e}")
            break   # non-transient error — don't retry

    return None, None


# ══════════════════════════════════════════════════════════════
# AUTOSAVE  — atomic write prevents corruption on crash
# ══════════════════════════════════════════════════════════════
def autosave() -> None:
    try:
        tmp = OUTPUT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        os.replace(tmp, OUTPUT_FILE)    # atomic rename
        log.info(f"💾 Checkpoint: {len(results)} docs saved → {OUTPUT_FILE}")
    except Exception as e:
        log.warning(f"Autosave failed: {e}")


# ══════════════════════════════════════════════════════════════
# ASYNC WORKER
# ══════════════════════════════════════════════════════════════
async def worker(queue: asyncio.Queue, session: aiohttp.ClientSession, pbar: tqdm) -> None:
    while True:
        url = await queue.get()
        try:
            # Hard page cap
            if len(visited_urls) >= MAX_PAGES:
                continue

            # Another worker already fetched this URL
            if url in visited_urls:
                continue
            visited_urls.add(url)

            content_type, content = await fetch(session, url)
            if not content:
                continue

            # ── HTML ──────────────────────────────────────────────────
            if content_type == "html":
                raw_text = extract_clean_text(content)
                if not raw_text or is_duplicate(raw_text):
                    continue

                # Deobfuscate stored content so RAG returns clean emails
                # e.g. "name[at]domain[dot]ac[dot]in" → "name@domain.ac.in"
                stored_text    = deobfuscate_emails_only(raw_text)
                emails, phones = extract_contacts(raw_text)
                tables         = extract_tables(content)

                results.append({
                    "url":     url,
                    "type":    "html",
                    "content": stored_text,
                    "emails":  emails,
                    "phones":  phones,
                    "tables":  tables,
                })

                # Enqueue newly discovered links (use queued_urls for dedup)
                for link in extract_links(content, url):
                    if link not in queued_urls:
                        queued_urls.add(link)
                        await queue.put(link)

            # ── PDF ───────────────────────────────────────────────────
            elif content_type == "pdf":
                raw_text = extract_pdf_text(content) or ocr_pdf(content)
                if not raw_text or is_duplicate(raw_text):
                    continue

                stored_text    = deobfuscate_emails_only(raw_text)
                emails, phones = extract_contacts(raw_text)

                results.append({
                    "url":     url,
                    "type":    "pdf",
                    "content": stored_text,
                    "emails":  emails,
                    "phones":  phones,
                    "tables":  [],
                })

            # Periodic checkpoint
            if results and len(results) % AUTOSAVE_EVERY == 0:
                autosave()

        except Exception as e:
            log.warning(f"Worker error [{url}]: {e}")
        finally:
            pbar.update(1)
            queue.task_done()


# ══════════════════════════════════════════════════════════════
# SEED URL LOADING
# ══════════════════════════════════════════════════════════════
def load_initial_links() -> set:
    urls: set = set()

    # 1 — Load from endpoints.txt
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, encoding="utf-8", errors="replace") as f:
            for line in f:
                link = normalise_url(line.strip())
                if is_valid_url(link):
                    urls.add(link)
        log.info(f"Loaded {len(urls)} seed URLs from {INPUT_FILE}")
    else:
        log.warning(f"{INPUT_FILE} not found — will rely on DuckDuckGo seeds only")

    # 2 — Supplement with DuckDuckGo discovery
    log.info("Searching DuckDuckGo for additional in-scope URLs...")
    ddg_added = 0
    for keyword in KEYWORDS:
        try:
            found = get_duckduckgo_links(keyword)
            added = 0
            for link in found:
                link = normalise_url(link)
                if is_valid_url(link) and is_in_scope(link) and link not in urls:
                    urls.add(link)
                    added += 1
            ddg_added += added
            log.info(f"  '{keyword}' → {added} new in-scope URLs")
        except Exception as e:
            log.warning(f"  DDG search failed for '{keyword}': {e}")

    if ddg_added == 0:
        log.warning(
            "DuckDuckGo found 0 new in-scope URLs. This is fine if endpoints.txt\n"
            "  is populated — the crawler discovers new pages by following links.\n"
            "  To improve DDG results: pip install ddgs"
        )

    log.info(f"Total seed URLs ready: {len(urls)}")
    return urls


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
async def main() -> None:
    os.makedirs("data", exist_ok=True)

    initial_urls = load_initial_links()

    queue: asyncio.Queue = asyncio.Queue()
    for u in initial_urls:
        queued_urls.add(u)    # enqueue dedup — NOT added to visited_urls here
        queue.put_nowait(u)   # visited_urls updated only when worker fetches

    log.info(f"Queue seeded: {queue.qsize()} URLs  |  page cap: {MAX_PAGES}")

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, force_close=False)
    pbar      = tqdm(desc="Pages crawled", unit="pg", dynamic_ncols=True)

    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [
            asyncio.create_task(worker(queue, session, pbar))
            for _ in range(MAX_CONCURRENT)
        ]
        await queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    pbar.close()
    autosave()

    log.info("═" * 50)
    log.info(f"  Pages fetched    : {len(visited_urls)}")
    log.info(f"  URLs discovered  : {len(queued_urls)}")
    log.info(f"  Docs extracted   : {len(results)}")
    log.info(f"  Output file      : {OUTPUT_FILE}")
    log.info("═" * 50)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    asyncio.run(main())