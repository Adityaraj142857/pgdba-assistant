import requests
from bs4 import BeautifulSoup
import time
import random
import logging
from urllib.parse import urljoin, urlparse

# --- CONFIGURATION ---
START_URLS = [
    "https://aspirants.pgdba.ml/"
]

# Strict domain locking: Only follow links within these domains
ALLOWED_DOMAINS = [
    "aspirants.pgdba.ml",
    "pgdba.ml",
    "blog.pgdba.ml",
    "magazine.pgdba.ml"
]

MAX_PAGES_TO_CRAWL = 50  # Safety limit to prevent infinite loops
OUTPUT_FILE = "pgdba_interview_data.txt"

# Headers to mimic a real browser
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("crawler.log"), logging.StreamHandler()]
)

def is_valid_link(url):
    """Checks if the URL belongs to the allowed domains and isn't a file/image."""
    parsed = urlparse(url)
    if parsed.netloc not in ALLOWED_DOMAINS:
        return False
    
    # Ignore non-html files
    lower_url = url.lower()
    if any(ext in lower_url for ext in ['.jpg', '.png', '.pdf', '.zip', '.css', '.js']):
        return False
        
    return True

def fetch_html(url, retries=3):
    session = requests.Session()
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(1.5, 3.5)) # Polite delay
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2)
            
    return None

def extract_content_and_links(html, base_url):
    """Extracts text content AND finds all internal links."""
    soup = BeautifulSoup(html, 'html.parser')

    # 1. Clean Noise
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
        tag.decompose()

    # 2. Extract Text
    text = soup.get_text(separator=' ')
    clean_text = '\n'.join(
        line.strip() for line in text.splitlines() if line.strip()
    )

    # 3. Extract Links
    found_links = set()
    for anchor in soup.find_all('a', href=True):
        full_url = urljoin(base_url, anchor['href'])
        # Remove fragments (e.g., #section-1) to avoid duplicates
        full_url = full_url.split('#')[0]
        
        if is_valid_link(full_url):
            found_links.add(full_url)

    return clean_text, found_links

def run_crawler():
    visited = set()
    queue = list(START_URLS)
    scraped_count = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        while queue and scraped_count < MAX_PAGES_TO_CRAWL:
            current_url = queue.pop(0)
            
            if current_url in visited:
                continue
            
            logging.info(f"Crawling ({scraped_count + 1}/{MAX_PAGES_TO_CRAWL}): {current_url}")
            visited.add(current_url)
            
            html = fetch_html(current_url)
            if not html:
                continue

            content, new_links = extract_content_and_links(html, current_url)
            
            # Save Data
            f.write(f"### SOURCE: {current_url}\n")
            f.write(content)
            f.write("\n\n" + ("=" * 50) + "\n\n")
            scraped_count += 1
            
            # Add new links to queue (if not visited and not already in queue)
            for link in new_links:
                if link not in visited and link not in queue:
                    queue.append(link)
                    
            logging.info(f"Found {len(new_links)} valid links on page.")

    logging.info(f"Crawler finished. Scraped {scraped_count} pages.")

if __name__ == "__main__":
    run_crawler()