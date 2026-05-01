import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import random

# HELPER: Reliable request with retry 
def get_with_retry(url, headers, stream=False, max_retries=3, timeout=20):
    base_wait = 10
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.3, 1.2))
            response = requests.get(url, headers=headers, timeout=timeout, stream=stream)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                wait_time = base_wait * (2 ** attempt)
                print(f" Rate limited (429). Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif response.status_code >= 500:
                wait_time = base_wait * (2 ** attempt)
                print(f" Server error {response.status_code}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f" Client error {response.status_code} for {url}")
                return None
        except requests.exceptions.RequestException as e:
            wait_time = 5 * (2 ** attempt)
            print(f" Network error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    print(f" Failed to fetch {url} after {max_retries} retries.")
    return None


# STEP 1: Collect all valid paper URLs 
def fetch_valid_paper_links(event_url, year="2025"):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = get_with_retry(event_url, headers=headers)
    if not response:
        print(f" Failed to fetch event page: {event_url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a", href=True)

    valid_pages = [
        a["href"] for a in links
        if a["href"].startswith(f"/{year}.")
        and not a["href"].endswith((".pdf", ".bib", ".zip", ".tgz"))
    ]
    return list(set(valid_pages))


# STEP 2: Download individual PDF 
def download_pdf(paper_url, save_dir, headers):
    paper_id = paper_url.strip("/")
    file_path = os.path.join(save_dir, f"{paper_id}.pdf")

    if os.path.exists(file_path):
        return f" Already exists: {paper_id}.pdf"

    paper_page_url = "https://aclanthology.org" + paper_url

    try:
        paper_page = get_with_retry(paper_page_url, headers=headers)
        if not paper_page:
            return f" Failed to fetch {paper_id} page"

        soup = BeautifulSoup(paper_page.text, "html.parser")
        pdf_tag = soup.find("a", href=True, string="PDF")
        if not pdf_tag:
            pdf_tag = soup.find("a", href=lambda h: h and h.endswith(".pdf"))
        if not pdf_tag:
            return f" No PDF link for {paper_id}"

        pdf_url = pdf_tag["href"]
        if not pdf_url.startswith("https://"):
            pdf_url = "https://aclanthology.org" + pdf_url

        pdf_response = get_with_retry(pdf_url, headers=headers, stream=True)
        if not pdf_response:
            return f" Failed download: {paper_id}"

        first_bytes = pdf_response.raw.read(4)
        if first_bytes != b"%PDF":
            return f" Invalid PDF content for {paper_id}"

        with open(file_path, "wb") as f:
            f.write(first_bytes)
            for chunk in pdf_response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f" Saved: {paper_id}.pdf"

    except Exception as e:
        return f" Error for {paper_id}: {e}"


# STEP 3: Download manager with batch support 
def download_papers_from_events(
    event_urls,
    save_dir="part1",
    max_workers=10,
    start_index=0,
    end_index=None
):
    headers = {"User-Agent": "Mozilla/5.0"}
    os.makedirs(save_dir, exist_ok=True)

    print("\n Fetching paper list...")
    all_papers = []
    for url in event_urls:
        print(f" Fetching from: {url}")
        papers = fetch_valid_paper_links(url)
        print(f"   → Found {len(papers)} papers.")
        all_papers.extend(papers)

    all_papers = sorted(list(set(all_papers)))
    total = len(all_papers)
    print(f"\n Total unique papers: {total}")

    # Apply batch slicing
    if end_index is None:
        selected = all_papers[start_index:]
    else:
        selected = all_papers[start_index:end_index]

    # Remove already downloaded
    existing = {f.replace(".pdf", "") for f in os.listdir(save_dir) if f.endswith(".pdf")}
    to_download = [p for p in selected if p.strip("/") not in existing]

    if not to_download:
        print(" All selected papers already downloaded.")
        return

    print(f" Downloading {len(to_download)} papers (Batch: {save_dir})...")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_pdf, p, save_dir, headers) for p in to_download]
        for f in tqdm(futures, total=len(to_download)):
            try:
                results.append(f.result())
            except Exception as e:
                results.append(f" Thread error: {e}")

    saved = sum(" Saved" in r for r in results)
    skipped = sum("Already exists" in r for r in results)
    failed = len(results) - saved - skipped

    print("\n --- SUMMARY ---")
    print(f" Downloaded: {saved}")
    print(f" Skipped: {skipped}")
    print(f" Failed: {failed}")
    print(f" Saved in: {os.path.abspath(save_dir)}")


# RUN SCRIPT 
if __name__ == "__main__":
    event_urls = [
        "https://aclanthology.org/events/acl-2025/",
        # Add more event URLs here (e.g., EMNLP, NAACL)
    ]

    # --- SETTINGS ---
    download_papers_from_events(
        event_urls=event_urls,
        save_dir="part1",   # change part name for each batch
        max_workers=8,                 # threads
        start_index=1,              # resume point
        end_index=10                 # None = all remaining
    )
