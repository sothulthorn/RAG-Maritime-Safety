"""
Maritime Safety Report Scraper
Downloads accident investigation reports from:
  - MAIB          (UK Marine Accident Investigation Branch)
  - NTSB          (US National Transportation Safety Board)
  - TSB           (Transportation Safety Board of Canada)
  - USCG          (US Coast Guard — Marine Safety Alerts)
  - ATSB          (Australian Transport Safety Bureau)
  - DMAIB         (Danish Maritime Accident Investigation Board)
  - BSU           (German Federal Bureau of Maritime Casualty Investigation)
  - MAIB Digests  (UK — annual lessons-learned compilations)
  - JTSB          (Japan Transport Safety Board)
  - SHK           (Swedish Accident Investigation Authority)

Usage:
    python scripts/scrape_reports.py                    # Download from all sources
    python scripts/scrape_reports.py --source maib      # Download MAIB only
    python scripts/scrape_reports.py --source uscg      # Download USCG only
    python scripts/scrape_reports.py --max 10           # Limit downloads per source
    python scripts/scrape_reports.py --list             # Show available sources
"""

import argparse
import hashlib
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent / "data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Maritime Safety Research Bot; academic use)"
}
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 1.0  # seconds, be polite to servers


def _download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF file if it doesn't already exist. Returns True on success."""
    if dest.exists():
        print(f"  [SKIP] Already exists: {dest.name}")
        return True

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
        if resp.status_code != 200:
            print(f"  [FAIL] HTTP {resp.status_code}: {url}")
            return False

        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and not url.endswith(".pdf"):
            print(f"  [SKIP] Not a PDF ({content_type}): {url}")
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = dest.stat().st_size / 1024
        if size_kb < 5:  # too small to be a real report
            dest.unlink()
            print(f"  [SKIP] File too small ({size_kb:.0f} KB): {url}")
            return False

        print(f"  [OK] {dest.name} ({size_kb:.0f} KB)")
        return True

    except requests.RequestException as e:
        print(f"  [ERR] {e}")
        return False


# ---------------------------------------------------------------------------
# MAIB — UK Marine Accident Investigation Branch
# ---------------------------------------------------------------------------
def scrape_maib(max_reports: int = 0):
    """
    Scrape MAIB reports from gov.uk.
    Strategy: paginate the report listing, visit each report page, find PDF links.
    """
    out_dir = BASE_DIR / "maib"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.gov.uk"
    list_url = f"{base_url}/maib-reports"

    print(f"\n{'='*60}")
    print("MAIB — UK Marine Accident Investigation Branch")
    print(f"{'='*60}")

    downloaded = 0
    page = 1

    while True:
        if 0 < max_reports <= downloaded:
            break

        url = f"{list_url}?page={page}"
        print(f"\nFetching listing page {page}...")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [ERR] Could not fetch listing page {page}: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find report links — they're in <a> tags within the listing
        report_links = []
        for a_tag in soup.select("a[href^='/maib-reports/']"):
            href = a_tag.get("href", "")
            if href and href != "/maib-reports/" and "/maib-reports?" not in href:
                full_url = urljoin(base_url, href)
                if full_url not in report_links:
                    report_links.append(full_url)

        if not report_links:
            print(f"  No more reports found on page {page}.")
            break

        print(f"  Found {len(report_links)} report links on page {page}")

        for report_url in report_links:
            if 0 < max_reports <= downloaded:
                break

            time.sleep(DELAY_BETWEEN_REQUESTS)

            try:
                resp = requests.get(report_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
            except requests.RequestException:
                continue

            report_soup = BeautifulSoup(resp.text, "html.parser")

            # Find PDF links on the report page
            pdf_links = []
            for a_tag in report_soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.endswith(".pdf"):
                    pdf_url = urljoin(report_url, href)
                    pdf_links.append(pdf_url)

            for pdf_url in pdf_links:
                if 0 < max_reports <= downloaded:
                    break

                # Generate filename from URL
                filename = pdf_url.split("/")[-1]
                if not filename.endswith(".pdf"):
                    filename += ".pdf"

                if _download_pdf(pdf_url, out_dir / filename):
                    downloaded += 1

                time.sleep(DELAY_BETWEEN_REQUESTS)

        # Check for next page
        next_link = soup.find("a", string=re.compile(r"Next page", re.IGNORECASE))
        if not next_link:
            # Also check rel="next"
            next_link = soup.find("a", rel="next")
        if not next_link:
            print(f"  No next page link found after page {page}.")
            break

        page += 1

    print(f"\nMAIB: Downloaded {downloaded} reports to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# NTSB — US National Transportation Safety Board
# ---------------------------------------------------------------------------
def scrape_ntsb(max_reports: int = 0):
    """
    Download NTSB marine reports by trying known PDF URL patterns.
    NTSB report IDs follow: MAR{YY}{NN} and MAB{YY}{NN}
    where YY=year (97-25), NN=report number (01-10).
    """
    out_dir = BASE_DIR / "ntsb"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.ntsb.gov/investigations/AccidentReports/Reports"

    print(f"\n{'='*60}")
    print("NTSB — US National Transportation Safety Board")
    print(f"{'='*60}")

    downloaded = 0

    # Generate candidate report IDs
    # MAR = Major Marine Accident Report
    # MAB = Marine Accident Brief
    # MSR = Marine Special Report / Marine Safety Recommendation
    prefixes = ["MAR", "MAB", "MSR"]
    years = list(range(97, 100)) + list(range(0, 27))  # 1997-2026

    for prefix in prefixes:
        for year in years:
            if 0 < max_reports <= downloaded:
                break

            for num in range(1, 15):
                if 0 < max_reports <= downloaded:
                    break

                year_str = f"{year:02d}"
                num_str = f"{num:02d}"
                report_id = f"{prefix}{year_str}{num_str}"
                pdf_url = f"{base_url}/{report_id}.pdf"
                filename = f"{report_id}.pdf"

                if (out_dir / filename).exists():
                    print(f"  [SKIP] Already exists: {filename}")
                    downloaded += 1
                    continue

                if _download_pdf(pdf_url, out_dir / filename):
                    downloaded += 1

                time.sleep(0.5)  # shorter delay since many will be 404s

    print(f"\nNTSB: Downloaded {downloaded} reports to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# TSB — Transportation Safety Board of Canada
# ---------------------------------------------------------------------------
def scrape_tsb(max_reports: int = 0):
    """
    Scrape TSB marine investigation reports.
    Strategy: parse the index page for report IDs, then find PDFs on report pages.
    """
    out_dir = BASE_DIR / "tsb"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.tsb.gc.ca"
    index_url = f"{base_url}/eng/rapports-reports/marine/index.html"

    print(f"\n{'='*60}")
    print("TSB — Transportation Safety Board of Canada")
    print(f"{'='*60}")

    print("\nFetching report index...")
    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERR] Could not fetch index: {e}")
        return 0

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find all report links — pattern: /eng/.../marine/YYYY/MXXCXXXX/MXXCXXXX.html
    report_ids = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        match = re.search(r"(m\d{2}[a-z]\d{4})", href, re.IGNORECASE)
        if match:
            report_id = match.group(1).upper()
            if report_id not in report_ids:
                report_ids.append(report_id)

    print(f"  Found {len(report_ids)} report IDs")

    downloaded = 0
    for report_id in report_ids:
        if 0 < max_reports <= downloaded:
            break

        # Extract year from report ID (e.g., M23C0305 -> 2023)
        year_digits = report_id[1:3]
        year = int(year_digits)
        full_year = 2000 + year if year < 90 else 1900 + year

        # Try the report page to find the PDF link
        report_page_url = f"{base_url}/eng/rapports-reports/marine/{full_year}/{report_id.lower()}/{report_id.lower()}.html"

        time.sleep(DELAY_BETWEEN_REQUESTS)

        try:
            resp = requests.get(report_page_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                # Try alternate URL pattern (investigation page)
                report_page_url = f"{base_url}/eng/enquetes-investigations/marine/{full_year}/{report_id.lower()}/{report_id.lower()}.html"
                resp = requests.get(report_page_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
                if resp.status_code != 200:
                    continue
        except requests.RequestException:
            continue

        report_soup = BeautifulSoup(resp.text, "html.parser")

        # Look for PDF download link
        pdf_url = None
        for a_tag in report_soup.find_all("a", href=True):
            href = a_tag["href"]
            if ".pdf" in href.lower() and report_id.lower() in href.lower():
                pdf_url = urljoin(base_url, href)
                break

        # Also check for common PDF pattern
        if not pdf_url:
            for a_tag in report_soup.find_all("a", href=True):
                href = a_tag["href"]
                if ".pdf" in href.lower() and ("ENG" in href or "eng" in href):
                    pdf_url = urljoin(base_url, href)
                    break

        if pdf_url:
            filename = f"{report_id}.pdf"
            if _download_pdf(pdf_url, out_dir / filename):
                downloaded += 1
        else:
            # Save the HTML report text as fallback
            text_content = report_soup.get_text(separator="\n", strip=True)
            if len(text_content) > 500:  # has meaningful content
                txt_path = out_dir / f"{report_id}.txt"
                if not txt_path.exists():
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text_content)
                    print(f"  [OK] {report_id}.txt (HTML text fallback)")
                    downloaded += 1

    print(f"\nTSB: Downloaded {downloaded} reports to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# Shared helpers for new scrapers
# ---------------------------------------------------------------------------
def _fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch a URL and return parsed BeautifulSoup, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            print(f"  [FAIL] HTTP {resp.status_code}: {url}")
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [ERR] {e}")
        return None


def _save_html_text(url: str, dest: Path, soup: BeautifulSoup) -> bool:
    """Save HTML page text as .txt fallback. Returns True if meaningful content saved."""
    if dest.exists():
        print(f"  [SKIP] Already exists: {dest.name}")
        return True
    text = soup.get_text(separator="\n", strip=True)
    if len(text) < 500:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [OK] {dest.name} (HTML text)")
    return True


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename by replacing non-alphanumeric characters."""
    return re.sub(r'[^\w\-.]', '_', name)


# ---------------------------------------------------------------------------
# USCG — US Coast Guard Marine Safety Alerts
# ---------------------------------------------------------------------------
def scrape_uscg(max_reports: int = 0):
    """
    Scrape USCG Marine Safety Alerts from the DCO website.
    Each alert is a one-page PDF linked from the listing page.
    """
    out_dir = BASE_DIR / "uscg"
    out_dir.mkdir(parents=True, exist_ok=True)
    list_url = "https://www.dco.uscg.mil/Featured-Content/Alerts/Marine-Safety-Alert/"

    print(f"\n{'='*60}")
    print("USCG — US Coast Guard Marine Safety Alerts")
    print(f"{'='*60}")

    print("\nFetching alert listing...")
    soup = _fetch_page(list_url)
    if not soup:
        return 0

    pdf_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.lower().endswith(".pdf"):
            full_url = urljoin(list_url, href)
            if full_url not in pdf_links:
                pdf_links.append(full_url)

    print(f"  Found {len(pdf_links)} PDF links")

    downloaded = 0
    for pdf_url in pdf_links:
        if 0 < max_reports <= downloaded:
            break
        filename = _sanitize_filename(pdf_url.split("/")[-1])
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        if _download_pdf(pdf_url, out_dir / filename):
            downloaded += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nUSCG: Downloaded {downloaded} alerts to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# ATSB — Australian Transport Safety Bureau
# ---------------------------------------------------------------------------
def scrape_atsb(max_reports: int = 0):
    """
    Scrape ATSB marine investigation reports.
    Strategy: paginate the marine investigations listing, find PDF links.
    """
    out_dir = BASE_DIR / "atsb"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.atsb.gov.au"
    list_url = f"{base_url}/marine/investigations"

    print(f"\n{'='*60}")
    print("ATSB — Australian Transport Safety Bureau")
    print(f"{'='*60}")

    downloaded = 0
    page = 0

    while True:
        if 0 < max_reports <= downloaded:
            break

        url = f"{list_url}?page={page}" if page > 0 else list_url
        print(f"\nFetching listing page {page + 1}...")

        soup = _fetch_page(url)
        if not soup:
            break

        report_links = []
        for a_tag in soup.select("a[href*='/marine/investigations/']"):
            href = a_tag.get("href", "")
            if href and href != "/marine/investigations/" and "?" not in href:
                full_url = urljoin(base_url, href)
                if full_url not in report_links:
                    report_links.append(full_url)

        if not report_links:
            print(f"  No more reports on page {page + 1}.")
            break

        print(f"  Found {len(report_links)} report links")

        for report_url in report_links:
            if 0 < max_reports <= downloaded:
                break

            time.sleep(DELAY_BETWEEN_REQUESTS)
            report_soup = _fetch_page(report_url)
            if not report_soup:
                continue

            pdf_found = False
            for a_tag in report_soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.lower().endswith(".pdf"):
                    pdf_url = urljoin(report_url, href)
                    filename = _sanitize_filename(pdf_url.split("/")[-1])
                    if _download_pdf(pdf_url, out_dir / filename):
                        downloaded += 1
                        pdf_found = True
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    if 0 < max_reports <= downloaded:
                        break

            if not pdf_found:
                slug = report_url.rstrip("/").split("/")[-1]
                if _save_html_text(report_url, out_dir / f"{slug}.txt", report_soup):
                    downloaded += 1

        # Check for next page
        next_link = soup.find("a", {"rel": "next"})
        if not next_link:
            next_link = soup.find("a", string=re.compile(r"Next|›|»", re.IGNORECASE))
        if not next_link:
            break

        page += 1

    print(f"\nATSB: Downloaded {downloaded} reports to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# DMAIB — Danish Maritime Accident Investigation Board
# ---------------------------------------------------------------------------
def scrape_dmaib(max_reports: int = 0):
    """
    Scrape DMAIB reports from dmaib.dk.
    Strategy: listing page with direct PDF links or report sub-pages.
    """
    out_dir = BASE_DIR / "dmaib"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://dmaib.dk"
    list_url = f"{base_url}/reports/"

    print(f"\n{'='*60}")
    print("DMAIB — Danish Maritime Accident Investigation Board")
    print(f"{'='*60}")

    downloaded = 0
    page = 1

    while True:
        if 0 < max_reports <= downloaded:
            break

        url = f"{list_url}page/{page}/" if page > 1 else list_url
        print(f"\nFetching listing page {page}...")

        soup = _fetch_page(url)
        if not soup:
            break

        pdf_links = []
        report_links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)
            if href.lower().endswith(".pdf"):
                if full_url not in pdf_links:
                    pdf_links.append(full_url)
            elif "/reports/" in href and href != "/reports/" and "page/" not in href:
                if full_url not in report_links and full_url != list_url:
                    report_links.append(full_url)

        if not pdf_links and not report_links:
            print(f"  No more reports on page {page}.")
            break

        for pdf_url in pdf_links:
            if 0 < max_reports <= downloaded:
                break
            filename = _sanitize_filename(pdf_url.split("/")[-1])
            if _download_pdf(pdf_url, out_dir / filename):
                downloaded += 1
            time.sleep(DELAY_BETWEEN_REQUESTS)

        for report_url in report_links:
            if 0 < max_reports <= downloaded:
                break
            time.sleep(DELAY_BETWEEN_REQUESTS)
            report_soup = _fetch_page(report_url)
            if not report_soup:
                continue
            for a_tag in report_soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.lower().endswith(".pdf"):
                    pdf_url = urljoin(report_url, href)
                    filename = _sanitize_filename(pdf_url.split("/")[-1])
                    if _download_pdf(pdf_url, out_dir / filename):
                        downloaded += 1
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    break

        next_link = soup.find("a", string=re.compile(r"Next|Næste|›|»", re.IGNORECASE))
        if not next_link:
            next_link = soup.find("a", class_=re.compile(r"next", re.IGNORECASE))
        if not next_link:
            break

        page += 1

    print(f"\nDMAIB: Downloaded {downloaded} reports to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# BSU — German Federal Bureau of Maritime Casualty Investigation
# ---------------------------------------------------------------------------
def scrape_bsu(max_reports: int = 0):
    """
    Scrape BSU investigation reports (English versions).
    Strategy: paginate the listing, visit report pages, find PDF links.
    """
    out_dir = BASE_DIR / "bsu"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.bsu-bund.de"
    list_url = f"{base_url}/EN/Publications/Investigation_reports/investigation_reports_node.html"

    print(f"\n{'='*60}")
    print("BSU — German Federal Bureau of Maritime Casualty Investigation")
    print(f"{'='*60}")

    print("\nFetching report listing...")
    soup = _fetch_page(list_url)
    if not soup:
        return 0

    # Collect all pagination URLs
    page_urls = [list_url]
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "investigation_reports_node" in href:
            full_url = urljoin(base_url, href)
            if full_url not in page_urls:
                page_urls.append(full_url)

    print(f"  Found {len(page_urls)} listing pages")

    downloaded = 0
    for page_url in page_urls:
        if 0 < max_reports <= downloaded:
            break

        if page_url != list_url:
            time.sleep(DELAY_BETWEEN_REQUESTS)
            soup = _fetch_page(page_url)
            if not soup:
                continue

        for a_tag in soup.find_all("a", href=True):
            if 0 < max_reports <= downloaded:
                break
            href = a_tag["href"]

            if href.lower().endswith(".pdf"):
                pdf_url = urljoin(base_url, href)
                filename = _sanitize_filename(pdf_url.split("/")[-1].split("?")[0])
                if not filename.endswith(".pdf"):
                    filename += ".pdf"
                if _download_pdf(pdf_url, out_dir / filename):
                    downloaded += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)

            elif "/Investigation_reports/" in href and "node" not in href:
                report_url = urljoin(base_url, href)
                time.sleep(DELAY_BETWEEN_REQUESTS)
                report_soup = _fetch_page(report_url)
                if not report_soup:
                    continue
                for inner_a in report_soup.find_all("a", href=True):
                    inner_href = inner_a["href"]
                    if inner_href.lower().endswith(".pdf"):
                        pdf_url = urljoin(base_url, inner_href)
                        filename = _sanitize_filename(pdf_url.split("/")[-1].split("?")[0])
                        if _download_pdf(pdf_url, out_dir / filename):
                            downloaded += 1
                        time.sleep(DELAY_BETWEEN_REQUESTS)
                        break

    print(f"\nBSU: Downloaded {downloaded} reports to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# MAIB Safety Digests — UK annual lessons-learned compilations
# ---------------------------------------------------------------------------
def scrape_maib_digests(max_reports: int = 0):
    """
    Scrape MAIB Safety Digest PDFs from gov.uk.
    These are annual compilations of lessons learned from marine accidents.
    """
    out_dir = BASE_DIR / "maib_digests"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.gov.uk"
    list_url = f"{base_url}/government/collections/maib-safety-digests"

    print(f"\n{'='*60}")
    print("MAIB Safety Digests — UK Lessons Learned Compilations")
    print(f"{'='*60}")

    print("\nFetching digest collection page...")
    soup = _fetch_page(list_url)
    if not soup:
        return 0

    digest_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "/government/publications/" in href and ("safety-digest" in href.lower() or "maib" in href.lower()):
            full_url = urljoin(base_url, href)
            if full_url not in digest_links:
                digest_links.append(full_url)

    print(f"  Found {len(digest_links)} digest publication links")

    downloaded = 0
    for digest_url in digest_links:
        if 0 < max_reports <= downloaded:
            break

        time.sleep(DELAY_BETWEEN_REQUESTS)
        digest_soup = _fetch_page(digest_url)
        if not digest_soup:
            continue

        for a_tag in digest_soup.find_all("a", href=True):
            if 0 < max_reports <= downloaded:
                break
            href = a_tag["href"]
            if href.lower().endswith(".pdf"):
                pdf_url = urljoin(base_url, href)
                filename = _sanitize_filename(pdf_url.split("/")[-1])
                if _download_pdf(pdf_url, out_dir / filename):
                    downloaded += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nMAIB Digests: Downloaded {downloaded} digests to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# JTSB — Japan Transport Safety Board
# ---------------------------------------------------------------------------
def scrape_jtsb(max_reports: int = 0):
    """
    Scrape JTSB marine accident reports (English versions).
    Strategy: parse the English report listing page, find PDF links.
    """
    out_dir = BASE_DIR / "jtsb"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.mlit.go.jp"
    list_url = f"{base_url}/jtsb/ship/rep-acci/ship-accident-list-e.html"

    print(f"\n{'='*60}")
    print("JTSB — Japan Transport Safety Board")
    print(f"{'='*60}")

    print("\nFetching English report listing...")
    soup = _fetch_page(list_url)
    if not soup:
        return 0

    pdf_links = []
    report_links = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.lower().endswith(".pdf"):
            full_url = urljoin(list_url, href)
            if full_url not in pdf_links:
                pdf_links.append(full_url)
        elif "/jtsb/ship/" in href and href != list_url and not href.lower().endswith(".pdf"):
            full_url = urljoin(base_url, href)
            if full_url not in report_links:
                report_links.append(full_url)

    print(f"  Found {len(pdf_links)} direct PDFs and {len(report_links)} report pages")

    downloaded = 0

    for pdf_url in pdf_links:
        if 0 < max_reports <= downloaded:
            break
        filename = _sanitize_filename(pdf_url.split("/")[-1])
        if _download_pdf(pdf_url, out_dir / filename):
            downloaded += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)

    for report_url in report_links:
        if 0 < max_reports <= downloaded:
            break
        time.sleep(DELAY_BETWEEN_REQUESTS)
        report_soup = _fetch_page(report_url)
        if not report_soup:
            continue
        for a_tag in report_soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.lower().endswith(".pdf"):
                pdf_url = urljoin(report_url, href)
                filename = _sanitize_filename(pdf_url.split("/")[-1])
                if _download_pdf(pdf_url, out_dir / filename):
                    downloaded += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)
                if 0 < max_reports <= downloaded:
                    break

    print(f"\nJTSB: Downloaded {downloaded} reports to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# SHK — Swedish Accident Investigation Authority
# ---------------------------------------------------------------------------
def scrape_shk(max_reports: int = 0):
    """
    Scrape SHK marine investigation reports (English versions).
    Strategy: parse the shipping investigations page, find report PDFs.
    """
    out_dir = BASE_DIR / "shk"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.havkom.se"
    list_url = f"{base_url}/en/investigations/shipping"

    print(f"\n{'='*60}")
    print("SHK — Swedish Accident Investigation Authority")
    print(f"{'='*60}")

    downloaded = 0
    page = 1

    while True:
        if 0 < max_reports <= downloaded:
            break

        url = f"{list_url}?page={page}" if page > 1 else list_url
        print(f"\nFetching listing page {page}...")

        soup = _fetch_page(url)
        if not soup:
            break

        report_links = []
        pdf_links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.lower().endswith(".pdf"):
                full_url = urljoin(base_url, href)
                if full_url not in pdf_links:
                    pdf_links.append(full_url)
            elif "/en/investigations/" in href and href != "/en/investigations/shipping" and "?" not in href:
                full_url = urljoin(base_url, href)
                if full_url not in report_links and full_url != list_url:
                    report_links.append(full_url)

        if not report_links and not pdf_links:
            print(f"  No more reports on page {page}.")
            break

        print(f"  Found {len(pdf_links)} PDFs and {len(report_links)} report pages")

        for pdf_url in pdf_links:
            if 0 < max_reports <= downloaded:
                break
            filename = _sanitize_filename(pdf_url.split("/")[-1])
            if _download_pdf(pdf_url, out_dir / filename):
                downloaded += 1
            time.sleep(DELAY_BETWEEN_REQUESTS)

        for report_url in report_links:
            if 0 < max_reports <= downloaded:
                break
            time.sleep(DELAY_BETWEEN_REQUESTS)
            report_soup = _fetch_page(report_url)
            if not report_soup:
                continue

            pdf_found = False
            for a_tag in report_soup.find_all("a", href=True):
                href = a_tag["href"]
                if href.lower().endswith(".pdf"):
                    pdf_url = urljoin(report_url, href)
                    filename = _sanitize_filename(pdf_url.split("/")[-1])
                    if _download_pdf(pdf_url, out_dir / filename):
                        downloaded += 1
                        pdf_found = True
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    break

            if not pdf_found:
                slug = report_url.rstrip("/").split("/")[-1]
                if _save_html_text(report_url, out_dir / f"{slug}.txt", report_soup):
                    downloaded += 1

        next_link = soup.find("a", string=re.compile(r"Next|Nästa|›|»", re.IGNORECASE))
        if not next_link:
            next_link = soup.find("a", class_=re.compile(r"next", re.IGNORECASE))
        if not next_link:
            next_link = soup.find("a", rel="next")
        if not next_link:
            break

        page += 1

    print(f"\nSHK: Downloaded {downloaded} reports to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SCRAPERS = {
    "maib": ("MAIB (UK)", scrape_maib),
    "ntsb": ("NTSB (US)", scrape_ntsb),
    "tsb": ("TSB (Canada)", scrape_tsb),
    "uscg": ("USCG Marine Safety Alerts", scrape_uscg),
    "atsb": ("ATSB (Australia)", scrape_atsb),
    "dmaib": ("DMAIB (Denmark)", scrape_dmaib),
    "bsu": ("BSU (Germany)", scrape_bsu),
    "maib_digests": ("MAIB Safety Digests", scrape_maib_digests),
    "jtsb": ("JTSB (Japan)", scrape_jtsb),
    "shk": ("SHK (Sweden)", scrape_shk),
}


def main():
    parser = argparse.ArgumentParser(description="Download maritime accident investigation reports")
    parser.add_argument(
        "--source",
        choices=list(SCRAPERS.keys()) + ["all"],
        default="all",
        help="Which source to download from (default: all)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Maximum reports per source (0 = unlimited)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available sources and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available sources:")
        for key, (name, _) in SCRAPERS.items():
            print(f"  {key:15s} — {name}")
        return

    print("Maritime Safety Report Scraper")
    print(f"Output directory: {BASE_DIR}")
    print(f"Source: {args.source} | Max per source: {args.max or 'unlimited'}")

    total = 0
    sources = SCRAPERS.keys() if args.source == "all" else [args.source]

    for source_key in sources:
        name, scrape_fn = SCRAPERS[source_key]
        total += scrape_fn(max_reports=args.max)

    print(f"\n{'='*60}")
    print(f"TOTAL: {total} reports downloaded")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
