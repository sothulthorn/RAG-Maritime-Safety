"""
Maritime Safety Report Scraper
Downloads accident investigation reports from:
  - MAIB (UK Marine Accident Investigation Branch)
  - NTSB (US National Transportation Safety Board)
  - TSB  (Transportation Safety Board of Canada)

Usage:
    python scripts/scrape_reports.py              # Download from all sources
    python scripts/scrape_reports.py --source maib # Download MAIB only
    python scripts/scrape_reports.py --source ntsb # Download NTSB only
    python scripts/scrape_reports.py --source tsb  # Download TSB only
    python scripts/scrape_reports.py --max 10      # Limit downloads per source
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download maritime accident investigation reports")
    parser.add_argument(
        "--source",
        choices=["maib", "ntsb", "tsb", "all"],
        default="all",
        help="Which source to download from (default: all)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Maximum reports per source (0 = unlimited)",
    )
    args = parser.parse_args()

    print("Maritime Safety Report Scraper")
    print(f"Output directory: {BASE_DIR}")
    print(f"Source: {args.source} | Max per source: {args.max or 'unlimited'}")

    total = 0

    if args.source in ("maib", "all"):
        total += scrape_maib(max_reports=args.max)

    if args.source in ("ntsb", "all"):
        total += scrape_ntsb(max_reports=args.max)

    if args.source in ("tsb", "all"):
        total += scrape_tsb(max_reports=args.max)

    print(f"\n{'='*60}")
    print(f"TOTAL: {total} reports downloaded")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
