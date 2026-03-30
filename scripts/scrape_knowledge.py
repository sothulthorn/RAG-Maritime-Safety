"""
Maritime Safety Knowledge Scraper
Downloads regulations, guidelines, reference articles, and research papers:
  - COLREGS       (International Collision Regulations — full text via USCG)
  - ISM Code      (International Safety Management Code)
  - COSWP         (UK Code of Safe Working Practices for Merchant Seafarers)
  - MCA MGNs      (UK Marine Guidance Notes — 300+ PDFs)
  - MCA MSNs      (UK Merchant Shipping Notices)
  - eCFR Title 46 (US Shipping Regulations)
  - Wikipedia     (Maritime safety articles via API)
  - Semantic Scholar (Open-access maritime safety research papers)
  - WMU Commons   (World Maritime University dissertations)
  - P&I Clubs     (Loss prevention publications — Gard, etc.)
  - FAO Fishing   (FAO fishing vessel safety guidelines)
  - EMSA          (European Maritime Safety Agency reports)

Usage:
    python scripts/scrape_knowledge.py                       # All sources
    python scripts/scrape_knowledge.py --source colregs      # Single source
    python scripts/scrape_knowledge.py --max 10              # Limit per source
    python scripts/scrape_knowledge.py --list                # Show available sources
"""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, quote

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent / "data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Maritime Safety Research Bot; academic use)"
}
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 1.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF file if it doesn't already exist."""
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
        if size_kb < 5:
            dest.unlink()
            print(f"  [SKIP] File too small ({size_kb:.0f} KB): {url}")
            return False

        print(f"  [OK] {dest.name} ({size_kb:.0f} KB)")
        return True
    except requests.RequestException as e:
        print(f"  [ERR] {e}")
        return False


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


def _save_text(text: str, dest: Path, label: str = "") -> bool:
    """Save text content to a file if it has meaningful length."""
    if dest.exists():
        print(f"  [SKIP] Already exists: {dest.name}")
        return True
    if len(text.strip()) < 200:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    size_kb = dest.stat().st_size / 1024
    print(f"  [OK] {dest.name} ({size_kb:.0f} KB) {label}")
    return True


def _sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-.]', '_', name)


# ---------------------------------------------------------------------------
# COLREGS — International Collision Regulations (full text via USCG)
# ---------------------------------------------------------------------------
def scrape_colregs(max_reports: int = 0):
    """Download the COLREGS full text PDF from the USCG Navigation Center."""
    out_dir = BASE_DIR / "regulations" / "colregs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("COLREGS — International Collision Regulations")
    print(f"{'='*60}")

    pdf_url = "https://www.navcen.uscg.gov/sites/default/files/pdf/navRules/navrules.pdf"
    downloaded = 0

    if _download_pdf(pdf_url, out_dir / "colregs_navigation_rules.pdf"):
        downloaded += 1

    # Also try to scrape the HTML version for cleaner text
    html_url = "https://www.navcen.uscg.gov/navigation-rules-amalgamated"
    time.sleep(DELAY_BETWEEN_REQUESTS)
    soup = _fetch_page(html_url)
    if soup:
        text = soup.get_text(separator="\n", strip=True)
        if _save_text(text, out_dir / "colregs_rules_text.txt", "(HTML text)"):
            downloaded += 1

    print(f"\nCOLREGS: Downloaded {downloaded} files to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# ISM Code — International Safety Management Code
# ---------------------------------------------------------------------------
def scrape_ism_code(max_reports: int = 0):
    """Download the ISM Code PDF."""
    out_dir = BASE_DIR / "regulations" / "ism_code"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("ISM Code — International Safety Management Code")
    print(f"{'='*60}")

    pdf_url = "https://maritimesafetyinnovationlab.org/wp-content/uploads/2014/02/ism-code.pdf"
    downloaded = 0

    if _download_pdf(pdf_url, out_dir / "ism_code.pdf"):
        downloaded += 1

    print(f"\nISM Code: Downloaded {downloaded} files to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# COSWP — Code of Safe Working Practices for Merchant Seafarers
# ---------------------------------------------------------------------------
def scrape_coswp(max_reports: int = 0):
    """Download the COSWP PDF from gov.uk."""
    out_dir = BASE_DIR / "regulations" / "coswp"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("COSWP — Code of Safe Working Practices for Merchant Seafarers")
    print(f"{'='*60}")

    # The latest edition from gov.uk
    pub_url = "https://www.gov.uk/government/publications/code-of-safe-working-practices-for-merchant-seafarers-2025-edition"

    downloaded = 0
    soup = _fetch_page(pub_url)
    if soup:
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.lower().endswith(".pdf"):
                pdf_url = urljoin("https://www.gov.uk", href)
                filename = _sanitize_filename(pdf_url.split("/")[-1])
                if _download_pdf(pdf_url, out_dir / filename):
                    downloaded += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)

    # Direct fallback URL
    if downloaded == 0:
        fallback = "https://assets.publishing.service.gov.uk/media/67e527b5a82c168e578c9b5a/12464_MCA_COSWP_BLACK_AND_WHITE_v3_0W.pdf"
        if _download_pdf(fallback, out_dir / "coswp_2025.pdf"):
            downloaded += 1

    print(f"\nCOSWP: Downloaded {downloaded} files to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# MCA MGNs — UK Marine Guidance Notes
# ---------------------------------------------------------------------------
def scrape_mca_mgns(max_reports: int = 0):
    """
    Scrape Marine Guidance Notes from gov.uk.
    Strategy: paginate the collection page, visit each publication, download PDFs.
    """
    out_dir = BASE_DIR / "regulations" / "mca_mgns"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.gov.uk"
    list_url = f"{base_url}/government/collections/marine-guidance-notices-mgns"

    print(f"\n{'='*60}")
    print("MCA MGNs — UK Marine Guidance Notes")
    print(f"{'='*60}")

    print("\nFetching MGN collection page...")
    soup = _fetch_page(list_url)
    if not soup:
        return 0

    # Find links to individual MGN publication pages
    pub_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "/government/publications/" in href and "mgn" in href.lower():
            full_url = urljoin(base_url, href)
            if full_url not in pub_links:
                pub_links.append(full_url)

    print(f"  Found {len(pub_links)} MGN publication links")

    downloaded = 0
    for pub_url in pub_links:
        if 0 < max_reports <= downloaded:
            break

        time.sleep(DELAY_BETWEEN_REQUESTS)
        pub_soup = _fetch_page(pub_url)
        if not pub_soup:
            continue

        for a_tag in pub_soup.find_all("a", href=True):
            if 0 < max_reports <= downloaded:
                break
            href = a_tag["href"]
            if href.lower().endswith(".pdf"):
                pdf_url = urljoin(base_url, href)
                filename = _sanitize_filename(pdf_url.split("/")[-1])
                if _download_pdf(pdf_url, out_dir / filename):
                    downloaded += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nMCA MGNs: Downloaded {downloaded} guidance notes to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# MCA MSNs — UK Merchant Shipping Notices
# ---------------------------------------------------------------------------
def scrape_mca_msns(max_reports: int = 0):
    """
    Scrape Merchant Shipping Notices from gov.uk.
    Strategy: same as MGNs — paginate collection, download PDFs.
    """
    out_dir = BASE_DIR / "regulations" / "mca_msns"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.gov.uk"
    list_url = f"{base_url}/government/collections/merchant-shipping-notices-msns"

    print(f"\n{'='*60}")
    print("MCA MSNs — UK Merchant Shipping Notices")
    print(f"{'='*60}")

    print("\nFetching MSN collection page...")
    soup = _fetch_page(list_url)
    if not soup:
        return 0

    pub_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "/government/publications/" in href and "msn" in href.lower():
            full_url = urljoin(base_url, href)
            if full_url not in pub_links:
                pub_links.append(full_url)

    print(f"  Found {len(pub_links)} MSN publication links")

    downloaded = 0
    for pub_url in pub_links:
        if 0 < max_reports <= downloaded:
            break

        time.sleep(DELAY_BETWEEN_REQUESTS)
        pub_soup = _fetch_page(pub_url)
        if not pub_soup:
            continue

        for a_tag in pub_soup.find_all("a", href=True):
            if 0 < max_reports <= downloaded:
                break
            href = a_tag["href"]
            if href.lower().endswith(".pdf"):
                pdf_url = urljoin(base_url, href)
                filename = _sanitize_filename(pdf_url.split("/")[-1])
                if _download_pdf(pdf_url, out_dir / filename):
                    downloaded += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nMCA MSNs: Downloaded {downloaded} notices to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# eCFR Title 46 — US Shipping Regulations
# ---------------------------------------------------------------------------
def scrape_ecfr(max_reports: int = 0):
    """
    Scrape US CFR Title 46 (Shipping) from Cornell LII.
    Strategy: fetch chapter index, then download each subchapter as text.
    """
    out_dir = BASE_DIR / "regulations" / "ecfr_title46"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://www.law.cornell.edu"
    index_url = f"{base_url}/cfr/text/46"

    print(f"\n{'='*60}")
    print("eCFR Title 46 — US Shipping Regulations")
    print(f"{'='*60}")

    print("\nFetching Title 46 index...")
    soup = _fetch_page(index_url)
    if not soup:
        return 0

    # Find chapter/subchapter links
    chapter_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "/cfr/text/46/" in href and href != "/cfr/text/46":
            full_url = urljoin(base_url, href)
            if full_url not in chapter_links:
                chapter_links.append(full_url)

    print(f"  Found {len(chapter_links)} chapter/part links")

    downloaded = 0
    for chapter_url in chapter_links:
        if 0 < max_reports <= downloaded:
            break

        time.sleep(DELAY_BETWEEN_REQUESTS)
        ch_soup = _fetch_page(chapter_url)
        if not ch_soup:
            continue

        # Extract the main content text
        content_div = ch_soup.find("div", class_="tab-pane") or ch_soup.find("div", id="content")
        if content_div:
            text = content_div.get_text(separator="\n", strip=True)
        else:
            text = ch_soup.get_text(separator="\n", strip=True)

        # Create filename from URL path
        slug = chapter_url.rstrip("/").replace(base_url, "").replace("/cfr/text/46/", "")
        slug = _sanitize_filename(slug) or "index"
        filename = f"cfr46_{slug}.txt"

        if _save_text(text, out_dir / filename):
            downloaded += 1

    print(f"\neCFR Title 46: Downloaded {downloaded} sections to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# Wikipedia — Maritime safety articles via API
# ---------------------------------------------------------------------------
WIKIPEDIA_ARTICLES = [
    # Conventions & Regulations
    "SOLAS", "MARPOL", "International_Regulations_for_Preventing_Collisions_at_Sea",
    "International_Convention_on_Standards_of_Training,_Certification_and_Watchkeeping_for_Seafarers",
    "ISM_Code", "Maritime_Labour_Convention", "International_Ship_and_Port_Facility_Security_Code",
    "International_Convention_on_Load_Lines",
    # Safety Equipment & Systems
    "Lifeboat_(shipboard)", "Inflatable_liferaft", "EPIRB",
    "Search_and_rescue_transponder", "Automatic_identification_system",
    "Global_Maritime_Distress_and_Safety_System", "Voyage_data_recorder",
    "Fire_protection_on_ships", "Personal_flotation_device",
    # Navigation
    "Marine_navigation", "Pilotage", "Radar_navigation",
    "Electronic_Chart_Display_and_Information_System", "Maritime_pilot",
    "Navigational_aid", "Collision_avoidance_(navigation)",
    # Ship Types
    "Tanker_(ship)", "Container_ship", "Bulk_carrier", "Roll-on/roll-off",
    "Passenger_ship", "Fishing_vessel", "Tugboat", "Oil_tanker",
    # Safety Concepts
    "Maritime_safety", "Capsizing", "Ship_grounding", "Foundering",
    "Ship_stability", "Free_surface_effect", "Metacentric_height",
    "Weathertight_integrity", "Watertight_compartment", "Freeboard_(nautical)",
    "Ballast_tank", "Man_overboard", "Distress_signal",
    # Maritime Incidents (notable for learning)
    "Sinking_of_MV_Sewol", "Costa_Concordia_disaster", "MS_Estonia",
    "MV_Doña_Paz", "SS_Edmund_Fitzgerald", "Herald_of_Free_Enterprise",
    # Organizations
    "International_Maritime_Organization", "Maritime_and_Coastguard_Agency",
    "United_States_Coast_Guard", "Classification_society",
    # Operations & Hazards
    "Mooring_(watercraft)", "Anchoring", "Ship_breaking",
    "Enclosed_space_rescue", "Carbon_monoxide_poisoning",
    "Hypothermia", "Sea_state", "Beaufort_scale",
]


def scrape_wikipedia(max_reports: int = 0):
    """
    Fetch maritime safety articles from Wikipedia via the REST API.
    Saves plain text extracts for each article.
    """
    out_dir = BASE_DIR / "knowledge" / "wikipedia"
    out_dir.mkdir(parents=True, exist_ok=True)
    api_url = "https://en.wikipedia.org/api/rest_v1/page/summary"

    print(f"\n{'='*60}")
    print("Wikipedia — Maritime Safety Articles")
    print(f"{'='*60}")

    downloaded = 0
    articles = WIKIPEDIA_ARTICLES
    if 0 < max_reports < len(articles):
        articles = articles[:max_reports]

    print(f"\n  Fetching {len(articles)} articles...")

    for title in articles:
        if 0 < max_reports <= downloaded:
            break

        filename = _sanitize_filename(title) + ".txt"
        dest = out_dir / filename

        if dest.exists():
            print(f"  [SKIP] Already exists: {filename}")
            downloaded += 1
            continue

        # Use the full page HTML endpoint for complete article text
        full_url = f"https://en.wikipedia.org/w/api.php?action=parse&page={quote(title)}&prop=text&format=json&redirects=1"
        time.sleep(0.5)  # Respectful rate for Wikipedia API

        try:
            resp = requests.get(full_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                print(f"  [FAIL] HTTP {resp.status_code}: {title}")
                continue

            data = resp.json()
            if "error" in data:
                print(f"  [SKIP] Not found: {title}")
                continue

            html_text = data.get("parse", {}).get("text", {}).get("*", "")
            if not html_text:
                continue

            # Parse HTML to plain text
            soup = BeautifulSoup(html_text, "html.parser")

            # Remove references, navigation boxes, edit links
            for tag in soup.find_all(["sup", "span"], class_=["reference", "mw-editsection"]):
                tag.decompose()
            for tag in soup.find_all("div", class_=["navbox", "metadata", "sistersitebox"]):
                tag.decompose()
            for tag in soup.find_all("table", class_="infobox"):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # Prepend the title
            display_title = data["parse"].get("title", title.replace("_", " "))
            text = f"# {display_title}\n\nSource: Wikipedia\n\n{text}"

            if _save_text(text, dest):
                downloaded += 1

        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  [ERR] {title}: {e}")

    print(f"\nWikipedia: Downloaded {downloaded} articles to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# Semantic Scholar — Open-access maritime safety research papers
# ---------------------------------------------------------------------------
SEMANTIC_SCHOLAR_QUERIES = [
    "maritime safety accident prevention",
    "ship collision avoidance safety",
    "fishing vessel safety capsize",
    "maritime crew fatigue human factors",
    "port state control inspection",
    "ship fire safety detection",
    "maritime risk assessment",
    "SOLAS maritime regulation compliance",
    "lifeboat evacuation passenger ship",
    "maritime cybersecurity autonomous vessel",
]


def scrape_semantic_scholar(max_reports: int = 0):
    """
    Fetch open-access maritime safety research papers via Semantic Scholar API.
    Downloads paper metadata + abstracts, and PDFs when available.
    """
    out_dir = BASE_DIR / "knowledge" / "research_papers"
    out_dir.mkdir(parents=True, exist_ok=True)
    api_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    print(f"\n{'='*60}")
    print("Semantic Scholar — Maritime Safety Research Papers")
    print(f"{'='*60}")

    downloaded = 0
    seen_ids = set()
    limit_per_query = max_reports if max_reports > 0 else 50

    for query in SEMANTIC_SCHOLAR_QUERIES:
        if 0 < max_reports <= downloaded:
            break

        print(f"\n  Searching: '{query}'...")
        time.sleep(DELAY_BETWEEN_REQUESTS)

        try:
            params = {
                "query": query,
                "limit": min(limit_per_query, 100),
                "fields": "title,abstract,year,authors,openAccessPdf,url",
                "openAccessPdf": "",
            }
            resp = requests.get(api_url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                print("  [WARN] Rate limited, waiting 30s...")
                time.sleep(30)
                resp = requests.get(api_url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                print(f"  [FAIL] HTTP {resp.status_code}")
                continue

            data = resp.json()
            papers = data.get("data", [])
            print(f"  Found {len(papers)} papers")

            for paper in papers:
                if 0 < max_reports <= downloaded:
                    break

                paper_id = paper.get("paperId", "")
                if not paper_id or paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)

                title = paper.get("title", "Untitled")
                abstract = paper.get("abstract", "")
                year = paper.get("year", "")
                authors = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:5])

                if not abstract:
                    continue

                # Save abstract as text
                slug = _sanitize_filename(f"{year}_{title[:80]}")
                txt_path = out_dir / f"{slug}.txt"

                content = (
                    f"# {title}\n\n"
                    f"Authors: {authors}\n"
                    f"Year: {year}\n"
                    f"Source: Semantic Scholar ({paper_id})\n\n"
                    f"## Abstract\n\n{abstract}\n"
                )

                if _save_text(content, txt_path):
                    downloaded += 1

                # Try to download open-access PDF
                oa_pdf = paper.get("openAccessPdf")
                if oa_pdf and oa_pdf.get("url"):
                    pdf_path = out_dir / f"{slug}.pdf"
                    time.sleep(0.5)
                    _download_pdf(oa_pdf["url"], pdf_path)

        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  [ERR] {query}: {e}")

    print(f"\nSemantic Scholar: Downloaded {downloaded} papers to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# WMU Maritime Commons — World Maritime University dissertations
# ---------------------------------------------------------------------------
def scrape_wmu(max_reports: int = 0):
    """
    Scrape open-access dissertations from WMU Maritime Commons.
    Strategy: paginate the dissertation listing, download PDFs.
    """
    out_dir = BASE_DIR / "knowledge" / "wmu_dissertations"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://commons.wmu.se"
    list_url = f"{base_url}/all_dissertations/"

    print(f"\n{'='*60}")
    print("WMU Maritime Commons — Dissertations")
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

        # Find dissertation links
        diss_links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "/all_dissertations/" in href and href != "/all_dissertations/":
                full_url = urljoin(base_url, href)
                if full_url not in diss_links:
                    diss_links.append(full_url)

        if not diss_links:
            print(f"  No more dissertations on page {page}.")
            break

        print(f"  Found {len(diss_links)} dissertation links")

        for diss_url in diss_links:
            if 0 < max_reports <= downloaded:
                break

            time.sleep(DELAY_BETWEEN_REQUESTS)
            diss_soup = _fetch_page(diss_url)
            if not diss_soup:
                continue

            # Find the PDF download link
            for a_tag in diss_soup.find_all("a", href=True):
                href = a_tag["href"]
                if ".pdf" in href.lower() or "/cgi/viewcontent" in href:
                    pdf_url = urljoin(base_url, href)
                    slug = diss_url.rstrip("/").split("/")[-1]
                    filename = _sanitize_filename(slug) + ".pdf"
                    if _download_pdf(pdf_url, out_dir / filename):
                        downloaded += 1
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    break

        # Check for next page
        next_link = soup.find("a", string=re.compile(r"Next|›|»", re.IGNORECASE))
        if not next_link:
            next_link = soup.find("a", class_=re.compile(r"next", re.IGNORECASE))
        if not next_link:
            break

        page += 1

    print(f"\nWMU: Downloaded {downloaded} dissertations to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# P&I Clubs — Loss prevention publications
# ---------------------------------------------------------------------------
def scrape_pni_clubs(max_reports: int = 0):
    """
    Download freely available P&I Club loss prevention publications.
    Includes Gard guidance documents and other club publications.
    """
    out_dir = BASE_DIR / "knowledge" / "pni_clubs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("P&I Clubs — Loss Prevention Publications")
    print(f"{'='*60}")

    # Known direct PDF URLs for major freely available P&I publications
    known_pdfs = {
        "gard_guidance_to_masters.pdf":
            "https://assets.eu.ctfassets.net/jchk06tdml2i/6Jswvno7jDSn4ayVD9kOxu/59100454047299edd36dac3f1dc1cf6e/Guidances_GTM_web.pdf",
        "gard_maritime_claims_guide.pdf":
            "https://assets.eu.ctfassets.net/jchk06tdml2i/2Ds7mlTgVDqo5zkGaT2VLb/485c0e3ad9da58b87f078972b029c50f/Handbooks_Gard_Guidance_on_Maritime_Claims_final.pdf",
    }

    downloaded = 0
    for filename, url in known_pdfs.items():
        if 0 < max_reports <= downloaded:
            break
        if _download_pdf(url, out_dir / filename):
            downloaded += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Scrape Japan P&I Club loss prevention guides
    jp_url = "https://www.piclub.or.jp/en/lossprevention/guide"
    time.sleep(DELAY_BETWEEN_REQUESTS)
    soup = _fetch_page(jp_url)
    if soup:
        for a_tag in soup.find_all("a", href=True):
            if 0 < max_reports <= downloaded:
                break
            href = a_tag["href"]
            if href.lower().endswith(".pdf"):
                pdf_url = urljoin(jp_url, href)
                filename = _sanitize_filename(pdf_url.split("/")[-1])
                if _download_pdf(pdf_url, out_dir / filename):
                    downloaded += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nP&I Clubs: Downloaded {downloaded} publications to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# FAO — Fishing vessel safety guidelines
# ---------------------------------------------------------------------------
def scrape_fao(max_reports: int = 0):
    """Download FAO fishing vessel safety guidelines and publications."""
    out_dir = BASE_DIR / "knowledge" / "fao_fishing"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("FAO — Fishing Vessel Safety Guidelines")
    print(f"{'='*60}")

    list_url = "https://www.fao.org/fishing-safety/capacity-building/guidelines-and-codes/en/"

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
        if _download_pdf(pdf_url, out_dir / filename):
            downloaded += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nFAO: Downloaded {downloaded} guidelines to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# EMSA — European Maritime Safety Agency reports
# ---------------------------------------------------------------------------
def scrape_emsa(max_reports: int = 0):
    """Download EMSA annual safety overview reports and publications."""
    out_dir = BASE_DIR / "knowledge" / "emsa"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("EMSA — European Maritime Safety Agency Reports")
    print(f"{'='*60}")

    list_url = "https://www.emsa.europa.eu/accident-investigation.html"

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

    # Also check for report page links
    report_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "casualt" in href.lower() or "annual-overview" in href.lower():
            full_url = urljoin(list_url, href)
            if full_url not in report_links and full_url not in pdf_links:
                report_links.append(full_url)

    print(f"  Found {len(pdf_links)} PDFs and {len(report_links)} report pages")

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

    print(f"\nEMSA: Downloaded {downloaded} reports to {out_dir}")
    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SCRAPERS = {
    "colregs": ("COLREGS (Collision Regulations)", scrape_colregs),
    "ism_code": ("ISM Code", scrape_ism_code),
    "coswp": ("COSWP (Safe Working Practices)", scrape_coswp),
    "mca_mgns": ("MCA Marine Guidance Notes", scrape_mca_mgns),
    "mca_msns": ("MCA Merchant Shipping Notices", scrape_mca_msns),
    "ecfr": ("eCFR Title 46 (US Shipping Regs)", scrape_ecfr),
    "wikipedia": ("Wikipedia Maritime Articles", scrape_wikipedia),
    "semantic_scholar": ("Semantic Scholar Research Papers", scrape_semantic_scholar),
    "wmu": ("WMU Maritime Commons Dissertations", scrape_wmu),
    "pni_clubs": ("P&I Club Loss Prevention Guides", scrape_pni_clubs),
    "fao": ("FAO Fishing Vessel Safety", scrape_fao),
    "emsa": ("EMSA Safety Reports", scrape_emsa),
}


def main():
    parser = argparse.ArgumentParser(
        description="Download maritime safety knowledge: regulations, guidelines, research papers"
    )
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
        help="Maximum items per source (0 = unlimited)",
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
            print(f"  {key:20s} — {name}")
        return

    print("Maritime Safety Knowledge Scraper")
    print(f"Output directory: {BASE_DIR}")
    print(f"Source: {args.source} | Max per source: {args.max or 'unlimited'}")

    total = 0
    sources = SCRAPERS.keys() if args.source == "all" else [args.source]

    for source_key in sources:
        name, scrape_fn = SCRAPERS[source_key]
        total += scrape_fn(max_reports=args.max)

    print(f"\n{'='*60}")
    print(f"TOTAL: {total} items downloaded")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
