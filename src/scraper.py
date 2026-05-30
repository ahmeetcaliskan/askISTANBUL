"""
scraper.py
----------
Fetches all 20 Istanbul-related Wikivoyage articles via the MediaWiki API
and saves raw wikitext + metadata to data/raw/<slug>.json
"""

import json
import time
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Target pages
# ---------------------------------------------------------------------------
PAGES = [
    "Istanbul",
    "Istanbul/Historical Peninsula",
    "Istanbul/New City",
    "Istanbul/European_Bosphorus",
    "Istanbul/Asian_Bosphorus",
    "Istanbul/Eastern_Suburbs",
    "Istanbul/Western_Suburbs",
    "Istanbul/Rural",
    "Istanbul/Princes' Islands",
    "Istanbul_Airport",
    "Istanbul/Kadıköy",
    "Istanbul/Galata",
    "Sabiha_Gökçen_International_Airport",
    "Istanbul/Sile",
    "Istanbul/Bosphorus",
    "Istanbul_with_children"
]

API_URL = "https://en.wikivoyage.org/w/api.php"
HEADERS = {"User-Agent": "askISTANBUL/1.0 (ahmetcaliskan3642@gmail.com) python-requests"}
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"


def fetch_page(title: str) -> Optional[dict]:
    """Fetch wikitext and basic metadata for a single Wikivoyage page."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions|info",
        "rvprop": "content",
        "rvslots": "main",
        "inprop": "url",
        "format": "json",
        "formatversion": "2",
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    pages = data["query"]["pages"]
    if not pages:
        return None

    page = pages[0]
    if "missing" in page:
        print(f"  [SKIP] Page not found: {title!r}")
        return None

    wikitext = page["revisions"][0]["slots"]["main"]["content"]
    return {
        "title": page["title"],
        "pageid": page["pageid"],
        "url": page.get("fullurl", ""),
        "wikitext": wikitext,
    }


def slug(title: str) -> str:
    """Convert page title to a safe filename slug."""
    return title.replace("/", "_").replace(" ", "_").replace("'", "").lower()


def scrape_all(pages: Optional[list] = None, out_dir: Path = OUTPUT_DIR) -> None:
    if pages is None:
        pages = PAGES
    out_dir.mkdir(parents=True, exist_ok=True)
    fetched, skipped = 0, 0

    for title in pages:
        out_file = out_dir / f"{slug(title)}.json"
        if out_file.exists():
            print(f"  [CACHED] {title}")
            fetched += 1
            continue

        print(f"  [FETCH]  {title} ...", end=" ", flush=True)
        try:
            result = fetch_page(title)
        except Exception as exc:
            print(f"ERROR — {exc}")
            skipped += 1
            continue

        if result is None:
            skipped += 1
            continue

        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"OK  ({len(result['wikitext'])} chars)")
        fetched += 1
        time.sleep(0.5)   # be polite to the API

    print(f"\nDone. Fetched: {fetched}  Skipped/missing: {skipped}")
    print(f"Raw data saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    scrape_all()
