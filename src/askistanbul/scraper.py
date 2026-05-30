"""Wikivoyage scraping via the MediaWiki API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests

from .config import config
from .models import RawPage
from .paths import RAW_DIR


DEFAULT_PAGES = [
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
    "Istanbul_with_children",
]

API_URL = "https://en.wikivoyage.org/w/api.php"


def slug(title: str) -> str:
    """Convert a page title into a safe filename slug."""
    return title.replace("/", "_").replace(" ", "_").replace("'", "").lower()


class Scraper:
    """Fetch Wikivoyage articles and cache them to ``output_dir``."""

    def __init__(
        self,
        pages: Optional[list[str]] = None,
        contact: Optional[str] = None,
        output_dir: Path = RAW_DIR,
        rate_limit_s: float = 0.5,
        max_retries: int = 3,
        timeout_s: int = 30,
    ):
        self.pages = list(pages) if pages is not None else list(DEFAULT_PAGES)
        contact = contact or config.askistanbul_contact
        self.headers = {"User-Agent": f"askISTANBUL/1.0 ({contact}) python-requests"}
        self.output_dir = Path(output_dir)
        self.rate_limit_s = rate_limit_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    def fetch_page(self, title: str) -> Optional[RawPage]:
        """Fetch wikitext + metadata for one page. Returns None on 'missing'."""
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
        resp = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(
                    API_URL,
                    params=params,
                    headers=self.headers,
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        assert resp is not None
        data = resp.json()
        pages = data["query"]["pages"]
        if not pages:
            return None
        page = pages[0]
        if "missing" in page:
            return None
        return RawPage(
            title=page["title"],
            pageid=page["pageid"],
            url=page.get("fullurl", ""),
            wikitext=page["revisions"][0]["slots"]["main"]["content"],
        )

    def scrape_all(self) -> list[RawPage]:
        """Scrape every page in ``self.pages``, caching to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results: list[RawPage] = []
        cached = fetched = skipped = 0

        for title in self.pages:
            out_file = self.output_dir / f"{slug(title)}.json"
            if out_file.exists():
                print(f"  [CACHED] {title}")
                results.append(RawPage.from_dict(json.loads(out_file.read_text(encoding="utf-8"))))
                cached += 1
                continue

            print(f"  [FETCH]  {title} ...", end=" ", flush=True)
            try:
                page = self.fetch_page(title)
            except Exception as exc:
                print(f"ERROR — {exc}")
                skipped += 1
                continue
            if page is None:
                print("SKIP (not found)")
                skipped += 1
                continue

            out_file.write_text(
                json.dumps(page.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"OK  ({len(page.wikitext)} chars)")
            results.append(page)
            fetched += 1
            time.sleep(self.rate_limit_s)

        print(f"\nDone. Cached: {cached}  Fetched: {fetched}  Skipped: {skipped}")
        print(f"Raw data: {self.output_dir.resolve()}")
        return results


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Scrape Wikivoyage Istanbul pages.")
    p.add_argument("--contact", default=None, help="Contact email for the User-Agent header.")
    p.add_argument("--rate-limit", type=float, default=0.5, help="Seconds between requests.")
    p.add_argument("--out-dir", default=None, help="Override the raw output directory.")
    args = p.parse_args()

    Scraper(
        contact=args.contact,
        output_dir=Path(args.out_dir) if args.out_dir else RAW_DIR,
        rate_limit_s=args.rate_limit,
    ).scrape_all()


if __name__ == "__main__":
    _main()
