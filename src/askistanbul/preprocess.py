"""Wikitext cleaning and section splitting.

The cleaner preserves Wikivoyage listing templates ({{see}}, {{do}}, {{eat}},
{{drink}}, {{sleep}}, {{buy}}, {{go}}, {{listing}}, {{vCard}}, {{marker}})
by rendering their parameters inline; other templates are dropped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .models import CleanedDocument, RawPage, Section
from .paths import CLEAN_DIR, RAW_DIR


# Wikivoyage listing-family templates carry the practical POI content
# (name, address, phone, hours, price). They must be rendered inline
# rather than dropped — other templates (routeboxes, banners, etc.) are dropped.
LISTING_TYPES = {
    "see", "do", "buy", "eat", "drink", "sleep", "go",
    "listing", "vcard", "marker",
}


# ---------------------------------------------------------------------------
# Template parsing / rendering — module-level helpers
# ---------------------------------------------------------------------------

def _parse_template_body(body: str) -> tuple[str, dict]:
    """Split a template body on top-level '|'; return (name, {param: value})."""
    parts, depth, buf, i, n = [], 0, [], 0, len(body)
    while i < n:
        two = body[i:i+2]
        if two in ("{{", "[["):
            depth += 1
            buf.append(two)
            i += 2
        elif two in ("}}", "]]"):
            depth = max(depth - 1, 0)
            buf.append(two)
            i += 2
        elif body[i] == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(body[i])
            i += 1
    parts.append("".join(buf))
    name = parts[0].strip()
    params: dict = {}
    for p in parts[1:]:
        if "=" in p:
            k, _, v = p.partition("=")
            params[k.strip().lower()] = v.strip()
    return name, params


def _render_listing(name: str, params: dict) -> str:
    label = name.lower()
    title = params.get("name", "").strip()
    alt = params.get("alt", "").strip()
    addr = params.get("address", "").strip()
    phone = params.get("phone", "").strip()
    hours = (params.get("hours") or params.get("checkin") or "").strip()
    price = params.get("price", "").strip()
    content = (params.get("content") or params.get("description") or "").strip()

    bits = []
    if title:
        bits.append(f"{title}" + (f" ({alt})" if alt else "") + f" [{label}].")
    if addr:
        bits.append(f"Address: {addr}.")
    if phone:
        bits.append(f"Phone: {phone}.")
    if hours:
        bits.append(f"Hours: {hours}.")
    if price:
        bits.append(f"Price: {price}.")
    if content:
        bits.append(content)
    return " ".join(bits).strip()


def remove_wikilinks(text: str) -> str:
    """[[Link|Display]] -> Display; [[Link]] -> Link; [[File:…]] -> ''."""
    text = re.sub(r"\[\[(?:File|Image|Fichier):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return text


def remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def remove_tables(text: str) -> str:
    """Strip MediaWiki tables ``{| ... |}`` (non-nested approximation)."""
    return re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# WikitextCleaner
# ---------------------------------------------------------------------------

class WikitextCleaner:
    """Strip MediaWiki markup; preserve Wikivoyage listing content inline."""

    def __init__(self, listing_types: Optional[set[str]] = None):
        self.listing_types = set(listing_types) if listing_types is not None else set(LISTING_TYPES)

    def _remove_templates(self, text: str) -> str:
        out: list[str] = []
        i, n = 0, len(text)
        while i < n:
            if text[i:i+2] == "{{":
                depth, j = 1, i + 2
                while j < n and depth:
                    if text[j:j+2] == "{{":
                        depth += 1
                        j += 2
                    elif text[j:j+2] == "}}":
                        depth -= 1
                        j += 2
                    else:
                        j += 1
                body = text[i+2:j-2] if depth == 0 else text[i+2:]
                name, params = _parse_template_body(body)
                if name.lower() in self.listing_types:
                    out.append(_render_listing(name, params))
                # else: drop entirely (infoboxes, banners, routeboxes, …)
                i = j if depth == 0 else n
            else:
                out.append(text[i])
                i += 1
        return "".join(out)

    @staticmethod
    def _clean_formatting(text: str) -> str:
        text = re.sub(r"'{2,3}", "", text)   # '''bold''' / ''italic''
        text = re.sub(r"-{4,}", "", text)    # ----
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        # Drop bullet lines left empty after non-listing templates were removed.
        text = re.sub(r"^\s*\*\s*$\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def clean(self, raw: str) -> str:
        text = remove_tables(raw)
        text = self._remove_templates(text)
        text = remove_wikilinks(text)
        text = remove_html(text)
        text = self._clean_formatting(text)
        text = self._normalize_whitespace(text)
        return text


# ---------------------------------------------------------------------------
# SectionSplitter
# ---------------------------------------------------------------------------

class SectionSplitter:
    """Split cleaned text into Section objects on heading boundaries."""

    INTRO_HEADING = "__intro__"

    def __init__(self, min_level: int = 2, max_level: int = 6):
        self.heading_re = re.compile(
            rf"^(={{{min_level},{max_level}}})\s*(.+?)\s*\1\s*$",
            re.MULTILINE,
        )

    def split(self, cleaned_text: str) -> list[Section]:
        matches = list(self.heading_re.finditer(cleaned_text))
        sections: list[Section] = []

        preamble = cleaned_text[:matches[0].start()].strip() if matches else cleaned_text.strip()
        if preamble:
            sections.append(Section(heading=self.INTRO_HEADING, text=preamble))

        for i, m in enumerate(matches):
            heading = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned_text)
            body = cleaned_text[start:end].strip()
            if body:
                sections.append(Section(heading=heading, text=body))

        return sections


# ---------------------------------------------------------------------------
# Preprocessor — composes cleaner + splitter and handles disk I/O
# ---------------------------------------------------------------------------

class Preprocessor:
    """Clean + split raw pages, writing CleanedDocument JSON to disk."""

    def __init__(
        self,
        cleaner: Optional[WikitextCleaner] = None,
        splitter: Optional[SectionSplitter] = None,
        raw_dir: Path = RAW_DIR,
        clean_dir: Path = CLEAN_DIR,
    ):
        self.cleaner = cleaner or WikitextCleaner()
        self.splitter = splitter or SectionSplitter()
        self.raw_dir = Path(raw_dir)
        self.clean_dir = Path(clean_dir)

    def preprocess_page(self, raw: RawPage) -> CleanedDocument:
        cleaned_text = self.cleaner.clean(raw.wikitext)
        sections = self.splitter.split(cleaned_text)
        full_text = "\n\n".join(f"[{s.heading}]\n{s.text}" for s in sections)
        return CleanedDocument(
            title=raw.title,
            url=raw.url,
            sections=sections,
            full_text=full_text,
        )

    def preprocess_file(self, raw_path: Path) -> CleanedDocument:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        raw = RawPage.from_dict(data)
        doc = self.preprocess_page(raw)
        out_path = self.clean_dir / raw_path.name
        out_path.write_text(
            json.dumps(doc.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  [OK] {doc.title:40s}  {len(doc.sections)} sections  {len(doc.full_text)} chars")
        return doc

    def preprocess_all(self) -> list[CleanedDocument]:
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(self.raw_dir.glob("*.json"))
        if not files:
            print(f"No raw files found in {self.raw_dir}. Run the scraper first.")
            return []
        print(f"Preprocessing {len(files)} files...\n")
        docs = [self.preprocess_file(f) for f in files]
        print(f"\nCleaned data: {self.clean_dir.resolve()}")
        return docs


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Clean raw wikitext and split into sections.")
    p.add_argument("--raw-dir", default=None, help="Override the input directory.")
    p.add_argument("--clean-dir", default=None, help="Override the output directory.")
    args = p.parse_args()

    Preprocessor(
        raw_dir=Path(args.raw_dir) if args.raw_dir else RAW_DIR,
        clean_dir=Path(args.clean_dir) if args.clean_dir else CLEAN_DIR,
    ).preprocess_all()


if __name__ == "__main__":
    _main()
