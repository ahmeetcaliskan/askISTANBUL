"""
preprocess.py
-------------
Cleans raw Wikivoyage wikitext and saves structured, plain-text documents
to data/cleaned/<slug>.json with the following schema:

{
  "title": str,
  "url": str,
  "sections": [
    {"heading": str, "text": str},
    ...
  ],
  "full_text": str   # all sections joined, used for chunking
}
"""

import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
CLEAN_DIR = Path(__file__).parent.parent / "data" / "cleaned"


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def remove_templates(text: str) -> str:
    """Remove {{...}} template blocks (handles nesting)."""
    result, depth = [], 0
    i = 0
    while i < len(text):
        if text[i:i+2] == "{{":
            depth += 1
            i += 2
        elif text[i:i+2] == "}}":
            depth = max(depth - 1, 0)
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    return "".join(result)


def remove_wikilinks(text: str) -> str:
    """[[Link|Display]] → Display   or   [[Link]] → Link"""
    # [[File:...]] and [[Image:...]] — remove entirely
    text = re.sub(r"\[\[(?:File|Image|Fichier):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # [[target|display]] → display
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    # [[target]] → target
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return text


def remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def clean_formatting(text: str) -> str:
    """Remove bold/italic markers and horizontal rules."""
    text = re.sub(r"'{2,3}", "", text)   # '''bold''', ''italic''
    text = re.sub(r"-{4,}", "", text)    # ----
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_wikitext(raw: str) -> str:
    text = remove_templates(raw)
    text = remove_wikilinks(text)
    text = remove_html(text)
    text = clean_formatting(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(={2,4})\s*(.+?)\s*\1\s*$", re.MULTILINE)


def split_sections(cleaned_text: str) -> list[dict]:
    """Split cleaned text into {heading, text} sections."""
    sections = []
    matches = list(HEADING_RE.finditer(cleaned_text))

    # Text before first heading
    preamble = cleaned_text[:matches[0].start()].strip() if matches else cleaned_text.strip()
    if preamble:
        sections.append({"heading": "__intro__", "text": preamble})

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned_text)
        body = cleaned_text[start:end].strip()
        if body:
            sections.append({"heading": heading, "text": body})

    return sections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess_file(raw_path: Path, clean_dir: Path) -> None:
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    wikitext = data.get("wikitext", "")

    cleaned = clean_wikitext(wikitext)
    sections = split_sections(cleaned)

    full_text = "\n\n".join(
        f"[{s['heading']}]\n{s['text']}" for s in sections
    )

    out = {
        "title": data["title"],
        "url": data.get("url", ""),
        "sections": sections,
        "full_text": full_text,
    }

    out_path = clean_dir / raw_path.name
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [OK] {data['title']:40s}  {len(sections)} sections  {len(full_text)} chars")


def preprocess_all(raw_dir: Path = RAW_DIR, clean_dir: Path = CLEAN_DIR) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        print("No raw files found. Run scraper.py first.")
        return

    print(f"Preprocessing {len(files)} files...\n")
    for f in files:
        preprocess_file(f, clean_dir)
    print(f"\nCleaned data saved to: {clean_dir.resolve()}")


if __name__ == "__main__":
    preprocess_all()
