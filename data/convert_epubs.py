"""
ASOIAF Epub → Structured Text Converter
========================================
Converts ASOIAF .epub files into plaintext files with BOOK: and CHAPTER: markers
that are directly compatible with ASOIAFIngestionPipeline.parse_file().

Usage:
    python data/convert_epubs.py

Output:
    One .txt file per epub, saved alongside the .epub in the /data directory.
    e.g. "A Game Of Thrones.txt"

Format produced:
    BOOK: A Game of Thrones
    CHAPTER: PROLOGUE
    <chapter text...>
    CHAPTER: BRAN
    <chapter text...>
    ...
"""

import os
import re
import glob
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, Tag
from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Epubs to skip (image-heavy, no readable prose)
SKIP_EPUBS = {
    "The Lands Of Ice And Fire",
}

# Front-matter titles to discard
FRONTMATTER_KEYWORDS = {
    "CONTENTS", "TABLE OF", "COPYRIGHT", "COVER", "TITLE PAGE",
    "DEDICATION", "A NOTE ON", "MAPS", "APPENDIX", "ACKNOWLEDGEMENT",
    "ABOUT THE AUTHOR", "ALSO BY", "INTRODUCTION",
}

# Minimum prose length to keep a chapter
MIN_CHAPTER_TEXT_LEN = 500


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace and replace common Unicode punctuation."""
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "--")
    text = text.replace("\u2026", "...")
    text = text.replace("\u00a0", " ")
    # Collapse runs of 3+ newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def is_frontmatter(title: str) -> bool:
    """Return True if the title string looks like front/back matter."""
    t = title.upper()
    return any(kw in t for kw in FRONTMATTER_KEYWORDS)


def find_chapter_title(soup: BeautifulSoup) -> str | None:
    """
    Try multiple strategies to find the chapter heading in an epub item.
    Returns the title string or None if not found.
    """
    # Strategy 1 — Calibre-style: h3 or h2 with any class
    for tag_name in ("h3", "h2", "h1"):
        tag = soup.find(tag_name)
        if tag:
            txt = tag.get_text(strip=True)
            if txt and len(txt) < 80 and not is_frontmatter(txt):
                return txt.upper()

    # Strategy 2 — class="ct" (A Storm of Swords / Fire and Blood style)
    ct = soup.find(class_="ct")
    if ct:
        txt = ct.get_text(strip=True)
        if txt and len(txt) < 80 and not is_frontmatter(txt):
            return txt.upper()

    # Strategy 3 — class contains "chapter"
    for tag in soup.find_all(True):
        classes = " ".join(tag.get("class", []))
        if "chapter" in classes.lower() or "chap" in classes.lower():
            txt = tag.get_text(strip=True)
            if txt and len(txt) < 80 and not is_frontmatter(txt):
                return txt.upper()

    return None


def extract_body_text(soup: BeautifulSoup, skip_heading_tag=None) -> str:
    """
    Extract all paragraph text from the soup body, optionally skipping
    the first heading tag (to avoid repeating the chapter title).
    """
    body = soup.find("body") or soup
    parts = []
    for tag in body.find_all(["p", "div"]):
        # Skip if this is the heading tag itself
        if skip_heading_tag and tag == skip_heading_tag:
            continue
        # Skip image-only containers
        if tag.find("img") and not tag.get_text(strip=True):
            continue
        text = tag.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def extract_chapters_from_epub(epub_path: str) -> list[dict]:
    """
    Extract all prose chapters from an epub file.
    Each returned dict has: { "title": str, "content": str, "item_index": int }
    
    Strategy: treat each ITEM_DOCUMENT as a potential chapter.
    Each item is one HTML file in the epub — in all ASOIAF epubs one item = one chapter.
    """
    book = epub.read_epub(epub_path, options={"ignore_ncx": True})
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    chapters = []

    for idx, item in enumerate(items):
        soup = BeautifulSoup(item.get_content(), "html.parser")

        # Find chapter title
        title = find_chapter_title(soup)
        if not title:
            continue  # No recognisable heading → skip (TOC, maps, etc.)

        if is_frontmatter(title):
            continue

        # Extract prose body
        # Find the heading tag to skip it in the body extraction
        heading_tag = (
            soup.find("h3") or soup.find("h2") or soup.find("h1")
            or soup.find(class_="ct")
        )
        content = extract_body_text(soup, skip_heading_tag=heading_tag)
        content = clean_text(content)

        if len(content) < MIN_CHAPTER_TEXT_LEN:
            continue  # Too short — probably a map or front matter page

        # Use item index to make each chapter unique even when POV names repeat
        chapters.append({
            "title": title,
            "content": content,
            "item_index": idx,
        })

    return chapters


def convert_epub(epub_path: str, book_name: str) -> str:
    """
    Convert a single epub to a structured text string with BOOK:/CHAPTER: markers.
    """
    logger.info(f"Converting: {os.path.basename(epub_path)}")
    chapters = extract_chapters_from_epub(epub_path)
    logger.info(f"  → Extracted {len(chapters)} chapters")

    lines = [f"BOOK: {book_name}", ""]
    for ch in chapters:
        lines.append(f"CHAPTER: {ch['title']}")
        lines.append(ch["content"])
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    epub_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.epub")))

    if not epub_files:
        logger.error("No .epub files found in the data directory.")
        return

    converted = 0
    skipped = 0

    for epub_path in epub_files:
        fname = os.path.basename(epub_path)
        book_name = os.path.splitext(fname)[0]

        if any(skip in book_name for skip in SKIP_EPUBS):
            logger.info(f"Skipping (non-prose): {fname}")
            skipped += 1
            continue

        out_path = os.path.join(DATA_DIR, book_name + ".txt")

        try:
            text = convert_epub(epub_path, book_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            size_kb = os.path.getsize(out_path) // 1024
            logger.success(f"  → Saved: {os.path.basename(out_path)} ({size_kb} KB)")
            converted += 1
        except Exception as e:
            logger.error(f"  → Failed to convert {fname}: {e}")

    logger.info(f"\nConversion complete. {converted} converted, {skipped} skipped.")


if __name__ == "__main__":
    try:
        from loguru import logger
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    main()
