# Data Directory

This directory contains the ASOIAF source data for ingestion into the Citadel Archival Search system.

## Contents

| File | Description |
|---|---|
| `convert_epubs.py` | One-time epub → structured .txt converter |
| `*.epub` | Source epub files (**not committed** — gitignored) |
| `*.txt` | Converted book text files (**not committed** — gitignored) |
| `game_of_thrones_sample.txt` | Legacy 1KB sample — ignored by ingestion |

## Epub → Text Conversion

The ingestion pipeline reads `.txt` files with `BOOK:` / `CHAPTER:` markers.
Run the converter once locally before ingesting:

```bash
# Install conversion deps (one-time, not needed on Streamlit Cloud)
pip install ebooklib beautifulsoup4

# Convert all .epub files in this directory
python data/convert_epubs.py
```

**Expected output:**
```
Converting: A Game Of Thrones.epub  →  74 chapters  (1568 KB)
Converting: A Clash of Kings.epub   →  71 chapters  (1711 KB)
Converting: A Storm of Swords.epub  →  83 chapters  (4412 KB)
Converting: A Feast for Crows.epub  →  48 chapters  (1577 KB)
Converting: A Dance With Dragons.epub → 94 chapters (4539 KB)
Converting: Fire and Blood.epub     →  27 chapters  (1458 KB)
Converting: The Tales of Dunk & Egg.epub → 4 chapters (557 KB)
```

## Structured Text Format

Each converted `.txt` file follows this format so `ASOIAFIngestionPipeline.parse_file()` can parse it:

```
BOOK: A Game of Thrones

CHAPTER: PROLOGUE
"We should start back," Gared urged as the woods began to grow dark...

CHAPTER: BRAN
The morning had dawned clear and cold...

CHAPTER: CATELYN
Catelyn had never liked this godswood...
```

**Rules:**
- `BOOK:` marker appears once at the top of each file
- `CHAPTER:` markers appear before each chapter (POV character name)
- No blank lines between `CHAPTER:` and chapter content
- UTF-8 encoding

## Cloud Ingestion

Once converted, point your local `.env` at Qdrant Cloud + Neo4j Aura and run once:

```bash
python backend/scripts/run_ingestion.py
```

This is idempotent — safe to re-run. Data persists in the cloud forever after.
No re-ingestion needed when deploying to Streamlit Cloud.

## Legal Note

The ASOIAF books are copyright © George R.R. Martin. Only process files you legally own.
