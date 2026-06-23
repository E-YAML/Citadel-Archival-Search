import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.ingestion import ASOIAFIngestionPipeline

def test_parse_file(tmp_path):
    """Test parsing a mock file with BOOK and CHAPTER markers."""
    mock_content = """BOOK: A Game of Thrones
CHAPTER: EDDARD I
Eddard stood on the battlements of Winterfell.
CHAPTER: JON I
Jon Snow felt the cold wind.
"""
    test_file = tmp_path / "test_book.txt"
    test_file.write_text(mock_content, encoding="utf-8")

    pipeline = ASOIAFIngestionPipeline()
    chapters = pipeline.parse_file(str(test_file))

    assert len(chapters) == 2
    assert chapters[0]["book_title"] == "A Game of Thrones"
    assert chapters[0]["chapter_title"] == "EDDARD I"
    assert chapters[0]["pov_character"] == "EDDARD"
    assert "Winterfell" in chapters[0]["content"]

    assert chapters[1]["book_title"] == "A Game of Thrones"
    assert chapters[1]["chapter_title"] == "JON I"
    assert chapters[1]["pov_character"] == "JON"
    assert "cold wind" in chapters[1]["content"]

def test_chunk_chapters():
    """Test chunking chapters into granular LangChain documents."""
    pipeline = ASOIAFIngestionPipeline()
    chapters = [
        {
            "book_title": "A Game of Thrones",
            "chapter_title": "EDDARD I",
            "pov_character": "EDDARD",
            "content": "Winter is coming. The direwolf is the sigil of House Stark."
        }
    ]
    chunks = pipeline.chunk_chapters(chapters)
    assert len(chunks) > 0
    assert chunks[0].metadata["book_title"] == "A Game of Thrones"
    assert chunks[0].metadata["chapter_title"] == "EDDARD I"
    assert chunks[0].metadata["pov_character"] == "EDDARD"
    assert "Winter is coming" in chunks[0].page_content

@pytest.mark.asyncio
async def test_seed_sample_lineage(mock_neo4j):
    """Test seeding the Neo4j instance with sample lineage nodes and relationships."""
    pipeline = ASOIAFIngestionPipeline()
    await pipeline.seed_sample_lineage()
    
    # Assert session.run was called for MERGE queries
    assert mock_neo4j.run.call_count > 0
    first_call_args = mock_neo4j.run.call_args_list[0][0]
    assert "MERGE" in first_call_args[0]
