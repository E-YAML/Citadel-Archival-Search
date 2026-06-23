import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document
from app.graph.state import AgentState
from app.services.retrieval import retrieve_graph_context
from app.graph.nodes import retrieve_node, grade_documents_node, generate_node, rewrite_node
from app.graph.workflow import decide_to_generate, check_hallucinations

@pytest.mark.asyncio
async def test_retrieve_graph_context_no_entities(mock_neo4j):
    """Test retrieve_graph_context when query contains no known entities."""
    # Find query returns no characters
    mock_neo4j.run.return_value.data = AsyncMock(return_value=[])
    
    result = await retrieve_graph_context("How is the weather in King's Landing?")
    assert result == ""
    mock_neo4j.run.assert_called_once()

@pytest.mark.asyncio
async def test_retrieve_graph_context_with_entities(mock_neo4j):
    """Test retrieve_graph_context when query contains a known character."""
    # First query (find characters) returns Jon Snow
    # Second query (fetch properties) returns Jon's properties
    # Third query (fetch relationships) returns Jon's relationships
    
    async def mock_run(query_str, **kwargs):
        mock_result = MagicMock()
        if "toLower($query) CONTAINS" in query_str:
            mock_result.data = AsyncMock(return_value=[{"name": "Jon Snow"}])
        elif "MATCH (c:Character {name: $name})" in query_str and "status" in query_str:
            mock_result.single = AsyncMock(return_value={"name": "Jon Snow", "house": "Stark", "status": "Bastard"})
        elif "MATCH (c:Character {name: $name})-[r]-(o:Character)" in query_str:
            mock_result.data = AsyncMock(return_value=[
                {"c_name": "Jon Snow", "rel_type": "SON_OF", "o_name": "Lyanna Stark", "is_outbound": True}
            ])
        return mock_result

    mock_neo4j.run.side_effect = mock_run
    
    result = await retrieve_graph_context("Who are Jon Snow's parents?")
    assert "Character Jon Snow" in result
    assert "House: Stark" in result
    assert "Jon Snow is SON_OF of Lyanna Stark" in result

@pytest.mark.asyncio
async def test_retrieve_node(mock_neo4j, monkeypatch):
    """Test retrieve_node queries both vector and graph contexts."""
    # Mock retrieve_vector_context
    mock_vector = AsyncMock(return_value=[
        {"text": "A Targaryen must rule.", "metadata": {"book_title": "A Dance with Dragons", "chapter_title": "Daenerys I"}}
    ])
    monkeypatch.setattr("app.graph.nodes.retrieve_vector_context", mock_vector)
    
    # Mock Neo4j find characters query to return empty
    mock_neo4j.run.return_value.data = AsyncMock(return_value=[])

    state: AgentState = {"question": "Who is Jon?", "generation": "", "documents": [], "is_hallucination": False, "search_retry_count": 0}
    result = await retrieve_node(state)
    
    assert len(result["documents"]) == 1
    assert result["documents"][0].page_content == "A Targaryen must rule."
    assert result["documents"][0].metadata["book_title"] == "A Dance with Dragons"

@pytest.mark.asyncio
async def test_grade_documents_node(monkeypatch):
    """Test grade_documents_node filters out irrelevant documents."""
    mock_grader = AsyncMock()
    # Grade first doc relevant, second irrelevant
    mock_grader.ainvoke.side_effect = [
        MagicMock(is_relevant=True),
        MagicMock(is_relevant=False)
    ]
    monkeypatch.setattr("app.graph.nodes.retrieval_grader_chain", mock_grader)

    docs = [
        Document(page_content="Jon is Lyanna's son.", metadata={}),
        Document(page_content="Balerion was a black dragon.", metadata={})
    ]
    state: AgentState = {"question": "Jon's mother", "generation": "", "documents": docs, "is_hallucination": False, "search_retry_count": 0}
    
    result = await grade_documents_node(state)
    assert len(result["documents"]) == 1
    assert result["documents"][0].page_content == "Jon is Lyanna's son."

@pytest.mark.asyncio
async def test_generate_node(monkeypatch):
    """Test generate_node formats documents and returns generation response."""
    mock_generator = AsyncMock()
    mock_generator.ainvoke.return_value = "According to records, Jon Snow is Lyanna's son."
    monkeypatch.setattr("app.graph.nodes.generator_chain", mock_generator)

    docs = [
        Document(page_content="Jon is Lyanna's son.", metadata={"book_title": "Citadel Knowledge Graph", "chapter_title": "Lineage Records"})
    ]
    state: AgentState = {"question": "Who is Jon?", "generation": "", "documents": docs, "is_hallucination": False, "search_retry_count": 0}
    
    result = await generate_node(state)
    assert result["generation"] == "According to records, Jon Snow is Lyanna's son."

@pytest.mark.asyncio
async def test_rewrite_node(monkeypatch):
    """Test rewrite_node optimizes query and increments search_retry_count."""
    mock_rewriter = AsyncMock()
    mock_rewriter.ainvoke.return_value = "Jon Snow mother lineage"
    monkeypatch.setattr("app.graph.nodes.question_rewriter_chain", mock_rewriter)

    state: AgentState = {"question": "Who is Jon's mother?", "generation": "", "documents": [], "is_hallucination": False, "search_retry_count": 1}
    result = await rewrite_node(state)
    
    assert result["question"] == "Jon Snow mother lineage"
    assert result["search_retry_count"] == 2

def test_decide_to_generate():
    """Test graph routing from decide_to_generate."""
    # Case 1: Max retries exceeded -> route to END
    state = {"search_retry_count": 3, "documents": []}
    assert decide_to_generate(state) == "__end__"

    # Case 2: No documents found -> route to rewrite
    state = {"search_retry_count": 0, "documents": []}
    assert decide_to_generate(state) == "rewrite"

    # Case 3: Documents found -> route to generate
    state = {"search_retry_count": 0, "documents": [Document(page_content="Context", metadata={})]}
    assert decide_to_generate(state) == "generate"

@pytest.mark.asyncio
async def test_check_hallucinations(monkeypatch):
    """Test graph routing from check_hallucinations."""
    # Mock hallucination and answer graders
    mock_hallucination = AsyncMock()
    mock_answer = AsyncMock()
    monkeypatch.setattr("app.graph.workflow.hallucination_grader_chain", mock_hallucination)
    monkeypatch.setattr("app.graph.workflow.answer_grader_chain", mock_answer)

    # Case 1: Hallucination detected -> route to rewrite
    mock_hallucination.ainvoke.return_value = MagicMock(has_hallucination=True)
    state = {"search_retry_count": 0, "question": "Q", "generation": "Gen", "documents": []}
    assert await check_hallucinations(state) == "rewrite"

    # Case 2: No hallucination, but answer is invalid -> route to rewrite
    mock_hallucination.ainvoke.return_value = MagicMock(has_hallucination=False)
    mock_answer.ainvoke.return_value = MagicMock(is_valid=False)
    assert await check_hallucinations(state) == "rewrite"

    # Case 3: Grounded and valid -> route to END
    mock_hallucination.ainvoke.return_value = MagicMock(has_hallucination=False)
    mock_answer.ainvoke.return_value = MagicMock(is_valid=True)
    assert await check_hallucinations(state) == "__end__"
