from typing import TypedDict, List, Any


class AgentState(TypedDict):
    """
    Represents the internal state of the Self-Corrective RAG LangGraph.

    This state is updated and passed sequentially between the nodes in the agent graph,
    orchestrating the query processing, document retrieval, evaluation, rewriting,
    and generation tasks.
    """
    
    question: str
    """
    The original query input supplied by the user.
    """
    
    generation: str
    """
    The generated text response or answer created by the generator node.
    """
    
    documents: List[Any]
    """
    A list of text context chunks (documents or nodes) fetched from the Qdrant 
    or Neo4j knowledge bases.
    """
    
    is_hallucination: bool
    """
    A validation flag indicating whether the generated response contains facts 
    unsupported by the retrieved documents (determined by the grader node).
    """
    
    search_retry_count: int
    """
    An integer counter tracking the number of times the system has rewritten 
    the search query and executed a new retrieval, preventing infinite retrieval loops.
    """
    
    original_question: str
    """
    The original, unmodified question query supplied by the user (preserving any typos).
    """
