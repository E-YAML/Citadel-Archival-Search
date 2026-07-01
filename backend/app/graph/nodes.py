from typing import Dict, Any, List
from langchain_core.documents import Document
from loguru import logger

from app.graph.state import AgentState
from app.services.retrieval import retrieve_vector_context, retrieve_graph_context
from app.graph.chains import (
    retrieval_grader_chain,
    generator_chain,
    question_rewriter_chain,
)


async def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves candidate document context from the Qdrant database.
    
    Args:
        state: The current LangGraph AgentState.
        
    Returns:
        A dictionary containing the updated list of fetched documents.
    """
    logger.info("LangGraph Node: [retrieve_node] - Initiated.")
    question = state.get("question", "")

    try:
        # Fetch matching documents via hybrid search
        search_results = await retrieve_vector_context(query=question, limit=4)
        documents = [
            Document(page_content=res["text"], metadata=res["metadata"])
            for res in search_results
        ]
        
        # Fetch knowledge graph context from Neo4j
        graph_context = await retrieve_graph_context(query=question)
        if graph_context:
            logger.info("Injecting Neo4j graph context into retrieved documents.")
            documents.append(
                Document(
                    page_content=graph_context,
                    metadata={"book_title": "Citadel Knowledge Graph", "chapter_title": "Lineage Records"}
                )
            )

        logger.info(f"LangGraph Node: [retrieve_node] - Retrieved {len(documents)} context documents (including graph context).")
        return {"documents": documents}
    except Exception as e:
        logger.error(f"LangGraph Node: [retrieve_node] - Execution failed: {str(e)}")
        return {"documents": []}


async def grade_documents_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates context documents concurrently. Filters out irrelevant documentation
    using the structured retrieval grader.
    
    Args:
        state: The current LangGraph AgentState.
        
    Returns:
        A dictionary containing the filtered list of relevant documents.
    """
    import asyncio
    logger.info("LangGraph Node: [grade_documents_node] - Initiated.")
    question = state.get("question", "")
    documents = state.get("documents", [])

    if not documents:
        return {"documents": []}

    async def grade_doc(idx: int, doc: Document) -> tuple:
        try:
            logger.info(f"Grading document {idx + 1}/{len(documents)}...")
            res = await retrieval_grader_chain.ainvoke({
                "question": question,
                "document": doc.page_content
            })
            grade = res.strip().lower()
            if "yes" in grade:
                logger.info(f"Document {idx + 1}: Relevance check - RELEVANT.")
                return idx, True
            else:
                logger.info(f"Document {idx + 1}: Relevance check - IRRELEVANT (Grade: {grade}).")
                return idx, False
        except Exception as e:
            logger.error(f"Error grading document {idx + 1}: {str(e)}")
            return idx, False

    tasks = [grade_doc(i, d) for i, d in enumerate(documents)]
    results = await asyncio.gather(*tasks)
    
    # Maintain ordering
    sorted_results = sorted(results, key=lambda x: x[0])
    filtered_documents = [
        documents[idx] for idx, is_relevant in sorted_results if is_relevant
    ]

    logger.info(f"LangGraph Node: [grade_documents_node] - Retained {len(filtered_documents)}/{len(documents)} relevant documents.")
    return {"documents": filtered_documents}


async def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates a response using the Maester citation rules.
    
    Args:
        state: The current LangGraph AgentState.
        
    Returns:
        A dictionary containing the generated text answer.
    """
    logger.info("LangGraph Node: [generate_node] - Initiated.")
    original_question = state.get("original_question") or state.get("question", "")
    documents = state.get("documents", [])

    # Format excerpts with metadata for prompt ingestion
    context_excerpts = []
    for doc in documents:
        meta = doc.metadata or {}
        book = meta.get("book_title", "Unknown Book")
        if " -- " in book:
            book = book.split(" -- ")[0]
        chapter = meta.get("chapter_title", "Unknown Chapter")
        context_excerpts.append(
            f"Excerpt from '{book}' (Chapter: {chapter}):\n{doc.page_content}"
        )
    context_str = "\n\n".join(context_excerpts)

    try:
        generation = await generator_chain.ainvoke({
            "context": context_str,
            "question": original_question
        })
        logger.info("LangGraph Node: [generate_node] - Text generated successfully.")
        return {"generation": generation}
    except Exception as e:
        logger.error(f"LangGraph Node: [generate_node] - Generator execution failed: {str(e)}")
        return {"generation": "Citadel records indicate an error occurred while processing the response."}


async def rewrite_node(state: AgentState) -> Dict[str, Any]:
    """
    Rewrites the search query to improve retrieval precision.
    Increments the loop retry counter.
    
    Args:
        state: The current LangGraph AgentState.
        
    Returns:
        A dictionary updating the question query and the search retry count.
    """
    logger.info("LangGraph Node: [rewrite_node] - Initiated.")
    original_question = state.get("original_question") or state.get("question", "")
    current_retry = state.get("search_retry_count", 0)

    try:
        # Call query rewriter chain
        rewritten = await question_rewriter_chain.ainvoke({"question": original_question})
        optimized_query = rewritten.strip()
        logger.info(f"LangGraph Node: [rewrite_node] - Query optimized: '{original_question}' -> '{optimized_query}'")
        return {
            "question": optimized_query,
            "search_retry_count": current_retry + 1
        }
    except Exception as e:
        logger.error(f"LangGraph Node: [rewrite_node] - Query rewrite failed: {str(e)}")
        return {
            "search_retry_count": current_retry + 1
        }
