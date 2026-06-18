from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.graph.state import AgentState
from app.graph.nodes import (
    retrieve_node,
    grade_documents_node,
    generate_node,
    rewrite_node,
)
from app.graph.chains import hallucination_grader_chain, answer_grader_chain


def decide_to_generate(state: AgentState) -> str:
    """
    Decides whether to route to the answer generation node or search query rewrite node,
    subject to max retry thresholds.
    
    Args:
        state: The current LangGraph AgentState.
        
    Returns:
        The name of the next node to transition to.
    """
    retry_count = state.get("search_retry_count", 0)
    if retry_count >= 3:
        logger.warning(f"Routing to END: Max retry threshold reached ({retry_count}).")
        return END

    documents = state.get("documents", [])
    if not documents:
        logger.info("Routing to rewrite: Zero relevant documents available.")
        return "rewrite"

    logger.info(f"Routing to generate: Found {len(documents)} relevant documents.")
    return "generate"


async def check_hallucinations(state: AgentState) -> str:
    """
    Evaluates grounding and accuracy of the generated answer. Routes back to query
    rewrite if the answer fails validation or contains hallucinations.
    
    Args:
        state: The current LangGraph AgentState.
        
    Returns:
        The name of the next node to transition to.
    """
    retry_count = state.get("search_retry_count", 0)
    if retry_count >= 3:
        logger.warning(f"Routing to END: Max retry threshold reached post-generation ({retry_count}).")
        return END

    question = state.get("question", "")
    generation = state.get("generation", "")
    documents = state.get("documents", [])

    # Concatenate context texts
    docs_context = "\n\n".join([doc.page_content for doc in documents])

    # 1. Grounding/Hallucination Check
    try:
        hallucination_result = await hallucination_grader_chain.ainvoke({
            "documents": docs_context,
            "generation": generation
        })
        if hallucination_result.has_hallucination:
            logger.info("Grader check: Hallucination detected. Routing to rewrite.")
            return "rewrite"
        logger.info("Grader check: Generation is grounded. Proceeding to answer relevancy evaluation...")
    except Exception as e:
        logger.error(f"Grader check: Hallucination check failed due to exception: {str(e)}. Defaulting to rewrite.")
        return "rewrite"

    # 2. Answer Relevancy Check (Addresses the question)
    try:
        answer_result = await answer_grader_chain.ainvoke({
            "question": question,
            "generation": generation
        })
        if answer_result.is_valid:
            logger.info("Grader check: Answer resolved the user question. Routing to END.")
            return END
        logger.info("Grader check: Answer is grounded but does not address the question. Routing to rewrite.")
        return "rewrite"
    except Exception as e:
        logger.error(f"Grader check: Relevancy check failed due to exception: {str(e)}. Defaulting to rewrite.")
        return "rewrite"


# --- StateGraph Construction & Compilation ---

workflow = StateGraph(AgentState)

# Register Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("rewrite", rewrite_node)

# Set up Standard Transitions
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Set up Conditional Routing from Grade Documents Node
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        END: END
    }
)

# Set up Conditional Routing from Generate Node
workflow.add_conditional_edges(
    "generate",
    check_hallucinations,
    {
        "rewrite": "rewrite",
        END: END
    }
)

# Set up Transition back to Retrieval from Query Rewrite Node
workflow.add_edge("rewrite", "retrieve")

# Enable checkpoint memory for thread tracking in local testing
memory = MemorySaver()

# Compile the runnable graph app
app = workflow.compile(checkpointer=memory)
