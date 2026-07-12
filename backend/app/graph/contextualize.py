from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from app.core.llm_fallback import build_llm_with_fallback


# --- LLM for contextualisation (lightweight, deterministic) ---
llm_contextualize = build_llm_with_fallback(temperature=0.0)


# --- Question Contextualizer Chain ---
# Resolves pronouns / anaphora ("he", "his mother", "that battle") in a
# follow-up question by examining the recent conversation history and
# rewriting it into a fully self-contained question suitable for vector search.

_contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and the latest user question, your ONLY "
     "job is to reformulate the latest question so it can be understood "
     "WITHOUT the conversation history.\n"
     "Rules:\n"
     "1. Replace ALL pronouns and references (he, she, they, it, his, her, "
     "that, the same, etc.) with the specific entities they refer to from "
     "the conversation history.\n"
     "2. If the question is ALREADY fully self-contained and understandable "
     "on its own, return it EXACTLY as-is.\n"
     "3. Do NOT answer the question.\n"
     "4. Do NOT add information that was not in the original question.\n"
     "5. Return ONLY the reformulated question text, nothing else."),
    ("human",
     "Conversation History:\n{chat_history}\n\n"
     "Latest Question: {question}\n\n"
     "Reformulated Question:"),
])

contextualize_question_chain = (
    _contextualize_prompt | llm_contextualize | StrOutputParser()
)


def _format_chat_history(messages: list[dict], max_turns: int = 6) -> str:
    """
    Formats the last *max_turns* exchanges (user+assistant pairs) from the
    Streamlit session message list into a plain-text transcript.

    Each entry is expected to be ``{"role": "user"|"assistant", "content": "..."}``.
    Only the last ``max_turns * 2`` messages are included to stay within
    context-window budgets.
    """
    # Take the tail of the history (last max_turns * 2 messages)
    recent = messages[-(max_turns * 2):]
    lines: list[str] = []
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def contextualize_question(
    chat_history: list[dict],
    question: str,
) -> str:
    """
    Resolve follow-up references in *question* using *chat_history*.

    If history is empty (first question in the session), the question is
    returned unchanged **without** making an LLM call.

    Args:
        chat_history: List of ``{"role": ..., "content": ...}`` dicts from the
                      Streamlit session state.
        question:     The latest raw user message.

    Returns:
        A self-contained question string.
    """
    if not chat_history:
        logger.debug("Contextualize: No chat history — returning question as-is.")
        return question

    history_text = _format_chat_history(chat_history)
    try:
        result = await contextualize_question_chain.ainvoke({
            "chat_history": history_text,
            "question": question,
        })
        contextualized = result.strip()
        if contextualized:
            logger.info(
                f"Contextualize: '{question}' → '{contextualized}'"
            )
            return contextualized
        # Fallback if the chain returns empty
        return question
    except Exception as exc:
        logger.error(f"Contextualize chain failed: {exc}. Using original question.")
        return question
