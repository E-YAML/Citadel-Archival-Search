"""
llm_fallback.py
---------------
Multi-provider LLM fallback chain for the Citadel Archival Search pipeline.

Provider priority order:
  1. Groq          — llama-3.3-70b-versatile          (primary: fastest, free tier)
  2. Cerebras      — llama-3.3-70b                    (fallback 1: high TPM, free tier)
  3. Together AI   — Llama-3.3-70B-Instruct-Turbo     (fallback 2: generous limits)

Cerebras and Together AI both expose an OpenAI-compatible REST API, so this
module uses ``langchain_openai.ChatOpenAI`` with a custom ``base_url`` for
those providers.  This means **no extra packages** are required beyond what is
already installed (``langchain-groq`` and ``langchain-openai``).

LangChain's ``with_fallbacks()`` chains the providers so that when the primary
raises a rate-limit error the next one takes over automatically — completely
transparent to nodes, workflows, and the Streamlit frontend.

Usage
-----
    from app.core.llm_fallback import build_llm_with_fallback

    llm = build_llm_with_fallback(temperature=0.0)
    # Use exactly like a plain ChatGroq / ChatOpenAI — supports .invoke(),
    # .ainvoke(), and LCEL pipe (|) syntax.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from app.core.config import settings


# ---------------------------------------------------------------------------
# Exceptions that trigger a fallback to the next provider.
# We catch both Groq's native RateLimitError and the openai-SDK variant used
# by Cerebras / Together, plus httpx for raw HTTP 429 responses.
# ---------------------------------------------------------------------------
def _rate_limit_exceptions() -> tuple[type[Exception], ...]:
    """
    Collect rate-limit exception classes from whichever SDKs are installed.
    Only importable classes are included; missing packages never raise at startup.
    """
    exc_types: list[type[Exception]] = []

    # Groq SDK
    try:
        from groq import RateLimitError as GroqRLE  # type: ignore
        exc_types.append(GroqRLE)
    except ImportError:
        pass

    # openai SDK (used by langchain-openai, Cerebras, Together)
    try:
        from openai import RateLimitError as OpenAIRLE  # type: ignore
        exc_types.append(OpenAIRLE)
    except ImportError:
        pass

    # httpx raw HTTP errors (covers any 429 not wrapped by the SDKs above)
    try:
        import httpx  # type: ignore
        exc_types.append(httpx.HTTPStatusError)
    except ImportError:
        pass

    # Deduplicate while preserving order
    seen: set[type] = set()
    unique: list[type[Exception]] = []
    for t in exc_types:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return tuple(unique) if unique else (Exception,)


# ---------------------------------------------------------------------------
# Individual provider builders
# ---------------------------------------------------------------------------

def _build_groq(temperature: float) -> BaseChatModel | None:
    """Return a ChatGroq instance if GROQ_API_KEY is configured."""
    api_key = getattr(settings, "GROQ_API_KEY", "") or ""
    if not api_key:
        logger.warning("[LLM Fallback] GROQ_API_KEY is empty — Groq provider disabled.")
        return None
    try:
        from langchain_groq import ChatGroq  # type: ignore
        logger.debug("[LLM Fallback] Groq provider ready (llama-3.3-70b-versatile).")
        return ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=temperature,
        )
    except ImportError:
        logger.warning(
            "[LLM Fallback] langchain-groq not installed — Groq provider disabled."
        )
        return None


def _build_cerebras(temperature: float) -> BaseChatModel | None:
    """
    Return a ChatOpenAI instance pointed at Cerebras' OpenAI-compatible endpoint.

    Cerebras exposes ``https://api.cerebras.ai/v1`` with the same request/response
    format as the OpenAI Chat Completions API, so ``langchain_openai.ChatOpenAI``
    works without any additional package.
    Free tier: https://cloud.cerebras.ai
    """
    api_key = getattr(settings, "CEREBRAS_API_KEY", "") or ""
    if not api_key:
        logger.debug(
            "[LLM Fallback] CEREBRAS_API_KEY not set — Cerebras provider skipped. "
            "Get a free key at https://cloud.cerebras.ai"
        )
        return None
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        logger.debug("[LLM Fallback] Cerebras provider ready (llama-3.3-70b).")
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1",
            model="llama-3.3-70b",
            temperature=temperature,
        )
    except ImportError:
        logger.warning(
            "[LLM Fallback] langchain-openai not installed — Cerebras provider disabled."
        )
        return None


def _build_together(temperature: float) -> BaseChatModel | None:
    """
    Return a ChatOpenAI instance pointed at Together AI's OpenAI-compatible endpoint.

    Together AI exposes ``https://api.together.xyz/v1`` with the same interface.
    Free: $1 credit on sign-up at https://api.together.xyz
    """
    api_key = getattr(settings, "TOGETHER_API_KEY", "") or ""
    if not api_key:
        logger.debug(
            "[LLM Fallback] TOGETHER_API_KEY not set — Together AI provider skipped. "
            "Get a free key at https://api.together.xyz"
        )
        return None
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        logger.debug(
            "[LLM Fallback] Together AI provider ready "
            "(meta-llama/Llama-3.3-70B-Instruct-Turbo)."
        )
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            temperature=temperature,
        )
    except ImportError:
        logger.warning(
            "[LLM Fallback] langchain-openai not installed — Together AI provider disabled."
        )
        return None


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_llm_with_fallback(temperature: float = 0.0) -> BaseChatModel:
    """
    Build an LLM runnable with automatic provider fallback on rate-limit errors.

    Returns a LangChain ``BaseChatModel`` (or a ``RunnableWithFallbacks`` wrapping
    one) that can be used exactly like a plain ``ChatGroq`` — supports ``.invoke()``,
    ``.ainvoke()``, and LCEL pipe (``|``) syntax.

    Provider priority:
        Groq  →  Cerebras  →  Together AI

    Any provider whose API key is absent is silently skipped. At least one
    provider must be available; otherwise a ``RuntimeError`` is raised at
    startup so the failure is loud and immediate rather than silent at query time.

    Args:
        temperature: Sampling temperature forwarded to all providers.

    Returns:
        A ready-to-use LangChain runnable (with fallbacks if multiple available).

    Raises:
        RuntimeError: If no provider is available (all API keys missing / unset).
    """
    builders = [_build_groq, _build_cerebras, _build_together]
    available: list[BaseChatModel] = [
        llm for builder in builders if (llm := builder(temperature)) is not None
    ]

    if not available:
        raise RuntimeError(
            "No LLM provider is configured. "
            "Set at least one of: GROQ_API_KEY, CEREBRAS_API_KEY, TOGETHER_API_KEY "
            "in your .env file."
        )

    provider_names = [type(llm).__name__ for llm in available]

    if len(available) == 1:
        logger.info(
            f"[LLM Fallback] Single provider active: {provider_names[0]}. "
            "Set CEREBRAS_API_KEY or TOGETHER_API_KEY to enable redundancy."
        )
        return available[0]

    primary, *fallbacks = available
    logger.info(
        f"[LLM Fallback] Fallback chain ready — "
        f"primary: {provider_names[0]}, "
        f"fallbacks: {provider_names[1:]}. "
        "Rate-limit errors will automatically retry on the next provider."
    )

    exceptions = _rate_limit_exceptions()
    return primary.with_fallbacks(
        fallbacks=fallbacks,
        exceptions_to_handle=exceptions,
    )
