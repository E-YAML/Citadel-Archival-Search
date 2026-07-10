"""
llm_fallback.py
---------------
Multi-provider LLM fallback chain for the Citadel Archival Search pipeline.

Provider priority order:
  1. Groq         — llama-3.3-70b-versatile     (primary: fastest, generous free tier)
  2. Google Gemini — gemini-1.5-flash            (fallback 1: truly free, no CC needed,
                                                  60 RPM / 1500 req/day)
  3. OpenRouter   — llama-3.1-8b-instruct:free  (fallback 2: OpenAI-compatible endpoint,
                                                  several permanently-free models)

Gemini uses ``langchain-google-genai`` (``ChatGoogleGenerativeAI``).
OpenRouter uses ``langchain-openai.ChatOpenAI`` with a custom ``base_url`` — no
extra package required.

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
# by OpenRouter / Gemini, plus httpx for raw HTTP 429 responses.
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

    # openai SDK (used by langchain-openai and OpenRouter)
    try:
        from openai import RateLimitError as OpenAIRLE  # type: ignore
        exc_types.append(OpenAIRLE)
    except ImportError:
        pass

    # Google API errors
    try:
        from google.api_core.exceptions import ResourceExhausted  # type: ignore
        exc_types.append(ResourceExhausted)
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
        logger.warning("[LLM Fallback] langchain-groq not installed — Groq provider disabled.")
        return None


def _build_gemini(temperature: float) -> BaseChatModel | None:
    """
    Return a ChatGoogleGenerativeAI instance if GOOGLE_API_KEY is configured.

    Gemini 1.5 Flash free tier (as of 2025):
      - 15 requests/min, 1 500 requests/day
      - 1 million token context window
      - No credit card required
    Sign up: https://aistudio.google.com/app/apikey
    """
    api_key = getattr(settings, "GOOGLE_API_KEY", "") or ""
    if not api_key:
        logger.debug(
            "[LLM Fallback] GOOGLE_API_KEY not set — Gemini provider skipped. "
            "Get a free key (no CC) at https://aistudio.google.com/app/apikey"
        )
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        logger.debug("[LLM Fallback] Gemini provider ready (gemini-1.5-flash).")
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-1.5-flash",
            temperature=temperature,
        )
    except ImportError:
        logger.warning(
            "[LLM Fallback] langchain-google-genai not installed — Gemini provider disabled. "
            "Install with: pip install langchain-google-genai"
        )
        return None


def _build_openrouter(temperature: float) -> BaseChatModel | None:
    """
    Return a ChatOpenAI instance pointed at OpenRouter's OpenAI-compatible endpoint.

    OpenRouter provides permanently-free access to several open models
    (e.g. meta-llama/llama-3.1-8b-instruct:free, mistralai/mistral-7b-instruct:free).
    Uses ``langchain_openai.ChatOpenAI`` with a custom ``base_url`` — no additional
    package required.
    Sign up: https://openrouter.ai  (free tier, no credit card for free models)
    """
    api_key = getattr(settings, "OPENROUTER_API_KEY", "") or ""
    if not api_key:
        logger.debug(
            "[LLM Fallback] OPENROUTER_API_KEY not set — OpenRouter provider skipped. "
            "Get a free key at https://openrouter.ai"
        )
        return None
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        logger.debug(
            "[LLM Fallback] OpenRouter provider ready "
            "(meta-llama/llama-3.1-8b-instruct:free)."
        )
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model="meta-llama/llama-3.1-8b-instruct:free",
            temperature=temperature,
            default_headers={
                # OpenRouter recommends these for app identification and rate-limit grouping
                "HTTP-Referer": "https://github.com/citadel-archival-search",
                "X-Title": "Citadel Archival Search",
            },
        )
    except ImportError:
        logger.warning(
            "[LLM Fallback] langchain-openai not installed — OpenRouter provider disabled."
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
        Groq  →  Google Gemini  →  OpenRouter

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
    builders = [_build_groq, _build_gemini, _build_openrouter]
    available: list[BaseChatModel] = [
        llm for builder in builders if (llm := builder(temperature)) is not None
    ]

    if not available:
        raise RuntimeError(
            "No LLM provider is configured. "
            "Set at least one of: GROQ_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY "
            "in your .env file."
        )

    provider_names = [type(llm).__name__ for llm in available]

    if len(available) == 1:
        logger.info(
            f"[LLM Fallback] Single provider active: {provider_names[0]}. "
            "Set GOOGLE_API_KEY or OPENROUTER_API_KEY to enable redundancy."
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
