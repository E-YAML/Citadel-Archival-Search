"""
conftest_evals.py
-----------------
Shared setup helpers for the Citadel Archival Search evaluation suite.

Handles:
- Environment variable loading from the project .env file
- sys.path patching so the backend package is importable
- Lazy graph app and retrieval service accessors
- Golden dataset loader
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path & env bootstrapping (must run before any app imports)
# ---------------------------------------------------------------------------

# Resolve the repository root (two levels up from this file: evals/ -> repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure the backend is importable
BACKEND_PATH = REPO_ROOT / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

# Load .env manually so app settings work outside of uvicorn/streamlit
_env_file = REPO_ROOT / ".env"
if _env_file.exists():
    from dotenv import dotenv_values  # type: ignore
    for key, value in dotenv_values(str(_env_file)).items():
        os.environ.setdefault(key, value)  # don't override already-set vars

# ---------------------------------------------------------------------------
# Golden dataset
# ---------------------------------------------------------------------------

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


def load_golden_dataset(
    categories: list[str] | None = None,
    ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load the golden ASOIAF QA dataset.

    Args:
        categories: Optional list of categories to filter by
                    (e.g. ['factual', 'lineage']).
        ids:        Optional list of specific example IDs to return.
        limit:      Optional maximum number of examples to return.

    Returns:
        A list of dataset example dicts.
    """
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    if categories:
        dataset = [ex for ex in dataset if ex["category"] in categories]
    if ids:
        dataset = [ex for ex in dataset if ex["id"] in ids]
    if limit:
        dataset = dataset[:limit]

    return dataset


# ---------------------------------------------------------------------------
# App accessors (lazy imports so settings are loaded first)
# ---------------------------------------------------------------------------

def get_graph_app():
    """
    Return the compiled LangGraph app.

    Imports are deferred so that app.core.config.Settings is only
    instantiated after the env vars above are available.
    """
    from app.graph.workflow import app as graph_app  # noqa: PLC0415
    return graph_app


def get_retrieval_functions():
    """
    Return the two retrieval coroutines used by the eval runner.

    Returns:
        (retrieve_vector_context, retrieve_graph_context)
    """
    from app.services.retrieval import (  # noqa: PLC0415
        retrieve_vector_context,
        retrieve_graph_context,
    )
    return retrieve_vector_context, retrieve_graph_context


# ---------------------------------------------------------------------------
# LangSmith helpers
# ---------------------------------------------------------------------------

def get_langsmith_client():
    """
    Return an authenticated LangSmith Client instance.

    Requires LANGCHAIN_API_KEY (or LANGSMITH_API_KEY) to be set.
    """
    try:
        from langsmith import Client  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "langsmith is not installed. Run: pip install langsmith"
        ) from exc

    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No LangSmith API key found. Set LANGSMITH_API_KEY or LANGCHAIN_API_KEY in your .env file."
        )
    return Client(api_key=api_key)


def get_langsmith_project() -> str:
    """Return the LangSmith project name from settings or default."""
    return os.environ.get("LANGSMITH_PROJECT", "citadel-archival-search-evals")


# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
