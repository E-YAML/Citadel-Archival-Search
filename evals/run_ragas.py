"""
run_ragas.py
------------
Offline RAGAS evaluation runner for the Citadel Archival Search Self-RAG system.

Runs each golden dataset example through:
  1. Retrieval (Qdrant hybrid + Neo4j graph) independently to capture raw context
  2. The full LangGraph Self-RAG pipeline to get the final generation
  3. RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall

Results are saved to evals/results/ragas_results_{timestamp}.json and also
printed as a summary table to stdout.

Usage
-----
# Full eval (all 30 examples, requires live Qdrant + Neo4j):
    python evals/run_ragas.py

# Quick smoke test with 5 examples:
    python evals/run_ragas.py --sample 5

# Filter to specific categories:
    python evals/run_ragas.py --categories factual lineage

# Skip graph-requiring examples (no Neo4j needed):
    python evals/run_ragas.py --skip-graph

# Save results with a custom tag:
    python evals/run_ragas.py --tag v1_baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap env + path before any app imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest_evals as ctx  # noqa: E402  (must be after sys.path insert)

# ---------------------------------------------------------------------------
# Lazy app imports (after env is loaded)
# ---------------------------------------------------------------------------
from langchain_core.documents import Document  # noqa: E402


# ---------------------------------------------------------------------------
# RAGAS metric helpers
# ---------------------------------------------------------------------------

def _build_ragas_llm():
    """
    Build a ragas 0.4-compatible LangchainLLMWrapper around the project's Groq LLM.
    ragas 0.4 requires wrapping langchain models with LangchainLLMWrapper.
    """
    from langchain_groq import ChatGroq  # noqa: PLC0415
    from ragas.llms.base import LangchainLLMWrapper  # noqa: PLC0415
    import os

    api_key = os.environ.get("GROQ_API_KEY", "")
    groq_llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.0,
    )
    return LangchainLLMWrapper(groq_llm)


def _build_ragas_embeddings():
    """
    Build a ragas 0.4-compatible LangchainEmbeddingsWrapper using fastembed.
    ragas 0.4 requires wrapping langchain embeddings with LangchainEmbeddingsWrapper.
    """
    from ragas.embeddings.base import LangchainEmbeddingsWrapper  # noqa: PLC0415
    from langchain_core.embeddings import Embeddings  # noqa: PLC0415
    from fastembed import TextEmbedding  # noqa: PLC0415

    class _FastEmbedWrapper(Embeddings):
        def __init__(self):
            self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [list(v) for v in self._model.embed(texts)]

        def embed_query(self, text: str) -> list[float]:
            return list(list(self._model.embed([text]))[0])

    return LangchainEmbeddingsWrapper(_FastEmbedWrapper())


# ---------------------------------------------------------------------------
# Per-example pipeline
# ---------------------------------------------------------------------------

async def _run_retrieval(
    question: str,
    requires_graph: bool,
    retrieve_vector_context,
    retrieve_graph_context,
) -> list[str]:
    """
    Run retrieval for a single question.
    Returns a flat list of context strings (one per chunk).
    """
    contexts: list[str] = []

    try:
        vector_results = await retrieve_vector_context(query=question, limit=4)
        for res in vector_results:
            text = res.get("text", "").strip()
            meta = res.get("metadata", {})
            book = meta.get("book_title", "")
            chapter = meta.get("chapter_title", "")
            if text:
                header = f"[{book} / {chapter}]" if book else ""
                contexts.append(f"{header}\n{text}".strip())
    except Exception as exc:
        print(f"    [WARN] Vector retrieval failed: {exc}")

    if requires_graph:
        try:
            graph_ctx = await retrieve_graph_context(query=question)
            if graph_ctx:
                contexts.append(f"[Citadel Knowledge Graph / Lineage Records]\n{graph_ctx}")
        except Exception as exc:
            print(f"    [WARN] Graph retrieval failed: {exc}")

    return contexts


async def _run_graph(question: str, graph_app) -> str:
    """
    Run the full LangGraph Self-RAG pipeline for a single question.
    Returns the final generation string.
    """
    thread_id = f"eval-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": question,
        "original_question": question,
        "generation": "",
        "documents": [],
        "is_hallucination": False,
        "search_retry_count": 0,
    }

    try:
        final_state = await graph_app.ainvoke(initial_state, config=config)
        return final_state.get("generation", "").strip()
    except Exception as exc:
        return f"[ERROR] Graph execution failed: {exc}"


async def _evaluate_example(
    example: dict[str, Any],
    graph_app,
    retrieve_vector_context,
    retrieve_graph_context,
) -> dict[str, Any]:
    """
    Run full retrieval + graph pipeline for a single golden example.
    Returns a dict ready for RAGAS EvaluationDataset ingestion.
    """
    question = example["question"]
    ground_truth = example["ground_truth"]
    requires_graph = example.get("requires_graph", False)

    print(f"  [{example['id']}] Retrieving contexts...")
    contexts = await _run_retrieval(
        question=question,
        requires_graph=requires_graph,
        retrieve_vector_context=retrieve_vector_context,
        retrieve_graph_context=retrieve_graph_context,
    )

    print(f"  [{example['id']}] Running Self-RAG graph ({len(contexts)} context chunks)...")
    answer = await _run_graph(question=question, graph_app=graph_app)

    print(f"  [{example['id']}] Done. Answer length: {len(answer)} chars.")

    return {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts if contexts else ["[No context retrieved]"],
        "reference": ground_truth,
        # Passthrough metadata for reporting
        "_id": example["id"],
        "_category": example["category"],
        "_requires_graph": requires_graph,
    }


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def _run_ragas_evaluation(rows: list[dict[str, Any]]) -> tuple[Any, dict[str, float]]:
    """
    Run RAGAS 0.4 evaluation on the collected rows.

    Uses the ragas 0.4 API:
      - SingleTurnSample for each row
      - EvaluationDataset wrapping the samples
      - LangchainLLMWrapper / LangchainEmbeddingsWrapper for the critic models
      - Class-based metrics (Faithfulness, AnswerRelevancy, etc.) with llm injected
      - llm + embeddings passed to evaluate() directly

    Returns:
        (ragas_result_object, aggregated_scores_dict)
    """
    from ragas import evaluate, EvaluationDataset  # noqa: PLC0415
    from ragas.dataset_schema import SingleTurnSample  # noqa: PLC0415
    from ragas.metrics.collections import (  # noqa: PLC0415
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    # Build SingleTurnSample objects (ragas 0.4 schema)
    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    # Instantiate class-based metrics with the wrapped LLM
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    print("\n[RAGAS] Running evaluation (this may take a few minutes)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )

    # Extract aggregate scores
    scores: dict[str, float] = {}
    try:
        result_df = result.to_pandas()
        metric_col_map = {
            "faithfulness": "faithfulness",
            "answer_relevancy": "answer_relevancy",
            "context_precision": "context_precision",
            "context_recall": "context_recall",
        }
        for key, col in metric_col_map.items():
            if col in result_df.columns:
                mean_val = result_df[col].dropna().mean()
                if mean_val == mean_val:  # not NaN
                    scores[key] = round(float(mean_val), 4)
    except Exception as exc:
        print(f"[WARN] Could not extract scores from result dataframe: {exc}")

    return result, scores


# ---------------------------------------------------------------------------
# Results serialization
# ---------------------------------------------------------------------------

def _save_results(
    rows: list[dict[str, Any]],
    ragas_scores: dict[str, float],
    tag: str,
    args: argparse.Namespace,
) -> Path:
    """Serialize and save results to evals/results/."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"ragas_results_{timestamp}"
    if tag:
        filename += f"_{tag}"
    filename += ".json"
    output_path = ctx.RESULTS_DIR / filename

    payload = {
        "run_metadata": {
            "timestamp": timestamp,
            "tag": tag,
            "total_examples": len(rows),
            "sample_size": args.sample,
            "categories_filter": args.categories,
            "skip_graph": args.skip_graph,
        },
        "aggregate_scores": ragas_scores,
        "per_example": [
            {
                "id": r["_id"],
                "category": r["_category"],
                "requires_graph": r["_requires_graph"],
                "question": r["user_input"],
                "answer": r["response"],
                "ground_truth": r["reference"],
                "context_count": len(r["retrieved_contexts"]),
                "contexts_preview": [c[:200] for c in r["retrieved_contexts"]],
            }
            for r in rows
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(ragas_scores: dict[str, float], rows: list[dict[str, Any]], output_path: Path):
    """Print a rich summary table to stdout."""
    bar = "=" * 60
    print(f"\n{bar}")
    print("  CITADEL ARCHIVAL SEARCH — RAGAS EVALUATION RESULTS")
    print(bar)
    print(f"  Total examples evaluated : {len(rows)}")
    print()

    metric_labels = {
        "faithfulness":       "Faithfulness         (answer grounded in context)",
        "answer_relevancy":   "Answer Relevancy     (answers the question)",
        "context_precision":  "Context Precision    (retrieved chunks are useful)",
        "context_recall":     "Context Recall       (all needed info retrieved)",
    }

    for key, label in metric_labels.items():
        score = ragas_scores.get(key)
        if score is not None:
            bar_width = int(score * 30)
            bar_str = "█" * bar_width + "░" * (30 - bar_width)
            flag = "✅" if score >= 0.7 else "⚠️ " if score >= 0.5 else "❌"
            print(f"  {flag} {label}")
            print(f"     [{bar_str}] {score:.4f}")
            print()

    # Per-category breakdown
    from collections import defaultdict
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["_category"]].append(r)

    if len(by_cat) > 1:
        print("  Per-category example counts:")
        for cat, examples in sorted(by_cat.items()):
            print(f"    {cat:<15} {len(examples)} examples")

    print()
    print(f"  Results saved to: {output_path}")
    print(bar)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS offline evaluation against the Citadel Archival Search system."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Only evaluate the first N examples (useful for smoke tests).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        metavar="CAT",
        choices=["factual", "lineage", "multi_hop", "negation", "hybrid", "adversarial"],
        help="Filter to specific categories.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="Run only specific example IDs (e.g. factual_001 lineage_002).",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip examples that require Neo4j graph context.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        metavar="TAG",
        help="Optional tag to append to the results filename (e.g. 'v1_baseline').",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Run retrieval + graph pipeline but skip RAGAS scoring (useful for pipeline debugging).",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace):
    # 1. Load dataset
    dataset = ctx.load_golden_dataset(
        categories=args.categories,
        ids=args.ids,
        limit=args.sample,
    )
    if args.skip_graph:
        dataset = [ex for ex in dataset if not ex.get("requires_graph", False)]

    if not dataset:
        print("[ERROR] No examples matched the provided filters. Exiting.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  CITADEL ARCHIVAL SEARCH — RAGAS EVAL RUNNER")
    print(f"{'='*60}")
    print(f"  Examples to evaluate : {len(dataset)}")
    if args.categories:
        print(f"  Categories           : {', '.join(args.categories)}")
    if args.skip_graph:
        print("  Graph examples       : skipped")
    print()

    # 2. Load app + retrieval functions
    print("[SETUP] Loading LangGraph app and retrieval services...")
    graph_app = ctx.get_graph_app()
    retrieve_vector_context, retrieve_graph_context = ctx.get_retrieval_functions()
    print("[SETUP] Ready.\n")

    # 3. Run retrieval + graph for each example sequentially
    # (Sequential is safer for rate-limited Groq API)
    rows: list[dict[str, Any]] = []
    for i, example in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] Processing: {example['id']} ({example['category']})")
        row = await _evaluate_example(
            example=example,
            graph_app=graph_app,
            retrieve_vector_context=retrieve_vector_context,
            retrieve_graph_context=retrieve_graph_context,
        )
        rows.append(row)

    print(f"\n[PIPELINE] All {len(rows)} examples processed.")

    # 4. Run RAGAS scoring
    ragas_scores: dict[str, float] = {}
    if not args.no_ragas:
        _, ragas_scores = _run_ragas_evaluation(rows)
    else:
        print("[RAGAS] Skipped (--no-ragas flag set).")

    # 5. Save results
    output_path = _save_results(rows, ragas_scores, args.tag, args)

    # 6. Print summary
    _print_summary(ragas_scores, rows, output_path)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
