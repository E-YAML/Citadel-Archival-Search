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
    Build a ragas 0.4 InstructorLLM using llm_factory pointed at Groq's
    OpenAI-compatible endpoint.

    Must use AsyncOpenAI (not OpenAI) so that the instructor-backed LLM
    can serve agenerate() calls made by the collections metrics' ascore() methods.
    """
    from openai import AsyncOpenAI  # noqa: PLC0415
    from ragas.llms import llm_factory  # noqa: PLC0415
    import os

    api_key = os.environ.get("GROQ_API_KEY", "")
    # AsyncOpenAI pointed at Groq's OpenAI-compatible endpoint
    groq_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )
    return llm_factory("llama-3.3-70b-versatile", client=groq_client)


def _build_ragas_embeddings():
    """
    Build a ragas 0.4-native embeddings object by subclassing BaseRagasEmbedding
    (singular — the ABC that collections metrics validate against) and backing
    it with the already-installed fastembed model.
    """
    from fastembed import TextEmbedding  # noqa: PLC0415
    from ragas.embeddings.base import BaseRagasEmbedding  # noqa: PLC0415

    class _FastEmbedRagasEmbedding(BaseRagasEmbedding):
        """fastembed-backed BaseRagasEmbedding for ragas 0.4 collections metrics."""

        def __init__(self):
            self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        def _embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [list(v) for v in self._model.embed(texts)]

        # ── Required sync interface (BaseRagasEmbedding ABC) ──────────────
        def embed_text(self, text: str, **kwargs) -> list[float]:
            return self._embed_batch([text])[0]

        def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
            return self._embed_batch(texts)

        # ── Required async interface ───────────────────────────────────────
        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            import asyncio  # noqa: PLC0415
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self._embed_batch([text])[0])

        async def aembed_texts(
            self, texts: list[str], **kwargs
        ) -> list[list[float]]:
            import asyncio  # noqa: PLC0415
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._embed_batch, texts)

    return _FastEmbedRagasEmbedding()



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

async def _run_ragas_evaluation(rows: list[dict[str, Any]]) -> tuple[list[dict], dict[str, float]]:
    """
    Run RAGAS 0.4 evaluation using direct per-sample ascore() calls.

    In ragas 0.4, the new collections metrics (Faithfulness, AnswerRelevancy,
    ContextPrecision, ContextRecall) inherit from BaseMetric/SimpleBaseMetric,
    NOT from the legacy Metric class. The ragas evaluate() function only accepts
    instances of the legacy Metric class and rejects collections metrics.

    The correct API for collections metrics is to call metric.ascore(**kwargs)
    directly for each sample with the specific kwargs each metric requires.

    Returns:
        (per_row_results_list, aggregated_scores_dict)
    """
    import asyncio  # noqa: PLC0415
    from ragas.metrics.collections import (  # noqa: PLC0415
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    faith_metric = Faithfulness(llm=ragas_llm)
    ar_metric    = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    cp_metric    = ContextPrecision(llm=ragas_llm)
    cr_metric    = ContextRecall(llm=ragas_llm)

    print("\n[RAGAS] Scoring each example with 4 metrics (this may take a few minutes)...")

    per_row: list[dict] = []
    all_scores: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    for i, r in enumerate(rows, 1):
        q   = r["user_input"]
        ans = r["response"]
        ctx = r["retrieved_contexts"]
        ref = r["reference"]
        print(f"  [{i}/{len(rows)}] Scoring: {q[:60]}...")

        row_scores: dict[str, float | None] = {}
        try:
            result = await faith_metric.ascore(
                user_input=q, response=ans, retrieved_contexts=ctx
            )
            row_scores["faithfulness"] = float(result.value) if result.value is not None else None
        except Exception as exc:
            print(f"    [WARN] faithfulness failed: {exc}")
            row_scores["faithfulness"] = None

        try:
            result = await ar_metric.ascore(user_input=q, response=ans)
            row_scores["answer_relevancy"] = float(result.value) if result.value is not None else None
        except Exception as exc:
            print(f"    [WARN] answer_relevancy failed: {exc}")
            row_scores["answer_relevancy"] = None

        try:
            result = await cp_metric.ascore(
                user_input=q, reference=ref, retrieved_contexts=ctx
            )
            row_scores["context_precision"] = float(result.value) if result.value is not None else None
        except Exception as exc:
            print(f"    [WARN] context_precision failed: {exc}")
            row_scores["context_precision"] = None

        try:
            result = await cr_metric.ascore(
                user_input=q, retrieved_contexts=ctx, reference=ref
            )
            row_scores["context_recall"] = float(result.value) if result.value is not None else None
        except Exception as exc:
            print(f"    [WARN] context_recall failed: {exc}")
            row_scores["context_recall"] = None

        per_row.append({**r, "ragas_scores": row_scores})

        for key, val in row_scores.items():
            if val is not None:
                all_scores[key].append(val)

    # Aggregate
    scores: dict[str, float] = {}
    for key, vals in all_scores.items():
        if vals:
            scores[key] = round(sum(vals) / len(vals), 4)

    print("\n[RAGAS] Scoring complete.")
    return per_row, scores



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
        rows, ragas_scores = await _run_ragas_evaluation(rows)
    else:
        print("[RAGAS] Skipped (--no-ragas flag set).")

    # 5. Save results
    output_path = _save_results(rows, ragas_scores, args.tag, args)

    # 6. Print summary
    _print_summary(ragas_scores, rows, output_path)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
