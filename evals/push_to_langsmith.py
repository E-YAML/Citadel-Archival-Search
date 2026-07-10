"""
push_to_langsmith.py
--------------------
LangSmith dataset push and experiment runner for Citadel Archival Search.

This script does two things:
  1. Push golden dataset → Creates/updates a LangSmith Dataset named
     'citadel-golden-dataset' with all 30 QA pairs as (input, output) examples.

  2. Run experiment → Wraps the full LangGraph Self-RAG pipeline in a
     LangSmith-evaluatable target function and runs langsmith.evaluate()
     against the dataset, with three LLM-as-judge evaluators:
       - correctness  : does the answer match the ground truth?
       - faithfulness : does the answer cite sources and stay grounded?
       - refusal_check: does the system correctly refuse adversarial questions?

Traces from every graph invocation are automatically linked to the experiment
in LangSmith (via LANGCHAIN_TRACING_V2=true).

Usage
-----
# Dry run — preview dataset without pushing to LangSmith:
    python evals/push_to_langsmith.py --dry-run

# Push dataset only (no experiment run):
    python evals/push_to_langsmith.py --push-only

# Full: push dataset + run experiment on all 30 examples:
    python evals/push_to_langsmith.py

# Run experiment on a sample of 5 examples:
    python evals/push_to_langsmith.py --sample 5

# Run experiment on specific categories:
    python evals/push_to_langsmith.py --categories factual lineage

# Custom experiment name:
    python evals/push_to_langsmith.py --experiment-name "v2_hybrid_search"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap env + path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest_evals as ctx  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LANGSMITH_DATASET_NAME = "citadel-golden-dataset"
LANGSMITH_DATASET_DESCRIPTION = (
    "Golden QA dataset for the Citadel Archival Search Self-RAG system. "
    "30 curated ASOIAF questions across 6 categories: "
    "factual, lineage, multi_hop, negation, hybrid, adversarial."
)


# ---------------------------------------------------------------------------
# Dataset push
# ---------------------------------------------------------------------------

def push_dataset(
    client,
    dry_run: bool = False,
) -> str | None:
    """
    Create or update the LangSmith dataset with all golden examples.

    Args:
        client:  Authenticated LangSmith Client.
        dry_run: If True, only prints what would be pushed without actually pushing.

    Returns:
        The dataset ID string (or None on dry run).
    """
    dataset = ctx.load_golden_dataset()

    inputs = [{"question": ex["question"]} for ex in dataset]
    outputs = [
        {
            "answer": ex["ground_truth"],
            "category": ex["category"],
            "requires_graph": ex["requires_graph"],
            "expected_retrieval_keywords": ex["expected_retrieval_keywords"],
        }
        for ex in dataset
    ]

    print(f"\n{'='*60}")
    print("  CITADEL ARCHIVAL SEARCH — LANGSMITH DATASET PUSH")
    print(f"{'='*60}")
    print(f"  Dataset name : {LANGSMITH_DATASET_NAME}")
    print(f"  Examples     : {len(dataset)}")
    print()

    if dry_run:
        print("  [DRY RUN] Would push the following examples:\n")
        for i, (inp, out) in enumerate(zip(inputs, outputs), 1):
            print(f"  [{i:02d}] Q: {inp['question'][:80]}...")
            print(f"       Category: {out['category']}")
        print(f"\n  [DRY RUN] Total: {len(inputs)} examples. No data was pushed.")
        return None

    # Check if dataset already exists
    existing_datasets = list(client.list_datasets(dataset_name=LANGSMITH_DATASET_NAME))
    if existing_datasets:
        dataset_id = str(existing_datasets[0].id)
        print(f"  Dataset already exists (ID: {dataset_id}).")
        print("  Checking for new examples to add...")

        existing_examples = list(client.list_examples(dataset_id=dataset_id))
        existing_questions = {
            ex.inputs.get("question", "") for ex in existing_examples
        }

        new_inputs = []
        new_outputs = []
        for inp, out in zip(inputs, outputs):
            if inp["question"] not in existing_questions:
                new_inputs.append(inp)
                new_outputs.append(out)

        if new_inputs:
            client.create_examples(
                inputs=new_inputs,
                outputs=new_outputs,
                dataset_id=dataset_id,
            )
            print(f"  Added {len(new_inputs)} new examples.")
        else:
            print("  All examples already present. No changes made.")
    else:
        # Create fresh dataset
        ls_dataset = client.create_dataset(
            dataset_name=LANGSMITH_DATASET_NAME,
            description=LANGSMITH_DATASET_DESCRIPTION,
        )
        dataset_id = str(ls_dataset.id)

        client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset_id,
        )
        print(f"  Created dataset ID: {dataset_id}")
        print(f"  Pushed {len(inputs)} examples.")

    print(f"\n  [OK] Dataset ready: https://smith.langchain.com/datasets/{dataset_id}")
    return dataset_id


# ---------------------------------------------------------------------------
# Target function (the system under test)
# ---------------------------------------------------------------------------

def build_target_fn():
    """
    Build a synchronous target function for langsmith.evaluate().
    LangSmith calls this with each example's inputs dict and expects
    a dict output with at minimum an 'answer' key.
    """
    graph_app = ctx.get_graph_app()

    async def _async_target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs.get("question", "")
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
            answer = final_state.get("generation", "").strip()
            retry_count = final_state.get("search_retry_count", 0)
        except Exception as exc:
            answer = f"[ERROR] Graph execution failed: {exc}"
            retry_count = -1

        return {
            "answer": answer,
            "search_retry_count": retry_count,
        }

    import time as _time  # noqa: PLC0415
    _example_count = [0]

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        """Sync wrapper for the async graph invocation.

        A 30-second cooldown is inserted between examples to stay within
        Groq's free-tier TPM (tokens-per-minute) rate limit. The Self-RAG
        graph issues ~8-12 Groq API calls per question (doc graders, rewrite,
        generate, hallucination check), so back-to-back examples quickly
        saturate the per-minute quota.
        """
        if _example_count[0] > 0:
            print(f"  [cooldown] Waiting 30s before example {_example_count[0] + 1} "
                  "to respect LLM provider rate limits...")
            _time.sleep(30)
        _example_count[0] += 1
        return asyncio.run(_async_target(inputs))

    return target


# ---------------------------------------------------------------------------
# LLM-as-judge evaluators
# ---------------------------------------------------------------------------

def build_evaluators():
    """
    Build LangSmith-compatible evaluator functions.
    Each evaluator receives (run, example) and returns a dict:
      { "key": str, "score": float (0-1), "comment": str }

    Uses build_llm_with_fallback() so that the eval judge automatically falls
    back to Gemini or OpenRouter if Groq hits a rate limit during scoring.
    """
    import sys as _sys  # noqa: PLC0415
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.core.llm_fallback import build_llm_with_fallback  # noqa: PLC0415
    from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415
    from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415

    llm = build_llm_with_fallback(temperature=0.0)

    # ── 1. Correctness evaluator ──────────────────────────────────────────
    correctness_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an objective judge evaluating whether an AI answer is correct.\n"
         "Compare the AI Answer to the Reference Answer.\n"
         "Score the answer:\n"
         "  1.0 = Fully correct, all key facts present\n"
         "  0.7 = Mostly correct, minor omissions\n"
         "  0.4 = Partially correct, key facts missing or wrong\n"
         "  0.0 = Incorrect or completely off-topic\n"
         "Output ONLY a JSON object: {{\"score\": <float>, \"reason\": \"<one sentence>\"}}"),
        ("human",
         "Reference Answer:\n{reference}\n\nAI Answer:\n{answer}")
    ])
    correctness_chain = correctness_prompt | llm | StrOutputParser()

    def correctness_evaluator(run, example) -> dict[str, Any]:
        import json as _json  # noqa: PLC0415
        answer = (run.outputs or {}).get("answer", "")
        reference = (example.outputs or {}).get("answer", "")

        try:
            raw = correctness_chain.invoke({"reference": reference, "answer": answer})
            parsed = _json.loads(raw.strip())
            score = float(parsed.get("score", 0.0))
            reason = parsed.get("reason", "")
        except Exception as exc:
            score = 0.0
            reason = f"Evaluation failed: {exc}"

        return {
            "key": "correctness",
            "score": max(0.0, min(1.0, score)),
            "comment": reason,
        }

    # ── 2. Faithfulness evaluator ─────────────────────────────────────────
    faithfulness_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader evaluating whether an AI answer cites sources and avoids hallucination.\n"
         "An answer is faithful if:\n"
         "  - It references book/chapter sources when making factual claims, OR\n"
         "  - It correctly states it cannot find the information in the archives\n"
         "An answer is unfaithful if:\n"
         "  - It makes specific factual claims without any source citation\n"
         "  - It fabricates plot details not in the ASOIAF books\n"
         "Score: 1.0 = Faithful | 0.5 = Partially faithful | 0.0 = Unfaithful\n"
         "Output ONLY JSON: {{\"score\": <float>, \"reason\": \"<one sentence>\"}}"),
        ("human",
         "AI Answer:\n{answer}")
    ])
    faithfulness_chain = faithfulness_prompt | llm | StrOutputParser()

    def faithfulness_evaluator(run, example) -> dict[str, Any]:
        import json as _json  # noqa: PLC0415
        answer = (run.outputs or {}).get("answer", "")

        try:
            raw = faithfulness_chain.invoke({"answer": answer})
            parsed = _json.loads(raw.strip())
            score = float(parsed.get("score", 0.0))
            reason = parsed.get("reason", "")
        except Exception as exc:
            score = 0.0
            reason = f"Evaluation failed: {exc}"

        return {
            "key": "faithfulness",
            "score": max(0.0, min(1.0, score)),
            "comment": reason,
        }

    # ── 3. Adversarial refusal check ─────────────────────────────────────
    refusal_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are checking whether an AI system correctly handled a trick question.\n"
         "The question contains a factual error or false premise.\n"
         "The AI should:\n"
         "  - Identify and correct the false premise, OR\n"
         "  - Decline to answer based on the false premise\n"
         "Score: 1.0 = Correctly identified false premise | 0.0 = Accepted false premise as true\n"
         "Output ONLY JSON: {{\"score\": <float>, \"reason\": \"<one sentence>\"}}"),
        ("human",
         "Question (contains false premise):\n{question}\n\nAI Answer:\n{answer}")
    ])
    refusal_chain = refusal_prompt | llm | StrOutputParser()

    def refusal_evaluator(run, example) -> dict[str, Any]:
        import json as _json  # noqa: PLC0415
        # Only applies to adversarial category
        category = (example.outputs or {}).get("category", "")
        if category != "adversarial":
            return {"key": "refusal_check", "score": None, "comment": "N/A — not adversarial"}

        answer = (run.outputs or {}).get("answer", "")
        question = (example.inputs or {}).get("question", "")

        try:
            raw = refusal_chain.invoke({"question": question, "answer": answer})
            parsed = _json.loads(raw.strip())
            score = float(parsed.get("score", 0.0))
            reason = parsed.get("reason", "")
        except Exception as exc:
            score = 0.0
            reason = f"Evaluation failed: {exc}"

        return {
            "key": "refusal_check",
            "score": max(0.0, min(1.0, score)),
            "comment": reason,
        }

    return [correctness_evaluator, faithfulness_evaluator, refusal_evaluator]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    client,
    dataset_id: str,
    experiment_name: str,
    sample: int | None = None,
    categories: list[str] | None = None,
) -> None:
    """
    Run a LangSmith experiment against the golden dataset.

    Args:
        client:           Authenticated LangSmith Client.
        dataset_id:       The LangSmith dataset ID to evaluate against.
        experiment_name:  Display name for this experiment run.
        sample:           Limit to first N examples.
        categories:       Filter examples by category.
    """
    try:
        from langsmith import evaluate as ls_evaluate  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "langsmith is not installed. Run: pip install langsmith>=0.1.0"
        ) from exc

    print(f"\n{'='*60}")
    print("  CITADEL ARCHIVAL SEARCH — LANGSMITH EXPERIMENT")
    print(f"{'='*60}")
    print(f"  Experiment name : {experiment_name}")
    print(f"  Dataset ID      : {dataset_id}")
    if sample:
        print(f"  Sample size     : {sample}")
    if categories:
        print(f"  Categories      : {', '.join(categories)}")
    print()

    target_fn = build_target_fn()
    evaluators = build_evaluators()

    # Build filter metadata for LangSmith (filter happens client-side via num_repetitions and example selection)
    kwargs: dict[str, Any] = {
        "data": dataset_id,
        "evaluators": evaluators,
        "experiment_prefix": experiment_name,
        "client": client,
        # Force strictly sequential execution so the 30s cooldown in target()
        # is effective and we never run two graph invocations concurrently.
        "max_concurrency": 1,
        "metadata": {
            "project": ctx.get_langsmith_project(),
            "categories_filter": categories or "all",
        },
    }
    if sample:
        kwargs["num_repetitions"] = 1

    print("  Starting experiment (each example runs the full Self-RAG graph)...")
    print("  This will take several minutes depending on Groq API rate limits.\n")

    results = ls_evaluate(target_fn, **kwargs)

    print(f"\n  ✅ Experiment complete!")
    print(f"  View results: https://smith.langchain.com/projects/{ctx.get_langsmith_project()}")

    # Print a quick score summary
    try:
        results_df = results.to_pandas()
        print("\n  Aggregate scores from this experiment run:")
        for col in results_df.columns:
            if col.startswith("feedback."):
                metric_name = col.replace("feedback.", "")
                mean_val = results_df[col].dropna().mean()
                if not (mean_val != mean_val):  # not NaN
                    print(f"    {metric_name:<20} {mean_val:.4f}")
    except Exception:
        pass  # Results summary is optional


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push golden dataset to LangSmith and optionally run an evaluation experiment."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be pushed without actually pushing to LangSmith.",
    )
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="Only push the dataset; skip running the evaluation experiment.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Run experiment on only the first N examples.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        metavar="CAT",
        choices=["factual", "lineage", "multi_hop", "negation", "hybrid", "adversarial"],
        help="Filter experiment to specific categories.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Custom experiment name (default: auto-generated with timestamp).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=LANGSMITH_DATASET_NAME,
        metavar="NAME",
        help=f"LangSmith dataset name (default: '{LANGSMITH_DATASET_NAME}').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build experiment name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    experiment_name = args.experiment_name or f"citadel-self-rag-eval-{timestamp}"

    # Dry run path — no LangSmith client needed
    if args.dry_run:
        push_dataset(client=None, dry_run=True)
        return

    # Authenticate
    print("[AUTH] Connecting to LangSmith...")
    client = ctx.get_langsmith_client()
    print("[AUTH] Connected.\n")

    # Push dataset
    dataset_id = push_dataset(client=client, dry_run=False)

    if args.push_only or dataset_id is None:
        print("\n[INFO] --push-only flag set. Skipping experiment run.")
        return

    # Run experiment
    run_experiment(
        client=client,
        dataset_id=dataset_id,
        experiment_name=experiment_name,
        sample=args.sample,
        categories=args.categories,
    )


if __name__ == "__main__":
    main()
