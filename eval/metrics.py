"""
Evaluation metrics for the textbook RAG retrieval pipeline.

Metrics:
  - Hit Rate   : fraction of queries where ≥1 correct chunk in top-K
  - MRR        : Mean Reciprocal Rank (average 1/rank of first relevant chunk)
  - Precision@K: fraction of top-K chunks that are relevant
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.rag_pipeline import query
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _is_relevant(chunk: dict, qa: dict) -> bool:
    """
    A chunk is relevant if it contains ANY of the expected keywords (case-insensitive).
    Optionally also checks source and chapter if provided.
    """
    text = chunk.get("text", "").lower()
    keywords = qa.get("expected_keywords", [])

    keyword_match = any(kw.lower() in text for kw in keywords)
    if not keyword_match:
        return False

    if qa.get("expected_source") and chunk.get("source") != qa["expected_source"]:
        return False

    return True


def evaluate(dataset_path: str, top_k: int = 5) -> dict:
    """
    Run evaluation over a QA dataset JSON file.

    Each item in the dataset:
      {
        "question":          str,
        "expected_keywords": [str, ...],
        "expected_source":   str  (optional),
        "expected_chapter":  str  (optional)
      }

    Returns:
      {
        "total_questions": int,
        "hit_rate":        float,   # 0–1
        "mrr":             float,   # 0–1
        "precision_at_k":  float,   # 0–1
        "per_question":    [...]
      }
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    hits, reciprocal_ranks, precisions = [], [], []
    per_question = []

    for qa in dataset:
        question = qa["question"]
        result   = query(question, top_k=top_k)
        chunks   = result["chunks"]

        relevant_flags = [_is_relevant(c, qa) for c in chunks]
        hit            = any(relevant_flags)
        first_relevant = next((i + 1 for i, r in enumerate(relevant_flags) if r), None)
        rr             = 1.0 / first_relevant if first_relevant else 0.0
        precision      = sum(relevant_flags) / top_k if chunks else 0.0

        hits.append(float(hit))
        reciprocal_ranks.append(rr)
        precisions.append(precision)

        per_question.append({
            "question":   question,
            "hit":        hit,
            "rr":         round(rr, 4),
            "precision":  round(precision, 4),
            "top_chunk":  chunks[0]["heading_path"] if chunks else None,
        })

        logger.info(f"Q: {question[:60]} | hit={hit} rr={rr:.2f} p={precision:.2f}")

    n = len(dataset)
    summary = {
        "total_questions": n,
        "hit_rate":        round(sum(hits) / n, 4),
        "mrr":             round(sum(reciprocal_ranks) / n, 4),
        "precision_at_k":  round(sum(precisions) / n, 4),
        "top_k":           top_k,
        "per_question":    per_question,
    }

    print("\n=== Evaluation Results ===")
    print(f"  Questions  : {summary['total_questions']}")
    print(f"  Hit Rate   : {summary['hit_rate']:.1%}")
    print(f"  MRR        : {summary['mrr']:.4f}")
    print(f"  Precision@{top_k}: {summary['precision_at_k']:.1%}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/datasets/sample_qa.json")
    parser.add_argument("--top_k",   type=int, default=5)
    parser.add_argument("--out",     default=None, help="Save results JSON")
    args = parser.parse_args()

    results = evaluate(args.dataset, top_k=args.top_k)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.out}")
