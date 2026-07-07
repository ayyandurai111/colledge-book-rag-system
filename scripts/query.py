"""
CLI: Query the textbook RAG system.

Usage:
    python scripts/query.py --question "What is Newton's Second Law?"
    python scripts/query.py --question "Define entropy" --type definition
    python scripts/query.py --question "Force formula" --type formula --source physics.pdf --top_k 3
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.rag_pipeline import query


def main():
    parser = argparse.ArgumentParser(description="Query the textbook RAG system")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top_k",   type=int, default=None)
    parser.add_argument("--type",    choices=["definition","formula","table","figure"],
                        default=None, dest="content_type")
    parser.add_argument("--source",  default=None, help="Filter by textbook filename")
    parser.add_argument("--chapter", default=None, help="Filter by chapter title")
    args = parser.parse_args()

    filters = {}
    if args.source:  filters["source"]  = args.source
    if args.chapter: filters["chapter"] = args.chapter

    result = query(
        question=args.question,
        top_k=args.top_k,
        filters=filters or None,
        content_type=args.content_type
    )

    print(f"\n=== Question ===\n{result['question']}\n")
    if result["content_type"]:
        print(f"Content type filter: {result['content_type']}")
    if result["filters"]:
        print(f"Metadata filters: {result['filters']}")
    print(f"\n=== {result['total_chunks']} Chunks Retrieved ===\n")

    for i, chunk in enumerate(result["chunks"], 1):
        print(f"{'─'*60}")
        print(f"[{i}] Score: {chunk['score']}  |  Type: {chunk['chunk_type']}")
        print(f"     {chunk['heading_path']}")
        print(f"     Source: {chunk['source']} | Page: {chunk['page_number']}")
        print()
        print(chunk["text"][:400])
        print()


if __name__ == "__main__":
    main()
