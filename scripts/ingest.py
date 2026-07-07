"""
CLI: Ingest textbook PDFs into ChromaDB.

Usage:
    python scripts/ingest.py --file data/raw/physics.pdf
    python scripts/ingest.py --dir data/raw/
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.ingest_pipeline import ingest_pdf, ingest_directory


def main():
    parser = argparse.ArgumentParser(description="Ingest textbook PDF(s) into ChromaDB")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single PDF")
    group.add_argument("--dir",  help="Directory containing PDFs")
    args = parser.parse_args()

    summary = ingest_pdf(args.file) if args.file else ingest_directory(args.dir)

    print("\n=== Ingestion Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
