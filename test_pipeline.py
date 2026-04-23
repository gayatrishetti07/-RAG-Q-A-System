"""
test_pipeline.py
----------------
Quick integration test for the RAG pipeline.

Usage:
    python test_pipeline.py                    # Uses test PDF in data/
    python test_pipeline.py --pdf path/to.pdf  # Use specific PDF
    python test_pipeline.py --query "Your question here"
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}")


def run_test(pdf_path: str, query: str):
    """
    End-to-end test: load PDF → chunk → embed → index → query.
    Prints results to stdout.
    """
    from utils.loader import load_pdf
    from utils.splitter import split_documents
    from utils.vector_store import build_vector_store, get_retriever
    from backend.rag_pipeline import build_rag_chain, query_rag

    print("\n" + "=" * 60)
    print("  🧪 RAG Pipeline Integration Test")
    print("=" * 60 + "\n")

    # Step 1: Load PDF
    print(f"📄 Step 1: Loading PDF → {pdf_path}")
    documents = load_pdf(pdf_path)
    print(f"   ✅ Loaded {len(documents)} page(s)\n")

    # Step 2: Split
    print("✂️  Step 2: Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks\n")

    # Step 3: Build vector store
    print("🗂️  Step 3: Building FAISS vector store...")
    vector_store = build_vector_store(chunks)
    print(f"   ✅ FAISS index built\n")

    # Step 4: Build RAG chain
    print("⛓️  Step 4: Building RAG chain...")
    retriever = get_retriever(vector_store)
    chain = build_rag_chain(retriever)
    print(f"   ✅ Chain ready\n")

    # Step 5: Run queries
    test_queries = [query]

    # Add generic follow-up to test memory
    test_queries.append("Can you summarize what you just told me?")

    print("=" * 60)
    for i, q in enumerate(test_queries, 1):
        print(f"\n💬 Query {i}: {q}")
        print("-" * 40)
        result = query_rag(chain, q)
        print(f"🤖 Answer: {result['answer']}")
        print(f"\n📎 Sources used: {len(result['sources'])}")
        for j, src in enumerate(result['sources'], 1):
            print(f"   [{j}] {src['file_name']} — Page {src['page']}")
        print()

    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the RAG pipeline end-to-end.")
    parser.add_argument(
        "--pdf",
        default="data/sample.pdf",
        help="Path to a PDF file for testing (default: data/sample.pdf)",
    )
    parser.add_argument(
        "--query",
        default="What is this document about?",
        help="Test question to ask (default: 'What is this document about?')",
    )
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        logger.error(
            f"PDF not found: {args.pdf}\n"
            "Place a PDF in the data/ folder or pass --pdf path/to/file.pdf"
        )
        sys.exit(1)

    run_test(args.pdf, args.query)
