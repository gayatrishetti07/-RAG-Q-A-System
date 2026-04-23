"""
utils/splitter.py
-----------------
Text chunking strategies for RAG preprocessing.

Why chunking matters:
- LLMs have fixed context windows (e.g., 4k–128k tokens)
- Smaller, focused chunks improve retrieval precision
- Overlap ensures context is not lost at chunk boundaries
"""

import os
from typing import List
from loguru import logger
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


def get_text_splitter(strategy: str = "recursive") -> RecursiveCharacterTextSplitter:
    """
    Factory function to create a text splitter based on the chosen strategy.

    Args:
        strategy: Splitting strategy. Options:
            - "recursive" (default): Splits on paragraph, sentence, word boundaries.
              Best general-purpose strategy.
            - "token": Splits by token count (respects LLM token limits precisely).

    Returns:
        A configured LangChain text splitter.
    """
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))

    logger.debug(f"Splitter config → strategy={strategy}, size={chunk_size}, overlap={chunk_overlap}")

    if strategy == "token":
        # Token-based splitting — respects actual LLM token limits
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Default: Recursive Character Splitter
    # Tries to split on: \n\n → \n → " " → "" (progressively smaller boundaries)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(documents: List[Document], strategy: str = "recursive") -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    Each chunk inherits metadata from its parent document (source, page number)
    and gets an additional `chunk_index` metadata field.

    Args:
        documents: List of LangChain Document objects to split.
        strategy: Splitting strategy ("recursive" or "token").

    Returns:
        List of chunked Document objects ready for embedding.
    """
    if not documents:
        logger.warning("No documents to split — received empty list.")
        return []

    splitter = get_text_splitter(strategy)
    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    logger.success(
        f"Split {len(documents)} document(s) → {len(chunks)} chunks "
        f"[strategy={strategy}, size={os.getenv('CHUNK_SIZE', 1000)}, "
        f"overlap={os.getenv('CHUNK_OVERLAP', 200)}]"
    )
    return chunks


def split_text(text: str, strategy: str = "recursive") -> List[str]:
    """
    Split a raw string into chunks (convenience wrapper).

    Args:
        text: Raw string to split.
        strategy: Splitting strategy.

    Returns:
        List of text chunk strings.
    """
    splitter = get_text_splitter(strategy)
    chunks = splitter.split_text(text)
    logger.debug(f"Split text → {len(chunks)} chunks")
    return chunks
