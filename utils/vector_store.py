"""
utils/vector_store.py
---------------------
FAISS vector store management: creation, persistence, and loading.

FAISS (Facebook AI Similarity Search) enables fast approximate nearest-neighbor
search over high-dimensional embedding vectors. It stores vectors in memory
and can persist them to disk for reuse across sessions.
"""

import os
from pathlib import Path
from typing import List, Optional
from loguru import logger
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever

from utils.embeddings import get_embeddings


def get_faiss_index_path() -> str:
    """Return the configured FAISS index directory path."""
    return os.getenv("FAISS_INDEX_PATH", "faiss_index")


def build_vector_store(documents: List[Document]) -> FAISS:
    """
    Create a new FAISS vector store from a list of chunked Documents.

    This function:
    1. Generates embeddings for every chunk (API call or local model)
    2. Stores them in a FAISS index in memory
    3. Persists the index to disk for future sessions

    Args:
        documents: List of chunked Documents (output of splitter.py).

    Returns:
        A FAISS VectorStore object ready for similarity search.

    Raises:
        ValueError: If the document list is empty.
    """
    if not documents:
        raise ValueError("Cannot build vector store from empty document list.")

    logger.info(f"Building FAISS index from {len(documents)} chunks...")

    embeddings = get_embeddings()

    # FAISS.from_documents:
    # - Extracts text from each Document
    # - Calls the embedding model (in batches for efficiency)
    # - Builds the index
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    # Persist to disk so we don't re-embed on every restart
    index_path = get_faiss_index_path()
    Path(index_path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(index_path)

    logger.success(
        f"FAISS index built and saved to '{index_path}' "
        f"({len(documents)} vectors)"
    )
    return vector_store


def load_vector_store() -> Optional[FAISS]:
    """
    Load an existing FAISS index from disk.

    Returns:
        FAISS VectorStore object if index exists, None otherwise.
    """
    index_path = get_faiss_index_path()
    index_file = Path(index_path) / "index.faiss"

    if not index_file.exists():
        logger.warning(f"No FAISS index found at '{index_path}'. Upload PDFs first.")
        return None

    logger.info(f"Loading FAISS index from '{index_path}'...")

    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,  # Required by recent LangChain versions
    )

    logger.success("FAISS index loaded successfully.")
    return vector_store


def add_documents_to_store(
    new_documents: List[Document],
    existing_store: Optional[FAISS] = None,
) -> FAISS:
    """
    Add new documents to an existing FAISS store (or create a new one).

    Supports incremental indexing — you can add new PDFs without
    rebuilding the entire index from scratch.

    Args:
        new_documents: New chunked Documents to add.
        existing_store: Optional existing FAISS store to extend.

    Returns:
        Updated FAISS VectorStore.
    """
    if existing_store is None:
        logger.info("No existing store provided, creating new one.")
        return build_vector_store(new_documents)

    logger.info(f"Adding {len(new_documents)} chunks to existing FAISS index...")
    embeddings = get_embeddings()
    new_store = FAISS.from_documents(new_documents, embeddings)
    existing_store.merge_from(new_store)

    # Re-save merged index
    index_path = get_faiss_index_path()
    existing_store.save_local(index_path)
    logger.success(f"Index updated and saved. Added {len(new_documents)} new chunks.")

    return existing_store


def get_retriever(
    vector_store: Optional[FAISS] = None,
    k: Optional[int] = None,
) -> VectorStoreRetriever:
    """
    Create a retriever from the vector store.

    The retriever takes a query string and returns the top-k most
    semantically similar document chunks.

    Args:
        vector_store: Optional pre-loaded FAISS store. Loads from disk if None.
        k: Number of top results to return. Defaults to TOP_K_RESULTS env var.

    Returns:
        A LangChain VectorStoreRetriever.

    Raises:
        RuntimeError: If no FAISS index is found.
    """
    if vector_store is None:
        vector_store = load_vector_store()

    if vector_store is None:
        raise RuntimeError(
            "No FAISS index available. "
            "Please upload PDFs and build the index first."
        )

    top_k = k or int(os.getenv("TOP_K_RESULTS", 4))

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    logger.debug(f"Retriever created (top_k={top_k})")
    return retriever


def index_exists() -> bool:
    """Check if a FAISS index has been built."""
    index_path = get_faiss_index_path()
    return (Path(index_path) / "index.faiss").exists()
