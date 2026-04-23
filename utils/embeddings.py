"""
utils/embeddings.py
-------------------
Embedding model factory supporting OpenAI and HuggingFace providers.

Embeddings convert text chunks into dense numerical vectors.
These vectors capture semantic meaning — similar texts get similar vectors.
This enables similarity search in the vector store.
"""

import os
from loguru import logger
from langchain.embeddings.base import Embeddings


def get_embeddings() -> Embeddings:
    """
    Return a configured embedding model based on the EMBEDDING_PROVIDER env var.

    Providers:
        - "openai" (default): Uses OpenAI's text-embedding-ada-002 model.
          Pros: Best quality, fast API.
          Cons: Costs money per token, requires API key.

        - "huggingface": Uses sentence-transformers locally (free).
          Pros: No API cost, works offline.
          Cons: Slower, lower quality than OpenAI.

    Returns:
        A LangChain-compatible Embeddings object.

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    logger.info(f"Initializing embedding model (provider={provider})")

    if provider == "openai":
        return _get_openai_embeddings()
    elif provider == "huggingface":
        return _get_huggingface_embeddings()
    else:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER: '{provider}'. "
            "Choose 'openai' or 'huggingface'."
        )


def _get_openai_embeddings() -> Embeddings:
    """Initialize OpenAI embedding model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Set it in your .env file or use EMBEDDING_PROVIDER=huggingface."
        )

    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=api_key,
        )
        logger.success("OpenAI embeddings initialized (model=text-embedding-ada-002)")
        return embeddings

    except ImportError:
        raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")


def _get_huggingface_embeddings() -> Embeddings:
    """Initialize HuggingFace sentence-transformers embedding model."""
    model_name = os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.success(f"HuggingFace embeddings initialized (model={model_name})")
        return embeddings

    except ImportError:
        raise ImportError(
            "langchain-huggingface not installed. "
            "Run: pip install langchain-huggingface sentence-transformers"
        )
