"""
backend/rag_pipeline.py
-----------------------
Core RAG pipeline: chains the retriever, prompt template, LLM, and memory.

Architecture:
    User Query
        │
        ▼
    Retriever ──► FAISS Vector Store ──► Top-K Relevant Chunks
        │
        ▼
    Prompt Template (query + chunks → structured prompt)
        │
        ▼
    LLM (OpenAI / HuggingFace)
        │
        ▼
    Answer (grounded in your documents)
"""

import os
from typing import Any, Dict, Optional
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

from utils.vector_store import get_retriever, load_vector_store


# ── Prompt Templates ──────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """You are an expert assistant that answers questions based ONLY on the provided context documents.

Instructions:
- Answer the question using information from the context below.
- If the answer is not found in the context, clearly state: "I don't have enough information in the provided documents to answer this question."
- Do NOT make up information or use outside knowledge.
- Be concise, accurate, and helpful.
- When possible, mention which part of the document supports your answer.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

CONDENSE_QUESTION_PROMPT_TEMPLATE = """Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question that captures full context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


def get_llm():
    """
    Return a configured LLM based on the LLM_PROVIDER environment variable.

    Supports:
        - "openai": ChatOpenAI with configurable model (gpt-3.5-turbo default)
        - "huggingface": HuggingFacePipeline with flan-t5-large (free, local)
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    logger.info(f"Initializing LLM (provider={provider})")

    if provider == "openai":
        return _get_openai_llm()
    elif provider == "huggingface":
        return _get_huggingface_llm()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'")


def _get_openai_llm():
    """Initialize OpenAI ChatLLM."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Set it in .env or switch to LLM_PROVIDER=huggingface"
        )

    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    llm = ChatOpenAI(
        model_name=model,
        temperature=0.0,          # Low temperature = factual, deterministic answers
        openai_api_key=api_key,
        max_tokens=1024,
        streaming=False,
    )
    logger.success(f"OpenAI LLM ready (model={model})")
    return llm


def _get_huggingface_llm():
    """Initialize a local HuggingFace LLM pipeline (no API key needed)."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain_community.llms import HuggingFacePipeline

    model_name = os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-large")
    logger.info(f"Loading HuggingFace model '{model_name}' locally (this may take a moment)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    logger.success(f"HuggingFace LLM ready (model={model_name})")
    return llm


def build_rag_chain(retriever: Optional[BaseRetriever] = None) -> ConversationalRetrievalChain:
    """
    Build the full ConversationalRetrievalChain.

    This chain:
    1. Condenses follow-up questions using chat history
    2. Retrieves relevant document chunks
    3. Generates grounded answers with the LLM

    Args:
        retriever: Optional pre-built retriever. Loads from FAISS if None.

    Returns:
        A configured ConversationalRetrievalChain.
    """
    if retriever is None:
        vector_store = load_vector_store()
        retriever = get_retriever(vector_store)

    llm = get_llm()

    # Memory: stores last N exchanges for multi-turn conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Custom prompt for the final answer generation step
    qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )

    # Prompt for condensing follow-up questions into standalone questions
    condense_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=CONDENSE_QUESTION_PROMPT_TEMPLATE,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,       # Include source chunks in response
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_prompt,
        verbose=False,
    )

    logger.success("RAG chain built successfully.")
    return chain


def query_rag(chain: ConversationalRetrievalChain, question: str) -> Dict[str, Any]:
    """
    Run a question through the RAG chain and return structured output.

    Args:
        chain: The ConversationalRetrievalChain instance.
        question: User's question string.

    Returns:
        Dict containing:
            - answer (str): The generated answer
            - sources (list): List of source document metadata
            - question (str): The original question

    Raises:
        RuntimeError: On chain invocation failure.
    """
    logger.info(f"Processing query: {question!r}")

    try:
        result = chain.invoke({"question": question})

        # Extract and format source documents
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "file_name": doc.metadata.get("file_name", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
            })

        output = {
            "question": question,
            "answer": result.get("answer", "No answer generated."),
            "sources": sources,
        }

        logger.success(f"Query answered. Sources used: {len(sources)}")
        return output

    except Exception as e:
        logger.error(f"RAG chain error: {e}")
        raise RuntimeError(f"Failed to process query: {e}") from e


class RAGPipeline:
    """
    Singleton-style RAG pipeline manager.

    Manages the lifecycle of the chain and memory across API requests.
    Use this class in the FastAPI app to maintain state between calls.
    """

    def __init__(self):
        self._chain: Optional[ConversationalRetrievalChain] = None
        logger.debug("RAGPipeline instance created.")

    def initialize(self) -> bool:
        """
        Build the RAG chain from the FAISS index.
        Returns True if successful, False if index doesn't exist.
        """
        from utils.vector_store import index_exists
        if not index_exists():
            logger.warning("Cannot initialize RAG pipeline: no FAISS index found.")
            return False

        try:
            self._chain = build_rag_chain()
            return True
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False

    def ask(self, question: str) -> Dict[str, Any]:
        """Query the pipeline. Auto-initializes if not ready."""
        if self._chain is None:
            initialized = self.initialize()
            if not initialized:
                return {
                    "question": question,
                    "answer": "⚠️ No documents indexed yet. Please upload PDFs first.",
                    "sources": [],
                }
        return query_rag(self._chain, question)

    def reset_memory(self):
        """Clear conversation history."""
        if self._chain and hasattr(self._chain, "memory"):
            self._chain.memory.clear()
            logger.info("Conversation memory cleared.")

    def rebuild(self):
        """Force rebuild the chain (after new documents are indexed)."""
        logger.info("Rebuilding RAG pipeline...")
        self._chain = None
        self.initialize()

    @property
    def is_ready(self) -> bool:
        """Check if the pipeline is initialized."""
        return self._chain is not None
