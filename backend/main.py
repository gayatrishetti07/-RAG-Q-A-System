"""
backend/main.py
---------------
FastAPI application — the backbone of the RAG Q&A system.

Endpoints:
    POST /upload     → Upload and index PDF files
    POST /ask        → Ask a question (RAG query)
    GET  /history    → Get conversation history
    POST /reset      → Clear conversation memory
    GET  /status     → System health check
    DELETE /index    → Delete FAISS index and rebuild

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
from loguru import logger

log_file = os.getenv("LOG_FILE", "logs/app.log")
Path(log_file).parent.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"), colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(log_file, rotation="10 MB", retention="7 days", compression="zip",
           level="DEBUG")

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Internal modules ──────────────────────────────────────────────────────────
from backend.models import (
    QuestionRequest, AnswerResponse, UploadResponse,
    StatusResponse, HistoryResponse, HistoryItem, ResetResponse, ErrorResponse,
    SourceDocument,
)
from backend.rag_pipeline import RAGPipeline
from utils.loader import load_multiple_pdfs
from utils.splitter import split_documents
from utils.vector_store import (
    build_vector_store,
    add_documents_to_store,
    load_vector_store,
    index_exists,
)

# ── App initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Q&A System API",
    description="""
## 🤖 Retrieval-Augmented Generation Q&A API

Upload PDF documents and ask questions grounded in their content.

### Features
- 📄 Multiple PDF support
- 🔍 Semantic search via FAISS
- 🧠 Conversational memory
- 🔄 OpenAI + HuggingFace LLM support
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global pipeline instance ──────────────────────────────────────────────────
# One instance shared across all requests (maintains conversation memory)
pipeline = RAGPipeline()


@app.on_event("startup")
async def startup_event():
    """Auto-initialize the RAG pipeline if an index already exists."""
    logger.info("FastAPI app starting up...")
    if index_exists():
        logger.info("Existing FAISS index found. Initializing RAG pipeline...")
        success = pipeline.initialize()
        if success:
            logger.success("RAG pipeline ready on startup.")
        else:
            logger.warning("Pipeline initialization failed on startup.")
    else:
        logger.info("No FAISS index found. Upload PDFs via POST /upload to begin.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint — quick health check."""
    return {
        "status": "ok",
        "message": "RAG Q&A System is running. Visit /docs for API documentation.",
        "version": "1.0.0",
    }


@app.get("/status", response_model=StatusResponse, tags=["Health"])
async def get_status():
    """
    Return the current system status.
    Useful for the frontend to know if the system is ready.
    """
    return StatusResponse(
        index_exists=index_exists(),
        pipeline_ready=pipeline.is_ready,
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files to be indexed.

    Steps:
    1. Validate file types (PDF only)
    2. Save to temp directory
    3. Load and chunk documents
    4. Add to FAISS vector store (creates or extends)
    5. Reinitialize RAG pipeline with updated index

    Args:
        files: List of PDF files (multipart/form-data).

    Returns:
        UploadResponse with processing stats.
    """
    # Validate file types
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are accepted. Got: {file.filename}",
            )

    logger.info(f"Received {len(files)} file(s) for upload.")

    # Save uploaded files to a temp directory
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    file_names = []

    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_paths.append(file_path)
            file_names.append(file.filename)
            logger.debug(f"Saved upload: {file.filename} ({len(content)} bytes)")

        # Load and chunk all PDFs
        documents = load_multiple_pdfs(saved_paths)
        if not documents:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from the uploaded PDFs.",
            )

        chunks = split_documents(documents)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not split documents into chunks.",
            )

        # Update vector store
        existing_store = load_vector_store()
        add_documents_to_store(chunks, existing_store)

        # Rebuild RAG pipeline with updated index
        pipeline.rebuild()

        logger.success(
            f"Indexed {len(chunks)} chunks from {len(files)} file(s): {file_names}"
        )

        return UploadResponse(
            message=f"Successfully indexed {len(files)} PDF(s).",
            files_processed=len(files),
            chunks_indexed=len(chunks),
            file_names=file_names,
        )

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/ask", response_model=AnswerResponse, tags=["QnA"])
async def ask_question(request: QuestionRequest):
    """
    Ask a question and receive a context-grounded answer.

    The RAG pipeline:
    1. Retrieves top-k relevant chunks from FAISS
    2. Builds a prompt with context + chat history
    3. Passes to LLM for answer generation
    4. Returns answer with source citations

    Args:
        request: QuestionRequest with the question string.

    Returns:
        AnswerResponse with answer and source documents.
    """
    if not index_exists():
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Please upload PDFs via POST /upload first.",
        )

    if not pipeline.is_ready:
        initialized = pipeline.initialize()
        if not initialized:
            raise HTTPException(
                status_code=500,
                detail="RAG pipeline could not be initialized. Check logs.",
            )

    try:
        result = pipeline.ask(request.question)

        sources = [
            SourceDocument(
                content=s["content"],
                source=s["source"],
                file_name=s["file_name"],
                page=str(s["page"]),
            )
            for s in result.get("sources", [])
        ]

        return AnswerResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
        )

    except RuntimeError as e:
        logger.error(f"/ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=HistoryResponse, tags=["QnA"])
async def get_history():
    """
    Retrieve the current conversation history.

    Returns all question-answer pairs from the current session.
    Memory is cleared when /reset is called or the server restarts.
    """
    if not pipeline.is_ready or pipeline._chain is None:
        return HistoryResponse(history=[])

    history = []
    try:
        messages = pipeline._chain.memory.chat_memory.messages
        for msg in messages:
            role = "user" if msg.type == "human" else "assistant"
            history.append(HistoryItem(role=role, content=msg.content))
    except Exception as e:
        logger.warning(f"Could not retrieve history: {e}")

    return HistoryResponse(history=history)


@app.post("/reset", response_model=ResetResponse, tags=["QnA"])
async def reset_memory():
    """
    Clear the conversation memory.

    Use this to start a fresh conversation while keeping
    the indexed documents intact.
    """
    pipeline.reset_memory()
    return ResetResponse(message="Conversation memory cleared successfully.")


@app.delete("/index", tags=["Documents"])
async def delete_index():
    """
    Delete the FAISS index and reset the pipeline.

    Use this to re-index with entirely new documents.
    """
    index_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    if Path(index_path).exists():
        shutil.rmtree(index_path)
        logger.info(f"Deleted FAISS index at '{index_path}'")

    pipeline._chain = None
    return {"message": "Index deleted. Upload new PDFs to rebuild."}


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler for unexpected errors."""
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "detail": str(exc)},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("BACKEND_PORT", 8000)),
        reload=True,
        log_level="info",
    )
