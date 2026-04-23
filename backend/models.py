"""
backend/models.py
-----------------
Pydantic schemas for FastAPI request/response validation.

Pydantic provides:
- Automatic type validation
- Clear API contracts
- Auto-generated OpenAPI docs
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request schema for the /ask endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question to ask the RAG system.",
        examples=["What is the main topic of the uploaded document?"],
    )


class SourceDocument(BaseModel):
    """Represents a retrieved source chunk used to generate the answer."""

    content: str = Field(..., description="Excerpt from the source chunk.")
    source: str = Field(..., description="Full file path of the source document.")
    file_name: str = Field(..., description="Base file name of the source PDF.")
    page: str = Field(..., description="Page number within the PDF.")


class AnswerResponse(BaseModel):
    """Response schema for the /ask endpoint."""

    question: str = Field(..., description="The original question asked.")
    answer: str = Field(..., description="The LLM-generated, context-grounded answer.")
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="List of document chunks used to generate the answer.",
    )


class UploadResponse(BaseModel):
    """Response schema for the /upload endpoint."""

    message: str = Field(..., description="Status message.")
    files_processed: int = Field(..., description="Number of PDFs processed.")
    chunks_indexed: int = Field(..., description="Total chunks added to vector store.")
    file_names: List[str] = Field(..., description="Names of uploaded files.")


class StatusResponse(BaseModel):
    """Response schema for the /status endpoint."""

    index_exists: bool = Field(..., description="Whether a FAISS index has been built.")
    pipeline_ready: bool = Field(..., description="Whether the RAG pipeline is initialized.")
    llm_provider: str = Field(..., description="Active LLM provider (openai/huggingface).")
    embedding_provider: str = Field(..., description="Active embedding provider.")


class HistoryItem(BaseModel):
    """A single turn in the conversation history."""

    role: str = Field(..., description="'user' or 'assistant'.")
    content: str = Field(..., description="Message content.")


class HistoryResponse(BaseModel):
    """Response schema for the /history endpoint."""

    history: List[HistoryItem] = Field(
        default_factory=list,
        description="List of conversation turns.",
    )


class ResetResponse(BaseModel):
    """Response schema for the /reset endpoint."""

    message: str = Field(..., description="Confirmation message.")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type.")
    detail: str = Field(..., description="Detailed error message.")
