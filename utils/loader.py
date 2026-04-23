"""
utils/loader.py
---------------
PDF document loader with support for single and multiple PDF files.
Uses LangChain's PyPDFLoader with fallback to PyMuPDF for better extraction.
"""

import os
from pathlib import Path
from typing import List
from loguru import logger
from langchain.schema import Document


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF file and return a list of LangChain Documents.

    Each page becomes its own Document object with metadata containing
    the source file path and page number.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects, one per page.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a PDF.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    logger.info(f"Loading PDF: {path.name}")

    try:
        # Primary loader: PyPDFLoader (fast, good for text-heavy PDFs)
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        logger.success(f"Loaded {len(documents)} pages from '{path.name}'")
        return documents

    except Exception as e:
        logger.warning(f"PyPDFLoader failed ({e}), trying PyMuPDF fallback...")

        # Fallback: PyMuPDF (better for scanned / complex PDFs)
        try:
            import fitz  # PyMuPDF
            documents = []
            pdf = fitz.open(str(path))
            for page_num, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():  # Skip empty pages
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "page": page_num + 1,
                            "total_pages": len(pdf),
                        }
                    )
                    documents.append(doc)
            pdf.close()
            logger.success(f"Loaded {len(documents)} pages via PyMuPDF from '{path.name}'")
            return documents

        except Exception as e2:
            logger.error(f"Both loaders failed for '{path.name}': {e2}")
            raise RuntimeError(f"Could not load PDF '{path.name}': {e2}") from e2


def load_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    """
    Load multiple PDF files and merge them into a single document list.

    Args:
        file_paths: List of paths to PDF files.

    Returns:
        Combined list of Documents from all PDFs.
    """
    all_documents = []
    failed = []

    for file_path in file_paths:
        try:
            docs = load_pdf(file_path)
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"Skipping '{file_path}': {e}")
            failed.append(file_path)

    if failed:
        logger.warning(f"Failed to load {len(failed)} file(s): {failed}")

    logger.info(f"Total documents loaded: {len(all_documents)} pages from {len(file_paths) - len(failed)} PDFs")
    return all_documents


def load_pdfs_from_directory(directory: str) -> List[Document]:
    """
    Recursively load all PDF files found in a directory.

    Args:
        directory: Path to the directory to scan for PDFs.

    Returns:
        Combined list of Documents from all discovered PDFs.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pdf_files = list(dir_path.rglob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in directory: {directory}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) in '{directory}'")
    return load_multiple_pdfs([str(p) for p in pdf_files])
