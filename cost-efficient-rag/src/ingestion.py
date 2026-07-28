# ============================================================
# src/ingestion.py
#
# WHAT THIS FILE DOES (the "reading documents in" part of RAG):
# 1. Reads text out of PDF, HTML, and Markdown files.
# 2. Splits that text into small overlapping "chunks".
# 3. Gives every chunk a unique, deterministic ID (a hash) so that
#    if you run ingestion twice on the same file, you get the SAME
#    IDs both times — meaning duplicates never get inserted.
#    This is called "idempotent ingestion".
# ============================================================

import hashlib
import os
from typing import List, Dict, Any

from pypdf import PdfReader
from bs4 import BeautifulSoup
import markdown as markdown_lib

from src.config import settings


# ------------------------------------------------------------
# STEP A: Loaders — one function per file type.
# Each one takes a file path and returns the RAW TEXT inside it.
# ------------------------------------------------------------

def load_pdf(file_path: str) -> str:
    """Extract all text from a PDF file, page by page."""
    reader = PdfReader(file_path)
    all_text = []
    for page in reader.pages:
        # page.extract_text() can return None for blank/image-only pages,
        # so we guard against that with `or ""`.
        all_text.append(page.extract_text() or "")
    return "\n".join(all_text)


def load_html(file_path: str) -> str:
    """Extract visible text from an HTML file, stripping out tags/scripts."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_html = f.read()
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove script/style tags entirely — we don't want their text content.
    for tag in soup(["script", "style"]):
        tag.decompose()

    # get_text() pulls out just the human-readable text.
    # separator=" " puts a space between text from different tags so
    # words don't accidentally get jammed together.
    return soup.get_text(separator=" ", strip=True)


def load_markdown(file_path: str) -> str:
    """Convert a Markdown file to plain text (strip formatting)."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_md = f.read()
    html = markdown_lib.markdown(raw_md)
    # Markdown -> HTML -> plain text is the simplest reliable way
    # to strip out #, *, [], etc. and keep just the words.
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def load_document(file_path: str) -> str:
    """
    Look at the file's extension and call the right loader.
    Raises an error for unsupported file types so problems are
    caught immediately instead of silently ignored.
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return load_pdf(file_path)
    elif extension in (".html", ".htm"):
        return load_html(file_path)
    elif extension in (".md", ".markdown", ".txt"):
        return load_markdown(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


# ------------------------------------------------------------
# STEP B: Chunking — split long text into overlapping pieces.
# ------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[str]:
    """
    Split `text` into a list of overlapping substrings.

    Why overlap? If a sentence is cut in half at a chunk boundary,
    the overlap means the next chunk still contains that sentence
    too, so we don't lose meaning at the edges.

    Example with chunk_size=10, chunk_overlap=3, text="ABCDEFGHIJKLMNOP":
      chunk 1: "ABCDEFGHIJ"      (characters 0-10)
      chunk 2: "HIJKLMNOPQ"      (starts at 10-3=7... etc.)
    """
    # Fall back to the values from .env if the caller didn't specify any.
    chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # skip empty/whitespace-only chunks
            chunks.append(chunk)
        # Move the start position forward, but back up by `chunk_overlap`
        # characters so consecutive chunks share some text.
        start += (chunk_size - chunk_overlap)

    return chunks


# ------------------------------------------------------------
# STEP C: Idempotent hashing — the anti-duplicate mechanism.
# ------------------------------------------------------------

def generate_chunk_id(source_path: str, chunk_index: int, text: str) -> str:
    """
    Build a unique, DETERMINISTIC id for a chunk.

    "Deterministic" is the key word: the same (source_path, chunk_index, text)
    will ALWAYS hash to the same ID, no matter how many times you run this.
    That's what lets us detect "we've already ingested this exact chunk before".
    """
    raw_identifier = f"{source_path}_{chunk_index}_{text}"
    return hashlib.sha256(raw_identifier.encode("utf-8")).hexdigest()


def process_and_deduplicate(
    chunks: List[str],
    source_path: str,
    file_type: str,
    existing_ids: set,
) -> List[Dict[str, Any]]:
    """
    Turn raw text chunks into full records ready for the vector store,
    SKIPPING any chunk whose ID we've already seen before.

    `existing_ids` should be the set of chunk IDs already stored in the
    vector database (fetched by vector_store.py before calling this).
    """
    new_records = []
    for idx, text in enumerate(chunks):
        chunk_id = generate_chunk_id(source_path, idx, text)

        if chunk_id in existing_ids:
            # We've seen this exact chunk before — skip it.
            # This is what makes re-running ingestion safe.
            continue

        new_records.append({
            "id": chunk_id,
            "text": text,
            "source": source_path,
            "chunk_index": idx,
            "file_type": file_type,
        })
    return new_records


# ------------------------------------------------------------
# STEP D: Put it all together for one file.
# ------------------------------------------------------------

def ingest_file(file_path: str, existing_ids: set) -> List[Dict[str, Any]]:
    """
    The full pipeline for ONE file: load -> chunk -> hash -> dedupe.
    Returns only the NEW records that need to be inserted into the
    vector store (duplicates already filtered out).
    """
    raw_text = load_document(file_path)
    chunks = chunk_text(raw_text)
    extension = os.path.splitext(file_path)[1].lower().lstrip(".")

    return process_and_deduplicate(
        chunks=chunks,
        source_path=file_path,
        file_type=extension,
        existing_ids=existing_ids,
    )
