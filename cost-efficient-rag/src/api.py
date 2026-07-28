# ============================================================
# src/api.py
#
# WHAT THIS FILE DOES:
# Exposes the whole system as a small HTTP web service with two endpoints:
#   POST /ingest  -> reads all files in data/raw_documents/ and adds
#                    any new chunks to the vector store
#   POST /query   -> answers a question using the RAG pipeline
#
# HOW TO RUN THIS FILE:
#   uvicorn src.api:app --reload
# Then open http://127.0.0.1:8000/docs in a browser — FastAPI gives
# you a free interactive page to test both endpoints by hand.
# ============================================================

import os
import glob

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.ingestion import ingest_file
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.logger import log_query_metrics

app = FastAPI(title="Cost-Efficient RAG Application")

# We create these ONCE when the server starts (not on every request) —
# loading the embedding model and connecting to the DB is slow, so we
# don't want to repeat it for every single API call.
vector_store = VectorStoreManager()
rag_pipeline = RAGPipeline()


# ------------------------------------------------------------
# Request/response "shapes" — Pydantic models describing exactly
# what JSON the endpoints expect and return. FastAPI uses these
# to validate incoming requests automatically.
# ------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    metadata_filter: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list
    chunks_retrieved: int
    retrieval_time_ms: float
    generation_time_ms: float


class IngestResponse(BaseModel):
    files_processed: int
    new_chunks_added: int


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
def ingest_documents():
    """
    Scans data/raw_documents/ for .pdf, .html, and .md files,
    and ingests any new chunks it finds. Safe to call repeatedly —
    already-ingested chunks are automatically skipped.
    """
    supported_patterns = ["*.pdf", "*.html", "*.htm", "*.md", "*.markdown", "*.txt"]
    file_paths = []
    for pattern in supported_patterns:
        file_paths.extend(glob.glob(os.path.join("data/raw_documents", pattern)))

    existing_ids = vector_store.get_existing_ids()
    total_new_chunks = 0

    for file_path in file_paths:
        new_records = ingest_file(file_path, existing_ids)
        vector_store.insert_records(new_records)
        # Update our in-memory set too, so if the SAME file appears
        # twice in this loop we still catch duplicates correctly.
        existing_ids.update(r["id"] for r in new_records)
        total_new_chunks += len(new_records)

    return IngestResponse(
        files_processed=len(file_paths),
        new_chunks_added=total_new_chunks,
    )


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Answers a question using retrieval-augmented generation.
    Logs latency/token/cost metrics for every call.
    """
    file_type_filter = None
    if request.metadata_filter and "file_type" in request.metadata_filter:
        file_type_filter = request.metadata_filter["file_type"]

    answer_text, metrics = rag_pipeline.answer_query(
        query=request.query,
        top_k=request.top_k,
        file_type_filter=file_type_filter,
    )

    # Build a simple citations list from the chunks that were used.
    citations = []
    for chunk in metrics.get("retrieved_chunks", []):
        citations.append({
            "source": chunk["source"],
            "chunk_id": chunk["id"],
        })

    log_query_metrics(
        query=request.query,
        retrieval_time_ms=metrics["retrieval_time_ms"],
        generation_time_ms=metrics["generation_time_ms"],
        chunks_retrieved=metrics["chunks_retrieved"],
        prompt_tokens=metrics["prompt_tokens"],
        completion_tokens=metrics["completion_tokens"],
        estimated_cost_usd=metrics["estimated_cost_usd"],
    )

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        chunks_retrieved=metrics["chunks_retrieved"],
        retrieval_time_ms=metrics["retrieval_time_ms"],
        generation_time_ms=metrics["generation_time_ms"],
    )


@app.get("/")
def health_check():
    """A simple endpoint to confirm the server is running."""
    return {"status": "ok", "message": "RAG API is running"}
