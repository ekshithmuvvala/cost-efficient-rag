# ============================================================
# src/logger.py
#
# WHAT THIS FILE DOES:
# 1. Sets up one shared "logger" object other files can import
#    to print nicely formatted log lines.
# 2. Provides log_query_metrics(), which appends one line of
#    JSON per query to results/query_log.jsonl — this is the
#    raw data we'll later use to compute p50/p95 latency.
# ============================================================

import json
import os
import time
from loguru import logger  # loguru is a drop-in replacement for Python's logging, easier to use

# Where we'll store one JSON line per query (JSONL = "JSON Lines" format,
# one valid JSON object per line — easy to append to and easy to read back).
QUERY_LOG_PATH = "results/query_log.jsonl"


def log_query_metrics(
    query: str,
    retrieval_time_ms: float,
    generation_time_ms: float,
    chunks_retrieved: int,
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost_usd: float,
):
    """
    Append one record describing everything about a single /query request.

    We call this function once at the end of every query in api.py.
    Later, eval scripts read this file to compute latency percentiles.
    """
    record = {
        "timestamp": time.time(),
        "query": query,
        "retrieval_time_ms": retrieval_time_ms,
        "generation_time_ms": generation_time_ms,
        "total_time_ms": retrieval_time_ms + generation_time_ms,
        "chunks_retrieved": chunks_retrieved,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": estimated_cost_usd,
    }

    # Make sure the results/ folder exists before we try to write into it.
    os.makedirs(os.path.dirname(QUERY_LOG_PATH), exist_ok=True)

    # "a" = append mode: adds a new line without erasing what's already there.
    with open(QUERY_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(f"Logged query metrics: {record}")
