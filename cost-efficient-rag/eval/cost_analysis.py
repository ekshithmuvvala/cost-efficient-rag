# ============================================================
# eval/cost_analysis.py
#
# WHAT THIS FILE DOES:
# 1. Estimates monthly $ cost of our embedded vector store
#    (LanceDB, disk-based) vs. a typical managed vector DB,
#    at three scales: 100K, 1M, and 10M vectors.
# 2. Reads results/query_log.jsonl (written by src/logger.py
#    every time /query is called) and computes p50/p95 latency.
#
# RUN IT WITH:
#   python -m eval.cost_analysis
# ============================================================

import json
import os

import numpy as np


# ------------------------------------------------------------
# PART 1: Cost model
# ------------------------------------------------------------
#
# ASSUMPTIONS (state these explicitly in your README too):
#  - Vector dimension = 384 (all-MiniLM-L6-v2), stored as float32 (4 bytes/number)
#  - ~500 bytes of metadata stored per vector (source, chunk_index, etc.)
#  - Disk storage cost (S3/EBS-style) ≈ $0.08 per GB per month
#  - Managed vector DB (e.g. Pinecone) bills for an always-on pod REGARDLESS
#    of query volume, with the pod size increasing at each scale tier below

def estimate_monthly_cost(vector_count: int, dim: int = 384) -> dict:
    # Bytes used just for the raw vector numbers themselves.
    raw_vector_bytes = vector_count * dim * 4
    # Bytes used for the metadata fields stored alongside each vector.
    metadata_bytes = vector_count * 500
    total_gb = (raw_vector_bytes + metadata_bytes) / (1024 ** 3)

    # Our embedded store only costs us disk space — no server rental.
    embedded_storage_cost = total_gb * 0.08

    # A managed DB's pod cost jumps in tiers as your data grows,
    # because you need more RAM/CPU to keep the index responsive.
    if vector_count <= 100_000:
        managed_cost = 70.00     # smallest always-on pod
    elif vector_count <= 1_000_000:
        managed_cost = 280.00    # bigger pod / more RAM
    else:
        managed_cost = 1200.00   # multi-node cluster

    savings_percentage = round((1 - (embedded_storage_cost / managed_cost)) * 100, 2)

    return {
        "vector_count": vector_count,
        "storage_size_gb": round(total_gb, 2),
        "embedded_db_cost_usd": round(embedded_storage_cost, 2),
        "managed_db_cost_usd": managed_cost,
        "savings_percentage": savings_percentage,
    }


def build_cost_table() -> list:
    scales = [100_000, 1_000_000, 10_000_000]
    return [estimate_monthly_cost(n) for n in scales]


def write_cost_markdown_table(cost_rows: list, path: str = "results/cost_benchmark_table.md"):
    lines = [
        "# Cost Benchmark: Embedded (LanceDB) vs. Managed Vector DB\n",
        "| Vector Count | Storage (GB) | Embedded DB Cost ($/mo) | Managed DB Cost ($/mo) | Savings |",
        "|---|---|---|---|---|",
    ]
    for row in cost_rows:
        lines.append(
            f"| {row['vector_count']:,} | {row['storage_size_gb']} | "
            f"${row['embedded_db_cost_usd']} | ${row['managed_db_cost_usd']} | "
            f"{row['savings_percentage']}% |"
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote cost table to {path}")


# ------------------------------------------------------------
# PART 2: Latency analysis from the query log
# ------------------------------------------------------------

def compute_latency_percentiles(log_path: str = "results/query_log.jsonl") -> dict:
    if not os.path.exists(log_path):
        print(f"No query log found at {log_path} yet — run some /query requests first.")
        return {}

    retrieval_times, generation_times, total_times = [], [], []
    with open(log_path, "r") as f:
        for line in f:
            record = json.loads(line)
            retrieval_times.append(record["retrieval_time_ms"])
            generation_times.append(record["generation_time_ms"])
            total_times.append(record["total_time_ms"])

    def p50_p95(values):
        if not values:
            return {"p50": None, "p95": None}
        return {
            "p50": round(float(np.percentile(values, 50)), 2),
            "p95": round(float(np.percentile(values, 95)), 2),
        }

    return {
        "num_queries_logged": len(total_times),
        "retrieval_ms": p50_p95(retrieval_times),
        "generation_ms": p50_p95(generation_times),
        "total_ms": p50_p95(total_times),
    }


if __name__ == "__main__":
    cost_rows = build_cost_table()
    print(json.dumps(cost_rows, indent=2))
    write_cost_markdown_table(cost_rows)

    latency_summary = compute_latency_percentiles()
    print(json.dumps(latency_summary, indent=2))

    with open("results/latency_summary.json", "w") as f:
        json.dump(latency_summary, f, indent=2)
