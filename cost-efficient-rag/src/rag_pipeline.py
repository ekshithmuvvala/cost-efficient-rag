# ============================================================
# src/rag_pipeline.py
#
# WHAT THIS FILE DOES (the "R" and "G" of RAG):
# 1. RETRIEVES the most relevant chunks for a user's question.
# 2. Builds a prompt that forces the LLM to answer ONLY using
#    those chunks, and to cite which chunk each fact came from.
# 3. If nothing relevant was found, returns a fixed fallback
#    message instead of letting the LLM guess/hallucinate.
# ============================================================

import time
from typing import List, Dict, Any, Tuple

from openai import OpenAI

from src.config import settings
from src.vector_store import VectorStoreManager

# The exact refusal text we return when we don't have enough context.
# Keeping this as a constant means eval/evaluate_answer.py can check
# for it exactly, to confirm the fallback logic actually triggered.
FALLBACK_MESSAGE = (
    "I do not have sufficient information in the provided context "
    "to answer this question."
)

SYSTEM_PROMPT_TEMPLATE = """You are a precise QA assistant. Answer the user question using ONLY the context provided below.
For every factual claim, cite the source using the chunk ID in brackets [Doc: <source>, Chunk: <id>].
If the provided context does not contain enough information to answer the question, output exactly this text:
"{fallback_message}"

Context:
{context_block}
"""


def build_prompt(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Turn a list of retrieved chunks into the system prompt text
    the LLM will see, with each chunk clearly labeled by its
    source file and chunk id (so the model can cite them).
    """
    formatted_chunks = []
    for chunk in retrieved_chunks:
        formatted_chunks.append(
            f"[Doc: {chunk['source']}, Chunk: {chunk['id']}]\n{chunk['text']}"
        )
    context_block = "\n\n".join(formatted_chunks)

    return SYSTEM_PROMPT_TEMPLATE.format(
        fallback_message=FALLBACK_MESSAGE,
        context_block=context_block,
    )


class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        file_type_filter: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        The full pipeline for a single question. Returns:
          - the answer text (string)
          - a dict of metrics (timing, token counts, chunks used)
            which api.py will log via src/logger.py
        """
        # ---- 1. RETRIEVAL ----
        retrieval_start = time.time()
        retrieved_chunks = self.vector_store.search(
            query_text=query,
            top_k=top_k,
            file_type_filter=file_type_filter,
        )
        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        # ---- 2. FALLBACK CHECK ----
        # LanceDB's "_distance" is a distance score: LOWER means MORE similar.
        # We convert it to a similarity-style check: if even the best match
        # is too far away (distance too high), we don't trust it.
        best_distance = retrieved_chunks[0]["_distance"] if retrieved_chunks else None
        # A normalized-embedding cosine distance of 0 = identical, 2 = opposite.
        # We treat "too dissimilar" as distance above (1 - threshold).
        distance_cutoff = 1 - settings.SIMILARITY_THRESHOLD

        if not retrieved_chunks or best_distance > distance_cutoff:
            metrics = {
                "retrieval_time_ms": retrieval_time_ms,
                "generation_time_ms": 0,
                "chunks_retrieved": len(retrieved_chunks),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "estimated_cost_usd": 0.0,
            }
            return FALLBACK_MESSAGE, metrics

        # ---- 3. BUILD PROMPT ----
        system_prompt = build_prompt(retrieved_chunks)

        # ---- 4. GENERATE ANSWER ----
        generation_start = time.time()
        response = self.llm_client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,  # 0 = deterministic, favors sticking to facts over creativity
        )
        generation_time_ms = (time.time() - generation_start) * 1000

        answer_text = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        # ---- 5. COST ESTIMATE ----
        # These per-token prices are an EXAMPLE for gpt-4o-mini — check
        # OpenAI's current pricing page and update these constants.
        input_price_per_1k = 0.00015
        output_price_per_1k = 0.0006
        estimated_cost_usd = (
            (prompt_tokens / 1000) * input_price_per_1k
            + (completion_tokens / 1000) * output_price_per_1k
        )

        metrics = {
            "retrieval_time_ms": retrieval_time_ms,
            "generation_time_ms": generation_time_ms,
            "chunks_retrieved": len(retrieved_chunks),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            # Keep the raw chunks too, so the API can return citations to the user.
            "retrieved_chunks": retrieved_chunks,
        }

        return answer_text, metrics
