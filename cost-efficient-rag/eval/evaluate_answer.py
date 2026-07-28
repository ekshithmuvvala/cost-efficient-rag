# ============================================================
# eval/evaluate_answer.py
#
# WHAT THIS FILE DOES:
# For each question in the eval set, runs it through the FULL
# RAG pipeline (retrieval + generation), then asks an LLM to
# JUDGE the generated answer on two axes:
#   - Faithfulness / Groundedness: is every claim actually
#     supported by the retrieved context? (catches hallucination)
#   - Answer Relevance: does the answer actually address the
#     question that was asked?
#
# Each score is 0.0-1.0. We ask the judge to also give a short
# reason, so a human can spot-check the scoring.
#
# RUN IT WITH:
#   python -m eval.evaluate_answer
# ============================================================

import json

from openai import OpenAI
import numpy as np

from src.config import settings
from src.rag_pipeline import RAGPipeline, FALLBACK_MESSAGE

JUDGE_PROMPT_TEMPLATE = """You are an evaluation judge. You will be given a QUESTION, the CONTEXT
that was retrieved to answer it, and the ANSWER that was generated.

Score the ANSWER on two dimensions, each from 0.0 to 1.0:

1. faithfulness: Is every factual claim in the ANSWER directly supported
   by the CONTEXT? 1.0 = fully supported, 0.0 = mostly unsupported/made up.
2. relevance: Does the ANSWER actually address the QUESTION that was asked?
   1.0 = fully addresses it, 0.0 = off-topic or non-responsive.

Respond with ONLY valid JSON in this exact shape, nothing else:
{{"faithfulness": <float>, "relevance": <float>, "reasoning": "<one short sentence>"}}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""


def judge_answer(llm_client: OpenAI, question: str, context: str, answer: str) -> dict:
    """Call the LLM once as a judge and parse its JSON verdict."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, context=context, answer=answer)

    response = llm_client.chat.completions.create(
        model=settings.LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw_text = response.choices[0].message.content.strip()
    # Judges sometimes wrap JSON in ```json fences — strip those if present.
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # If the judge didn't return clean JSON, record it as a failure
        # rather than crashing the whole evaluation run.
        return {"faithfulness": 0.0, "relevance": 0.0, "reasoning": "JUDGE_PARSE_ERROR"}


def run_answer_evaluation():
    with open("data/eval_dataset.json", "r") as f:
        eval_questions = json.load(f)

    rag_pipeline = RAGPipeline()
    llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    faithfulness_scores, relevance_scores = [], []
    per_question_results = []

    for item in eval_questions:
        question = item["question"]

        answer_text, metrics = rag_pipeline.answer_query(question, top_k=5)

        # If the pipeline correctly triggered the fallback (e.g. for a
        # deliberately unanswerable question), we don't send it to the
        # judge — we just record that the fallback fired correctly.
        if answer_text.strip() == FALLBACK_MESSAGE:
            per_question_results.append({
                "question": question,
                "answer": answer_text,
                "faithfulness": None,
                "relevance": None,
                "note": "fallback triggered",
            })
            continue

        context_text = "\n\n".join(
            chunk["text"] for chunk in metrics.get("retrieved_chunks", [])
        )

        verdict = judge_answer(llm_client, question, context_text, answer_text)

        faithfulness_scores.append(verdict.get("faithfulness", 0.0))
        relevance_scores.append(verdict.get("relevance", 0.0))

        per_question_results.append({
            "question": question,
            "answer": answer_text,
            "faithfulness": verdict.get("faithfulness"),
            "relevance": verdict.get("relevance"),
            "reasoning": verdict.get("reasoning"),
        })

    summary = {
        "num_questions": len(eval_questions),
        "avg_faithfulness": round(float(np.mean(faithfulness_scores)), 4) if faithfulness_scores else None,
        "avg_relevance": round(float(np.mean(relevance_scores)), 4) if relevance_scores else None,
        "per_question": per_question_results,
    }

    print(json.dumps(summary, indent=2))

    with open("results/answer_eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_answer_evaluation()
