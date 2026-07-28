# Cost-Efficient RAG Application

A Retrieval-Augmented Generation (RAG) question-answering service built on
**LanceDB**, an embedded, disk-based vector store — chosen instead of a
managed cloud vector database (e.g. Pinecone) because it has **zero idle
compute cost**: it runs as files on your own disk, with no always-on pod
billed monthly whether or not you're querying it. You only pay for the
disk space the vectors actually occupy.

---

## 0. Zero-Knowledge Setup (do this once)

You need Python 3.10+ installed. Check with:

```bash
python3 --version
```

Then, from inside this project folder:

```bash
# 1. Create an isolated Python environment just for this project
python3 -m venv venv

# 2. Activate it (you'll need to run this every time you open a new terminal)
source venv/bin/activate          # Windows (PowerShell): venv\Scripts\activate

# 3. Install every package the project needs
pip install -r requirements.txt

# 4. Create your real secrets file from the template
cp .env.example .env
```

Now open `.env` in any text editor and paste your OpenAI API key in place
of `your_openai_api_key_here`. (Get a key at
https://platform.openai.com/api-keys — you'll need billing enabled on
your OpenAI account.)

---

## 1. Run the Server

```bash
uvicorn src.api:app --reload
```

You should see something like `Uvicorn running on http://127.0.0.1:8000`.
Leave this terminal window open — it's your running server.

Open **http://127.0.0.1:8000/docs** in your browser. This is an
auto-generated page where you can test both endpoints by clicking
"Try it out" — no coding needed.

---

## 2. Ingest the Sample Document

A sample file is already in `data/raw_documents/sample.md` so you can test
immediately. In the `/docs` page, click on `POST /ingest`, then
"Try it out", then "Execute". You should get back something like:

```json
{ "files_processed": 1, "new_chunks_added": 3 }
```

**Try running it again** — you'll get `"new_chunks_added": 0`, proving
re-ingestion doesn't create duplicates. That's the idempotence requirement.

To ingest your own files: drop `.pdf`, `.html`, or `.md` files into
`data/raw_documents/`, then call `POST /ingest` again.

---

## 3. Ask a Question

Click `POST /query`, "Try it out", and enter a body like:

```json
{
  "query": "How many vacation days do full-time employees get?",
  "top_k": 3
}
```

Click "Execute". You'll get back an answer, the chunk IDs it cited, and
timing info.

Try an off-topic question too (e.g. "What is the capital of France?") —
you should get the fixed fallback message instead of a hallucinated
answer, because there's nothing relevant in the vector store.

---

## 4. Build Your Real Evaluation Set

1. Ingest the documents you actually care about.
2. Run `python list_chunks.py` to print every stored chunk with its ID
   and text — use this to find which chunk(s) answer each question you
   want to test.
3. Edit `data/eval_dataset.json`: replace the placeholder entries with
   15-30 real questions, their gold answers, and the real chunk IDs from
   step 2. Include a few questions your documents genuinely can't answer,
   to test the fallback path.

---

## 5. Run the Evaluation Harness

With the server **not necessarily running** (these scripts talk to the
vector store and OpenAI directly):

```bash
# Retrieval quality: Recall@k, Hit Rate, MRR, nDCG@k
python -m eval.evaluate_retrieval

# Answer quality: faithfulness + relevance, via LLM-as-judge
python -m eval.evaluate_answer

# Cost model (100K/1M/10M vectors) + latency percentiles from
# whatever queries you've already logged by using /query
python -m eval.cost_analysis
```

Each writes its results into `results/` as JSON (and, for cost, also as
a Markdown table you can paste straight into a report).

---

## 6. Project Structure

```
cost-efficient-rag/
├── data/
│   ├── raw_documents/          # Drop PDF, HTML, MD files here
│   └── eval_dataset.json       # Your 15-30 gold Q&A set
├── src/
│   ├── config.py                # Loads settings from .env
│   ├── ingestion.py             # File loading, chunking, idempotent hashing
│   ├── vector_store.py          # Embeddings + LanceDB interface
│   ├── rag_pipeline.py          # Retrieval, prompting, generation, fallback
│   ├── logger.py                # Per-query metrics logging
│   └── api.py                   # FastAPI app (/ingest, /query)
├── eval/
│   ├── evaluate_retrieval.py    # Recall@k, Hit Rate, MRR, nDCG@k
│   ├── evaluate_answer.py       # Faithfulness + relevance (LLM-as-judge)
│   └── cost_analysis.py         # Cost model + latency percentiles
├── results/                     # All eval output lands here
├── list_chunks.py               # Helper to look up chunk IDs
├── .env.example
├── requirements.txt
└── README.md
```

---

## 7. Architecture

```
       PDF / HTML / MD files
                │
                ▼
        [ ingestion.py ]  ── chunk + SHA-256 hash each chunk
                │
                ▼
      [ vector_store.py ]  ── embed (sentence-transformers)
                │             insert into LanceDB (skip existing IDs)
                ▼
        LanceDB (disk files, no server)
                ▲
                │ top-k similarity search
                │
        [ rag_pipeline.py ]  ── build grounded prompt with citations
                │               OR return fallback if nothing relevant
                ▼
             OpenAI LLM
                │
                ▼
        [ api.py: POST /query ]  ── answer + citations + metrics
                │
                ▼
        [ logger.py ]  ── results/query_log.jsonl (feeds latency eval)
```

---

## 8. Cost & Assumptions

See `eval/cost_analysis.py` for the exact formulas. In short:

- Vector dimension = 384 (`all-MiniLM-L6-v2`), stored as float32 (4 bytes each)
- ~500 bytes of metadata per vector
- Disk storage ≈ $0.08/GB/month (S3/EBS-style pricing)
- Managed vector DB assumed to bill an always-on pod: ~$70/mo up to 100K
  vectors, ~$280/mo up to 1M, ~$1,200/mo beyond that — regardless of how
  many queries you actually run
- Query volume assumption used for the write-up: 50,000 queries/month

Run `python -m eval.cost_analysis` to regenerate `results/cost_benchmark_table.md`
with these numbers computed at 100K / 1M / 10M vectors.

---

## 9. Discussion

**At what scale would you switch back to a managed DB?**
*(Fill this in with your own numbers once you've run the cost analysis —
e.g., "around N million vectors, the operational burden of managing our
own disk-backed store and scaling read throughput starts to outweigh the
$X/month we're saving versus a managed pod.")*

**Was retrieval or generation the weaker link?**
*(Fill this in using your `results/retrieval_eval_results.json` and
`results/answer_eval_results.json` — e.g., "Recall@5 averaged 0.91, but
average faithfulness was only 0.74, suggesting the LLM sometimes drifted
from the retrieved context rather than retrieval failing to find it.")*
