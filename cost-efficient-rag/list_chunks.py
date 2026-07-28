# ============================================================
# list_chunks.py
#
# WHY THIS EXISTS:
# eval_dataset.json needs real chunk IDs (the sha256 hashes) in its
# "relevant_chunk_ids" field. Those IDs don't exist until AFTER you've
# run ingestion. This script prints every chunk currently stored,
# so you can read the text, find which chunk(s) answer each of your
# eval questions, and copy their IDs into eval_dataset.json by hand.
#
# RUN IT WITH (after you've called POST /ingest at least once):
#   python list_chunks.py
# ============================================================

from src.vector_store import VectorStoreManager

vector_store = VectorStoreManager()

if not vector_store._table_exists():
    print("No chunks found yet. Run ingestion first (POST /ingest).")
else:
    table = vector_store.db.open_table(vector_store.table_name)
    df = table.to_pandas()

    for _, row in df.iterrows():
        print("=" * 60)
        print(f"id:     {row['id']}")
        print(f"source: {row['source']} (chunk #{row['chunk_index']})")
        print(f"text:   {row['text'][:200]}...")  # first 200 chars only
