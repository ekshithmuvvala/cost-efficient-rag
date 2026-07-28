# ============================================================
# src/vector_store.py
#
# WHAT THIS FILE DOES:
# 1. Turns text into numeric vectors ("embeddings") using a free,
#    local model (sentence-transformers).
# 2. Wraps LanceDB (our embedded, disk-based vector database) so
#    the rest of the app can just call simple methods like
#    `.insert_records(...)` and `.search(...)` without knowing
#    any LanceDB-specific details.
#
# WHY LANCEDB / AN EMBEDDED STORE?
# LanceDB stores everything as files on your own disk — there's no
# separate always-on server/pod to pay for, unlike a managed cloud
# vector database (e.g. Pinecone), which bills you monthly whether
# or not you're actually querying it. That's the whole "cost-efficient"
# premise of this project.
# ============================================================

from typing import List, Dict, Any, Optional

import lancedb
from sentence_transformers import SentenceTransformer

from src.config import settings


class EmbeddingModel:
    """
    A thin wrapper around sentence-transformers so the rest of the
    codebase doesn't need to know which embedding library we picked.
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or settings.EMBEDDING_MODEL_NAME
        # Downloads the model the first time (then caches it locally).
        self.model = SentenceTransformer(model_name)
        # all-MiniLM-L6-v2 produces 384-number vectors per piece of text.
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        """Turn a single string into a list of numbers (a vector)."""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed many strings at once — much faster than one at a time."""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


class VectorStoreManager:
    """
    Everything related to storing and searching vectors lives here.
    Think of this class as "our database", even though under the hood
    it's really just files on disk that LanceDB manages for us.
    """

    def __init__(self, db_path: str = None, table_name: str = "rag_chunks"):
        db_path = db_path or settings.VECTOR_STORE_PATH
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.embedder = EmbeddingModel()

    # --------------------------------------------------------
    # Table setup
    # --------------------------------------------------------

    def _table_exists(self) -> bool:
        return self.table_name in self.db.table_names()

    def get_existing_ids(self) -> set:
        """
        Return the set of every chunk ID already stored.
        ingestion.py uses this to decide what's new vs. already ingested.
        If the table doesn't exist yet (first run ever), return an empty set.
        """
        if not self._table_exists():
            return set()
        table = self.db.open_table(self.table_name)
        # .to_pandas() pulls the whole table into memory as a DataFrame.
        # For very large tables you'd want a smarter approach, but for
        # this assignment's scale it's simple and fast enough.
        df = table.to_pandas()
        return set(df["id"].tolist()) if len(df) > 0 else set()

    # --------------------------------------------------------
    # Insertion
    # --------------------------------------------------------

    def insert_records(self, records: List[Dict[str, Any]]):
        """
        Take a list of chunk records (from ingestion.py), embed their
        text, and insert them into the table. Creates the table on
        the very first insert if it doesn't exist yet.
        """
        if not records:
            print("No new records to insert (everything was already ingested).")
            return

        texts = [r["text"] for r in records]
        vectors = self.embedder.embed_batch(texts)

        # LanceDB wants a list of dicts, each containing a "vector" field
        # plus whatever other columns/metadata we want to store alongside it.
        rows = []
        for record, vector in zip(records, vectors):
            rows.append({
                "id": record["id"],
                "vector": vector,
                "text": record["text"],
                "source": record["source"],
                "chunk_index": record["chunk_index"],
                "file_type": record["file_type"],
            })

        if self._table_exists():
            table = self.db.open_table(self.table_name)
            table.add(rows)
        else:
            self.db.create_table(self.table_name, data=rows)

        print(f"Inserted {len(rows)} new chunks into '{self.table_name}'.")

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        file_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed the query, then find the top_k most similar chunks.
        Optionally filter to only one file_type (e.g. only "pdf" chunks).

        Returns a list of dicts, each with the chunk's text, source,
        id, and a "_distance" field (lower = more similar).
        """
        if not self._table_exists():
            return []

        query_vector = self.embedder.embed_text(query_text)
        table = self.db.open_table(self.table_name)

        search_query = table.search(query_vector).limit(top_k)

        if file_type_filter:
            # LanceDB accepts a SQL-style WHERE clause string.
            search_query = search_query.where(f"file_type = '{file_type_filter}'")

        results = search_query.to_pandas()
        return results.to_dict(orient="records")
