"""Chunk 2: per-table SQLite DDL, CPU BGE vectors, strict database filtering."""
from contextlib import closing
import hashlib
import logging
from pathlib import Path
import re
import sqlite3

from notebook2.environment import EMBEDDING_MODEL, open_collection


def database_path(root, db_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", db_id):
        raise ValueError("Invalid Spider database ID")
    root = Path(root).resolve()
    path = (root / db_id / f"{db_id}.sqlite").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(path)
    return path


def extract_schemas(path, db_id):
    """Original CREATE TABLE retains column types, composite PKs and FK clauses.

    PRAGMA metadata supplements DDL and explicitly records key relationships.
    No data values, validation questions, or gold SQL enter the vector index.
    """
    records = []
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        tables = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' "
                              "AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name").fetchall()
        for name, ddl in tables:
            quoted = '"' + name.replace('"', '""') + '"'
            columns = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
            keys = conn.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
            primary = [r[1] for r in sorted(columns, key=lambda r: r[5]) if r[5]]
            lines = [f"-- Database: {db_id}; table: {name}", ddl.rstrip("; ") + ";"]
            if primary:
                lines.append("-- Primary key: " + ", ".join(primary))
            for fk in keys:
                lines.append(f"-- Foreign key {fk[0]}.{fk[1]}: {name}.{fk[3]} -> {fk[2]}.{fk[4] or '(primary key)'}")
            document = "\n".join(lines)
            records.append({"id": hashlib.sha256(f"{db_id}\0{name}".encode()).hexdigest(),
                "document": document, "metadata": {"db_id": db_id, "table_name": name,
                    "schema_sha256": hashlib.sha256(document.encode()).hexdigest()}})
    if not records:
        raise ValueError(f"No tables in {db_id}")
    return records


class SchemaIndex:
    def __init__(self, directory):
        from fastembed import TextEmbedding
        self.client, self.collection = open_collection(directory)
        # ONNX CPU embeddings leave GPU memory exclusively for the language model.
        self.encoder = TextEmbedding(model_name=EMBEDDING_MODEL,
            cache_dir="/tmp/bge-cache", providers=["CPUExecutionProvider"], threads=2)

    def ingest(self, path, db_id):
        records = extract_schemas(path, db_id)
        vectors = [v.tolist() for v in self.encoder.embed([r["document"] for r in records])]
        assert all(len(v) == 384 for v in vectors)
        self.collection.upsert(ids=[r["id"] for r in records], embeddings=vectors,
            documents=[r["document"] for r in records], metadatas=[r["metadata"] for r in records])
        stale = set(self.collection.get(where={"db_id": db_id})["ids"]) - {r["id"] for r in records}
        if stale:
            self.collection.delete(ids=sorted(stale))
        logging.info("Indexed %s: %d tables", db_id, len(records))
        return records

    def retrieve(self, user_query, db_id, top_k=3):
        if not user_query.strip() or top_k < 1:
            raise ValueError("Non-empty question and positive top_k required")
        available = self.collection.get(where={"db_id": db_id})["ids"]
        if not available:
            raise KeyError(f"Database {db_id!r} has not been indexed")
        vector = next(iter(self.encoder.query_embed(user_query))).tolist()
        result = self.collection.query(query_embeddings=[vector],
            where={"db_id": db_id}, n_results=min(top_k, len(available)),
            include=["documents", "metadatas", "distances"])
        matches = [{"document": doc, "metadata": meta, "distance": distance}
            for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]
        assert matches and all(m["metadata"]["db_id"] == db_id for m in matches), "Cross-database retrieval"
        return "\n\n".join(m["document"] for m in matches), matches

    def retrieve_relevant_schemas(self, user_query: str, db_id: str, top_k: int = 3) -> str:
        return self.retrieve(user_query, db_id, top_k)[0]


_active_index = None


def initialize_schema_index(directory):
    global _active_index
    _active_index = SchemaIndex(directory)
    return _active_index


def retrieve_relevant_schemas(user_query: str, db_id: str, top_k: int = 3) -> str:
    if _active_index is None:
        raise RuntimeError("Call initialize_schema_index first")
    return _active_index.retrieve_relevant_schemas(user_query, db_id, top_k)


def verify_schemas():
    """Real BGE + Chroma assertion test on remote synthetic databases."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        index = SchemaIndex(Path(directory) / "chroma")
        for db_id in ("alpha", "beta"):
            path = Path(directory) / f"{db_id}.sqlite"
            with sqlite3.connect(path) as conn:
                conn.executescript("CREATE TABLE parent(id INTEGER PRIMARY KEY, name TEXT);"
                    "CREATE TABLE child(id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parent(id));")
            records = index.ingest(path, db_id)
            assert len(records) == 2 and "REFERENCES" in records[0]["document"]
            index.ingest(path, db_id)  # Upserts must be idempotent.
        assert index.collection.count() == 4
        for db_id in ("alpha", "beta"):
            text, matches = index.retrieve("List child and parent names", db_id, 20)
            assert len(matches) == 2 and all(m["metadata"]["db_id"] == db_id for m in matches)
            assert f"Database: {db_id}" in text
        try:
            index.retrieve("parent names", "missing")
            raise AssertionError("Unknown database must not fall back to all databases")
        except KeyError:
            pass
    logging.info("CHUNK 2 PASS | real BGE/Chroma isolation, PK/FK extraction, idempotence")
