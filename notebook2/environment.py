"""Chunk 1: persistent Chroma configuration and remote-only verification."""
import importlib.metadata
import json
import logging
import os
from pathlib import Path

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "spider_ddl_bge_small_v1"
VRAM_LIMIT_BYTES = 10_000_000_000  # Decimal GB, stricter than 10 GiB.


def require_kaggle():
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this notebook on Kaggle with a T4 GPU; local execution is disabled.")


def configure_logging(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "evaluation.log")])


def open_collection(directory):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(directory),
        settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": EMBEDDING_MODEL, "schema_format": "sqlite-ddl-v1"})
    if collection.metadata.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("Existing collection uses a different embedding model")
    return client, collection


def verify_environment():
    """Remote assertion test, before any schema or model work."""
    require_kaggle()
    import sqlite3
    import tempfile
    import torch
    import sqlglot
    assert torch.cuda.is_available(), "Enable the Kaggle GPU accelerator"
    assert "T4" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
    assert sqlglot.parse_one("SELECT 1", read="sqlite") is not None
    with sqlite3.connect(":memory:") as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)
    with tempfile.TemporaryDirectory() as directory:
        client, collection = open_collection(directory)
        collection.add(ids=["probe"], embeddings=[[1.0, 0.0]], documents=["SELECT 1"],
                       metadatas=[{"db_id": "probe"}])
        _, reopened = open_collection(directory)
        assert reopened.get(where={"db_id": "probe"})["ids"] == ["probe"]
    versions = {name: importlib.metadata.version(name) for name in
        ("torch", "chromadb", "fastembed", "transformers", "peft", "bitsandbytes", "pandas", "sqlglot")}
    logging.info("CHUNK 1 PASS | GPU=%s | versions=%s", torch.cuda.get_device_name(0), versions)
    return versions
