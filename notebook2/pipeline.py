"""Chunk 5: identical Spider samples, audited retrieval, EX and comparison exports."""
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import time

import pandas as pd
import sqlglot

from notebook2.model import MemoryBudgetExceeded, BASE_MODEL, BASE_REVISION
from notebook2.schemas import database_path
from notebook2.sql_engine import execute_dataframe, score_prediction

ADAPTER_SHA256 = "961d13a4c49a15b92e1a0801f72e86253f4c4f5f7665786e9e758fd6ad1f5a9a"


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def discover_inputs():
    root = Path("/kaggle/input")
    adapters = list(root.rglob("lora_sql_adapter_3epochs/adapter_model.safetensors"))
    priors = list(root.rglob("qlora-spider-benchmark-100/run_provenance.json"))
    databases = list(root.rglob("concert_singer/concert_singer.sqlite"))
    assert len(adapters) == len(priors) == len(databases) == 1, (adapters, priors, databases)
    assert sha256_file(adapters[0]) == ADAPTER_SHA256, "Wrong final adapter"
    return adapters[0].parent, priors[0].parent, databases[0].parent.parent


def load_comparable_inputs(prior_root, db_root):
    prior_root = Path(prior_root)
    provenance = json.loads((prior_root / "run_provenance.json").read_text())
    summary = json.loads((prior_root / "spider_benchmark_100/summary.json").read_text())
    rows = json.loads((prior_root / "validation.json").read_text())
    indices = provenance["sample_indices"]
    previous = [json.loads(line) for line in (prior_root / "spider_benchmark_100/predictions.jsonl").read_text().splitlines()]
    assert len(indices) == len(previous) == 100 and len(set(indices)) == 100
    assert indices == summary["metadata"]["sample_indices"] == [r["index"] for r in previous]
    assert provenance["adapter_sha256"] == ADAPTER_SHA256
    assert summary["metric"] == "strict_pandas_dataframe_equals_single_database"
    assert summary["metadata"]["base_revision"] == BASE_REVISION
    assert summary["gold_errors"] == 0 and summary["metadata"]["arguments"]["schema"] == "none"
    for i, old in zip(indices, previous):
        assert (rows[i]["db_id"], rows[i]["question"], rows[i]["query"]) == (old["db_id"], old["question"], old["gold_sql"])
    for db_id in sorted({rows[i]["db_id"] for i in indices}):
        assert sha256_file(database_path(db_root, db_id)) == provenance["database_sha256"][db_id]
    for i in indices:
        execute_dataframe(database_path(db_root, rows[i]["db_id"]), rows[i]["query"])
    logging.info("Preflight PASS | 100 identical samples, 19 matching DB hashes, final adapter and base revision")
    return rows, indices, previous, summary, provenance


def aggregate(records):
    if not records:
        raise ValueError("Empty evaluation")
    n = len(records)
    return {"samples": n, "metric": "strict_pandas_dataframe_equals_single_database",
        "execution_accuracy_pct": 100 * sum(r["rag"]["execution_match"] for r in records) / n,
        "execution_validity_pct": 100 * sum(r["rag"]["execution_valid"] for r in records) / n,
        "syntax_validity_pct": 100 * sum(r["rag"]["syntax_valid"] for r in records) / n,
        "syntax_error_rate_pct": 100 * sum(not r["rag"]["syntax_valid"] for r in records) / n,
        "gold_errors": sum(r["gold_error"] is not None for r in records),
        "generation_errors": sum(r["rag"]["error_type"] == "generation" for r in records),
        "retrieval_errors": sum(r["rag"]["error_type"] == "retrieval" for r in records)}


def evaluate(rows, indices, db_root, index, generator, output, top_k=3):
    """SQL errors count as failures; resource-budget violations stop the run."""
    records = []
    with (Path(output) / "predictions.jsonl").open("w") as stream:
        for position, i in enumerate(indices, 1):
            row = rows[i]
            path = database_path(db_root, row["db_id"])
            record = {"index": i, "db_id": row["db_id"], "question": row["question"],
                "gold_sql": row["query"], "gold_error": None, "retrieval": [], "raw_output": "", "generation": {}}
            try:
                gold = execute_dataframe(path, row["query"])
            except (ValueError, sqlite3.Error, sqlglot.errors.SqlglotError) as exc:
                gold = None
                record["gold_error"] = str(exc)
            stage = "retrieval"
            try:
                schema, matches = index.retrieve(row["question"], row["db_id"], top_k)
                record["retrieval"] = matches
                stage = "generation"
                answer, metadata = generator(row["question"], row["db_id"], schema)
                record.update(raw_output=answer, generation=metadata)
                record["rag"] = score_prediction(path, answer, gold)
            except MemoryBudgetExceeded:
                raise
            except (ValueError, KeyError, RuntimeError) as exc:
                # Never turn CUDA OOM into 100 misleading ordinary SQL failures.
                if "out of memory" in str(exc).lower() or "NVML" in str(exc):
                    raise
                record["rag"] = {"sql": "", "syntax_valid": False, "execution_valid": False,
                    "execution_match": False, "error_type": stage, "error": str(exc)}
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            logging.info("[%d/%d] db=%s tables=%s | EX=%s | error=%s", position, len(indices), row["db_id"],
                [m["metadata"]["table_name"] for m in record["retrieval"]],
                record["rag"]["execution_match"], record["rag"]["error"])
    return records


def save_results(records, previous, previous_summary, output, metadata):
    output = Path(output)
    summary = {**aggregate(records), "metadata": metadata}
    prior_error = previous_summary["finetuned"]["syntax_error_rate_pct"]
    summary["syntax_error_rate_reduction_vs_3epoch_pct"] = (
        100 * (prior_error - summary["syntax_error_rate_pct"]) / prior_error if prior_error else None)
    summary["ex_gain_vs_3epoch_percentage_points"] = summary["execution_accuracy_pct"] - previous_summary["finetuned"]["execution_accuracy_pct"]
    old = {r["index"]: r for r in previous}
    table = pd.DataFrame([{"Index": r["index"], "Database": r["db_id"], "Question": r["question"],
        "Gold SQL": r["gold_sql"], "Base Model SQL": old[r["index"]]["base"]["sql"],
        "Schema-Free 3-Epoch SQL": old[r["index"]]["finetuned"]["sql"], "RAG SQL": r["rag"]["sql"],
        "Base Execution Match": old[r["index"]]["base"]["execution_match"],
        "3-Epoch Execution Match": old[r["index"]]["finetuned"]["execution_match"],
        "RAG Execution Match": r["rag"]["execution_match"], "RAG Syntax Valid": r["rag"]["syntax_valid"],
        "Retrieved Tables": ", ".join(m["metadata"]["table_name"] for m in r["retrieval"]),
        "RAG Error": r["rag"]["error"]} for r in records])
    comparison = pd.DataFrame([
        {"Model": "Unadapted base / schema-free", "EX (%)": previous_summary["base"]["execution_accuracy_pct"], "Status": "Measured, same 100"},
        {"Model": "Notebook 1 / 60 steps / schema-free", "EX (%)": None, "Status": "Unmeasured; 15% is the 3-epoch result"},
        {"Model": "Notebook 1 / 3 epochs / schema-free", "EX (%)": previous_summary["finetuned"]["execution_accuracy_pct"], "Status": "Measured, same 100"},
        {"Model": "Notebook 2 / 3 epochs / schema RAG", "EX (%)": summary["execution_accuracy_pct"], "Status": "Measured, same 100"}])
    table.to_csv(output / "comparison.csv", index=False)
    comparison.to_csv(output / "performance_summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "comparison.html").write_text('<!doctype html><html><head><meta charset="utf-8">'
        '<title>Spider Schema RAG</title><style>body{font-family:system-ui}table{border-collapse:collapse}'
        'th,td{border:1px solid #bbb;padding:8px;vertical-align:top;max-width:450px;overflow-wrap:anywhere}'
        '</style></head><body><h1>Notebook 2: Schema-aware RAG</h1>'
        '<p>Strict Pandas equality on identical SQLite databases. Not official Spider test-suite accuracy. '
        'Column labels, dtypes and row/column order must match exactly.</p>' + comparison.to_html(index=False, escape=True)
        + table.to_html(index=False, escape=True) + '</body></html>')
    logging.info("PERFORMANCE\n%s", comparison.to_string(index=False))
    return summary


def verify_pipeline():
    """Small deterministic remote fixture: error accounting and metric denominator."""
    import tempfile
    class FixtureIndex:
        def retrieve(self, question, db_id, top_k):
            return "CREATE TABLE item(id INTEGER);", [{"metadata": {"db_id": db_id, "table_name": "item"}}]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "fixture").mkdir()
        with sqlite3.connect(root / "fixture/fixture.sqlite") as conn:
            conn.executescript("CREATE TABLE item(id INTEGER); INSERT INTO item VALUES(1);")
        rows = [{"db_id": "fixture", "question": q, "query": "SELECT id FROM item"} for q in ("good", "bad")]
        generator = lambda question, db_id, schema: ("SELECT id FROM item" if question == "good" else "SELECT * FROM absent", {})
        records = evaluate(rows, [0, 1], root, FixtureIndex(), generator, root)
        result = aggregate(records)
        assert result["samples"] == 2 and result["execution_accuracy_pct"] == 50
        assert result["syntax_validity_pct"] == 100 and result["gold_errors"] == 0
        assert len((root / "predictions.jsonl").read_text().splitlines()) == 2
    logging.info("CHUNK 5 PASS | complete denominator, failed SQL continuation, JSONL output")
