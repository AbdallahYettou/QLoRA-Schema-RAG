"""Remote-only HTTP/API smoke verification of the Gradio app."""
import json
from pathlib import Path
import tempfile

import pandas as pd

from notebook2.environment import require_kaggle
from notebook2.sql_engine import execute_dataframe
from gradio_app.app import SQLApp


def verify_app(service, demo, output):
    require_kaggle()
    from gradio_client import Client
    import requests
    checks = []
    for db, question, top_k in (("../invalid", "test", 3), ("concert_singer", "", 3), ("concert_singer", "test", 99)):
        answer = service.ask(db, question, top_k)
        assert answer[0].startswith("Request failed") and answer[3].empty
    checks.append("invalid database, empty question, invalid retrieval count")

    # Verify unsafe/invalid SQL clears prior results using the real SQLite harness.
    class FixtureIndex:
        def ingest(self, path, db_id):
            pass
        def retrieve(self, question, db_id, top_k):
            return "CREATE TABLE singer(id INTEGER);", [{"metadata": {"db_id": db_id, "table_name": "singer"}}]
    for sql in ("DROP TABLE singer", "SELECT * FROM nonexistent", "SELECT 1; DELETE FROM singer"):
        fixture = SQLApp(lambda *args: (sql, {}), FixtureIndex(), service.db_root)
        answer = fixture.ask("concert_singer", "test", 3)
        assert answer[0].startswith("Request failed") and answer[3].empty
    checks.append("mutations, multiple statements and missing-table failures")

    # Exercise the actual Gradio server and serialized event queue on the T4.
    _, local_url, _ = demo.launch(server_name="127.0.0.1", server_port=7860,
        share=False, prevent_thread_lock=True, quiet=True)
    try:
        response = requests.get(local_url, timeout=30)
        assert response.status_code == 200 and "gradio" in response.text.lower()
        client = Client(local_url, verbose=False)
        cases = [
            ("Simple aggregate", "How many singers are there?", "SELECT count(*) FROM singer"),
            ("GROUP BY", "What is the year that had the most concerts?", "SELECT year FROM concert GROUP BY year ORDER BY count(*) DESC LIMIT 1"),
            ("JOIN + GROUP BY", "Show the name and theme for all concerts and the number of singers in each concert.",
             "SELECT T2.concert_name, T2.theme, count(*) FROM singer_in_concert AS T1 JOIN concert AS T2 ON T1.concert_id = T2.concert_id GROUP BY T2.concert_id"),
        ]
        results = []
        for tier, question, gold_sql in cases:
            status, sql, schema, frame_payload, details = client.predict("concert_singer", question, 3, api_name="/ask_sql")
            assert status.startswith("Executed successfully"), status
            assert "Database: concert_singer" in schema
            assert details["database"] == "concert_singer" and details["total_rows"] >= 0
            # Compare values separately from labels to make this a practical app smoke test.
            # This is not a replacement for the existing strict benchmark metric.
            gold = execute_dataframe(service.db_root / "concert_singer/concert_singer.sqlite", gold_sql)
            predicted = execute_dataframe(service.db_root / "concert_singer/concert_singer.sqlite", sql)
            value_match = predicted.shape == gold.shape and predicted.to_numpy().tolist() == gold.to_numpy().tolist()
            results.append({"tier": tier, "question": question, "sql": sql,
                "reference_sql": gold_sql, "value_match": value_match,
                "strict_dataframe_match": predicted.equals(gold), "details": details})
        assert all(r["value_match"] for r in results), results
        checks.append("HTTP page and Gradio client API: three real GPU questions")
        proof = {"execution_location": "Kaggle", "checks_passed": checks,
            "results": results, "memory": service.generator.memory.check()}
        Path(output).write_text(json.dumps(proof, indent=2) + "\n")
        print("GRADIO APP PASS: page, API, three GPU questions and error handling", flush=True)
        return proof
    finally:
        demo.close()
