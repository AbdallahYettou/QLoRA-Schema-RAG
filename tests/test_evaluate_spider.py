import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from scripts.evaluate_spider import database_path, equivalent, execute_sql, score_one


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "demo").mkdir()
        self.db = self.root / "demo/demo.sqlite"
        with sqlite3.connect(self.db) as conn:
            conn.executescript("CREATE TABLE t(x INTEGER, name TEXT); INSERT INTO t VALUES (1,'a'),(1,'b'),(2,'c'),(NULL,'d');")

    def tearDown(self):
        self.temp.cleanup()

    def test_fenced_aggregate(self):
        score = score_one(self.db, "SELECT COUNT(*) FROM t", "```sql\nSELECT COUNT(*) FROM t\n```")
        self.assertTrue(score["syntax_valid"] and score["execution_valid"] and score["execution_match"])

    def test_duplicates_are_significant(self):
        score = score_one(self.db, "SELECT x FROM t", "SELECT DISTINCT x FROM t")
        self.assertTrue(score["execution_valid"])
        self.assertFalse(score["execution_match"])

    def test_unordered_rows_and_nulls(self):
        self.assertTrue(score_one(self.db, "SELECT x FROM t", "SELECT x FROM t ORDER BY x DESC")["execution_match"])

    def test_outer_order_is_significant(self):
        self.assertFalse(score_one(self.db, "SELECT x FROM t ORDER BY x", "SELECT x FROM t ORDER BY x DESC")["execution_match"])

    def test_nested_order_does_not_require_outer_order(self):
        self.assertTrue(score_one(self.db, "SELECT x FROM (SELECT x FROM t ORDER BY x)", "SELECT x FROM t ORDER BY x DESC")["execution_match"])

    def test_unknown_table_is_parseable_but_not_executable(self):
        score = score_one(self.db, "SELECT x FROM t", "SELECT x FROM imaginary")
        self.assertTrue(score["syntax_valid"])
        self.assertFalse(score["execution_valid"])

    def test_reject_malformed_and_multiple_statements(self):
        for sql in ("SELECT FROM", "SELECT 1; SELECT 2", "DELETE FROM t", "ATTACH DATABASE '/tmp/x' AS x"):
            with self.subTest(sql=sql):
                self.assertFalse(score_one(self.db, "SELECT x FROM t", sql)["syntax_valid"])

    def test_sqlite_blocks_writes_even_without_parser(self):
        with self.assertRaises(sqlite3.DatabaseError):
            execute_sql(self.db, "DELETE FROM t")
        self.assertEqual(execute_sql(self.db, "SELECT COUNT(*) FROM t")[0], [(4,)])

    def test_resource_limits(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            execute_sql(self.db, "SELECT * FROM t", max_rows=2)
        with self.assertRaises(sqlite3.OperationalError):
            execute_sql(self.db, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT sum(x) FROM c", timeout=0.01)

    def test_empty_results_still_require_same_column_count(self):
        self.assertFalse(equivalent(([], 1), ([], 2), False))

    def test_missing_gold_does_not_count_as_correct(self):
        score = score_one(self.db, "SELECT * FROM missing", "SELECT 1")
        self.assertFalse(score["gold_valid"])
        self.assertFalse(score["execution_match"])
        self.assertTrue(score["execution_valid"])

    def test_database_path_validation(self):
        with self.assertRaises(ValueError):
            database_path(self.root, "../demo")
        with self.assertRaises(FileNotFoundError):
            database_path(self.root, "missing")

    def test_cli_offline_scoring_and_alignment(self):
        rows = [{"db_id": "demo", "question": "How many rows?", "query": "SELECT COUNT(*) FROM t"}]
        (self.root / "data.json").write_text(json.dumps(rows))
        prediction = {"index": 0, "db_id": "demo", "question": rows[0]["question"], "prediction": "SELECT COUNT(*) FROM t"}
        (self.root / "pred.json").write_text(json.dumps([prediction]))
        cmd = [sys.executable, "scripts/evaluate_spider.py", "--db-root", str(self.root),
               "--data", str(self.root / "data.json"), "--predictions", str(self.root / "pred.json"),
               "--output", str(self.root / "output")]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        summary = json.loads((self.root / "output/summary.json").read_text())
        self.assertEqual(summary["execution_accuracy_all_samples"], 1)
        prediction["question"] = "Wrong sample"
        (self.root / "pred.json").write_text(json.dumps([prediction]))
        cmd[-1] = str(self.root / "misaligned_output")
        failed = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("misaligned", failed.stderr)


if __name__ == "__main__":
    unittest.main()
