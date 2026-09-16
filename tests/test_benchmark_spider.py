from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from scripts.benchmark_spider import PairedGenerator, aggregate, benchmark, execute_dataframe, score_prediction


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "demo").mkdir()
        self.db = self.root / "demo/demo.sqlite"
        with sqlite3.connect(self.db) as conn:
            conn.executescript("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (1),(1),(2),(NULL);")
        self.gold = execute_dataframe(self.db, "SELECT x FROM t")

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_dataframes_and_fences(self):
        score = score_prediction(self.db, "```sql\nSELECT x FROM t\n```", self.gold)
        self.assertTrue(score["execution_match"])

    def test_alias_order_dtype_and_duplicates_are_significant(self):
        for sql in ("SELECT x AS renamed FROM t", "SELECT x FROM t ORDER BY x DESC",
                    "SELECT DISTINCT x FROM t", "SELECT CAST(x AS TEXT) AS x FROM t"):
            with self.subTest(sql=sql):
                score = score_prediction(self.db, sql, self.gold)
                self.assertTrue(score["execution_valid"])
                self.assertFalse(score["execution_match"])

    def test_syntax_errors_do_not_escape(self):
        for sql in ("SELECT FROM", "SELECT x FROM t; DELETE FROM t", "No SQL answer"):
            with self.subTest(sql=sql):
                score = score_prediction(self.db, sql, self.gold)
                self.assertFalse(score["syntax_valid"])
                self.assertFalse(score["execution_match"])

    def test_missing_table_is_execution_error(self):
        score = score_prediction(self.db, "SELECT x FROM unknown", self.gold)
        self.assertTrue(score["syntax_valid"])
        self.assertFalse(score["execution_valid"])
        self.assertEqual(score["error_type"], "sqlite_execution")

    def test_time_and_row_limits(self):
        score = score_prediction(self.db, "SELECT * FROM t", self.gold, max_rows=2)
        self.assertFalse(score["execution_valid"])
        self.assertEqual(score["error_type"], "resource_or_query_limit")
        sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT sum(x) FROM c"
        self.assertFalse(score_prediction(self.db, sql, self.gold, timeout=0.01)["execution_valid"])

    def test_gold_and_generation_failure_remain_in_denominator(self):
        rows = [{"db_id": "demo", "question": "Example", "query": "SELECT x FROM missing"}]

        def broken_generator(row, path, use_adapter):
            if use_adapter:
                raise RuntimeError("Synthetic generation failure")
            return "SELECT x FROM t", {}

        records = benchmark(rows, [0], {"demo": self.db}, broken_generator, self.root)
        summary = aggregate(records)
        self.assertEqual(summary["gold_errors"], 1)
        self.assertEqual(summary["finetuned"]["generation_errors"], 1)
        self.assertEqual(summary["base"]["execution_accuracy_pct"], 0)
        self.assertEqual(summary["finetuned"]["execution_accuracy_pct"], 0)
        self.assertIsNone(summary["syntax_error_rate_reduction_pct"])

    def test_reduction_zero_and_negative(self):
        good = score_prediction(self.db, "SELECT x FROM t", self.gold)
        bad = score_prediction(self.db, "SELECT FROM", self.gold)
        records = [{"gold_error": None, "base": bad, "finetuned": good}]
        self.assertEqual(aggregate(records)["syntax_error_rate_reduction_pct"], 100)
        records.append({"gold_error": None, "base": good, "finetuned": bad})
        records[0]["finetuned"] = bad
        self.assertEqual(aggregate(records)["syntax_error_rate_reduction_pct"], -100)

    def test_offline_cli_outputs_and_rescoring(self):
        rows = [{"db_id": "demo", "question": f"Question {i}", "query": "SELECT x FROM t"} for i in range(3)]
        predictions = [{"index": i, "db_id": "demo", "question": row["question"],
                        "base_sql": "SELECT FROM" if i == 0 else "SELECT x FROM t",
                        "finetuned_sql": "SELECT x FROM t"} for i, row in enumerate(rows)]
        data_path, pred_path = self.root / "dev.json", self.root / "pred.json"
        data_path.write_text(json.dumps(rows))
        pred_path.write_text(json.dumps(predictions))
        cmd = [sys.executable, "scripts/benchmark_spider.py", "--data", str(data_path),
               "--predictions", str(pred_path), "--db-root", str(self.root), "--samples", "3",
               "--output", str(self.root / "output")]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        summary = json.loads((self.root / "output/summary.json").read_text())
        self.assertAlmostEqual(summary["base"]["execution_accuracy_pct"], 200 / 3)
        self.assertEqual(summary["finetuned"]["execution_accuracy_pct"], 100)
        self.assertEqual(summary["syntax_error_rate_reduction_pct"], 100)
        self.assertIn("Fine-Tuned Execution Match", (self.root / "output/comparison.csv").read_text())
        cmd[cmd.index("--predictions") + 1] = str(self.root / "output/predictions.jsonl")
        cmd[-1] = str(self.root / "rescore")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        rescored = json.loads((self.root / "rescore/summary.json").read_text())
        self.assertEqual(summary["base"], rescored["base"])
        self.assertEqual(summary["finetuned"], rescored["finetuned"])


class AdapterToggleTests(unittest.TestCase):
    def test_base_disables_adapter_and_finetuned_restores_it(self):
        class Tensor:
            shape = (1, 4)

            def to(self, device):
                return self

            def __getitem__(self, key):
                return [10, 11]

        class Model:
            disabled = False
            calls = []

            def get_input_embeddings(self):
                return SimpleNamespace(weight=SimpleNamespace(device="cuda:0"))

            @contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False

            def generate(self, **kwargs):
                self.calls.append((self.disabled, kwargs))
                return Tensor()

        grad_entries = []

        @contextmanager
        def no_grad():
            grad_entries.append(True)
            yield

        generator = object.__new__(PairedGenerator)
        generator.args = SimpleNamespace(schema="none", max_new_tokens=8, max_context=32)
        generator.torch = SimpleNamespace(no_grad=no_grad)
        generator.model = Model()
        generator.tokenizer = SimpleNamespace(
            apply_chat_template=lambda *a, **k: {"input_ids": Tensor(), "attention_mask": Tensor()},
            decode=lambda *a, **k: "SELECT 1", eos_token_id=0)
        for enabled in (False, True):
            generator({"db_id": "demo", "question": "Example"}, Path("unused"), enabled)
        self.assertEqual([disabled for disabled, _ in generator.model.calls], [True, False])
        self.assertEqual(len(grad_entries), 2)
        for _, kwargs in generator.model.calls:
            self.assertFalse(kwargs["do_sample"])
            self.assertEqual(kwargs["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
