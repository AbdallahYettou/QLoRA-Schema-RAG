"""Gradio interface; GPU inference and verification run on Kaggle only."""
import getpass
import logging
from pathlib import Path
import sqlite3
import threading
import time

import gradio as gr
import pandas as pd
import sqlglot

from notebook2.model import MemoryBudgetExceeded
from notebook2.schemas import database_path
from notebook2.sql_engine import sanitize_sql, execute_dataframe

EXAMPLES = [
    ["concert_singer", "How many singers are there?", 3],
    ["concert_singer", "What is the year that had the most concerts?", 3],
    ["concert_singer", "Show the name and theme for all concerts and the number of singers in each concert.", 3],
]


class SQLApp:
    """One shared model; serialize index/model access across all browser sessions."""
    def __init__(self, generator, schema_index, db_root):
        self.generator, self.index = generator, schema_index
        self.db_root = Path(db_root)
        self.databases = sorted(p.parent.name for p in self.db_root.glob("*/*.sqlite")
                                if p.name == p.parent.name + ".sqlite")
        if not self.databases:
            raise ValueError("No Spider databases attached")
        self.indexed = set()
        self.lock = threading.Lock()
        self.unavailable = False

    def ask(self, db_id, question, top_k=3):
        """Return status, SQL, schema, result preview and diagnostic details."""
        sql, schema = "", ""
        empty = pd.DataFrame()
        started = time.monotonic()
        try:
            if db_id not in self.databases:
                raise ValueError("Choose a database from the dropdown")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Enter a question first")
            if len(question) > 2000:
                raise ValueError("Keep the question under 2,000 characters")
            if top_k not in (1, 2, 3, 4, 5):
                raise ValueError("Choose between 1 and 5 schema tables")
            with self.lock:
                if self.unavailable:
                    raise RuntimeError("GPU limit reached earlier. Restart the Kaggle session and reload the app.")
                path = database_path(self.db_root, db_id)
                if db_id not in self.indexed:
                    self.index.ingest(path, db_id)
                    self.indexed.add(db_id)
                schema, matches = self.index.retrieve(question.strip(), db_id, int(top_k))
                assert all(m["metadata"]["db_id"] == db_id for m in matches)
                raw, metadata = self.generator(question.strip(), db_id, schema)
                sql = sanitize_sql(raw)
                frame = execute_dataframe(path, sql, timeout=5.0, max_rows=20000)
            details = {"database": db_id, "retrieved_tables": [m["metadata"]["table_name"] for m in matches],
                "retrieval_distances": [m.get("distance") for m in matches],
                "total_rows": len(frame), "displayed_rows": min(len(frame), 1000),
                "seconds": round(time.monotonic() - started, 2), "generation": metadata,
                "correctness": "Not evaluated: a new question has no reference answer."}
            status = f"Executed successfully · {len(frame):,} rows · {details['seconds']:.2f}s."
            if len(frame) > 1000:
                status += " Showing the first 1,000 rows."
            status += " Check the SQL and result: successful execution does not guarantee a correct answer."
            logging.info("App request db=%s tables=%s rows=%d", db_id, details["retrieved_tables"], len(frame))
            return status, sql, schema, frame.head(1000), details
        except (MemoryBudgetExceeded, RuntimeError) as exc:
            if isinstance(exc, MemoryBudgetExceeded) or "out of memory" in str(exc).lower() or "NVML" in str(exc):
                self.unavailable = True
            return f"Request failed: {exc}", sql, schema, empty, {"error_type": type(exc).__name__}
        except (ValueError, KeyError, sqlite3.Error, sqlglot.errors.SqlglotError) as exc:
            return f"Request failed: {exc}", sql, schema, empty, {"error_type": type(exc).__name__}


def build_demo(service, benchmark_root):
    benchmark_root = Path(benchmark_root)
    performance = pd.read_csv(benchmark_root / "performance_summary.csv")
    comparison = pd.read_csv(benchmark_root / "comparison.csv")
    with gr.Blocks(title="Ask Spider · Fine-tuned Qwen + schema RAG") as demo:
        gr.Markdown("# Ask Spider\nTurn a question into SQL and inspect the database answer.\n\n"
            "**Notebook 1 trained the SQL adapter. Notebook 2 retrieves schemas and uses that adapter to generate SQL.**")
        with gr.Tab("Ask the database"):
            with gr.Row():
                database = gr.Dropdown(service.databases,
                    value="concert_singer" if "concert_singer" in service.databases else service.databases[0], label="Database")
                top_k = gr.Slider(1, 5, value=3, step=1, label="Tables to retrieve")
            question = gr.Textbox(label="Question", placeholder="How many singers are there?", lines=2)
            run = gr.Button("Generate SQL and run", variant="primary")
            gr.Examples(EXAMPLES, inputs=[database, question, top_k], label="Try an example")
            status = gr.Textbox(label="Status", interactive=False)
            sql = gr.Code(language="sql", label="Generated SQL", interactive=False)
            result = gr.Dataframe(label="Query result (up to 1,000 rows)", interactive=False, type="pandas")
            with gr.Accordion("Retrieved schema", open=False):
                schema = gr.Code(language="sql", interactive=False)
            with gr.Accordion("Timing and model details", open=False):
                details = gr.JSON()
            run.click(service.ask, inputs=[database, question, top_k],
                outputs=[status, sql, schema, result, details], api_name="ask_sql",
                concurrency_limit=1, concurrency_id="single_gpu")
        with gr.Tab("Benchmark results"):
            gr.Markdown("## Saved 100-example benchmark\n"
                "These are the completed November 9 results, not a new benchmark or a confidence score for a live question.\n\n"
                "EX requires exact Pandas result equality, including column labels, types and row order. "
                "The 60-step adapter was not measured on this split.")
            gr.Dataframe(performance, label="Execution accuracy", interactive=False)
            gr.Dataframe(comparison, label="Questions, SQL and execution matches", interactive=False)
        with gr.Tab("How it works"):
            gr.Markdown("""## Two notebooks, one adapted model
1. **Notebook 1:** trains LoRA adapter weights on Spider question-to-SQL examples.
2. **BGE + Chroma:** embed the question and retrieve schema documents from the selected database.
3. **Base Qwen + Notebook 1 adapter:** read the question and retrieved schema through Qwen's chat template, then generate SQL.
4. **SQLite:** executes the cleaned query in read-only mode. Pandas displays the returned rows.

The adapter is loaded once per session. Asking a question does not retrain the model.
BGE retrieves schemas; Qwen writes SQL. No gold SQL is included in a live question's prompt.

The GPU stays on Kaggle. Keep the interactive session running while using this app.
""")
    return demo.queue(default_concurrency_limit=1, max_size=8)


def launch_app(demo):
    """Call from an interactive Kaggle cell; never embed credentials in source."""
    password = getpass.getpass("Choose a temporary app password (not a Kaggle password or API token): ")
    if len(password) < 8:
        raise ValueError("Use at least 8 characters for the app password")
    return demo.launch(share=True, auth=("demo", password), server_name="0.0.0.0",
                       inline=False, prevent_thread_lock=True, show_error=False)
