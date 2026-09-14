#!/usr/bin/env python3
"""Generate SQL with a PEFT adapter and score it on Spider SQLite databases.

This reports single-database result equivalence, NOT official Spider test-suite
accuracy. Dependencies for scoring: sqlglot. Generation additionally requires
torch, transformers, peft, accelerate, bitsandbytes, datasets and safetensors.
"""
import argparse
from collections import Counter
from contextlib import closing
import importlib.metadata
import json
from pathlib import Path
import random
import re
import sqlite3
import time

import sqlglot
from sqlglot import exp

SYSTEM_PROMPT = "You are an expert SQL assistant. Given a database schema and a question, write a valid SQL query that answers the question."


def extract_sql(text):
    """Accept a single fenced SQL answer or a bare query; do not repair SQL."""
    fences = re.findall(r"```(?:sql|sqlite)?\s*\n?(.*?)```", text, re.I | re.S)
    if len(fences) > 1:
        raise ValueError("Multiple code blocks in answer")
    return (fences[0] if fences else text).strip()


def parse_query(sql):
    statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("Expected exactly one SELECT/compound query")
    return statements[0]


def database_path(root, db_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", db_id):
        raise ValueError(f"Invalid database ID: {db_id!r}")
    path = (root / db_id / f"{db_id}.sqlite").resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Missing Spider database: {path}")
    return path


def execute_sql(path, sql, timeout=5.0, max_rows=100000):
    """Read-only SQLite execution with an authorizer, VM timeout and row cap."""
    started = time.monotonic()
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
               sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

    def authorize(action, arg1, arg2, db, source):
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() == "load_extension":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=timeout)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.set_authorizer(authorize)
        conn.set_progress_handler(lambda: int(time.monotonic() - started > timeout), 1000)
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError(f"Result exceeds {max_rows} rows")
        return rows, len(cursor.description or ())


def equivalent(gold, pred, ordered):
    gold_rows, gold_width = gold
    pred_rows, pred_width = pred
    if gold_width != pred_width:
        return False
    return gold_rows == pred_rows if ordered else Counter(gold_rows) == Counter(pred_rows)


def score_one(path, gold_sql, answer, timeout=5.0, max_rows=100000):
    record = {"prediction": "", "syntax_valid": False, "execution_valid": False,
              "execution_match": False, "gold_valid": False}
    try:
        record["prediction"] = extract_sql(answer)
        parse_query(record["prediction"])
        record["syntax_valid"] = True
    except (ValueError, sqlglot.errors.SqlglotError) as exc:
        record["syntax_error"] = str(exc)
    try:
        gold_ast = parse_query(gold_sql)
        gold = execute_sql(path, gold_sql, timeout, max_rows)
        record["gold_valid"] = True
    except (ValueError, sqlite3.Error, sqlglot.errors.SqlglotError) as exc:
        record["gold_error"] = str(exc)
        gold = None
    if record["syntax_valid"]:
        try:
            pred = execute_sql(path, record["prediction"], timeout, max_rows)
            record["execution_valid"] = True
            if gold is not None:
                record["execution_match"] = equivalent(gold, pred, gold_ast.args.get("order") is not None)
        except (ValueError, sqlite3.Error) as exc:
            record["execution_error"] = str(exc)
    return record


def read_json_rows(path):
    text = path.read_text()
    return json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]


def schema_text(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        return "\n".join(row[0] + ";" for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name"))


def load_generator(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import PeftConfig, PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires a CUDA GPU. Use --predictions for CPU-only scoring.")
    set_seed(args.seed)
    config = PeftConfig.from_pretrained(args.adapter)
    base = args.base_model or config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        base, device_map="auto", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.float16))
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    torch.cuda.reset_peak_memory_stats()

    def generate(row, path):
        prompt = f"Database: {row['db_id']}\n"
        if args.schema == "ddl":
            prompt += f"Schema:\n{schema_text(path)}\n"
        prompt += f"Question: {row['question']}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        inputs = {key: value.to(model.get_input_embeddings().weight.device) for key, value in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]
        if prompt_tokens + args.max_new_tokens > args.max_context:
            raise ValueError(f"Prompt plus output budget exceeds {args.max_context} tokens; no silent truncation")
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    use_cache=True, pad_token_id=tokenizer.eos_token_id)
        tokens = output[0, prompt_tokens:]
        return tokenizer.decode(tokens, skip_special_tokens=True), {
            "generation_seconds": time.monotonic() - started,
            "prompt_tokens": prompt_tokens, "generated_tokens": len(tokens),
            "hit_token_limit": len(tokens) == args.max_new_tokens,
        }

    return generate, base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter", help="Local final adapter directory or Hub adapter repository")
    source.add_argument("--predictions", type=Path, help="JSON/JSONL rows with index, db_id, question, prediction")
    parser.add_argument("--db-root", type=Path, required=True, help="Spider database/ directory")
    parser.add_argument("--data", type=Path, help="Local Spider dev.json; defaults to HF validation split")
    parser.add_argument("--dataset-revision", default="main", help="Use a commit SHA for reproducibility")
    parser.add_argument("--base-model", help="Override base recorded in adapter config")
    parser.add_argument("--schema", choices=["none", "ddl"], default="none")
    parser.add_argument("--limit", type=int, help="Seeded random validation subset; omit for full validation")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=Path("spider_evaluation"))
    args = parser.parse_args()
    for name in ("timeout", "max_rows", "max_new_tokens", "max_context"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.data:
        rows = read_json_rows(args.data)
    else:
        from datasets import load_dataset
        rows = list(load_dataset("xlangai/spider", split="validation", revision=args.dataset_revision))
    indices = list(range(len(rows)))
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            parser.error("--limit must be between 1 and dataset size")
        indices = sorted(random.Random(args.seed).sample(indices, args.limit))
    if not indices:
        parser.error("Dataset is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("predictions.jsonl", "summary.json", "gold.txt", "predicted.sql")):
        parser.error("Output already contains evaluation results; choose a new --output directory")
    # Validate every database before loading a costly model or reporting metrics.
    paths = {rows[i]["db_id"]: database_path(args.db_root, rows[i]["db_id"]) for i in indices}
    saved = {}
    if args.predictions:
        for item in read_json_rows(args.predictions):
            index = item["index"]
            if index in saved:
                parser.error(f"Duplicate prediction index {index}")
            saved[index] = item
        for i in indices:
            if i not in saved or any(saved[i].get(k) != rows[i][k] for k in ("db_id", "question")):
                parser.error(f"Missing or misaligned prediction at validation index {i}")
            if not isinstance(saved[i].get("prediction"), str):
                parser.error(f"Prediction at index {i} must be a string")
        generator, base = None, None
    else:
        generator, base = load_generator(args)
    results = []
    started = time.monotonic()
    with (args.output / "predictions.jsonl").open("w") as out, (args.output / "gold.txt").open("w") as gold_out, (args.output / "predicted.sql").open("w") as sql_out:
        for i in indices:
            row, metadata = rows[i], {}
            path = paths[row["db_id"]]
            try:
                if generator:
                    answer, metadata = generator(row, path)
                else:
                    answer = saved[i]["prediction"]
            except (ValueError, RuntimeError) as exc:
                answer = ""
                metadata["generation_error"] = str(exc)
            record = {"index": i, "db_id": row["db_id"], "question": row["question"],
                      "gold_sql": row["query"], "raw_prediction": answer, **metadata,
                      **score_one(path, row["query"], answer, args.timeout, args.max_rows)}
            results.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            # Single-line SQL for the official evaluator; sqlglot preserves quoted whitespace.
            try:
                single_line = parse_query(record["prediction"]).sql(dialect="sqlite", pretty=False)
            except (ValueError, sqlglot.errors.SqlglotError):
                single_line = "SELECT * FROM __invalid_prediction__"
            sql_out.write(single_line.replace("\n", " ") + "\n")
            gold_out.write(row["query"].replace("\n", " ") + "\t" + row["db_id"] + "\n")
            print(f"{len(results)}/{len(indices)} | match={record['execution_match']}", flush=True)
    n = len(results)
    eligible = sum(r["gold_valid"] for r in results)
    matches = sum(r["execution_match"] for r in results)
    versions = {}
    for package in ("sqlglot", "torch", "transformers", "peft", "datasets", "bitsandbytes"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    summary = {
        "metric": "single_database_result_equivalence_not_official_spider_test_suite",
        "samples": n, "gold_executable_samples": eligible, "gold_errors": n - eligible,
        "syntax_validity": sum(r["syntax_valid"] for r in results) / n,
        "execution_validity": sum(r["execution_valid"] for r in results) / n,
        "execution_matches": matches, "execution_accuracy_all_samples": matches / n,
        "execution_accuracy_gold_executable": matches / eligible if eligible else None,
        "generation_errors": sum("generation_error" in r for r in results),
        "evaluation_seconds_excluding_model_load": time.monotonic() - started,
        "base_model": base, "versions": versions,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "comparison": "Column order and duplicates preserved; row order required only for outer gold ORDER BY; exact SQLite values, no float tolerance.",
    }
    if generator:
        import torch
        summary["peak_allocated_vram_gib"] = [torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
