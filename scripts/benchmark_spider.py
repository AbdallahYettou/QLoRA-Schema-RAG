#!/usr/bin/env python3
"""Compare the unadapted base and final QLoRA adapter on identical Spider rows.

Default: 100 seeded validation samples, identical prompts, greedy generation,
SQLite execution, and strict pandas.DataFrame.equals result comparison.
Run with --help. See BENCHMARK.md for metric definitions and commands.
"""
import argparse
from contextlib import closing, nullcontext
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import random
import sqlite3
import time

import pandas as pd
import sqlglot

if __package__:
    from .evaluate_spider import SYSTEM_PROMPT, database_path, extract_sql, parse_query, read_json_rows, schema_text
else:
    from evaluate_spider import SYSTEM_PROMPT, database_path, extract_sql, parse_query, read_json_rows, schema_text

LOG = logging.getLogger("spider-benchmark")


def execute_dataframe(path, sql, timeout=5.0, max_rows=100000):
    """Execute one read-only query and preserve SQLite result labels/order."""
    parse_query(sql)
    start = time.monotonic()
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
               sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

    def authorize(action, arg1, arg2, db, source):
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() == "load_extension":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=timeout)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.set_authorizer(authorize)
        conn.set_progress_handler(lambda: int(time.monotonic() - start > timeout), 1000)
        # Use a cursor to bound the result before constructing a DataFrame;
        # pandas infers dtypes over the entire result, rather than per chunk.
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError(f"Result exceeds row limit ({max_rows})")
        return pd.DataFrame.from_records(rows, columns=[column[0] for column in cursor.description])


def score_prediction(path, answer, gold_df, timeout=5.0, max_rows=100000, generation_error=None):
    result = {"sql": "", "syntax_valid": False, "execution_valid": False,
              "execution_match": False, "error_type": None, "error": None}
    if generation_error:
        return {**result, "error_type": "generation", "error": generation_error}
    try:
        result["sql"] = extract_sql(answer)
        parse_query(result["sql"])
    except (ValueError, sqlglot.errors.SqlglotError) as exc:
        return {**result, "error_type": "invalid_sql", "error": str(exc)}
    result["syntax_valid"] = True
    try:
        predicted = execute_dataframe(path, result["sql"], timeout, max_rows)
        result["execution_valid"] = True
        result["execution_match"] = gold_df is not None and predicted.equals(gold_df)
        result["result_shape"] = list(predicted.shape)
    except sqlite3.Error as exc:
        message = str(exc)
        # SQLite uses SQLITE_ERROR for both grammar and name-resolution errors.
        # Keep missing table/column errors separate from reported syntax errors.
        is_syntax = any(s in message.lower() for s in (
            "syntax error", "incomplete input", "unrecognized token"))
        result.update(syntax_valid=not is_syntax,
                      error_type="sqlite_syntax" if is_syntax else "sqlite_execution", error=message)
    except (ValueError, sqlglot.errors.SqlglotError) as exc:
        result.update(error_type="resource_or_query_limit", error=str(exc))
    return result


class PairedGenerator:
    """Keep one quantized base in memory; disable/enable LoRA for the comparison."""
    def __init__(self, args):
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for generation; use --predictions for CPU-only scoring.")
        self.args, self.torch = args, torch
        set_seed(args.seed)
        config = PeftConfig.from_pretrained(args.adapter)
        if config.peft_type != "LORA" or config.bias != "none" or config.modules_to_save:
            raise ValueError("This paired comparison requires LoRA with bias='none' and no modules_to_save.")
        self.base_name = args.base_model or config.base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(args.adapter)
        LOG.info("Loading base %s in NF4; adapter=%s", self.base_name, args.adapter)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_name, device_map={"": args.device}, torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
        self.model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
        self.model_revision = getattr(self.model.config, "_commit_hash", None)

    def __call__(self, row, path, use_adapter):
        args, torch = self.args, self.torch
        prompt = f"Database: {row['db_id']}\n"
        if args.schema == "ddl":
            prompt += "Schema:\n" + schema_text(path) + "\n"
        prompt += "Question: " + row["question"]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_tensors="pt", return_dict=True)
        device = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length + args.max_new_tokens > args.max_context:
            raise ValueError("Prompt plus generation budget exceeds --max-context; not truncated")
        context = nullcontext() if use_adapter else self.model.disable_adapter()
        start = time.monotonic()
        with context, torch.no_grad():
            tokens = self.model.generate(**inputs, do_sample=False, temperature=0.0, num_beams=1,
                max_new_tokens=args.max_new_tokens, use_cache=True,
                return_dict_in_generate=False, pad_token_id=self.tokenizer.eos_token_id)
        generated = tokens[0, prompt_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True), {
            "seconds": time.monotonic() - start, "prompt_tokens": prompt_length,
            "generated_tokens": len(generated), "hit_token_limit": len(generated) == args.max_new_tokens}


def aggregate(records):
    n = len(records)
    if not n:
        raise ValueError("Cannot score an empty benchmark")
    summary = {"samples": n, "gold_errors": sum(r["gold_error"] is not None for r in records),
               "metric": "strict_pandas_dataframe_equals_single_database"}
    for name in ("base", "finetuned"):
        scores = [r[name] for r in records]
        invalid = sum(not s["syntax_valid"] for s in scores)
        summary[name] = {
            "syntax_validity_pct": 100 * (n - invalid) / n,
            "syntax_error_rate_pct": 100 * invalid / n,
            "execution_validity_pct": 100 * sum(s["execution_valid"] for s in scores) / n,
            "execution_accuracy_pct": 100 * sum(s["execution_match"] for s in scores) / n,
            "syntax_invalid_count": invalid,
            "generation_errors": sum(s["error_type"] == "generation" for s in scores),
        }
    base_errors = summary["base"]["syntax_invalid_count"]
    ft_errors = summary["finetuned"]["syntax_invalid_count"]
    summary["syntax_error_rate_reduction_pct"] = 100 * (base_errors - ft_errors) / base_errors if base_errors else None
    summary["syntax_error_rate_drop_percentage_points"] = 100 * (base_errors - ft_errors) / n
    summary["syntax_reduction_note"] = "Undefined when base error rate is zero; negative means regression."
    return summary


def benchmark(rows, indices, paths, generator, output, timeout=5.0, max_rows=100000):
    """Generator(row, path, use_adapter) returns (raw answer, metadata)."""
    records = []
    with (output / "predictions.jsonl").open("w") as stream:
        for position, i in enumerate(indices, start=1):
            row = rows[i]
            path = paths[row["db_id"]]
            record = {"index": i, "db_id": row["db_id"], "question": row["question"],
                      "gold_sql": row["query"], "gold_error": None}
            try:
                gold_df = execute_dataframe(path, row["query"], timeout, max_rows)
            except (sqlite3.Error, ValueError, sqlglot.errors.SqlglotError) as exc:
                gold_df = None
                record["gold_error"] = str(exc)
                LOG.warning("Sample %s gold query failed: %s", i, exc)
            for name in ("base", "finetuned"):
                try:
                    answer, metadata = generator({**row, "index": i}, path, name == "finetuned")
                    error = metadata.get("generation_error")
                except (RuntimeError, ValueError) as exc:
                    answer, metadata, error = "", {}, str(exc)
                score = score_prediction(path, answer, gold_df, timeout, max_rows, error)
                record[name] = {**score, "raw_output": answer, "generation": metadata}
                record[name + "_sql"] = score["sql"]
                if score["error"]:
                    LOG.warning("Sample %s %s %s: %s", i, name, score["error_type"], score["error"])
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            LOG.info("[%d/%d] db=%s | base EX=%s | fine-tuned EX=%s", position, len(indices),
                     row["db_id"], record["base"]["execution_match"], record["finetuned"]["execution_match"])
    return records


def save_results(records, output, metadata):
    table = pd.DataFrame([{
        "Index": r["index"], "Database": r["db_id"], "Question": r["question"],
        "Gold SQL": r["gold_sql"], "Base Model SQL": r["base_sql"],
        "Fine-Tuned Model SQL": r["finetuned_sql"],
        "Base Execution Match": r["base"]["execution_match"],
        "Fine-Tuned Execution Match": r["finetuned"]["execution_match"],
        "Base Syntax Valid": r["base"]["syntax_valid"],
        "Fine-Tuned Syntax Valid": r["finetuned"]["syntax_valid"],
        "Base Error": r["base"]["error"], "Fine-Tuned Error": r["finetuned"]["error"],
        "Gold Error": r["gold_error"],
    } for r in records])
    table.to_csv(output / "comparison.csv", index=False)
    summary = {**aggregate(records), "metadata": metadata}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "comparison.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Spider benchmark</title>'
        '<style>body{font-family:system-ui}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc;'
        'vertical-align:top}td{min-width:120px;max-width:380px;overflow-wrap:anywhere}</style></head><body>'
        '<h1>Base versus fine-tuned Spider benchmark</h1><p>Strict Pandas DataFrame equality; '
        'single-database evaluation.</p>' + table.to_html(index=False, escape=True) + '</body></html>')
    LOG.info("Base EX: %.2f%% | Fine-tuned EX: %.2f%% | Syntax error reduction: %s",
             summary["base"]["execution_accuracy_pct"], summary["finetuned"]["execution_accuracy_pct"],
             "N/A (base error rate zero)" if summary["syntax_error_rate_reduction_pct"] is None
             else f"{summary['syntax_error_rate_reduction_pct']:.2f}%")
    print(table.head(5).to_string(index=False, max_colwidth=70))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="results_3epochs/lora_sql_adapter_3epochs")
    parser.add_argument("--base-model", help="Defaults to adapter's saved base model name")
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="Local Spider dev.json; otherwise HF validation split")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--schema", choices=["none", "ddl"], default="none")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--predictions", type=Path, help="CPU rescore JSON/JSONL: index, db_id, question, base_sql, finetuned_sql")
    parser.add_argument("--output", type=Path, default=Path("spider_benchmark"))
    args = parser.parse_args()
    for key in ("samples", "max_new_tokens", "max_context", "timeout", "max_rows"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; choose a new --output to preserve prior runs")
    if args.data:
        rows = read_json_rows(args.data)
    else:
        from datasets import load_dataset
        rows = list(load_dataset("xlangai/spider", split="validation", revision=args.dataset_revision))
    if args.samples > len(rows):
        parser.error("--samples exceeds the validation split size")
    indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.samples))
    paths = {rows[i]["db_id"]: database_path(args.db_root, rows[i]["db_id"]) for i in indices}
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "benchmark.log")])
    LOG.info("Selected %d validation samples, seed=%d, schema=%s", len(indices), args.seed, args.schema)
    if args.predictions:
        saved = {}
        for item in read_json_rows(args.predictions):
            if item["index"] in saved:
                parser.error(f"Duplicate prediction index {item['index']}")
            saved[item["index"]] = item
        for i in indices:
            if i not in saved or any(saved[i].get(k) != rows[i][k] for k in ("db_id", "question")):
                parser.error(f"Missing or misaligned predictions at index {i}")
            if any(not isinstance(saved[i].get(k), str) for k in ("base_sql", "finetuned_sql")):
                parser.error(f"SQL predictions must be strings at index {i}")

        def generator(row, path, use_adapter):
            name = "finetuned" if use_adapter else "base"
            item = saved[row["index"]]
            prior = item.get(name, {})
            # Keep raw answers and generation failures when rescoring our JSONL.
            metadata = {}
            if prior.get("error_type") == "generation":
                metadata["generation_error"] = prior["error"]
            return prior.get("raw_output", item[name + "_sql"]), metadata
    else:
        generator = PairedGenerator(args)
    started = time.monotonic()
    records = benchmark(rows, indices, paths, generator, args.output, args.timeout, args.max_rows)
    versions = {}
    for package in ("pandas", "sqlglot", "torch", "transformers", "peft", "bitsandbytes", "datasets"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sample_indices": indices, "base_model": getattr(generator, "base_name", None),
        "base_revision": getattr(generator, "model_revision", None), "versions": versions,
        "sample_sha256": hashlib.sha256(json.dumps([rows[i] for i in indices], sort_keys=True).encode()).hexdigest(),
        "seconds_excluding_model_load": time.monotonic() - started,
        "generation": {"do_sample": False, "temperature": 0.0, "num_beams": 1},
        "comparison": "DataFrame.equals: exact labels, row/column order, dtypes and values; no sorting or alias normalization.",
    }
    save_results(records, args.output, metadata)


if __name__ == "__main__":
    main()
