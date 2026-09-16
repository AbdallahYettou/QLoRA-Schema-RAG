"""Package the local evaluator into a self-contained Kaggle notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


setup = '''import os, sys, subprocess
os.environ["HF_HOME"] = "/tmp/hf-benchmark-cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "transformers==5.17.0", "peft==0.21.2", "datasets==4.8.5", "tokenizers==0.23.3",
    "accelerate>=1.0", "bitsandbytes>=0.45", "safetensors>=0.5", "sqlglot==30.30.0",
    "pandas>=2.2"], check=True)
'''
bundle = 'from pathlib import Path\nimport sys\nscript_dir = Path("/kaggle/working/benchmark_scripts")\nscript_dir.mkdir(exist_ok=True)\n'
for name in ("evaluate_spider.py", "benchmark_spider.py"):
    bundle += f'(script_dir / {name!r}).write_text({(ROOT / "scripts" / name).read_text()!r})\n'
bundle += 'sys.path.insert(0, str(script_dir))\n'

run = '''import hashlib, json, random, time, zipfile
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from benchmark_spider import execute_dataframe

assert torch.cuda.is_available(), "Kaggle CUDA GPU not attached"
print("GPU:", torch.cuda.get_device_name(0), flush=True)
input_root = Path("/kaggle/input")
adapters = list(input_root.rglob("lora_sql_adapter_3epochs/adapter_model.safetensors"))
assert len(adapters) == 1, f"Expected one final adapter, found {adapters}"
adapter = adapters[0].parent
with adapters[0].open("rb") as stream:
    adapter_sha = hashlib.file_digest(stream, "sha256").hexdigest()
assert adapter_sha == "961d13a4c49a15b92e1a0801f72e86253f4c4f5f7665786e9e758fd6ad1f5a9a", "Unexpected adapter weights"
db_candidates = list(input_root.rglob("concert_singer/concert_singer.sqlite"))
assert len(db_candidates) == 1, f"Ambiguous database root: {db_candidates}"
db_root = db_candidates[0].parent.parent
revision = HfApi().dataset_info("xlangai/spider").sha
rows = list(load_dataset("xlangai/spider", split="validation", revision=revision))
indices = sorted(random.Random(3407).sample(range(len(rows)), 100))
data_path = Path("/kaggle/working/validation.json")
data_path.write_text(json.dumps(rows) + "\\n")
db_hashes = {}
for i in indices:
    row = rows[i]
    db = db_root / row["db_id"] / (row["db_id"] + ".sqlite")
    execute_dataframe(db, row["query"])
    if row["db_id"] not in db_hashes:
        with db.open("rb") as stream:
            db_hashes[row["db_id"]] = hashlib.file_digest(stream, "sha256").hexdigest()
print(f"Preflight passed: {len(indices)} gold queries, {len(db_hashes)} databases", flush=True)
with zipfile.ZipFile("/kaggle/working/benchmark_databases.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for db_id in db_hashes:
        db = db_root / db_id / (db_id + ".sqlite")
        archive.write(db, f"database/{db_id}/{db_id}.sqlite")
provenance = {"adapter_sha256": adapter_sha, "adapter_source": "yettouabdallah/qlora-text2sql-3epochs",
    "database_source": "yousufqureshi5/spider-sqlite-databases", "database_sha256": db_hashes,
    "dataset": "xlangai/spider", "dataset_revision": revision, "samples": 100, "seed": 3407,
    "schema_mode": "none", "gpu": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__, "sample_indices": indices,
    "script_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in script_dir.glob("*.py")}}
Path("/kaggle/working/run_provenance.json").write_text(json.dumps(provenance, indent=2))
started = time.monotonic()
subprocess.run([sys.executable, str(script_dir / "benchmark_spider.py"),
    "--adapter", str(adapter), "--db-root", str(db_root), "--data", str(data_path),
    "--samples", "100", "--seed", "3407", "--schema", "none", "--device", "cuda:0",
    "--output", "/kaggle/working/spider_benchmark_100"], check=True)
summary = json.loads(Path("/kaggle/working/spider_benchmark_100/summary.json").read_text())
summary["kaggle_run_seconds_including_model_load"] = time.monotonic() - started
Path("/kaggle/working/spider_benchmark_100/summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
assert summary["gold_errors"] == 0
assert summary["base"]["generation_errors"] == 0 and summary["finetuned"]["generation_errors"] == 0, "Generation errors require investigation"
'''

notebook = {"nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": [{"cell_type": "markdown", "metadata": {}, "source": [
        "# QLoRA Spider benchmark: base versus 3-epoch adapter\n",
        "100 identical validation examples; original schema-free training prompt; strict Pandas result equality.\n",
        "Credentials are supplied by Kaggle, never embedded in notebook source.\n"]},
        code(setup), code(bundle), code(run)]}
(DEST / "benchmark.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
(DEST / "kernel-metadata.json").write_text(json.dumps({
    "id": "yettouabdallah/qlora-spider-benchmark-100",
    "title": "QLoRA Spider Benchmark 100", "code_file": "benchmark.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_internet": True,
    "dataset_sources": ["yousufqureshi5/spider-sqlite-databases"],
    "kernel_sources": ["yettouabdallah/qlora-text2sql-3epochs"],
    "competition_sources": [], "model_sources": []}, indent=2) + "\n")
print(DEST / "benchmark.ipynb")
