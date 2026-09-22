"""Package readable notebook cells. This builder performs no evaluation/inference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


requirements = (ROOT / "notebook2/requirements.txt").read_text().splitlines()
setup = '''import os, sys, subprocess
from pathlib import Path
assert Path('/kaggle/input').is_dir(), 'Execute on Kaggle, not locally'
os.environ['HF_HOME'] = '/tmp/hf-rag-cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.chdir('/kaggle/working')
Path('notebook2').mkdir(exist_ok=True)
Path('notebook2/__init__.py').write_text('')
sys.path.insert(0, '/kaggle/working')
'''
setup += f"subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *{requirements!r}], check=True)\n"
cells = [cell("markdown", """# Notebook 2: Schema-Aware RAG & SQL Execution Agent Pipeline

Run all cells **on Kaggle with a Tesla T4**. Each chunk's assertion test must pass
before advancing. BGE embeddings run on CPU; Qwen + final LoRA run on one GPU.
The 100 examples, final adapter, base revision and databases match Notebook 1.
No validation gold SQL is provided to retrieval or generation.

Results use strict `pandas.DataFrame.equals`: labels, dtypes and row/column order
matter. This is single-database EX, not official Spider test-suite accuracy.
The measured schema-free **3-epoch** score is 15%; the 60-step score is unmeasured.
"""), cell("markdown", "## Chunk 1: Environment, Dependencies & ChromaDB Setup\n"), cell("code", setup)]


def module(name):
    return cell("code", f"%%writefile /kaggle/working/notebook2/{name}.py\n" + (ROOT / f"notebook2/{name}.py").read_text())


cells += [module("environment"), cell("code", """from notebook2.environment import configure_logging, verify_environment
OUTPUT = Path('/kaggle/working/rag_evaluation')
configure_logging(OUTPUT)
versions = verify_environment()
passed_chunks = [1]
""")]
cells += [cell("markdown", """## Chunk 2: Schema Extraction & Metadata-Filtered Vector Indexing

One original SQLite DDL document per table, including PK/FK relationships.
Chroma applies `where={"db_id": db_id}` before top-k retrieval. Unknown databases
raise an error. No global fallback. BGE's 512-token embedding window may truncate
very wide DDL for ranking; the full stored DDL is injected into the model prompt.
"""), module("schemas"), cell("code", """from notebook2.schemas import verify_schemas, initialize_schema_index, retrieve_relevant_schemas
verify_schemas()
passed_chunks.append(2)
schema_index = initialize_schema_index('/kaggle/working/chroma_schema_store')
""")]
cells += [cell("markdown", """## Chunk 3: Model & LoRA Adapter Loading with Dynamic Prompting

NF4, double quantization, FP16 compute, batch size 1, greedy decoding, 2048-token
context and 256 new tokens. Prompts exceeding the limit are recorded as failures,
never silently truncated. GPU allocator cap: 8.8 GB. Both allocator high-water
marks and sampled NVML usage must remain below **10 decimal GB**.
"""), module("model"), cell("code", """import hashlib
from notebook2.model import SQLGenerator, verify_model
adapter_files = list(Path('/kaggle/input').rglob('lora_sql_adapter_3epochs/adapter_model.safetensors'))
assert len(adapter_files) == 1
with adapter_files[0].open('rb') as stream:
    adapter_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
assert adapter_sha == '961d13a4c49a15b92e1a0801f72e86253f4c4f5f7665786e9e758fd6ad1f5a9a'
generator = SQLGenerator(str(adapter_files[0].parent))
verify_model(generator)
passed_chunks.append(3)
""")]
cells += [cell("markdown", """## Chunk 4: SQL Sanitizer & SQLite Execution Harness

Accept one SQL code block or one bare query. Remove comments outside quoted
strings without rewriting SQL spelling. SQLite opens existing files read-only;
an authorizer, five-second VM deadline, and 100,000-row cap limit execution.
Syntax validity is separate from missing-table/column execution errors.
"""), module("sql_engine"), cell("code", """from notebook2.sql_engine import verify_sql_engine
verify_sql_engine()
passed_chunks.append(4)
""")]
cells += [cell("markdown", """## Chunk 5: Spider Evaluation Pipeline & Comparative Analysis

Replay Notebook 1's validation snapshot and exact 100 indices. Verify every
database hash and the adapter hash. Index only schema DDL, retrieve three tables
per question, generate SQL once, and execute against the same gold query.
All selected examples remain in the denominator. Raw predictions, retrieved
tables, errors, timings and memory metrics are saved for auditing.

Top-3 retrieval can omit join bridge tables. No retries or gold-guided schema
selection are used. Comparative improvements include the new system prompt as
well as schema retrieval; this is not a controlled retrieval-only ablation.
"""), module("pipeline"), cell("code", """import json, time, logging
from notebook2.pipeline import (discover_inputs, load_comparable_inputs, evaluate,
    save_results, verify_pipeline, sha256_file, BASE_MODEL, BASE_REVISION)
verify_pipeline()
passed_chunks.append(5)
assert passed_chunks == [1, 2, 3, 4, 5]
adapter, prior_root, db_root = discover_inputs()
rows, indices, previous, previous_summary, prior_provenance = load_comparable_inputs(prior_root, db_root)
source_hashes = {p.name: sha256_file(p) for p in Path('notebook2').glob('*.py') if p.name != '__init__.py'}
verification = {'passed_chunks': passed_chunks, 'execution_location': 'Kaggle',
    'source_sha256': source_hashes, 'versions': versions}
Path('/kaggle/working/verification.json').write_text(json.dumps(verification, indent=2))
started = time.monotonic()
for db_id in sorted({rows[i]['db_id'] for i in indices}):
    from notebook2.schemas import database_path
    schema_index.ingest(database_path(db_root, db_id), db_id)
records = evaluate(rows, indices, db_root, schema_index, generator, OUTPUT, top_k=3)
memory = generator.memory.check()
generator.memory.close()
metadata = {'sample_indices': indices, 'seed': 3407, 'top_k': 3, 'versions': versions,
    'base_model': BASE_MODEL, 'base_revision': BASE_REVISION, 'adapter_sha256': adapter_sha,
    'prior_provenance': prior_provenance, 'source_sha256': source_hashes,
    'max_context': 2048, 'max_new_tokens': 256, 'do_sample': False, 'temperature': 0.0,
    'seconds_indexing_and_evaluation': time.monotonic() - started, 'memory': memory,
    'embedding_model': 'BAAI/bge-small-en-v1.5', 'embedding_device': 'CPU',
    'metric_note': 'Strict labels/dtypes/row and column order; not official Spider test-suite accuracy'}
summary = save_results(records, previous, previous_summary, OUTPUT, metadata)
assert summary['samples'] == 100 and summary['gold_errors'] == 0
assert summary['generation_errors'] == 0 and summary['retrieval_errors'] == 0, summary
verification['evaluation_complete'] = True
verification['memory'] = memory
Path('/kaggle/working/verification.json').write_text(json.dumps(verification, indent=2))
print(json.dumps(summary, indent=2), flush=True)
from IPython.display import display
import pandas as pd
display(pd.read_csv(OUTPUT / 'performance_summary.csv'))
display(pd.read_csv(OUTPUT / 'comparison.csv').head(10))
""")]
notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
for i, item in enumerate(cells):
    item["id"] = f"rag-cell-{i:02}"
(DEST / "notebook2_schema_rag.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
(DEST / "kernel-metadata.json").write_text(json.dumps({
    "id": "yettouabdallah/qlora-schema-aware-rag-sql", "title": "QLoRA Schema Aware RAG SQL",
    "code_file": "notebook2_schema_rag.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": ["yousufqureshi5/spider-sqlite-databases"],
    "kernel_sources": ["yettouabdallah/qlora-text2sql-3epochs", "yettouabdallah/qlora-spider-benchmark-100"],
    "competition_sources": [], "model_sources": []}, indent=2) + "\n")
print(DEST / "notebook2_schema_rag.ipynb")
