# Run evaluation and publish the adapter

**GPU benchmark completed on Kaggle:** see the [verified 100-example results](reports/KAGGLE_BENCHMARK.md).

For a paired **unadapted base versus final adapter** benchmark with exact Pandas
DataFrame comparison, use [the paired benchmark guide](BENCHMARK.md) and
[benchmark_spider.py](scripts/benchmark_spider.py).

The measured training comparison is in [the report](reports/COMPARISON.md).
Commands below run from the project root. Evaluation requires the original
Spider SQLite databases, available through the
[official Spider release](https://yale-lily.github.io/spider).
The [Hugging Face dataset](https://huggingface.co/datasets/xlangai/spider)
supplies questions and reference SQL; it does not provide the SQLite database
files through `load_dataset`.

## Environment and data

For CPU scoring of saved predictions or adapter publishing:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-eval.txt
```

For generation on a CUDA host, first install the PyTorch build appropriate for
that host, then install [the GPU dependencies](requirements-gpu.txt):

```bash
.venv/bin/pip install -r requirements-gpu.txt
```

The downloaded training run reports PyTorch 2.11.0+cu130, Transformers 5.17.0,
PEFT 0.21.2, Datasets 4.8.5, and Tokenizers 0.23.3. Core generation versions are
pinned to those observed versions. The paired benchmark's GPU loading and
generation have now been verified on Kaggle. Scoring was tested with SQLGlot 30.30.0.

Extract Spider so the database root has this layout:

```text
data/spider/
  dev.json
  tables.json
  database/
    concert_singer/concert_singer.sqlite
    ...
```

## Evaluate the final adapter

Run a seeded 20-example smoke test, then omit `--limit` for all validation samples:

```bash
.venv/bin/python scripts/evaluate_spider.py \
  --adapter results_3epochs/lora_sql_adapter_3epochs \
  --db-root data/spider/database \
  --data data/spider/dev.json \
  --schema none --limit 20 --seed 3407 \
  --output spider_evaluation_smoke

.venv/bin/python scripts/evaluate_spider.py \
  --adapter results_3epochs/lora_sql_adapter_3epochs \
  --db-root data/spider/database \
  --data data/spider/dev.json \
  --schema none \
  --output spider_evaluation_3epochs

.venv/bin/python scripts/evaluate_spider.py \
  --adapter results_baseline/lora_sql_adapter \
  --db-root data/spider/database \
  --data data/spider/dev.json \
  --schema none \
  --output spider_evaluation_baseline
```

Omit `--data` to load `xlangai/spider`'s validation split. Use
`--dataset-revision COMMIT_SHA` to pin the Hub dataset. Keep the same data,
sample selection, prompt mode, context limit and generation settings across
models. Greedy generation is explicit; outputs are capped at 256 new tokens.
The base model defaults to the exact name saved in the adapter config:
`unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`. Tokenizer files are loaded from
the adapter directory. The script loads the base using NF4 with double quantization
and FP16 compute, then attaches the adapter with PEFT.

`--schema none` reproduces the training prompt. A separate run with `--schema ddl`
adds table definitions from each SQLite database. This is a distinct prompting
experiment: do not compare a baseline without schema to a full model with schema
as though only the weights changed. Prompts exceeding the context budget are
recorded as generation failures; no silent truncation is performed.

Outputs include per-example JSONL (SQL, errors, timings and match decisions),
a JSON summary, and paired SQL/gold text files for the official evaluator.
Existing result files are never overwritten. The JSONL can be rescored on CPU:

```bash
.venv/bin/python scripts/evaluate_spider.py \
  --predictions spider_evaluation_3epochs/predictions.jsonl \
  --db-root data/spider/database \
  --data data/spider/dev.json \
  --output spider_evaluation_rescored
```

For external prediction files, each JSON/JSONL row must include `index`, `db_id`,
`question`, and `prediction`. Indices refer to the original validation ordering.
Rescoring a subset requires the same `--limit` and `--seed` used during generation.

## Metric definitions

- **Syntax validity:** exactly one SELECT/compound query parsed by SQLGlot's
  SQLite dialect. This is parser validity; SQLite may still reject names or syntax.
- **Execution validity:** the extracted prediction actually executes against
  the corresponding SQLite database, within timeout and row limits.
- **Execution accuracy:** predicted and gold result rows match, preserving
  duplicates and column order. Row order matters only for an outermost gold
  `ORDER BY`. Values compare exactly, including numeric equality; no float
  tolerance or column permutation search is applied.

The script reports accuracy over all selected samples and, separately, over
samples with successfully executed gold queries. Gold errors are exposed rather
than silently discarded. Missing database files fail preflight. Generation
failures, invalid SQL and prediction timeouts remain failures in the denominator.
The evaluator runs SQLite read-only, denies writes/attachment, uses a VM time
budget (default 5 seconds), and caps results (default 100,000 rows).

This is **single-database result equivalence**, not the official Spider test-suite
metric. Empty-result coincidences can create false positives; exact ordering of
ties, strict column order, and floating-point behavior can create differences
from official scoring. The formal benchmark uses additional database instances.
Use the [official test-suite evaluator](https://github.com/taoyds/test-suite-sql-eval)
and its downloadable database suite for a comparable official score:

```bash
python /path/to/test-suite-sql-eval/evaluation.py \
  --gold spider_evaluation_3epochs/gold.txt \
  --pred spider_evaluation_3epochs/predicted.sql \
  --db /path/to/test-suite-databases \
  --etype exec --keep_distinct
```

Do not add `--plug_value`: this model predicts literal values. Record the
evaluator revision and options along with the result. The emitted SQL files
are normalized to one line for this interface; retain the JSONL as the primary
record of original outputs. SQL containing literal newlines needs manual review
before using the one-line export.

## Publish only the final adapter

Inspect offline first:

```bash
.venv/bin/python scripts/publish_adapter.py \
  --adapter results_3epochs/lora_sql_adapter_3epochs
```

Authenticate using `hf auth login` from the virtual environment, or supply
`HF_TOKEN` through the environment. Then replace the example namespace with
the target Hugging Face account or organization:

```bash
.venv/bin/hf auth login
.venv/bin/python scripts/publish_adapter.py \
  --adapter results_3epochs/lora_sql_adapter_3epochs \
  --repo-id YOUR_HF_NAMESPACE/qwen2.5-coder-7b-text2sql-3epochs \
  --push
```

New repositories default to private; add `--public` to create a public repository.
Existing repository visibility is preserved. The Kaggle username does not
establish the Hugging Face namespace, so no account name is assumed.

The publisher validates the adapter and uploads only its weights/config,
available tokenizer assets and a generated model card. It does not upload
checkpoints, optimizer state, logs, credentials or base-model weights. It also
does not delete pre-existing files in a destination repository; use a dedicated
adapter repository to keep its contents limited to adapters.
The current FP32 weights are 161.53 MB. Merging is unnecessary for this upload
and would create a much larger full model.
[Hugging Face upload API documentation](https://huggingface.co/docs/huggingface_hub/guides/upload).

## Reproduce extraction and tests

```bash
.venv/bin/pip install kaggle six matplotlib
.venv/bin/kaggle kernels status yettouabdallah/qlora-text2sql-qwen2-5-coder
.venv/bin/kaggle kernels status yettouabdallah/qlora-text2sql-3epochs
.venv/bin/kaggle kernels output yettouabdallah/qlora-text2sql-qwen2-5-coder -p ./results_baseline -q
.venv/bin/kaggle kernels output yettouabdallah/qlora-text2sql-3epochs -p ./results_3epochs -q
.venv/bin/python scripts/compare_runs.py
.venv/bin/python -m unittest discover -s tests -v
```

Keep Kaggle credentials outside notebooks. Both local notebook sources have
been cleaned; rotate the token that was previously embedded in them.
