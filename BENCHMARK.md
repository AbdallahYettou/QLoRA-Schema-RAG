# Base model versus 3-epoch adapter benchmark

**Kaggle run completed:** strict execution accuracy was **3% base versus 15%
fine-tuned** on 100 validation samples. Syntax error rate fell from 6% to 1%.
See the [verified results report](reports/KAGGLE_BENCHMARK.md) and
[full side-by-side table](results_benchmark/spider_benchmark_100/comparison.html).

## Kaggle execution

The benchmark is packaged in [a self-contained notebook](kaggle_benchmark/benchmark.ipynb)
with [Kaggle metadata](kaggle_benchmark/kernel-metadata.json). It attaches the
completed training kernel output and the Spider SQLite dataset directly.
The submitted run is [QLoRA Spider Benchmark 100](https://www.kaggle.com/code/yettouabdallah/qlora-spider-benchmark-100).

To reproduce after changing the local evaluator:

```bash
python kaggle_benchmark/build_kernel.py
kaggle kernels push -p kaggle_benchmark --accelerator NvidiaTeslaT4 -t 7200
kaggle kernels logs yettouabdallah/qlora-spider-benchmark-100 -f
kaggle kernels output yettouabdallah/qlora-spider-benchmark-100 -p results_benchmark -q
```

The notebook verifies the final adapter checksum and executes all sampled gold
queries before loading the model. Its outputs include the pinned validation-data
snapshot, dataset/model provenance, and a ZIP of the 19 sampled databases for
independent CPU rescoring. Credentials are read by the local Kaggle CLI and are
not included in the uploaded notebook.

The final adapter is downloaded in
[results_3epochs/lora_sql_adapter_3epochs](results_3epochs/lora_sql_adapter_3epochs/).
Its 161.53 MB weights match checkpoint 2625; see the
[integrity manifest](reports/adapter_manifest.json).

[benchmark_spider.py](scripts/benchmark_spider.py) compares the **unadapted base**
against the **base plus final adapter**, rather than the earlier 60-step adapter.
It uses the same 100 seeded validation samples and prompts for both models.

## Run on a CUDA host

Install a CUDA-compatible PyTorch build, then:

```bash
python -m pip install -r requirements-gpu.txt
python scripts/benchmark_spider.py \
  --adapter results_3epochs/lora_sql_adapter_3epochs \
  --db-root /path/to/spider/database \
  --samples 100 --seed 3407 \
  --output spider_benchmark_100
```

Run from the project root. Use `--samples 50` for a smaller run. By default,
questions and gold queries come from `xlangai/spider`, split `validation`.
Use `--dataset-revision COMMIT_SHA` for a pinned dataset, or
`--data /path/to/spider/dev.json` for an offline copy. Obtain SQLite files from
the [Spider release](https://yale-lily.github.io/spider); `--db-root` must contain
directories such as `concert_singer/concert_singer.sqlite`. Missing databases
stop the run before model loading.

The script reuses prompt/path helpers from [evaluate_spider.py](scripts/evaluate_spider.py).
Copy both scripts if transferring the benchmark into a Kaggle notebook environment.
Pass `--adapter /kaggle/input/.../lora_sql_adapter_3epochs` and the mounted database
root as appropriate. No retraining is required.

Only one NF4 base model is loaded in GPU memory. The saved PEFT config identifies
it as `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`. The base pass runs inside
`model.disable_adapter()`; the fine-tuned pass enables the adapter. The script
checks that this is a bias-free LoRA adapter without separately saved modules,
so disabling it recovers the unadapted base. Both passes use the same tokenizer
and quantization. [PEFT API](https://huggingface.co/docs/peft/package_reference/peft_model).

Inference uses `model.eval()`, `torch.no_grad()`, `do_sample=False`,
`temperature=0.0`, and one beam. Greedy decoding is controlled by `do_sample=False`;
temperature is irrelevant in that mode and some Transformers versions may warn
that it is ignored. These settings remove sampling randomness, but do not promise
bitwise-identical outputs across different GPU/software environments.
[Generation API](https://huggingface.co/docs/transformers/main_classes/text_generation).

`--schema none` reproduces the training prompt (database ID plus question).
`--schema ddl` supplies table definitions to **both** models as a separate
experiment. Prompts are not silently truncated; an exceeded context budget is
recorded as a generation failure.

## Results

The output directory contains:

- `comparison.csv`: question, gold SQL, base SQL, fine-tuned SQL, separate Boolean
  execution matches for each model, syntax flags and errors.
- `comparison.html`: the same side-by-side table for a browser, with escaped SQL.
- `summary.json`: aggregate percentages, sample indices, version information,
  settings and a hash of selected samples.
- `predictions.jsonl`: raw answers, extracted SQL, generation timing/token counts,
  gold errors and per-model scoring decisions; flushed after every example.
- `benchmark.log`: progress, errors and aggregate scores. The console also prints
  progress and a five-row table preview.

The completed Kaggle run's downloaded outputs are in
[results_benchmark](results_benchmark/). GPU inference completed on a Tesla T4,
and independent CPU rescoring reproduced every per-model decision.

## Metrics

**Execution accuracy (EX)** is `100 × exact_matches / selected_samples`.
SQLite executes gold and predicted queries in read-only connections. The full
result is converted to a Pandas DataFrame, and `predicted.equals(gold)` is used
directly. This deliberately preserves row/column order, labels, duplicates and
dtypes. Matching missing values compare equal. No sorting, alias normalization,
column permutation or float tolerance is applied.
[Pandas equality semantics](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.equals.html).

Consequently, `COUNT(*) AS total` can fail against `COUNT(*)` despite identical
counts. Queries without `ORDER BY` can also return equivalent rows in different
orders and fail this strict metric. This uses strict DataFrame
comparison; it is **not official Spider test-suite accuracy**.

**Syntax validity** requires an extracted single SELECT/compound query that
SQLGlot accepts and no SQLite-reported syntax error during execution. Name
resolution errors (`no such table/column`) are execution errors, not labeled
syntax errors. Parser acceptance alone is not proof of executable SQL. The
summary additionally reports execution-validity percentages.

**Syntax error rate** is the complement of syntax validity over all selected
samples. An empty/non-SQL answer or failed generation contributes an invalid
sample, rather than disappearing from the denominator. Generation failure
counts and error categories are exposed separately.

```text
relative syntax error reduction (%) =
    100 × (base_error_rate - fine_tuned_error_rate) / base_error_rate
```

The script also reports the percentage-point difference. Relative reduction is
`null`/N/A if the base error rate is zero; negative values indicate regression.
Gold-query failures count as EX failures for both models and are reported
explicitly. Any gold failures should be resolved before interpreting the score.

Malformed SQL and ordinary generation failures are caught per example. SQLite
has a VM execution time budget (default 5 s), a result cap (100,000 rows), and
an authorizer that rejects writes and database attachment. These limits prevent
a bad query from stopping the remaining evaluation examples.

## Rescore without a GPU

```bash
python -m pip install -r requirements-eval.txt
python scripts/benchmark_spider.py \
  --predictions spider_benchmark_100/predictions.jsonl \
  --db-root /path/to/spider/database \
  --samples 100 --seed 3407 \
  --output spider_benchmark_rescored
```

Use the same dataset revision/local data and sampling arguments. External
JSON/JSONL predictions need `index`, `db_id`, `question`, `base_sql` and
`finetuned_sql`; index/identity checks prevent misaligned comparisons.

Run CPU verification with:

```bash
python -m unittest discover -s tests -v
```
