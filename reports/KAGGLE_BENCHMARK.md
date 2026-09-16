# Kaggle benchmark: unadapted base versus 3-epoch adapter

**Completed on Kaggle on 2026-11-09.**
[Run (version 1)](https://www.kaggle.com/code/yettouabdallah/qlora-spider-benchmark-100)
was launched through the Kaggle API using the existing local credential.
The final API status was `KernelWorkerStatus.COMPLETE`.

The evaluation uses 100 seeded examples from `xlangai/spider` validation,
covering 19 SQLite databases. The baseline here is the **unadapted base model**,
not the earlier 60-step adapter. Both models use the same NF4 base, tokenizer,
sample, schema-free prompt and greedy generation settings.

## Measured scores

| Metric | Base, no adapter | Final 3-epoch adapter |
|---|---:|---:|
| Strict Pandas execution accuracy | **3% (3/100)** | **15% (15/100)** |
| Syntax validity | 94% | 99% |
| Syntax error rate | 6% | 1% |
| Successful SQLite execution | 10% | 16% |
| Generation failures | 0 | 0 |
| Outputs reaching the 256-token limit | 15 | 0 |
| Mean generated tokens per example | 141.80 | 34.85 |
| Total measured generation time | 1,469.10 s | 422.22 s |

The fine-tuned model gains **12 percentage points** of strict execution accuracy.
The relative syntax-error-rate reduction is **83.33%**, calculated as
`100 × (6% − 1%) / 6%`; the absolute drop is 5 percentage points.
There were **zero gold-query failures**. Every selected example remains in the
denominator, including invalid predictions.

Paired outcomes: 13 examples improve, 1 regresses, 2 pass with both models, and
84 fail with both models. Evaluation took 1,892.83 seconds (31m 33s), excluding
model loading. The benchmark subprocess took 1,976.06 seconds (32m 56s), including
loading. The final kernel log timestamp was 2,101.83 seconds (35m 2s), including
setup and output conversion. GPU: Tesla T4, with inference directed to `cuda:0`.

## Side-by-side outputs

The complete 100-row table contains each question, gold SQL, both generated SQL
queries, separate Boolean execution matches, syntax flags and error messages:

- [Open the HTML comparison](../results_benchmark/spider_benchmark_100/comparison.html)
- [CSV comparison](../results_benchmark/spider_benchmark_100/comparison.csv)
- [Aggregate JSON](../results_benchmark/spider_benchmark_100/summary.json)
- [Raw predictions and per-example decisions](../results_benchmark/spider_benchmark_100/predictions.jsonl)
- [Evaluation log](../results_benchmark/spider_benchmark_100/benchmark.log)

For example, validation index 3 asks for singers' names, countries and ages in
descending age order. The base model invents `birth_date` and uses MySQL-style
`TIMESTAMPDIFF`/`CURDATE`. The fine-tuned model selects the real `Name`, `Country`
and `Age` fields and matches the gold result.

## Interpretation and limitations

The primary metric uses `predicted_dataframe.equals(gold_dataframe)` exactly. It checks values, labels, dtypes, duplicates and row/column order.
Column aliases or harmless ordering differences can therefore turn an otherwise
equivalent query into a strict mismatch. This is a 100-example single-database
benchmark, **not official Spider test-suite accuracy**.

As a separate sensitivity check, ignoring result labels and ignoring row order
when the gold query has no outer `ORDER BY` gives **7% base versus 15% fine-tuned**.
Four base predictions change outcome (indices 509, 968, 974, 1025). For example,
`SELECT COUNT(*) AS TotalCourses FROM courses` returns the right count for index
509 but fails strict equality against `SELECT count(*) FROM Courses` because
the result column label differs. The supplementary check preserves duplicate
rows and column order; it does not replace the primary metric.
[Supplementary audit](../results_benchmark/supplementary_result_audit.json).

The dominant problem is **schema resolution**. The base has 84 SQLite execution
errors and the fine-tuned model has 83; common errors are nonexistent tables or
columns. The remaining invalid predictions are extraction/parser/SQLite syntax
failures. Thus 99% syntax validity for the adapter does not mean 99% usable SQL.

No schema definitions were supplied, matching the original training prompt.
These results do not establish the models' performance when given schemas.
The 256-token output cap was reached by 15 base responses and zero adapter
responses; longer base responses and output formatting are additional factors
in this comparison. A useful next controlled experiment is the same sample with
`--schema ddl` for both models, followed by official test-suite evaluation.

## Independent verification and reproducibility

All outputs were downloaded. A CPU rescore using the exported database snapshot
reproduced **all 200 per-model syntax/execution/match decisions** and all reported
aggregate scores. All 19 database SHA-256 checksums and both uploaded script
hashes match the run provenance. The mounted adapter hash matches the previously
verified final checkpoint 2625.

- [Independent rescore](../results_benchmark/local_rescore/summary.json)
- [Provenance and database checksums](../results_benchmark/run_provenance.json)
- [Validation-data snapshot](../results_benchmark/validation.json)
- [Exported SQLite databases](../results_benchmark/benchmark_databases.zip)
- [Kaggle execution log](../results_benchmark/qlora-spider-benchmark-100.log)
- [Executable notebook](../kaggle_benchmark/benchmark.ipynb)
- [Benchmark script](../scripts/benchmark_spider.py) and [run guide](../BENCHMARK.md)

Reproduction identifiers:

- Seed: `3407`; sample size: `100`; prompt mode: `none`.
- Dataset revision: `0c350918f3f29ec754f1181c65cdce76cd6c133c`.
- Base: `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`.
- Base revision: `4858886f896bd05db823557fc5cbc4ac28342af7`.
- Adapter SHA-256: `961d13a4c49a15b92e1a0801f72e86253f4c4f5f7665786e9e758fd6ad1f5a9a`.
- Database mirror: [yousufqureshi5/spider-sqlite-databases](https://www.kaggle.com/datasets/yousufqureshi5/spider-sqlite-databases).
- Remote versions: PyTorch `2.11.0+cu130`, Transformers `5.17.0`, PEFT `0.21.2`,
  BitsAndBytes `0.50.2`, Datasets `4.8.5`, Pandas `2.3.3`, SQLGlot `30.30.0`.
- Greedy generation: `do_sample=False`, `temperature=0.0`, `num_beams=1`,
  `max_new_tokens=256`, context budget `2048`, `torch.no_grad()`.

The existing suite of 25 CPU tests passed before this GPU run. The successful
Kaggle execution additionally validates model loading, adapter disabling and
re-enabling, and generation with the actual downloaded adapter.
