# Notebook 2 remote evaluation

Kernel: [QLoRA Schema Aware RAG SQL](https://www.kaggle.com/code/yettouabdallah/qlora-schema-aware-rag-sql).

**Kaggle version 1 completed successfully on November 9, 2026.** All five chunk
assertion tests passed remotely, followed by all 100 evaluation examples. Kaggle
reported COMPLETE and its tables, logs, verification evidence and persistent
Chroma index were downloaded to [results_notebook2](../results_notebook2/).
No Notebook 2 inference, embedding generation, SQL execution or benchmark was
run locally. Local inspection only read downloaded results and checked hashes.

The evaluation reuses the prior benchmark's exact 100 validation examples across
19 databases. The 3-epoch adapter SHA-256 and all database hashes match the prior
run. BGE embeddings run on CPU; NF4 Qwen and LoRA run on one Tesla T4.

| Configuration | Strict Pandas EX |
|---|---:|
| Unadapted base, schema-free | 3% |
| Notebook 1, 60-step adapter, schema-free | Unmeasured |
| Notebook 1, 3-epoch adapter, schema-free | 15% |
| Notebook 2, 3-epoch adapter, top-3 schema RAG | **62%** |

The RAG configuration improves strict EX by **47 percentage points** over the
schema-free 3-epoch run. Paired results: 48 previously incorrect examples became
correct, 1 previously correct example became incorrect, 14 stayed correct and
37 stayed incorrect. The 60-step score remains unmeasured; the earlier 15% value
must not be attributed to that adapter.

## Execution and memory

| Measurement | Result |
|---|---:|
| SQL syntax validity | 100/100 (100%) |
| Successful SQLite execution | 95/100 (95%) |
| Exact gold result matches | 62/100 (62%) |
| Syntax error reduction vs schema-free 3 epochs | 100% relative (1 error to 0) |
| Gold query errors | 0 |
| Generation / retrieval failures | 0 / 0 |
| Generated queries hitting the 256-token cap | 0 |
| Prompt tokens | 218–603 |
| Peak PyTorch allocated memory | 5,968,054,272 bytes (5.97 GB) |
| Peak PyTorch reserved memory | 6,178,209,792 bytes (6.18 GB) |
| Sampled peak total GPU memory | **6,789,726,208 bytes (6.79 GB)** |
| Limit | 10,000,000,000 bytes (10 decimal GB) |
| Indexing + evaluation | 669.05 seconds (11.15 minutes) |
| Sum of generation times | 659.58 seconds |
| Last notebook execution-log timestamp, including setup/export | 835.87 seconds (13.93 minutes) |

NVML sampled device usage every 50 ms; this does not exclude shorter transient
peaks. The allocator was additionally capped at 8.8 GB. Final export/upload time
after the last notebook log is not included in the execution-log duration.

## Verification and remaining errors

[verification.json](../results_notebook2/verification.json) records all five
passed chunks. The downloaded hashes of all five source modules match the local
deliverables. Inspection confirmed 100 prediction records, 62 true EX flags and
no retrieved metadata from a different database.

Five predictions failed SQLite name resolution:

| Validation index | Database | Missing column |
|---|---|---|
| 100 | car_1 | T2.car_id |
| 178 | car_1 | T2.Country |
| 271 | employee_hire_evaluation | Number_of_products |
| 485 | wta_1 | winner_id |
| 578 | student_transcripts_tracking | T1.address_details |

Another 33 queries executed but did not exactly match gold. Reading the saved
SQL shows concrete semantic errors as well as possible strict-comparison effects:

- Index 744 intersects two aggregate counts instead of counting countries that
  appear in both language sets.
- Indices 752 and 753 omit the gold query's `DISTINCT` on regions.
- Index 761 selects only the city name while gold also selects population.

All three world-database tables were retrieved for those examples, so these
particular errors cannot be explained by a missing table document. No local SQL
rescore or alternative metric was run.

The primary metric is exact `DataFrame.equals` on one database: labels, dtypes,
values, and row/column ordering must match. It is not official Spider test-suite
accuracy. Retrieval and the system prompt both change relative to Notebook 1.

Top-3 retrieval can omit join bridge tables in larger databases. The syntax-error
reduction reflects only one prior error on this sample; it is not evidence of a
general zero-error rate.

## Deliverables

- [Self-contained five-chunk notebook](../kaggle_rag_kernel/notebook2_schema_rag.ipynb)
- [Side-by-side HTML comparison](../results_notebook2/rag_evaluation/comparison.html)
- [Full CSV comparison](../results_notebook2/rag_evaluation/comparison.csv)
- [Aggregate summary and provenance](../results_notebook2/rag_evaluation/summary.json)
- [Raw predictions and retrieved DDL](../results_notebook2/rag_evaluation/predictions.jsonl)
- [Evaluation log](../results_notebook2/rag_evaluation/evaluation.log)
- [Kaggle execution log](../results_notebook2/qlora-schema-aware-rag-sql.log)
See [the implementation guide](../NOTEBOOK2.md) for the modules, remote assertion
tests and execution bounds.
