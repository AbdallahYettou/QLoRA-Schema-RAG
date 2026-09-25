# QLoRA Schema RAG

Text-to-SQL with Qwen2.5-Coder-7B, a QLoRA adapter trained on Spider, and
database-filtered schema retrieval. Training and inference run on Kaggle T4 GPUs.

## Pipeline

1. Fine-tune the base model with QLoRA on Spider question-to-SQL examples.
2. Index SQLite table definitions, columns, primary keys and foreign keys in Chroma.
3. Retrieve relevant schemas with BGE embeddings, filtered by database ID.
4. Generate SQL with the base Qwen model and the trained 3-epoch adapter.
5. Sanitize and execute the query in read-only SQLite, then display the results.

The base model uses 4-bit NF4 quantization. LoRA uses rank 16 and alpha 16 on the
query, key, value, output, gate, up and down projections. Schema embeddings run
on CPU; model inference runs on one T4.

## Notebooks

| Notebook | Purpose |
|---|---|
| [60-step training](notebook_qlora_text2sql.ipynb) | Initial QLoRA training and adapter export |
| [3-epoch training](notebook_qlora_text2sql_3epochs.ipynb) | Full training over 7,000 examples for 2,625 optimizer steps |
| [Base/adapter benchmark](kaggle_benchmark/benchmark.ipynb) | Paired evaluation on 100 validation examples |
| [Schema RAG](kaggle_rag_kernel/notebook2_schema_rag.ipynb) | Schema retrieval, SQL generation and execution evaluation |
| [Gradio app](kaggle_gradio_kernel/gradio_text2sql.ipynb) | Interactive database questions and benchmark results |

## Results

The same 100 Spider validation examples produce the following strict Pandas
execution accuracy. This metric includes column labels, dtypes, and row/column
order; it is not official Spider test-suite accuracy.

| Configuration | EX |
|---|---:|
| Base model, schema-free | 3% |
| 3-epoch adapter, schema-free | 15% |
| 3-epoch adapter, schema RAG | 62% |

The 60-step adapter has no measured EX on this split. The RAG evaluation reports
100% syntax validity, 95% successful SQLite execution and 6.79 GB sampled peak
device memory. Detailed results and actual run dates are recorded in the reports.

## Run on Kaggle

Each kernel folder includes a notebook and Kaggle metadata with its input sources.
Upload the notebook or submit a folder through the CLI:

```bash
kaggle kernels push -p kaggle_rag_kernel --accelerator NvidiaTeslaT4 -t 7200
kaggle kernels push -p kaggle_gradio_kernel --accelerator NvidiaTeslaT4 -t 3600
```

For interactive Gradio use, enable T4 and Internet, set `LAUNCH_APP = True` in
the app notebook, and run all cells. Use the temporary app password prompt and
keep the Kaggle session running.

Weights, credentials, model caches and downloaded run outputs are excluded from
Git. The Kaggle notebooks attach the training outputs directly. Some report links
point to downloaded artifacts available after running the documented download
commands. The local history helper is also excluded from version control.

## Guides

- [Evaluation and adapter publishing](EVALUATION.md)
- [Paired benchmark](BENCHMARK.md)
- [Schema RAG implementation](NOTEBOOK2.md)
- [Gradio setup](GRADIO_APP.md)
- [Training comparison](reports/COMPARISON.md)
- [Base/adapter results](reports/KAGGLE_BENCHMARK.md)
- [Schema RAG results](reports/NOTEBOOK2_RESULTS.md)
