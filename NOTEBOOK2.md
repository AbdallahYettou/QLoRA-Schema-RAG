# Notebook 2: Schema-aware RAG and SQLite execution

The self-contained [Kaggle notebook](kaggle_rag_kernel/notebook2_schema_rag.ipynb)
contains five numbered chunks. Each chunk writes its readable Python module and
runs an assertion test **on Kaggle** before advancing. Do not execute it locally.
The [source package](notebook2/) supports reviewing each module separately.

## Run remotely

The [metadata](kaggle_rag_kernel/kernel-metadata.json) enables internet and a GPU,
and attaches the final training output, prior benchmark output and Spider SQLite
dataset. JSON booleans are `true`, not strings. The CLI explicitly selects T4.

```bash
# Packaging only: no model, embeddings, tests or evaluation run locally.
python kaggle_rag_kernel/build_kernel.py
kaggle kernels push -p ./kaggle_rag_kernel --accelerator NvidiaTeslaT4 -t 7200
kaggle kernels status yettouabdallah/qlora-schema-aware-rag-sql
kaggle kernels logs yettouabdallah/qlora-schema-aware-rag-sql -f
# Download after status COMPLETE:
kaggle kernels output yettouabdallah/qlora-schema-aware-rag-sql -p ./results_notebook2
```

Kaggle CLI reads the existing host credential. No token is embedded in notebook
source or committed. Hugging Face/model caches are stored under `/tmp` remotely
to avoid exporting gigabytes of cached weights.

## Five modules and verification

| Chunk | Module | Remote assertions |
|---|---|---|
| 1 | [environment.py](notebook2/environment.py) | T4/CUDA, SQLite, parser, persistent Chroma reopen |
| 2 | [schemas.py](notebook2/schemas.py) | Real BGE vectors, PK/FK extraction, idempotence, two-database isolation, unknown DB rejection |
| 3 | [model.py](notebook2/model.py) | Adapter shape/config, NF4, native Qwen template, real GPU inference, memory limit |
| 4 | [sql_engine.py](notebook2/sql_engine.py) | Quotes/comments/fences, exact result equality, bad SQL, missing tables, mutation rejection, timeout, row cap |
| 5 | [pipeline.py](notebook2/pipeline.py) | Failure accounting, metric denominator, JSONL persistence; then full 100-example evaluation |

Original SQLite `CREATE TABLE` statements retain columns, types and constraints.
PRAGMA key metadata supplements each document. CPU FastEmbed uses
`BAAI/bge-small-en-v1.5`; Chroma persists explicit 384-dimensional vectors with
cosine distance. Retrieval always filters `db_id` before selecting up to three
tables, and independently asserts that all returned metadata has that ID.

```python
from notebook2.schemas import initialize_schema_index, retrieve_relevant_schemas

index = initialize_schema_index('/kaggle/working/chroma_schema_store')
index.ingest('/kaggle/input/.../database/concert_singer/concert_singer.sqlite',
             'concert_singer')
context = retrieve_relevant_schemas('How many singers are there?',
                                    db_id='concert_singer', top_k=3)
```

The model defaults to the final local adapter directory; its constructor also
accepts a real Hugging Face adapter repository ID. The packaged run discovers
the attached final adapter and verifies its SHA-256. It pins the same quantized
base revision as Notebook 1. No gold SQL or database rows are given to retrieval
or generation. The system and user messages follow the schema-aware
Qwen format; `assistant_target()` demonstrates the fenced target format without
inserting an answer into an inference prompt.

## Memory and execution bounds

NF4 + double quantization, FP16 compute, batch size one, 2048 total context tokens,
and 256 new tokens limit memory. Greedy generation uses `torch.no_grad()`,
`do_sample=False`, `temperature=0.0` and one beam. CPU embeddings consume no CUDA
memory. The PyTorch allocator is capped at 8.8 decimal GB; the run checks both
allocator peaks and total device usage against **10,000,000,000 bytes**. NVML is
sampled every 50 ms, so that measurement cannot rule out shorter transient peaks.
The notebook stops on an observed budget violation or CUDA OOM.

Generated SQL is not repaired. One fenced or bare SELECT/CTE is accepted; multiple
statements, mutations and malformed quotes/comments are rejected. Comments are
removed without changing literal strings or SQL spelling. SQLite opens existing
files read-only, enables `query_only`, denies unauthorized operations and extension
loading, and imposes a VM timeout and row cap. This is a bounded local benchmark
runner, not a general-purpose adversarial SQL sandbox.

## Interpretation and outputs

The run reuses the exact 100 validation indices, questions, gold SQL, adapter and
database hashes from the completed schema-free benchmark. Primary EX uses
`DataFrame.equals`, including labels, dtypes and row/column order. Equivalent SQL
with different aliases can therefore fail. This is **not official Spider
test-suite accuracy**. SQL syntax validity and successful execution are reported
separately: missing tables or columns can be syntactically valid but unexecutable.
Generation/retrieval failures count against all-example validity and EX; counts
are also reported separately. The syntax-rate reduction is relative and undefined
when the previous error rate is zero.

The prior scores are 3% EX for the unadapted base and 15% for the 3-epoch adapter.
The **60-step adapter has no measured score** on this split; the table marks it
unmeasured. New EX is populated only from completed inference, never estimated.
Both schema context and the system prompt change, so this comparison is not a
retrieval-only ablation. Top-3 retrieval can omit join bridge tables. BGE truncates
very long DDL for embedding at its 512-token window; full documents remain stored
and are injected into prompts. Context overflow is reported instead of silently
truncating schemas.

Downloaded outputs include:

- `rag_evaluation/comparison.csv` and `.html`: question, gold, base, schema-free
  3-epoch and RAG SQL, individual EX flags and retrieved tables.
- `rag_evaluation/performance_summary.csv` and `summary.json`: aggregate scores,
  comparison, sample/model provenance, timings and GPU memory.
- `rag_evaluation/predictions.jsonl`: raw responses, selected DDL, distances and errors.
- `rag_evaluation/evaluation.log`, Kaggle execution log and `verification.json`.
- `chroma_schema_store/`: reusable persistent vector collection.

See [the results report](reports/NOTEBOOK2_RESULTS.md) for the completed run.

## Primary documentation

- [Chroma metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)
- [Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure)
- [BGE small English v1.5 model](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
