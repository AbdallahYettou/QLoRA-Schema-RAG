"""Package the Gradio demo without running inference or evaluation locally."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


requirements = (ROOT / "notebook2/requirements.txt").read_text().splitlines()
requirements = ["fastembed==0.9.1" if r.startswith("fastembed") else r for r in requirements]
requirements.append("gradio==6.30.0")
cells = [cell("markdown", """# Ask Spider: Gradio + fine-tuned Qwen

Notebook 1 trained the **3-epoch LoRA adapter**. This app reuses Notebook 2's BGE
schema retrieval, NF4 Qwen + that adapter, and read-only SQLite execution.
The app also displays the saved 100-example benchmark (3% → 15% → 62% EX).

**Run on Kaggle with T4 GPU + Internet.** To interact with the app, edit this
notebook, set `LAUNCH_APP = True` below, and run all cells. Choose a temporary
app password when prompted, then open the printed Gradio URL. Username: `demo`.
Keep the interactive Kaggle session running. The API-submitted verification run
uses `False`, performs three GPU smoke tests, and exits without a public server.
"""), cell("code", """LAUNCH_APP = False  # Set True in an INTERACTIVE Kaggle session.
import os, sys, subprocess
from pathlib import Path
assert Path('/kaggle/input').is_dir(), 'Run this notebook on Kaggle only'
os.environ['HF_HOME'] = '/tmp/hf-gradio-cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GRADIO_TEMP_DIR'] = '/tmp/gradio-app-files'
os.chdir('/kaggle/working')
for package in ('notebook2', 'gradio_app'):
    Path(package).mkdir(exist_ok=True)
    Path(package, '__init__.py').write_text('')
sys.path.insert(0, '/kaggle/working')
""" + f"subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *{requirements!r}], check=True)\n")]
cells.append(cell("markdown", "## 1. Load the existing pipeline and the Gradio frontend\nThese modules are bundled; no separate Python file uploads are needed."))
files = ["notebook2/environment.py", "notebook2/schemas.py", "notebook2/model.py", "notebook2/sql_engine.py",
         "gradio_app/app.py", "gradio_app/verify.py"]
for name in files:
    cells.append(cell("code", f"%%writefile /kaggle/working/{name}\n" + (ROOT / name).read_text()))
cells += [cell("markdown", "## 2. Verify inputs and load the trained adapter\nNo retraining and no full benchmark rerun."), cell("code", """import hashlib, json
from notebook2.environment import configure_logging, verify_environment
from notebook2.model import SQLGenerator
from notebook2.schemas import initialize_schema_index
from gradio_app.app import SQLApp, build_demo, launch_app
from gradio_app.verify import verify_app

configure_logging('/kaggle/working/app_logs')
versions = verify_environment()
inputs = Path('/kaggle/input')
adapters = list(inputs.rglob('lora_sql_adapter_3epochs/adapter_model.safetensors'))
databases = list(inputs.rglob('concert_singer/concert_singer.sqlite'))
benchmarks = list(inputs.rglob('rag_evaluation/summary.json'))
assert len(adapters) == len(databases) == len(benchmarks) == 1
with adapters[0].open('rb') as stream:
    adapter_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
assert adapter_sha == '961d13a4c49a15b92e1a0801f72e86253f4c4f5f7665786e9e758fd6ad1f5a9a'
history_root = benchmarks[0].parent
history = json.loads(benchmarks[0].read_text())
assert history['metadata']['adapter_sha256'] == adapter_sha
assert history['samples'] == 100 and history['execution_accuracy_pct'] == 62.0
db_root = databases[0].parent.parent
generator = SQLGenerator(str(adapters[0].parent))
index = initialize_schema_index('/kaggle/working/app_chroma_store')
service = SQLApp(generator, index, db_root)
demo = build_demo(service, history_root)
print(f'Loaded the final adapter and {len(service.databases)} available databases.')
"""), cell("markdown", "## 3. Verify the actual Gradio HTTP/API and three live questions\nTests execute on Kaggle. Historical benchmark files are only displayed."), cell("code", """proof = verify_app(service, demo, '/kaggle/working/gradio_verification.json')
proof['adapter_sha256'] = adapter_sha
proof['versions'] = versions
proof['source_sha256'] = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for package in ('notebook2', 'gradio_app') for p in Path(package).glob('*.py')
    if p.name != '__init__.py'
}
Path('gradio_verification.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2), flush=True)
# A fresh interface after closing the smoke-test server.
demo = build_demo(service, history_root)
"""), cell("markdown", "## 4. Launch the interactive app\nSet `LAUNCH_APP = True` in the first cell, or run `launch_app(demo)` manually here.\nUse a new app password, not a Kaggle token. Username: **demo**."), cell("code", """if LAUNCH_APP:
    launch_app(demo)
else:
    print('Verification finished. For live use, open an interactive T4 session, set LAUNCH_APP=True, and run all cells.')
    print('After setup, additional questions reuse the same loaded model; no retraining is needed.')
""")]
for i, item in enumerate(cells):
    item["id"] = f"gradio-cell-{i:02}"
notebook = {"nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells}
(DEST / "gradio_text2sql.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
(DEST / "kernel-metadata.json").write_text(json.dumps({
    "id": "yettouabdallah/qlora-schema-rag-gradio", "title": "QLoRA Schema RAG Gradio",
    "code_file": "gradio_text2sql.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": ["yousufqureshi5/spider-sqlite-databases"],
    "kernel_sources": ["yettouabdallah/qlora-text2sql-3epochs", "yettouabdallah/qlora-schema-aware-rag-sql"],
    "competition_sources": [], "model_sources": []}, indent=2) + "\n")
print(DEST / "gradio_text2sql.ipynb")
