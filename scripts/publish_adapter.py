#!/usr/bin/env python3
"""Validate and optionally publish only a final LoRA adapter to Hugging Face.

Default is an offline dry run. Add --push and --repo-id YOUR_HF_USER/REPO to
upload. Authentication uses HF_TOKEN or a cached `hf auth login` session.
No base weights, checkpoints, optimizer states, notebooks or logs are uploaded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import shutil

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")


def inspect_adapter(directory):
    config_path = directory / "adapter_config.json"
    weights = directory / "adapter_model.safetensors"
    config = json.loads(config_path.read_text())
    if config.get("peft_type") != "LORA":
        raise ValueError("Expected a LoRA adapter")
    with weights.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length > 16 * 1024 * 1024 or length > weights.stat().st_size - 8:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(handle.read(length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors or not all("lora_A" in k or "lora_B" in k for k in tensors):
        raise ValueError("Expected adapter-only LoRA A/B tensors")
    dtype_bytes = {"F32": 4, "F16": 2, "BF16": 2}
    offset = 0
    for tensor in sorted(tensors.values(), key=lambda value: value["data_offsets"][0]):
        start, end = tensor["data_offsets"]
        expected = math.prod(tensor["shape"]) * dtype_bytes[tensor["dtype"]]
        if start != offset or end - start != expected:
            raise ValueError("Invalid tensor offsets or shape in adapter")
        offset = end
    if weights.stat().st_size != 8 + length + offset:
        raise ValueError("Adapter file is truncated or has unexpected trailing data")
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    info = {"base_model": config["base_model_name_or_path"], "rank": config["r"],
            "alpha": config["lora_alpha"], "target_modules": sorted(config["target_modules"]),
            "parameters": sum(math.prod(v["shape"]) for v in tensors.values()),
            "tensor_dtypes": sorted({v["dtype"] for v in tensors.values()}),
            "weights_bytes": weights.stat().st_size, "sha256": digest.hexdigest()}
    return config, info


def model_card(info, repo_id):
    return f'''---
base_model: {info["base_model"]}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- text-to-sql
- qlora
datasets:
- xlangai/spider
---

# {repo_id or "Qwen2.5-Coder-7B Spider LoRA adapter"}

LoRA adapter trained for text-to-SQL on the Spider training split.
The base model is required separately. This repository contains adapter weights
and tokenizer assets only; it is not a merged model.

- Base recorded in adapter: `{info["base_model"]}`
- Rank: {info["rank"]}; alpha: {info["alpha"]}
- Target modules: {", ".join(info["target_modules"])}
- Adapter parameters: {info["parameters"]:,}
- Saved tensor dtypes: {", ".join(info["tensor_dtypes"])}
- Weights: {info["weights_bytes"]:,} bytes
- Weights SHA-256: `{info["sha256"]}`

The training prompt contains a database ID and question but no database schema.
Training loss and hand-written examples do not establish validation accuracy.
No Spider execution-accuracy score is claimed by this card.

Use the base-model and dataset source pages for their respective licenses and
attribution requirements. The base-model reference follows the saved PEFT config.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="Final exported adapter folder, not its parent output directory")
    parser.add_argument("--repo-id", help="Explicit Hugging Face namespace/repository")
    parser.add_argument("--push", action="store_true", help="Create the repo if needed and upload")
    parser.add_argument("--public", action="store_true", help="Create a public repo; default new repos are private")
    args = parser.parse_args()
    if args.push and (not args.repo_id or args.repo_id.count("/") != 1):
        parser.error("--push requires --repo-id YOUR_HF_NAMESPACE/REPOSITORY")
    _, info = inspect_adapter(args.adapter)
    files = ["adapter_config.json", "adapter_model.safetensors"]
    files += [name for name in TOKENIZER_FILES if (args.adapter / name).is_file()]
    print(json.dumps({**info, "upload_files": files + ["README.md"], "repo_id": args.repo_id,
                      "action": "push" if args.push else "dry_run"}, indent=2))
    if not args.push:
        print("Validated. Add --push --repo-id YOUR_HF_NAMESPACE/REPOSITORY to publish.")
        return
    from huggingface_hub import HfApi
    api = HfApi()  # Uses HF_TOKEN or cached login; never hard-code or print tokens.
    api.whoami()  # Fail before creating a repository if authentication is missing.
    with tempfile.TemporaryDirectory(prefix="sql-lora-publish-") as temporary:
        staged = Path(temporary)
        for name in files:
            shutil.copyfile(args.adapter / name, staged / name)
        (staged / "README.md").write_text(model_card(info, args.repo_id))
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
        commit = api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=staged,
                                   allow_patterns=files + ["README.md"],
                                   commit_message="Upload final Spider text-to-SQL LoRA adapter")
    print(f"Published: {commit.commit_url}")


if __name__ == "__main__":
    main()
