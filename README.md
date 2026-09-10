# QLoRA Schema RAG

Qwen2.5-Coder-7B fine-tuning for text-to-SQL on Spider.

The [baseline notebook](notebook_qlora_text2sql.ipynb) configures a 60-step QLoRA
run on a Kaggle T4: 4-bit NF4, rank 16, alpha 16, and all seven projection modules.
It formats database IDs, questions and SQL targets with Qwen's native chat template.

Model weights and credentials are excluded from Git. Run training on Kaggle with
GPU and internet enabled.
