# Benchmark Baseline: 60-Step QLoRA Fine-Tuning Run

### 1. Training Configuration & Compute Profile
- **Base Model**: `unsloth/Qwen2.5-Coder-7B-Instruct` (4-bit NF4)
- **Dataset**: `xlangai/spider` (Train split: 7,000 samples)
- **LoRA Parameters**: $r=16$, $\alpha=16$, dropout=0.0
- **Target Modules**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Trainable Parameters**: 40,370,176 / 7,655,986,688 (0.53%)
- **Hardware**: NVIDIA Tesla T4 (14.56 GB VRAM)
- **Batch Setup**: Per-device batch size = 2, Gradient accumulation = 4 (Effective batch size = 8)
- **Total Training Steps**: 60 steps
- **Samples Seen**: 480 / 7,000 (~0.068 of 1 epoch)
- **Active Training Time**: ~211 seconds (~3.5 minutes)
- **Peak Memory**: 5.97 GB / 14.56 GB (41.0% VRAM utilization)

---

### 2. Qualitative Validation Test
**Input Prompt**:
```text
System: You are an expert SQL assistant. Given a database schema and a question, write a valid SQL query that answers the question.
User: Database: concert_singer
Question: How many singers do we have?
```

**Model Generated Response**:
```sql
SELECT count(*) FROM singer
```

---

### 3. Stored Artifacts
- LoRA Adapter weights: `baseline_60steps/lora_sql_adapter/adapter_model.safetensors` (161 MB)
- Config: `baseline_60steps/lora_sql_adapter/adapter_config.json`
- Full execution log: `baseline_60steps/qlora_training.log`
