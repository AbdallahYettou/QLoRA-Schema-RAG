"""Chunk 3: native Qwen chat, NF4 + LoRA, bounded single-example inference."""
import logging
import threading
import time

from notebook2.environment import VRAM_LIMIT_BYTES, require_kaggle

BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
BASE_REVISION = "4858886f896bd05db823557fc5cbc4ac28342af7"
SYSTEM_PROMPT = ("You are an expert SQL assistant. Given the following database schema, "
    "write a valid SQL query that accurately answers the question.\n\n"
    "### Database Schema:\n{retrieved_schema}")


def chat_messages(question, db_id, retrieved_schema):
    if not retrieved_schema.strip():
        raise ValueError("Refusing schema-free inference in Notebook 2")
    return [{"role": "system", "content": SYSTEM_PROMPT.format(retrieved_schema=retrieved_schema)},
            {"role": "user", "content": f"Database: {db_id}\nQuestion: {question}"}]


def assistant_target(query):
    """Reference formatting only; never append a gold target during inference."""
    return {"role": "assistant", "content": f"```sql\n{query}\n```"}


class MemoryBudgetExceeded(RuntimeError):
    pass


class MemoryMonitor:
    """Sample NVML device usage and record exact PyTorch allocator high-water marks.

    NVML includes context and non-PyTorch allocations. Its peak is sampled at
    50 ms; the allocator is additionally capped at 8.8 GB. This is a measured
    budget check, not a promise about every possible unsampled transient.
    """
    def __init__(self, torch):
        import pynvml
        self.torch, self.nvml = torch, pynvml
        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        self.peak_device_bytes = 0
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _watch(self):
        try:
            while not self.stop_event.is_set():
                self.sample()
                self.stop_event.wait(0.05)
        except Exception as exc:
            self.error = str(exc)

    def sample(self):
        used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
        self.peak_device_bytes = max(self.peak_device_bytes, used)

    def snapshot(self):
        self.sample()
        return {"peak_allocated_bytes": self.torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(0),
            "peak_device_used_bytes_sampled": self.peak_device_bytes,
            "limit_bytes": VRAM_LIMIT_BYTES, "nvml_sampling_seconds": 0.05}

    def check(self):
        if self.error:
            raise RuntimeError("NVML monitoring failed: " + self.error)
        stats = self.snapshot()
        if max(stats[k] for k in ("peak_reserved_bytes", "peak_device_used_bytes_sampled")) >= VRAM_LIMIT_BYTES:
            raise MemoryBudgetExceeded(f"10 GB VRAM budget exceeded: {stats}")
        return stats

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)


class SQLGenerator:
    def __init__(self, adapter="./lora_sql_adapter_3epochs", max_context=2048, max_new_tokens=256):
        require_kaggle()
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        assert torch.cuda.is_available()
        self.torch = torch
        self.max_context, self.max_new_tokens = max_context, max_new_tokens
        set_seed(3407)
        torch.cuda.set_per_process_memory_fraction(8_800_000_000 / torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats(0)
        self.memory = MemoryMonitor(torch)
        config = PeftConfig.from_pretrained(adapter)
        assert config.r == 16 and config.lora_alpha == 16 and config.bias == "none"
        assert set(config.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        self.tokenizer = AutoTokenizer.from_pretrained(adapter)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION,
            device_map={"": "cuda:0"}, torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
        self.model = PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()
        assert self.model.is_loaded_in_4bit
        quant = self.model.config.quantization_config
        assert (quant.get("bnb_4bit_quant_type") if isinstance(quant, dict) else quant.bnb_4bit_quant_type) == "nf4"
        logging.info("Loaded NF4 base + final LoRA | memory=%s", self.memory.check())

    def __call__(self, question, db_id, schema):
        inputs = self.tokenizer.apply_chat_template(chat_messages(question, db_id, schema),
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        prompt_tokens = inputs["input_ids"].shape[1]
        if prompt_tokens + self.max_new_tokens > self.max_context:
            raise ValueError(f"Context limit: {prompt_tokens}+{self.max_new_tokens}>{self.max_context}; no silent truncation")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        start = time.monotonic()
        with self.torch.no_grad():
            output = self.model.generate(**inputs, do_sample=False, temperature=0.0,
                num_beams=1, max_new_tokens=self.max_new_tokens, use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id)
        tokens = output[0, prompt_tokens:]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True)
        stats = {"seconds": time.monotonic() - start, "prompt_tokens": prompt_tokens,
            "generated_tokens": len(tokens), "hit_token_limit": len(tokens) == self.max_new_tokens}
        del inputs, output, tokens
        stats["memory"] = self.memory.check()
        return answer, stats


def verify_model(generator):
    """Actual GPU inference and native-template assertions before evaluation."""
    schema = "CREATE TABLE item(id INTEGER PRIMARY KEY);"
    messages = chat_messages("How many items?", "probe", schema)
    assert len(messages) == 2 and messages[0]["content"].endswith(schema)
    assert assistant_target("SELECT 1")["content"] == "```sql\nSELECT 1\n```"
    rendered = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert "<|im_start|>system" in rendered and "<|im_start|>assistant" in rendered
    answer, metadata = generator("How many items?", "probe", schema)
    assert answer.strip() and metadata["generated_tokens"] > 0
    generator.memory.check()
    logging.info("CHUNK 3 PASS | GPU generation=%r | memory=%s", answer, metadata["memory"])
