#!/usr/bin/env python3
"""Extract loss curves and compute/inference evidence from Kaggle outputs."""
import argparse
import csv
import json
import re
from pathlib import Path

from publish_adapter import inspect_adapter


def read_run(root, adapter_name):
    states = []
    for path in root.rglob("trainer_state.json"):
        state = json.loads(path.read_text())
        states.append((state.get("global_step", 0), path, state))
    if not states:
        raise ValueError(f"No trainer_state.json found in {root}")
    _, state_path, state = max(states, key=lambda item: item[0])
    losses = [{k: row[k] for k in ("step", "epoch", "loss", "learning_rate") if k in row}
              for row in state["log_history"] if "loss" in row]
    events = []
    log_paths = sorted(root.glob("*.log"))
    for path in log_paths:
        try:
            items = json.loads(path.read_text())
            if isinstance(items, list):
                events.extend(item for item in items if isinstance(item, dict) and "data" in item)
        except json.JSONDecodeError:
            pass
    if not events:
        raise ValueError(f"No Kaggle JSON execution log found in {root}")
    events.sort(key=lambda event: event.get("time", 0))
    stdout = "".join(event["data"] for event in events if event.get("stream_name") == "stdout")
    all_text = "".join(event["data"] for event in events)
    peak = re.search(r"Peak memory:\s*([0-9.]+)\s*GB", stdout)
    start = next((event["time"] for event in events if "[Training] Launching" in event["data"]), None)
    end = next((event["time"] for event in events if "[Training Complete]" in event["data"]), None)
    inference = {}
    for name, sql in re.findall(r"^=== (Test [^\n]+?) ===\s*```sql\s*(.*?)```", stdout, re.S | re.M):
        inference[name] = sql.strip()
    baseline_sql = re.search(r"--- Model SQL Output ---\s*```sql\s*(.*?)```", stdout, re.S)
    if baseline_sql:
        inference["Test 1 (Simple Aggregate)"] = baseline_sql.group(1).strip()
    _, adapter = inspect_adapter(root / adapter_name)
    runtime = next((row for row in reversed(state["log_history"]) if "train_runtime" in row), None)
    result = {
        "source_root": str(root), "trainer_state": str(state_path), "logs": [str(p) for p in log_paths],
        "global_step": state["global_step"], "epoch": state.get("epoch"),
        "first_logged_loss": losses[0] if losses else None,
        "last_logged_loss": losses[-1] if losses else None,
        "peak_allocated_vram_gib": float(peak.group(1)) if peak else None,
        "training_wall_seconds_from_log_markers": end - start if start is not None and end is not None else None,
        "last_kernel_log_elapsed_seconds": max(event["time"] for event in events),
        "trainer_final_summary": runtime,
        "inference": inference, "adapter": adapter,
        "hardware_log_lines": [line for line in all_text.splitlines() if any(s in line for s in ("VRAM:", "GPU:", "Num GPUs used", "Total steps"))],
        "loss_points": len(losses),
    }
    return result, losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results_baseline"))
    parser.add_argument("--full", type=Path, default=Path("results_3epochs"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary, curves = {}, {}
    for label, root, adapter in (("baseline_60steps", args.baseline, "lora_sql_adapter"),
                                 ("full_3epochs", args.full, "lora_sql_adapter_3epochs")):
        summary[label], curves[label] = read_run(root, adapter)
    (args.output / "training_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "loss_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "step", "epoch", "loss", "learning_rate"])
        writer.writeheader()
        for label, rows in curves.items():
            for row in rows:
                writer.writerow({"run": label, **row})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (label, rows) in zip(axes, curves.items()):
        ax.plot([row["step"] for row in rows], [row["loss"] for row in rows], linewidth=1.6)
        ax.scatter([rows[0]["step"], rows[-1]["step"]], [rows[0]["loss"], rows[-1]["loss"]], s=25)
        ax.set(title=label.replace("_", " "), xlabel="Optimizer step", ylabel="Logged training loss")
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
    fig.suptitle("Spider QLoRA: logged training loss (different step scales)")
    fig.tight_layout()
    fig.savefig(args.output / "loss_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
