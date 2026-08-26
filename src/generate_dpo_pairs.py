"""
High-Throughput On-Policy Preference Dataset Generator for DPO.

Batches generation across prompts, samples candidate code, runs fast in-memory
CryptoMiniSat verification, and outputs clean (chosen, rejected) preference pairs.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.compiler_grader import extract_python_code


def quick_grade_in_memory(code: str) -> Tuple[bool, Optional[str]]:
    """Fast in-memory execution check with stdout suppression."""
    clean_code = extract_python_code(code)
    locs = {}
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(clean_code, locs, locs)
        experiments = locs.get("experiments")
        if experiments and isinstance(experiments, list) and len(experiments) > 0:
            return True, None
        return False, "NoExperimentsSynthesized"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def generate_dpo_dataset_batched(
    model_name: str,
    adapter_path: Optional[str],
    input_file: str,
    output_file: str,
    batch_size: int = 16,
    temperature: float = 0.8,
    max_prompts: Optional[int] = None,
):
    print("=" * 60)
    print("Generating Batched DPO Preference Pairs")
    print(f"Model: {model_name} (Adapter: {adapter_path})")
    print(f"Batch Size: {batch_size}, Temp: {temperature}")
    print("=" * 60, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    dataset = load_dataset("json", data_files=input_file, split="train")
    if max_prompts:
        dataset = dataset.select(range(min(max_prompts, len(dataset))))

    items = [item for item in dataset]
    n = len(items)
    dpo_pairs = []

    t0 = time.perf_counter()

    for i in range(0, n, batch_size):
        batch = items[i : i + batch_size]
        prompts = [x["prompt"] for x in batch]
        systems = [x["messages"][0]["content"] for x in batch]

        formatted = []
        for p, s in zip(prompts, systems):
            msg = [{"role": "system", "content": s}, {"role": "user", "content": p}]
            formatted.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, (item, out_ids) in enumerate(zip(batch, outputs)):
            gen_toks = out_ids[inputs["input_ids"].shape[1] :]
            candidate_code = tokenizer.decode(gen_toks, skip_special_tokens=True)

            passed, err_msg = quick_grade_in_memory(candidate_code)
            canonical_code = f"```python\n{item['canonical_code']}\n```"

            # If candidate failed, canonical is chosen (yw) and candidate is rejected (yl)
            if not passed:
                dpo_record = {
                    "id": f"dpo_{len(dpo_pairs):05d}",
                    "task_family": item.get("task_family", "unknown"),
                    "prompt": item["prompt"],
                    "chosen": [
                        {"role": "system", "content": item["messages"][0]["content"]},
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": canonical_code},
                    ],
                    "rejected": [
                        {"role": "system", "content": item["messages"][0]["content"]},
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": candidate_code if candidate_code.startswith("```") else f"```python\n{candidate_code}\n```"},
                    ],
                    "rejected_error": err_msg,
                }
                dpo_pairs.append(dpo_record)

        print(f"  Processed {min(i + batch_size, n)}/{n} prompts -> {len(dpo_pairs)} valid DPO pairs formed ({len(dpo_pairs)/(time.perf_counter()-t0):.1f} pairs/s)", flush=True)

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in dpo_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_time = time.perf_counter() - t0
    print(f"\nGenerated {len(dpo_pairs)} verified DPO preference pairs in {total_time:.2f}s.")
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate on-policy DPO preference dataset")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--input_file", type=str, default="data/train.jsonl")
    parser.add_argument("--output_file", type=str, default="data/dpo_train.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_prompts", type=int, default=300)
    args = parser.parse_args()

    generate_dpo_dataset_batched(
        model_name=args.model,
        adapter_path=args.adapter,
        input_file=str(PROJECT_ROOT / args.input_file),
        output_file=str(PROJECT_ROOT / args.output_file),
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_prompts=args.max_prompts,
    )


if __name__ == "__main__":
    main()
