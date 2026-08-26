"""
High-Throughput Evaluation Engine for SweetPea Post-Training.

Features:
1. Batched GPU generation (16-32 samples/batch) on RTX 4080
2. Parallel multiprocessing compiler grading
3. Comprehensive diagnostic tracking (Pass@1, AST validity, CryptoMiniSat synthesis, family breakdowns)
4. Outputs structured JSONL trials for statistical inference harness
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.compiler_grader import grade_sweetpea_code


def load_model_and_tokenizer(
    model_name_or_path: str,
    adapter_path: Optional[str] = None,
    torch_dtype: str = "bfloat16",
):
    print(f"Loading model: {model_name_or_path}...")
    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_batch_completions(
    model,
    tokenizer,
    prompts: List[str],
    system_prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> List[str]:
    formatted_prompts = []
    for p, sp_text in zip(prompts, system_prompts):
        messages = [
            {"role": "system", "content": sp_text},
            {"role": "user", "content": p},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(formatted)

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    do_sample = temperature > 1e-4
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode each completion
    completions = []
    input_lens = [len(ids) for ids in inputs["input_ids"]]
    for i, out_ids in enumerate(outputs):
        # Extract new tokens after input length
        # Note: left padding means new tokens start at inputs['input_ids'].shape[1]
        gen_tokens = out_ids[inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        completions.append(text)

    return completions


def _grade_worker(args: Tuple[dict, str, float]) -> dict:
    item, completion, gen_time = args
    grade = grade_sweetpea_code(completion, timeout_sec=4.0)

    return {
        "id": item.get("id", "unknown"),
        "task_family": item.get("task_family", "unknown"),
        "complexity": item.get("complexity", "unknown"),
        "prompt": item["prompt"],
        "generated_completion": completion,
        "passed": grade.passed,
        "score": grade.score,
        "syntax_valid": grade.syntax_valid,
        "ast_parsed": grade.ast_parsed,
        "block_created": grade.block_created,
        "synthesized": grade.synthesized,
        "num_trials": grade.num_trials,
        "error_type": grade.error_type,
        "error_message": grade.error_message,
        "gen_time_sec": gen_time,
        "exec_time_sec": grade.execution_time_sec,
    }


def evaluate_checkpoint(
    model,
    tokenizer,
    test_file: str,
    output_log_file: str,
    batch_size: int = 16,
    temperature: float = 0.0,
    max_samples: Optional[int] = None,
) -> Dict:
    dataset = load_dataset("json", data_files=test_file, split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    n = len(dataset)
    print(f"\nEvaluating on {n} examples (batch_size={batch_size}, temperature={temperature})...", flush=True)

    items = [item for item in dataset]
    all_completions = []
    t_gen_start = time.perf_counter()

    # 1. Batched GPU Generation
    print("Running Batched GPU Generation...", flush=True)
    for i in range(0, n, batch_size):
        batch_items = items[i : i + batch_size]
        prompts = [x["prompt"] for x in batch_items]
        systems = [x["messages"][0]["content"] for x in batch_items]

        t0 = time.perf_counter()
        completions = generate_batch_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            system_prompts=systems,
            temperature=temperature,
        )
        batch_dt = time.perf_counter() - t0
        dt_per_sample = batch_dt / len(batch_items)

        for item, comp in zip(batch_items, completions):
            all_completions.append((item, comp, dt_per_sample))

        print(f"  Generated {min(i + batch_size, n)}/{n} completions ({len(all_completions)/(time.perf_counter()-t_gen_start):.1f} samples/s)", flush=True)

    gen_time_total = time.perf_counter() - t_gen_start
    print(f"GPU Generation completed in {gen_time_total:.2f}s ({n/gen_time_total:.1f} samples/s)", flush=True)

    # 2. Parallel Multiprocessing Compiler Grading
    print("\nRunning Parallel Compiler Execution Grading...", flush=True)
    t_grade_start = time.perf_counter()
    results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_grade_worker, comp_tuple) for comp_tuple in all_completions]
        for f in as_completed(futures):
            results.append(f.result())

    # Sort back by original order
    id_to_idx = {items[i]["id"]: i for i in range(n)}
    results.sort(key=lambda x: id_to_idx.get(x["id"], 0))

    grade_time_total = time.perf_counter() - t_grade_start
    print(f"Compiler Grading completed in {grade_time_total:.2f}s ({n/grade_time_total:.1f} samples/s)", flush=True)

    # 3. Compute Summary Statistics
    pass_count = sum(1 for r in results if r["passed"])
    total_score = sum(r["score"] for r in results)
    pass_rate = pass_count / n if n > 0 else 0.0
    mean_score = total_score / n if n > 0 else 0.0

    family_stats = {}
    error_counts = {}
    for r in results:
        fam = r["task_family"]
        if fam not in family_stats:
            family_stats[fam] = {"total": 0, "passed": 0, "score_sum": 0.0}
        family_stats[fam]["total"] += 1
        if r["passed"]:
            family_stats[fam]["passed"] += 1
        family_stats[fam]["score_sum"] += r["score"]

        if r["error_type"]:
            error_counts[r["error_type"]] = error_counts.get(r["error_type"], 0) + 1

    total_time = gen_time_total + grade_time_total
    summary = {
        "num_eval_samples": n,
        "pass_rate_pass_at_1": pass_rate,
        "mean_stepped_score": mean_score,
        "total_eval_time_sec": total_time,
        "generation_time_sec": gen_time_total,
        "grading_time_sec": grade_time_total,
        "error_distribution": error_counts,
        "family_breakdown": {
            k: {
                "pass_rate": v["passed"] / v["total"] if v["total"] > 0 else 0.0,
                "mean_score": v["score_sum"] / v["total"] if v["total"] > 0 else 0.0,
                "n": v["total"],
            }
            for k, v in family_stats.items()
        },
    }

    # Save detailed JSONL trials log
    out_path = Path(output_log_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save summary JSON
    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS SUMMARY: {out_path.stem}")
    print(f"Pass@1 Accuracy: {pass_rate * 100:.2f}% ({pass_count}/{n})")
    print(f"Mean Stepped Score: {mean_score:.3f}")
    print("Family Breakdown:")
    for fam, stats in summary["family_breakdown"].items():
        print(f"  {fam:30s}: Pass@1 = {stats['pass_rate']*100:5.1f}% (N={stats['n']})")
    print("Error Distribution:", error_counts)
    print(f"Detailed trials saved to: {output_log_file}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints on SweetPea DSL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base model name/path")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--test_file", type=str, default="data/test_heldout.jsonl", help="Path to test JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output trials JSONL")
    parser.add_argument("--batch_size", type=int, default=16, help="GPU batch size")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of eval samples")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, adapter_path=args.adapter)
    test_path = str(PROJECT_ROOT / args.test_file)
    out_path = str(PROJECT_ROOT / args.output_file)

    evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        test_file=test_path,
        output_log_file=out_path,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
