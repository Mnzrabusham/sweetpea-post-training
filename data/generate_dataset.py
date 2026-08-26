"""
Fast Dataset Generator & Compiler Verifier for SweetPea Post-Training.

Generates prompt-code pairs across diverse experimental paradigms,
verifies each sample directly with the SweetPea SAT solver,
and formats data into standard ChatML / HuggingFace format with cluster metadata.
"""

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.sweetpea_templates import GENERATORS, generate_sample

SYSTEM_PROMPT = (
    "You are an expert in experimental design and cognitive science using SweetPea, "
    "a domain-specific language for synthesizing randomized constrained factorial designs in Python.\n"
    "When given an experimental specification, produce complete, valid, executable Python code "
    "that imports sweetpea as sp, constructs all factors, derivations, and constraints, builds the "
    "CrossBlock, and synthesizes trial sequences using synthesize_trials."
)


def verify_code_in_memory(code: str) -> tuple[bool, int, int, float]:
    """Executes code in isolated local namespace, suppressing stdout."""
    locs = {}
    t0 = time.perf_counter()
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(code, locs, locs)
            
        experiments = locs.get("experiments")
        if experiments and isinstance(experiments, list) and len(experiments) > 0:
            first_exp = experiments[0]
            num_factors = len(first_exp.keys()) if isinstance(first_exp, dict) else 0
            num_trials = len(list(first_exp.values())[0]) if isinstance(first_exp, dict) and num_factors > 0 else 0
            elapsed = time.perf_counter() - t0
            return True, num_factors, num_trials, elapsed
        return False, 0, 0, time.perf_counter() - t0
    except Exception:
        return False, 0, 0, time.perf_counter() - t0


def generate_verified_dataset(
    num_samples: int,
    base_seed: int,
    split_name: str,
    progress_interval: int = 100
) -> list:
    print(f"Generating {num_samples} verified samples for '{split_name}' (seed={base_seed})...", flush=True)
    rng = random.Random(base_seed)
    dataset = []
    attempts = 0
    t0 = time.perf_counter()

    while len(dataset) < num_samples:
        attempts += 1
        example = generate_sample(rng)
        
        ok, num_factors, num_trials, exec_time = verify_code_in_memory(example.code)
        if not ok:
            continue

        sample_id = f"{split_name}_{len(dataset):05d}"
        formatted_code = f"```python\n{example.code}\n```"

        record = {
            "id": sample_id,
            "task_family": example.task_family,
            "complexity": example.complexity,
            "constraint_types": example.constraint_types,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example.prompt},
                {"role": "assistant", "content": formatted_code}
            ],
            "prompt": example.prompt,
            "canonical_code": example.code,
            "meta": {
                "num_trials": num_trials,
                "num_factors": num_factors,
                "exec_time_sec": exec_time
            }
        }
        dataset.append(record)

        if len(dataset) % 50 == 0 or len(dataset) == num_samples:
            elapsed = time.perf_counter() - t0
            rate = len(dataset) / elapsed if elapsed > 0 else 0
            print(f"  [{split_name}] {len(dataset)}/{num_samples} verified ({rate:.1f} samples/s)", flush=True)

    return dataset


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate and verify SweetPea dataset")
    parser.add_argument("--num_train", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=150, help="Number of validation samples")
    parser.add_argument("--num_test", type=int, default=250, help="Number of held-out test samples")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data"

    train_data = generate_verified_dataset(args.num_train, base_seed=args.seed, split_name="train")
    val_data = generate_verified_dataset(args.num_val, base_seed=args.seed + 10000, split_name="val")
    test_data = generate_verified_dataset(args.num_test, base_seed=args.seed + 20000, split_name="test_heldout")

    save_jsonl(train_data, data_dir / "train.jsonl")
    save_jsonl(val_data, data_dir / "val.jsonl")
    save_jsonl(test_data, data_dir / "test_heldout.jsonl")

    # Summary
    from collections import Counter
    train_families = Counter(d["task_family"] for d in train_data)
    test_families = Counter(d["task_family"] for d in test_data)
    print("\n" + "="*50)
    print("Dataset Distribution Summary:")
    print("Train task families:", dict(train_families))
    print("Test task families:", dict(test_families))
    print("="*50)


if __name__ == "__main__":
    main()
