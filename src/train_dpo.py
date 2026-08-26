"""
Direct Preference Optimization (DPO) Training Script for SweetPea DSL.

Optimizes language model policy directly from (chosen, rejected) preference pairs
derived from CryptoMiniSat execution outcomes without requiring a separate reward model.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_dpo_dataset(data_path: str, tokenizer):
    """Loads JSONL and formats chosen/rejected messages into chat strings."""
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_dpo(example):
        # Format prompt
        prompt_msgs = [
            {"role": "system", "content": example["chosen"][0]["content"]},
            {"role": "user", "content": example["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Format chosen and rejected responses
        chosen_text = example["chosen"][2]["content"] + tokenizer.eos_token
        rejected_text = example["rejected"][2]["content"] + tokenizer.eos_token

        return {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Run DPO training for SweetPea DSL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo_qwen_1.5b.yaml",
        help="Path to YAML training configuration",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    output_dir = args.output_dir or cfg["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Starting DPO Training Pipeline (Phase 2)")
    print(f"Model: {cfg['model']['name_or_path']}")
    print(f"Output Dir: {output_dir}")
    print(f"Beta: {cfg['dpo']['beta']}")
    print(f"Torch Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60, flush=True)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 2. Load Model
    torch_dtype = getattr(torch, cfg["model"]["torch_dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name_or_path"],
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # 3. Configure LoRA
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
        task_type="CAUSAL_LM",
    )

    # 4. Prepare Preference Datasets
    train_path = str(PROJECT_ROOT / cfg["data"]["train_file"])
    val_path = str(PROJECT_ROOT / cfg["data"].get("val_file", cfg["data"]["train_file"]))

    train_dataset = prepare_dpo_dataset(train_path, tokenizer)
    val_dataset = prepare_dpo_dataset(val_path, tokenizer)
    print(f"Loaded {len(train_dataset)} training preference pairs.")

    # 5. DPO Config
    t_cfg = cfg["training"]
    dpo_args = DPOConfig(
        output_dir=output_dir,
        beta=float(cfg["dpo"]["beta"]),
        max_length=int(cfg["dpo"].get("max_length", 1024)),
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=float(t_cfg["learning_rate"]),
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_steps=int(t_cfg.get("warmup_steps", 10)),
        weight_decay=float(t_cfg["weight_decay"]),
        logging_steps=t_cfg["logging_steps"],
        eval_strategy=t_cfg["eval_strategy"],
        eval_steps=t_cfg["eval_steps"],
        save_strategy=t_cfg["save_strategy"],
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        greater_is_better=t_cfg["greater_is_better"],
        bf16=t_cfg["bf16"],
        fp16=t_cfg["fp16"],
        seed=t_cfg["seed"],
        report_to=t_cfg["report_to"],
        remove_unused_columns=False,
    )

    # 6. DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Automatically uses frozen base weights when peft_config is provided
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    print("\nStarting DPO optimization on GPU...", flush=True)
    dpo_result = dpo_trainer.train()

    print("\nDPO Training completed successfully!", flush=True)
    print(f"Final Train Loss: {dpo_result.training_loss:.4f}", flush=True)

    # 7. Save best DPO adapter
    best_adapter_dir = f"{output_dir}/final_adapter"
    dpo_trainer.save_model(best_adapter_dir)
    tokenizer.save_pretrained(best_adapter_dir)
    print(f"Saved best DPO LoRA adapter to: {best_adapter_dir}", flush=True)

    # 8. Save metrics log
    metrics_path = Path(output_dir) / "dpo_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(dpo_trainer.state.log_history, f, indent=2)
    print(f"Saved DPO training metrics to: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
