"""
Supervised Fine-Tuning (SFT) Training Script for SweetPea DSL.

Features:
1. LoRA Parameter-Efficient Fine-Tuning (PEFT) on Qwen2.5-Coder
2. Custom ResponseOnlyDataCollator for strict assistant-only loss masking
3. Standard YAML configuration and seed control
4. Resumable checkpointing, evaluation tracking, and metrics export
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ResponseOnlyDataCollator:
    """Collates chat sequences and masks all user/prompt tokens with -100

    so that loss is computed strictly on assistant completions.
    """

    def __init__(self, tokenizer, response_template: str = "<|im_start|>assistant\n", max_length: int = 1024):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [e["text"] for e in examples]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()

        t_len = len(self.response_template_ids)
        for i in range(len(texts)):
            input_ids = batch["input_ids"][i].tolist()
            resp_start_idx = None

            # Find the starting index of the assistant response
            for j in range(len(input_ids) - t_len + 1):
                if input_ids[j : j + t_len] == self.response_template_ids:
                    resp_start_idx = j + t_len
                    break

            if resp_start_idx is not None:
                # Mask all tokens prior to assistant response
                labels[i, :resp_start_idx] = -100
            
            # Mask padding tokens
            labels[i, batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch


def prepare_chat_dataset(data_path: str, tokenizer):
    """Loads JSONL and applies chat template."""
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Run SFT training for SweetPea DSL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_qwen_1.5b.yaml",
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
    print("Starting SFT Training Pipeline")
    print(f"Model: {cfg['model']['name_or_path']}")
    print(f"Output Dir: {output_dir}")
    print(f"Torch Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 4. Prepare Datasets
    train_path = str(PROJECT_ROOT / cfg["data"]["train_file"])
    val_path = str(PROJECT_ROOT / cfg["data"]["val_file"])
    max_seq_len = cfg["data"]["max_seq_length"]

    train_dataset = prepare_chat_dataset(train_path, tokenizer)
    val_dataset = prepare_chat_dataset(val_path, tokenizer)
    print(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")

    # 5. Response-Only Masking Collator
    response_template = cfg["data"]["response_template"]
    collator = ResponseOnlyDataCollator(
        tokenizer=tokenizer,
        response_template=response_template,
        max_length=max_seq_len,
    )

    # 6. Training Arguments
    t_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        remove_unused_columns=False,
        seed=t_cfg["seed"],
        report_to=t_cfg["report_to"],
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print("\nStarting SFT training loop on GPU...", flush=True)
    train_result = trainer.train()
    
    print("\nTraining completed successfully!", flush=True)
    print(f"Final Train Loss: {train_result.training_loss:.4f}", flush=True)

    # 8. Save best adapter & tokenizer
    best_adapter_dir = f"{output_dir}/final_adapter"
    trainer.save_model(best_adapter_dir)
    tokenizer.save_pretrained(best_adapter_dir)
    print(f"Saved best LoRA adapter to: {best_adapter_dir}", flush=True)

    # 9. Save metrics JSON
    metrics_path = Path(output_dir) / "sft_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Saved training log history to: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
