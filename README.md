# Verifiable Post-Training & Statistical Evaluation Harness

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace PEFT & TRL](https://img.shields.io/badge/%F0%9F%A4%97%20TRL-DPO%20%26%20SFT-yellow)](https://github.com/huggingface/trl)
[![SweetPea 0.2.14](https://img.shields.io/badge/SweetPea-SAT%20DSL-purple.svg)](https://github.com/sweetpea-org/sweetpea)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end post-training pipeline and evaluation harness that grounds open-weight language model synthesis in **verifiable compiler execution** (CryptoMiniSat SAT solver) and evaluates treatment effects using **cluster-robust statistical inference**.

---

## 📊 Core Empirical Benchmark

Every checkpoint was evaluated on the identical held-out test suite ($N = 150$ experimental specifications across 7 distinct task families), graded strictly by the **SweetPea CryptoMiniSat execution compiler**:

| Model Checkpoint | Compiler Pass@1 | Mean Stepped Score | $\Delta$ vs Base (95% Cluster CI) | Permutation $p$-value | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Base:** `Qwen2.5-Coder-1.5B-Instruct` | **0.00%** (0/150) | 0.283 | — | — | Baseline (124 ValueErrors) |
| **SFT (LoRA):** `checkpoints/sft_qwen_1.5b` | **100.00%** (150/150) | 1.000 | **+100.0%** [100.0%, 100.0%] | **$p < 0.0001$** | **100% SAT execution ($\checkmark$)** |
| **DPO (from Base):** `checkpoints/dpo_qwen_1.5b` | **0.00%** (0/150) | 0.300 | +0.0% [0.0%, 0.0%] | $p = 1.0000$ | Fails without SFT warm-start |

---

## 🎯 Key Technical Highlights

1. **Verifiable Ground-Truth Rewards ($0/1$):**
   Unlike subjective human annotators or biased LLM-as-a-judge evaluators, correctness is verified deterministically via AST validation and SAT constraint satisfaction in an isolated sandbox.
2. **Post-Training Recipe Space:**
   - **SFT:** LoRA fine-tuning ($r=16, \alpha=32$) on all linear projections (Attention + MLP) with custom `ResponseOnlyDataCollator` enforcing $-100$ loss masking on prompts.
   - **DPO:** Direct Preference Optimization ($\beta=0.1$, sigmoid loss) on on-policy compiler-verified failure pairs $(x, y_w, y_l)$, achieving implicit reward margins of $+14.10$.
3. **Statistical Rigour vs. Naive Evals:**
   Replaces un-clustered $N=50$ sample-and-pick-winner practices with:
   - **Cluster-Robust Bootstrap:** Resampling $K=7$ task families with replacement ($B=5,000$) to account for intra-family prompt correlation.
   - **Exact Paired Permutation Testing:** 10,000 sign flips for distribution-free hypothesis testing ($p < 0.0001$).
   - **Power Analysis:** Pre-calculated Minimum Detectable Effect ($\text{MDE} = \pm 2.29\%$ at $\alpha=0.05, 1-\beta=0.80$).

---

## 📁 Repository Structure

```text
.
├── configs/
│   ├── sft_qwen_1.5b.yaml               # Reproducible SFT recipe (LoRA r=16, alpha=32, BF16)
│   └── dpo_qwen_1.5b.yaml               # Reproducible DPO recipe (beta=0.1, sigmoid loss)
├── data/
│   ├── sweetpea_templates.py            # Parametric cognitive science experiment generator
│   ├── generate_dataset.py              # Parallel compiler-verified dataset builder
│   ├── train.jsonl (600 samples)        # Formatted ChatML SFT training set
│   ├── val.jsonl (100 samples)          # SFT validation set
│   ├── test_heldout.jsonl (150 samples) # Held-out test benchmark
│   └── dpo_train.jsonl (300 pairs)      # On-policy compiler-verified preference pairs
├── src/
│   ├── compiler_grader.py               # Subprocess sandbox with CryptoMiniSat AST checks
│   ├── train_sft.py                     # SFT trainer with custom response-only token masking
│   ├── generate_dpo_pairs.py            # On-policy sampler & compiler preference pair extractor
│   ├── train_dpo.py                     # Direct Preference Optimization trainer (TRL + PEFT)
│   ├── evaluate.py                      # Batched GPU inference + parallel compiler execution grader
│   └── statistical_analysis.py          # Cluster bootstrap, permutation tests, MDE & simulation
├── reports/
│   ├── phase-1-sft-mechanics-deep-dive.md              # Detailed SFT theory & practitioner guide
│   ├── phase-2-dpo-preference-optimization-deep-dive.md # DPO math, reward dynamics & findings
│   ├── phase-3-statistical-rigour-and-evaluation-deep-dive.md # Statistical inference & power guide
│   └── statistical_evaluation_report.md                # Publication-style benchmark report
└── eval_results/
    ├── base_model_eval.jsonl            # Baseline raw trials log
    ├── sft_model_eval.jsonl             # SFT raw trials log
    └── dpo_model_eval.jsonl             # DPO raw trials log
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Mnzrabusham/verifiable-post-training.git
cd verifiable-post-training

# Create virtual environment with uv or venv
uv venv .venv --python 3.11
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers peft trl datasets bitsandbytes accelerate sweetpea scipy
```

### 2. Generate Verified Datasets
```bash
python data/generate_dataset.py
```

### 3. Run Supervised Fine-Tuning (SFT)
```bash
python src/train_sft.py --config configs/sft_qwen_1.5b.yaml
```

### 4. Run Direct Preference Optimization (DPO)
```bash
# Generate on-policy preference pairs
python src/generate_dpo_pairs.py --input_file data/train.jsonl --output_file data/dpo_train.jsonl

# Train DPO adapter
python src/train_dpo.py --config configs/dpo_qwen_1.5b.yaml
```

### 5. Evaluate Checkpoints on Held-Out Benchmark
```bash
# Evaluate Base Model
python src/evaluate.py --model "Qwen/Qwen2.5-Coder-1.5B-Instruct" --output_file eval_results/base_model_eval.jsonl

# Evaluate SFT Checkpoint
python src/evaluate.py --model "Qwen/Qwen2.5-Coder-1.5B-Instruct" --adapter checkpoints/sft_qwen_1.5b/final_adapter --output_file eval_results/sft_model_eval.jsonl
```

### 6. Run Statistical Power & Bootstrap Inference
```bash
python src/statistical_analysis.py
```

---

## 📚 Technical Deep Dives

For exhaustive practitioner notes, mathematical derivations, and interview talking points, refer to the reports:
- [**Phase 1 Deep Dive: SFT Mechanics & Recipe Space**](reports/phase-1-sft-mechanics-deep-dive.md)
- [**Phase 2 Deep Dive: DPO Preference Optimization & Verifiable Data**](reports/phase-2-dpo-preference-optimization-deep-dive.md)
- [**Phase 3 Deep Dive: Statistical Rigour, Power Analysis, & Noise Floors**](reports/phase-3-statistical-rigour-and-evaluation-deep-dive.md)

---

## 👤 Author
**Monthir Ali, PhD**  
Maintainer of `sweetpea` | Author of `eval-harness`  
[GitHub Profile](https://github.com/Mnzrabusham)
