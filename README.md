# IEMS_490_Assignment3
Assignment 3 for Northwestern University IEMS_490 Fall course, focusing on LLM.

This repository implements an end-to-end (small-scale) RLHF-style pipeline on the **Anthropic/hh-rlhf** dataset using **GPT-2** (`openai-community/gpt2`) as the base model. The project is organized to match the assignment structure:

- **Part 1**: Dataset exploration + preprocessing + **Reward Model** training (pairwise preference ranking)
- **Part 2**: Policy optimization with **SFT initialization**, then **PPO** and **GRPO**
- **Part 3**: **DPO** (Direct Preference Optimization)
- **Part 4**: Evaluation and analysis (reward scores / KL drift / **DeepSeek judge** win-rate)

All outputs (datasets, checkpoints, logs, plots) are saved under `artifacts/`.

---

## 0. Repository layout

Typical structure:
```
.
├── scripts/
│   ├── part1_1_explore_hhrlhf.py
│   ├── part1_1_prepare_hhrlhf.py
│   ├── part1_2_train_reward_model.py
│   ├── part1_2_error_analysis.py
│   ├── part2_1_sft_init_policy.py
│   ├── part2_1_train_ppo.py
│   ├── part2_1_sweep_ppo.py
│   ├── part2_2_train_grpo.py
│   └── part3_train_dpo.py
├── src/
│   └── data/
│       └── hhrlhf.py
├── artifacts/
│   └── (generated)
├── RUNNING.md
├── ANALYSIS.md
└── Dockerfile
```

---

## 1. Environment setup

### 1.1 Python + packages
Recommended:
- Python **3.10+** (tested with 3.11)
- `torch`, `transformers`, `datasets`, `numpy`, `pandas`, `tqdm`, `matplotlib`, `safetensors`
- For judge: `openai` (OpenAI SDK, used for DeepSeek)

Install:
```bash
pip install -U pip
pip install torch transformers datasets numpy pandas tqdm matplotlib safetensors openai
```

---
# 2. Lack of computing power
Since I only have an RTX 3060 graphics card, I encountered out-of-memory errors multiple times during the project.  Therefore, I had to reduce the data size, which might have prevented me from achieving optimal results. I hope you understand.



# 3. Part 1 — Dataset + Preprocessing + Reward Model

### 3.1 Part 1.1 Explore HH-RLHF
Explores token length statistics, tie/refusal rates, etc.
```bash
python -m scripts.part1_1_explore_hhrlhf --max_samples 2000
```
- ties_count ~ 5 / 2000
- prompt token p99 ~ 700+
- max prompt length can exceed 1024 (long tail)

### 3.2 Part 1.1 Prepare dataset (tokenize + train/val split)
```bash
python -m scripts.part1_1_prepare_hhrlhf --max_samples 5000 --max_length 512 --val_size 0.05 --seed 42
```

Outputs a processed DatasetDict under:
```
artifacts/part1_1/processed/hhrlhf_train_gpt2len512
```

### 3.3 Part 1.2 Train Reward Model (Task A)
Trains a GPT-2 **sequence classification** head to predict which response is preferred.
```bash
python -m scripts.part1_2_train_reward_model \
  --data_dir artifacts/part1_1/processed/hhrlhf_train_gpt2len512 \
  --out_dir artifacts/part1_2/reward_model_gpt2 \
  --epochs 1 --batch_size 8 --lr 1e-5 \
  --eval_every 2000 --log_every 50 --num_workers 0
```

Expected artifacts:
```
artifacts/part1_2/reward_model_gpt2/last/
artifacts/part1_2/reward_model_gpt2/logs.jsonl
```

Pilot reference (example):
- `val_acc ≈ 0.494`
- `val_loss ≈ 0.716`

### 3.4 Reward Model Error Analysis (Task B)
Exports misclassified examples for qualitative inspection.
```bash
python scripts/part1_2_error_analysis.py \
  --model_dir artifacts/part1_2/reward_model_gpt2/last \
  --data_dir artifacts/part1_1/processed/hhrlhf_train_gpt2len512 \
  --fp16 --batch_size 16 --seq_len 512
```

Outputs:
```
artifacts/part1_2/analysis_len512/
  ├── summary.json
  └── errors.jsonl   (>=20 required; typical export 50)
```

---

## 4. Part 2 — Policy Optimization (SFT → PPO / GRPO)

### 4.1 Part 2.1 SFT init policy
Fine-tunes GPT-2 on the “chosen” responses to initialize a policy for RL.
```bash
python -m scripts.part2_1_sft_init_policy \
  --data_dir artifacts/part1_1/processed/hhrlhf_train_gpt2len512 \
  --out_dir artifacts/part2_1/sft_gpt2 \
  --max_train_samples 2000 --max_val_samples 256 \
  --epochs 1 --batch_size 8 --lr 5e-5 \
  --max_length 512 --num_workers 0
```

Output:
```
artifacts/part2_1/sft_gpt2/last/
```

### 4.2 Part 2.1 PPO training (baseline)

```bash
python -m scripts.part2_1_train_ppo \
  --data_dir artifacts/part1_1/processed/hhrlhf_train_gpt2len512 \
  --reward_model_dir artifacts/part1_2/reward_model_gpt2/last \
  --init_policy_dir artifacts/part2_1/sft_gpt2/last \
  --out_dir artifacts/part2_1/ppo_run_baseline \
  --total_steps 200 --batch_size 4 \
  --max_new_tokens 64 --seq_len 512 --prompt_max_length 384 \
  --clip_ratio 0.2 --kl_coef 0.1 --ent_coef 0.01 \
  --lr 1e-6 --log_every 10 --save_every 100
```

### 4.3 PPO hyperparameter sweep
```bash
python -m scripts.part2_1_sweep_ppo \
  --base_out artifacts/part2_1/ppo_sweeps \
  --data_dir artifacts/part1_1/processed/hhrlhf_train_gpt2len512 \
  --reward_model_dir artifacts/part1_2/reward_model_gpt2/last \
  --init_policy_dir artifacts/part2_1/sft_gpt2/last \
  --total_steps 100 --batch_size 4 --max_new_tokens 64 --seed 42 \
  --clip_ratios 0.1 0.2 \
  --kl_coefs 0.05 0.1 0.2 \
  --lrs 5e-7 1e-6
```

### 4.4 Part 2.2 GRPO training (baseline)
GRPO can be memory-heavy (group rollouts). Start conservative:
```bash
python -m scripts.part2_2_train_grpo \
  --data_dir artifacts/part1_1/processed/hhrlhf_train_gpt2len512 \
  --reward_model_dir artifacts/part1_2/reward_model_gpt2/last \
  --init_policy_dir artifacts/part2_1/sft_gpt2/last \
  --out_dir artifacts/part2_2/grpo_run_baseline \
  --total_steps 200 --prompt_batch 1 --group_size 4 \
  --max_new_tokens 32 --seq_len 512 --prompt_max_length 384 \
  --lr 1e-6 --kl_coef 0.05 --ent_coef 0.01 \
  --log_every 10 --save_every 100 --num_workers 0
```

---

## 5. Part 3 — DPO

DPO is also memory-sensitive because it computes log-probs for both chosen/rejected.
Start conservative:
```bash
python -m scripts.part3_train_dpo \
  --data_dir artifacts/part1_1/processed/hhrlhf_train_gpt2len512 \
  --init_policy_dir artifacts/part2_1/sft_gpt2/last \
  --out_dir artifacts/part3/dpo_gpt2 \
  --beta 0.1 --seq_len 512 --batch_size 1 \
  --lr 5e-6 --epochs 1 \
  --max_train_samples 2000 --max_val_samples 256 \
  --log_every 50 --eval_every 500 --save_every 100 --num_workers 0
```

OOM fixes:
- `--seq_len 384/256`
- `--batch_size 1`
- disable fp16 with `--no_fp16` 

---

## 6. Part 4 — DeepSeek-as-Judge evaluation

A writeup scaffold is provided in `ANALYSIS.md`.

---

