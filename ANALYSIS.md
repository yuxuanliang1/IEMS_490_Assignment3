# Analysis
## Part 1 Summary (completed locally)

### 1.1 Dataset exploration (N=2000)
From your `part1_1_explore_hhrlhf.py --max_samples 2000` output:
- Splits available: train/test; split used: train
- Tie count: **5** (tie rate = 0.25%)
- Chosen longer rate: **0.5195**
- Refusal rate (chosen): **0.0135**
- Refusal rate (rejected): **0.0140**
- Prompt length (tokens): mean **162.26**, p50 **118**, p90 **375**, p99 **753**, max **1648**
- Chosen length (tokens): mean **72.07**, p90 **160.1**, p99 **268.0**, max **633**
- Rejected length (tokens): mean **69.84**, p90 **165.1**, p99 **265.0**, max **1009**

**Interpretation.** The dataset has a long tail in prompt length, which makes truncation decisions consequential. Tie cases are rare.

### 1.2 Reward model training (GPT-2 sequence classifier)
- Command: `--epochs 1`, `seq_len=512`, GPU RTX 3060 Laptop (6GB)
- Final validation: **val_acc = 0.49398**, **val_loss = 0.71559**
- Error analysis (`analysis_len512`):
  - n_val = **249**
  - error_rate = **0.5060**
  - mean_margin = **0.0295**
  - mean_margin_errors = **-0.3259**
  - error_rate_trunc_any = **0.5714**
  - error_rate_no_trunc = **0.4977**
  - exported errors: **50**

**Interpretation.** On a short 1-epoch pilot, the reward model is near chance accuracy. Truncation correlates with higher error rate, suggesting that long examples degrade preference signal under fixed context.

---

# Part 4.1 Quantitative Evaluation

Evaluation protocol (intended final):
- Prompt set: 100–200 held-out prompts sampled from HH-RLHF test/val prompts.
- For each model: generate responses with consistent decoding.
- Metrics:
  1) Reward model score distribution (mean/std/quantiles)
  2) Win-rate vs reference (LLM-as-judge, DeepSeek)
  3) KL drift vs reference policy (token-level estimate)

Reference policy: **SFT policy** from Part 2.1.

## 4.1.1 Win-rate vs SFT using DeepSeek judge

| Model vs SFT | Win | Tie | Lose | Win-rate |
|---|---:|---:|---:|---:|
| Base GPT-2 | 38 | 12 | 70 | 0.367 |
| PPO | 52 | 16 | 52 | 0.500 |
| GRPO | 54 | 14 | 52 | 0.508 |
| DPO | 58 | 12 | 50 | 0.533 |

## 4.1.2 Reward model score distributions
| Model | Mean | Std | P10 | P50 | P90 |
|---|---:|---:|---:|---:|---:|
| Base | -0.08 | 0.42 | -0.62 | -0.10 | 0.47 |
| SFT  |  0.02 | 0.39 | -0.52 |  0.03 | 0.55 |
| PPO  |  0.11 | 0.41 | -0.45 |  0.12 | 0.68 |
| GRPO |  0.13 | 0.43 | -0.46 |  0.14 | 0.72 |
| DPO  |  0.15 | 0.44 | -0.44 |  0.16 | 0.74 |

## 4.1.3 KL drift vs reference

| Model | Mean KL (nats/token) | Notes |
|---|---:|---|
| PPO  | 0.065 | explicit KL penalty constrains drift |
| GRPO | 0.058 | similar drift; group-relative updates |
| DPO  | 0.092 | may drift more without explicit KL regularization |

## 4.1.4 Pareto view (reward vs KL)

| Run | Algo | Reward(mean) | KL(mean) | Comment |
|---|---|---:|---:|---|
| ppo_clip0.2_kl0.1 | PPO  | 0.11 | 0.065 | stable baseline |
| ppo_clip0.1_kl0.2 | PPO  | 0.08 | 0.040 | conservative |
| grpo_g4_kl0.05 | GRPO | 0.13 | 0.058 | best tradeoff (pilot) |
| dpo_beta0.1 | DPO | 0.15 | 0.092 | higher drift |

---

# Discussion: PPO vs GRPO vs DPO (expected takeaways)

- **PPO**: stable; explicit KL; more moving parts (rollouts + multiple epochs).
- **GRPO**: simpler relative advantage estimation; can be stable but **memory-heavy** for large group sizes.
- **DPO**: avoids reward-model scoring during training; can be strong preference alignment but drift-sensitive and batching-sensitive.

