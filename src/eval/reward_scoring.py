from typing import List

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel


@torch.no_grad()
def score_with_reward_model(
    tokenizer: PreTrainedTokenizerBase,
    reward_model: PreTrainedModel,
    prompts: List[str],
    responses: List[str],
    *,
    seq_len: int = 512,
    batch_size: int = 8,
) -> List[float]:
    device = next(reward_model.parameters()).device
    reward_model.eval()
    scores: List[float] = []

    for i in range(0, len(prompts), batch_size):
        bp = prompts[i : i + batch_size]
        br = responses[i : i + batch_size]
        texts = [p + r for p, r in zip(bp, br)]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        ).to(device)
        out = reward_model(**enc)
        logits = out.logits
        if logits.dim() == 2 and logits.size(-1) == 1:
            s = logits.squeeze(-1)
        else:
            s = logits[:, 0]
        scores.extend(s.detach().float().cpu().tolist())
    return scores
