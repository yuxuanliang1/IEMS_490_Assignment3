from typing import List, Dict, Any

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel


@torch.no_grad()
def generate_responses(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    prompts: List[str],
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    seq_len: int = 512,
    prompt_max_length: int = 384,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    device = next(model.parameters()).device
    out: List[Dict[str, Any]] = []
    model.eval()

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        prompt_lens = attn.sum(dim=1).tolist()

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for j in range(gen.size(0)):
            full_ids = gen[j]
            pl = int(prompt_lens[j])
            new_ids = full_ids[pl:]
            response_text = tokenizer.decode(new_ids, skip_special_tokens=True)

            # keep last seq_len tokens for logprob/KL
            if full_ids.numel() > seq_len:
                cut = full_ids.numel() - seq_len
                full_ids = full_ids[cut:]
                pl = max(0, pl - cut)

            full_attn = torch.ones_like(full_ids, device=full_ids.device)

            out.append(
                dict(
                    prompt=batch_prompts[j],
                    response_text=response_text,
                    input_ids=full_ids.detach().cpu(),
                    attention_mask=full_attn.detach().cpu(),
                    prompt_len=pl,
                )
            )

    return out
