import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


@torch.no_grad()
def token_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,          # [B, T]
    attention_mask: torch.Tensor,     # [B, T]
) -> torch.Tensor:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [B, T, V]
    logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
    next_ids = input_ids[:, 1:]  # [B, T-1]
    logp = torch.gather(logp_all, dim=-1, index=next_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    return logp


@torch.no_grad()
def kl_on_samples(
    policy: PreTrainedModel,
    ref: PreTrainedModel,
    input_ids: torch.Tensor,          # [B, T]
    attention_mask: torch.Tensor,     # [B, T]
    prompt_lens: torch.Tensor,        # [B]
) -> torch.Tensor:
    logp_pi = token_logprobs(policy, input_ids, attention_mask)    # [B, T-1]
    logp_ref = token_logprobs(ref, input_ids, attention_mask)      # [B, T-1]

    B, Tm1 = logp_pi.shape
    idx = torch.arange(Tm1, device=logp_pi.device).unsqueeze(0).expand(B, -1)
    start = (prompt_lens.to(logp_pi.device) - 1).clamp(min=0).unsqueeze(1)
    mask = idx >= start
    mask = mask & attention_mask.to(logp_pi.device)[:, 1:].bool()

    diff = (logp_pi - logp_ref) * mask.float()
    denom = mask.float().sum(dim=1).clamp(min=1.0)
    return diff.sum(dim=1) / denom
