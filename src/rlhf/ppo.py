# src/rlhf/ppo.py
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class PPOLossOutput:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_mean: torch.Tensor
    entropy_mean: torch.Tensor
    clip_frac: torch.Tensor


def ppo_loss(
    logp_new: torch.Tensor,      # [B] sum log prob of actions under current policy
    logp_old: torch.Tensor,      # [B] sum log prob under behavior (rollout) policy
    logp_ref: torch.Tensor,      # [B] sum log prob under reference policy (frozen)
    advantages: torch.Tensor,    # [B]
    entropy: torch.Tensor,       # [B] mean entropy over action tokens (or sum/avg; just be consistent)
    clip_ratio: float,
    kl_coef: float,
    ent_coef: float,
) -> PPOLossOutput:
    """
    PPO objective (sequence-level):
      L_clip = -E[min(r*A, clip(r,1-eps,1+eps)*A)]
      L = L_clip + kl_coef * KL(policy || ref) - ent_coef * Entropy

    KL here is approximated on sampled actions: KL ≈ E[logp_new - logp_ref].
    """
    # normalize advantages for stability
    adv = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    ratio = torch.exp(logp_new - logp_old)  # [B]
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -torch.mean(torch.minimum(unclipped, clipped))

    # approx KL on sampled actions
    kl = torch.mean(logp_new - logp_ref)

    # entropy bonus (maximize entropy -> subtract in loss)
    entropy_mean = torch.mean(entropy)

    total = policy_loss + kl_coef * kl - ent_coef * entropy_mean

    clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_ratio).float())

    return PPOLossOutput(
        total_loss=total,
        policy_loss=policy_loss.detach(),
        kl_mean=kl.detach(),
        entropy_mean=entropy_mean.detach(),
        clip_frac=clip_frac.detach(),
    )
