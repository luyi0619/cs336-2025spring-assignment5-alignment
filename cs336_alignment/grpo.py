import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Any, Callable, Literal

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
  rollout_batch_size = len(rollout_responses)
  assert rollout_batch_size % group_size == 0
  n_prompts_per_rollout_batch = rollout_batch_size // group_size
  rewards = [ reward_fn(response, truth)['reward'] for response, truth in zip(rollout_responses, repeated_ground_truths) ]
  raw_rewards = torch.tensor(rewards, dtype=torch.float32).reshape(n_prompts_per_rollout_batch, group_size)
  mean = torch.mean(raw_rewards, dim = -1, keepdim = True)
  advantages = raw_rewards - mean
  if normalize_by_std:
    advantages = advantages / (torch.std(raw_rewards, dim = -1, keepdim=True) + advantage_eps)
  return (advantages.reshape(-1, ), raw_rewards.reshape(-1, ), {})


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
  # policy_log_probs [b s]
  # raw_rewards_or_advantages (b, )
  return - raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clip_ratio = torch.clip(ratio, min = 1 - cliprange, max = 1 + cliprange)
    loss = -torch.min(advantages * ratio, advantages * clip_ratio)
    return (loss, {})


def compute_policy_gradient_loss(
  policy_log_probs: torch.Tensor,
  loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
  raw_rewards: torch.Tensor | None= None,
  advantages: torch.Tensor | None= None,
  old_log_probs: torch.Tensor | None= None,
  cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  if loss_type == 'no_baseline':
    return (compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {})
  elif loss_type == 'reinforce_with_baseline':
    return (compute_naive_policy_gradient_loss(advantages, policy_log_probs), {})
  else:
    return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
