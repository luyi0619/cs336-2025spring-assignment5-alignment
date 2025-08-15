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
  return - raw_rewards_or_advantages * policy_log_probs;