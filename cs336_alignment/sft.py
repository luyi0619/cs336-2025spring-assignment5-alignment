import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
  assert len(prompt_strs) == len(output_strs)


  prompt_and_output_list = []
  for i, (prompt_str, output_str) in enumerate(zip(prompt_strs, output_strs, strict=True)):
    prompt_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt_str))
    output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output_str))
    prompt_and_output_list.append((prompt_ids, output_ids))

  batch_size = len(prompt_and_output_list)
  max_element = max(prompt_and_output_list, key = lambda x: len(x[0]) + len(x[1]))
  seq = len(max_element[0]) + len(max_element[1])

  input_ids = torch.zeros(batch_size, seq - 1, dtype=torch.int32)
  labels = torch.zeros(batch_size, seq - 1, dtype=torch.int32)
  response_mask = torch.zeros(batch_size, seq - 1, dtype=torch.bool)

  for i, prompt_and_output in enumerate(prompt_and_output_list):
    prompt_len = len(prompt_and_output[0])
    output_len = len(prompt_and_output[1])
    sequence = prompt_and_output[0] + prompt_and_output[1] + [tokenizer.pad_token_id] * (seq - prompt_len - output_len)
    input_ids[i, :seq-1] = torch.tensor(sequence[:-1])
    labels[i, :seq-1] = torch.tensor(sequence[1:])
    response_mask[i, prompt_len-1:prompt_len + output_len - 1 ] = True
  return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
  # [b s v] -> [b s]
  #
  # H(p) =− \sum p(x) log p(x), log p(x) = log ( exp(x) / \sum exp(i)) = x - logsumexp(i)
  x = logits
  logsumexp = torch.logsumexp(logits, dim = -1, keepdim = True)
  logpx = logits - logsumexp
  px = torch.exp(logpx)
  return -torch.sum(px * logpx, dim = -1) 


def get_response_log_probs(
  model: PreTrainedModel,
  input_ids: torch.Tensor,
  labels: torch.Tensor,
  return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
  # input_ids, labels [b s]
  logits = model(input_ids).logits # [b s v]
  log_softmax = torch.nn.functional.log_softmax(logits, dim = -1)
  log_probs = torch.gather(log_softmax, -1, labels.unsqueeze(-1)).squeeze(-1)
  result = {'log_probs': log_probs}
  if return_token_entropy:
    result['token_entropy'] = compute_entropy(logits)
  return result


def masked_normalize(
  tensor: torch.Tensor,
  mask: torch.Tensor,
  dim: int | None = None,
  normalize_constant: float = 1.0,
) -> torch.Tensor:
  return torch.sum(tensor * mask, dim = dim) / normalize_constant


def sft_microbatch_train_step(
  policy_log_probs: torch.Tensor,
  response_mask: torch.Tensor,
  gradient_accumulation_steps: int,
  normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  loss = -masked_normalize(tensor = policy_log_probs, mask = response_mask, dim = -1, normalize_constant = normalize_constant).mean() / gradient_accumulation_steps
  loss.backward()
  return (loss, {})
