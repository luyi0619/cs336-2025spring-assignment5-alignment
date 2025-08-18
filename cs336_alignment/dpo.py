import torch
from transformers import PreTrainedTokenizerBase
from sft import get_response_log_probs

propmt_template = \
"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""  

def get_logs_prob(lm: torch.nn.Module, prompt: list[int]):
  input_ids = torch.tensor(prompt[:-1], dtype=torch.int64)
  labels = torch.tensor(prompt[1:], dtype=torch.int64)
  log_probs = get_response_log_probs(lm, input_ids, labels)['log_probs']
  return torch.sum(log_probs)


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    y_win = propmt_template.format(prompt=prompt, response=response_chosen) 
    y_loss = propmt_template.format(prompt=prompt, response=response_rejected)
    y_win = tokenizer.encode(y_win) + [tokenizer.eos_token_id]
    y_loss = tokenizer.encode(y_loss) + [tokenizer.eos_token_id]
    y_win_logs_prob = get_logs_prob(lm, y_win)
    y_loss_logs_prob = get_logs_prob(lm, y_loss)
    y_ref_win_logs_prob = get_logs_prob(lm_ref, y_win)
    y_ref_loss_logs_prob = get_logs_prob(lm_ref, y_loss)
    return -torch.nn.functional.logsigmoid(beta*((y_win_logs_prob - y_loss_logs_prob)-(y_ref_win_logs_prob - y_ref_loss_logs_prob)))
