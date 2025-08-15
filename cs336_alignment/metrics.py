import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Any, Callable, Literal
import re

def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
  match = re.search(r"The correct answer is ([A-Z])", model_output)
  if match:
    answer = match.group(1)
    return answer
  return None


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
  words = model_output.split()
  for word in reversed(words):
    if word.isdigit():
      return word
  return None