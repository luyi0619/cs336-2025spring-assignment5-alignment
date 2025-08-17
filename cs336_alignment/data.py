import json
import os
import random
import torch

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


propmt_template = \
"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""   

class SftDataset(Dataset):
  def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_path: str | os.PathLike, seq_length: int, shuffle: bool):
    self.seq_length = seq_length

    docs = []
    with open(dataset_path, "r", encoding="utf-8") as f:
      for line in f:
        data = json.loads(line)
        docs.append(propmt_template.format(prompt=data["prompt"], response=data["response"]))
    
    if shuffle:
      random.shuffle(docs)

    text = "<|end_of_text|><|begin_of_text|>".join(docs)
    self.all_tokens = tokenizer.encode(text)

  def __len__(self):
    return (len(self.all_tokens) - 1) // self.seq_length
  
  def __getitem__(self, i):
    input_ids = self.all_tokens[i * self.seq_length: (i+1) *  self.seq_length]
    labels = self.all_tokens[i * self.seq_length + 1: (i+1) *  self.seq_length + 1]
    return {"input_ids":torch.tensor(input_ids), "labels":torch.tensor(labels)}

def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return None