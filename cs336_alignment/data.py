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
    docs = []
    with open(dataset_path, "r", encoding="utf-8") as f:
      for line in f:
        data = json.loads(line)
        docs.append(propmt_template.format(prompt=data["prompt"], response=data["response"]))
    if shuffle:
      random.shuffle(docs)

    text = "<|end_of_text|><|begin_of_text|>".join(docs)
    all_tokens = tokenizer.encode(text)


    length = (len(all_tokens)-1) // seq_length
    self.input_ids = torch.tensor(all_tokens[0:seq_length * length], dtype=torch.int32).reshape(length, seq_length)
    self.labels = torch.tensor(all_tokens[1:seq_length * length+1], dtype=torch.int32).reshape(length, seq_length)

  def __len__(self):
    return self.input_ids.shape[0]
  
  def __getitem__(self, i):
    return {"input_ids":self.input_ids[i], "labels":self.labels[i]}

def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return None
