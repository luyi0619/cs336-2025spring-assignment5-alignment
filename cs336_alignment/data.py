import os

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def get_packed_sft_dataset_impl(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    return None


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return None