"""
data_loading.py

Utility functions to load the ANLI Round 2 dataset.
"""

from datasets import load_dataset
from typing import Tuple
from datasets.arrow_dataset import Dataset

LABEL2NAME = {0: "entailment", 1: "neutral", 2: "contradiction"}
NAME2LABEL = {v: k for k, v in LABEL2NAME.items()}


def load_anli_r2() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load ANLI Round 2 (train_r2, dev_r2, test_r2) from Hugging Face.

    Returns
    -------
    train : datasets.Dataset
    val   : datasets.Dataset
    test  : datasets.Dataset
    """
    ds = load_dataset("facebook/anli", "plain_text")
    train = ds["train_r2"]
    val = ds["dev_r2"]
    test = ds["test_r2"]
    return train, val, test
