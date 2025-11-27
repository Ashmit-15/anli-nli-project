"""
preprocessing.py

Preprocessing utilities shared between baseline and transformer models.
"""

from typing import Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def combine_premise_hypothesis(example: Dict[str, Any]) -> str:
    """
    Combine premise and hypothesis into a single string for classical ML models.
    """
    return example["premise"] + " [SEP] " + example["hypothesis"]


def get_tokenizer(model_name: str = "distilroberta-base") -> PreTrainedTokenizerBase:
    """
    Load a Hugging Face tokenizer by model name.
    """
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_batch(batch, tokenizer: PreTrainedTokenizerBase, max_length: int = 256):
    """
    Tokenize a batch of examples (premise, hypothesis) for transformer models.
    """
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
