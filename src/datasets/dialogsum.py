from typing import Any, Dict, Optional

from transformers import AutoTokenizer

from datasets import DatasetDict, load_dataset
from src.utils.utils import format_prompt


def tokenize_function(example, tokenizer, prompt_template: str = None) -> dict:
    """
    Tokenize examples using the provided prompt template.
    Args:
        example: Dictionary containing 'dialogue' and 'summary'
        tokenizer: The tokenizer to use
        prompt_template: Optional prompt template. If None, uses default template.
    Returns:
        Tokenized example with input_ids and labels
    """
    prompts = [format_prompt(prompt_template, dialogue=dialogue) for dialogue in example['dialogue']]
    example['input_ids'] = tokenizer(
        prompts, padding="max_length", truncation=True
    ).input_ids
    example['labels'] = tokenizer(
        example['summary'], padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    
    return example

def subsample_dataset(tokenized_datasets: DatasetDict, subsample_factor: int)-> DatasetDict:
    """ Subsample the dataset by a given factor.
    Args:
        tokenized_datasets: The tokenized dataset to subsample.
        subsample_factor: The factor by which to subsample the dataset.
    Returns:
        Subsampled dataset.
    """ 
    return tokenized_datasets.filter(
        lambda example, index: index % subsample_factor == 0, 
        with_indices=True
    )