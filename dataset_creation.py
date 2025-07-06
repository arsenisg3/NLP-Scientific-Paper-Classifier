from datasets import Dataset
from transformers import PreTrainedTokenizer
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DatasetCreator:
    """Creates Hugging Face tokenized datasets from pandas DataFrames."""
    def __init__(self, tokenizer: PreTrainedTokenizer, text_column: str, label_column: str = 'label_id'):
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column

    def _tokenize_function(self, examples):
        """Internal tokenization function for use with dataset.map."""
        return self.tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

    def create_dataset(self, df: pd.DataFrame) -> Dataset:
        """Creates and tokenizes a Hugging Face Dataset."""

        hf_dataset = Dataset.from_dict({
            'text': df[self.text_column].tolist(),
            'labels': df[self.label_column].tolist()
        })

        # Apply tokenization
        tokenized_dataset = hf_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=['text']  # Remove original text column to save memory
        )

        # Set the format to PyTorch tensors for the Trainer
        tokenized_dataset.set_format("torch")

        logger.info(f"DatasetCreator: Created tokenized dataset with {len(tokenized_dataset)} examples.")
        return tokenized_dataset
