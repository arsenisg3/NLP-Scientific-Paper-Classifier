from constants import Constants as C

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ModelLoader:
    """Handles loading the pre-trained tokenizer and model."""
    def __init__(self, num_labels: int, label2id: dict, id2label: dict):
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = id2label
        self.tokenizer = None
        self.model = None

    def load(self):
        """Loads the tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            C.MODEL_NAME,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        logger.info(f"ModelLoader: Tokenizer and model '{C.MODEL_NAME}' loaded.")
        logger.info(f"ModelLoader: Number of output labels: {self.model.config.num_labels}")
        logger.info(f"ModelLoader: Model config labels: {self.model.config.id2label}")
        return self.tokenizer, self.model

    def move_to_device(self, device: str = None):
        """Moves the model to the specified device (GPU if available, else CPU)."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        logger.info(f"Model moved to: {device}")
        return device
