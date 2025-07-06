from constants import Constants as C

from transformers import (AutoModelForSequenceClassification, PreTrainedTokenizer, Trainer, TrainingArguments,
                          DataCollatorWithPadding, PreTrainedModel)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from torch import nn
import os
import logging
import torch
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ModelTrainer(Trainer):
    """Manages the training and evaluation processes of the model."""
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 train_dataset: Dataset, eval_dataset: Dataset, class_weights: torch.Tensor = None):

        self.class_weights = class_weights

        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir=C.OUTPUT_DIR,
            num_train_epochs=C.NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=C.PER_DEVICE_BATCH_SIZE,
            per_device_eval_batch_size=C.PER_DEVICE_BATCH_SIZE,
            dataloader_num_workers=C.DATALOADER_NUM_WORKERS,
            warmup_steps=C.WARMUP_STEPS,
            weight_decay=C.WEIGHT_DECAY,
            logging_dir=os.path.join(C.OUTPUT_DIR, 'logs'),  # Use subdir of output_dir
            logging_steps=C.LOGGING_STEPS,
            eval_strategy=C.EVAL_STRATEGY,
            save_strategy=C.SAVE_STRATEGY,
            save_steps=C.SAVE_STEPS,
            save_total_limit=C.SAVE_TOTAL_LIMIT,
            metric_for_best_model=C.METRIC_FOR_BEST_MODEL,
            greater_is_better=C.GREATER_IS_BETTER,
            load_best_model_at_end=C.LOAD_BEST_MODEL_AT_END,
            report_to=C.REPORT_TO,
            seed=C.SEED
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, Union[torch.Tensor, Any]],
                     return_outputs: bool = False, num_items_in_batch: Optional[int] = None)\
            -> Union[torch.Tensor, tuple[torch.Tensor, Dict[str, Any]]]:
        """Custom loss computation method to apply class weights."""

        if self.class_weights is None:
            raise ValueError("Class weights must be provided to ModelTrainer for weighted loss computation.")

        # Get labels from inputs and remove them before passing to model
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Ensure class weights are on the same device as the model's logits
        weights_on_device = self.class_weights.to(logits.device)

        # Instantiate CrossEntropyLoss with the weights
        loss_fct = nn.CrossEntropyLoss(weight=weights_on_device)

        # Calculate the loss: reshape logits and labels for CrossEntropyLoss
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))  # Use model.config.num_labels

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def compute_metrics(eval_pred):
        """Computes evaluation metrics (accuracy, F1, precision, recall)."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        # Macro
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        # Weighted
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        # Per class metrics
        p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        metrics_dict = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_weighted": f1_weighted,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
        }

        for i in range(len(f1_per_class)):
            metrics_dict[f"f1_class_{i}"] = f1_per_class[i]
            metrics_dict[f"precision_class_{i}"] = p_per_class[i]
            metrics_dict[f"recall_class_{i}"] = r_per_class[i]
            metrics_dict[f"support_class_{i}"] = support_per_class[i]

        return metrics_dict

    def train(self, *args, **kwargs):
        """Model training process."""
        logger.info(f"ModelTrainer: Model device before training: {self.model.device}")
        logger.info("--- ModelTrainer: Starting Fine-tuning ---")
        super().train()
        logger.info("--- ModelTrainer: Fine-tuning Complete! ---")

    def evaluate(self, test_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval") -> Dict[str, float]:
        """Evaluates the trained model on a test dataset."""
        logger.info("--- ModelTrainer: Evaluating on Test Set ---")

        dataset_to_evaluate = test_dataset if test_dataset is not None else self.eval_dataset

        results = super().evaluate(
            eval_dataset=dataset_to_evaluate,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        logger.info(f"ModelTrainer: Test Set Evaluation Results: {results}")
        return results
