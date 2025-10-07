#!/usr/bin/env python3
"""Evaluate or run predictions with the LoRA ModernBERT prompt injection detector.

This script loads the LoRA adapters from ``outputs/modernbert-lora/final`` and
exposes two Fire commands:

* ``predict`` – run a single prompt through the classifier.
* ``evaluate`` – score an entire dataset saved with ``Dataset.save_to_disk``.

Example usage::

    python scripts/eval.py predict "Ignore previous instructions"
    python scripts/eval.py evaluate --dataset_path data/processed/dataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fire
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS: Tuple[str, str] = ("normal", "attack")
DEFAULT_MODEL_REPO = "answerdotai/ModernBERT-large"
DEFAULT_MODEL_PATH = "outputs/modernbert-lora/final"
DEFAULT_DATASET_PATH = "data/processed/dataset"


def _resolve_device(device_override: Optional[str] = None) -> torch.device:
    """Pick a torch.device, preferring overrides, then CUDA, MPS, and CPU."""

    if device_override is not None:
        return torch.device(device_override)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _ensure_columns(dataset: Dataset, text_column: Optional[str], label_column: Optional[str]) -> Tuple[str, str]:
    """Select the best matching text and label columns present in the dataset."""

    candidate_texts = [
        text_column,
        "text",
        "prompt",
        "Prompt",
        "context",
    ]
    candidate_labels = [
        label_column,
        "labels",
        "label",
        "Label",
    ]

    for candidate in candidate_texts:
        if candidate and candidate in dataset.column_names:
            chosen_text = candidate
            break
    else:
        raise ValueError(
            f"Could not locate a text column in dataset; looked for {candidate_texts}."
        )

    for candidate in candidate_labels:
        if candidate and candidate in dataset.column_names:
            chosen_label = candidate
            break
    else:
        raise ValueError(
            f"Could not locate a label column in dataset; looked for {candidate_labels}."
        )

    return chosen_text, chosen_label


def _batched(iterable: Sequence[int], batch_size: int) -> Iterable[range]:
    total = len(iterable)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield range(start, end)


def _compute_binary_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length.")

    total = len(labels)
    correct = sum(int(p == l) for p, l in zip(predictions, labels))
    tp = sum(int(p == 1 and l == 1) for p, l in zip(predictions, labels))
    tn = sum(int(p == 0 and l == 0) for p, l in zip(predictions, labels))
    fp = sum(int(p == 1 and l == 0) for p, l in zip(predictions, labels))
    fn = sum(int(p == 0 and l == 1) for p, l in zip(predictions, labels))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    precision_neg = tn / (tn + fn) if (tn + fn) else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = (
        2 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if (precision_neg + recall_neg)
        else 0.0
    )

    f1_macro = (f1 + f1_neg) / 2.0

    return {
        "samples": float(total),
        "accuracy": correct / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
    }


class PromptInjectionEvaluator:
    """CLI helper that loads the LoRA adapters and provides evaluation utilities."""

    def __init__(
        self,
        *,
        model_path: str = DEFAULT_MODEL_PATH,
        base_model: str = DEFAULT_MODEL_REPO,
        device: Optional[str] = None,
        max_seq_length: int = 2048,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_path.resolve()}"
            )

        self.base_model = base_model
        self.device = _resolve_device(device)
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        base = AutoModelForSequenceClassification.from_pretrained(self.base_model)
        self.model = PeftModel.from_pretrained(base, self.model_path)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict(
        self,
        prompt: str,
        *,
        max_length: Optional[int] = None,
        return_probabilities: bool = True,
    ) -> Dict[str, object]:
        """Classify a single prompt as ``normal`` or ``attack``."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        effective_max_length = max_length or self.max_seq_length

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=effective_max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = _softmax(logits).squeeze(0).tolist()

        predicted_index = int(torch.argmax(logits, dim=-1).item())
        predicted_label = LABELS[predicted_index]
        confidence = probabilities[predicted_index]

        result: Dict[str, object] = {
            "label": predicted_label,
            "score": float(confidence),
        }
        if return_probabilities:
            result["probabilities"] = {
                label: float(probabilities[idx]) for idx, label in enumerate(LABELS)
            }
        return result

    def evaluate(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        *,
        splits: Optional[Sequence[str]] = None,
        batch_size: int = 32,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate one or multiple dataset splits saved via ``Dataset.save_to_disk``."""

        dataset_root = Path(dataset_path)
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_root.resolve()}"
            )

        dataset_dict: DatasetDict = load_from_disk(str(dataset_root))

        available_splits = list(dataset_dict.keys())
        if splits is None:
            target_splits = available_splits
        elif isinstance(splits, str):
            target_splits = [splits]
        else:
            target_splits = list(splits)

        missing = [name for name in target_splits if name not in dataset_dict]
        if missing:
            raise ValueError(
                f"Requested splits {missing} not found in dataset archive."
            )

        effective_max_length = max_length or self.max_seq_length

        summary: Dict[str, Dict[str, float]] = {}
        for split_name in target_splits:
            dataset = dataset_dict[split_name]
            chosen_text, chosen_label = _ensure_columns(
                dataset, text_column, label_column
            )
            predictions: List[int] = []
            references: List[int] = []

            indices = range(len(dataset))
            for batch_range in _batched(indices, batch_size):
                batch = dataset.select(list(batch_range))
                texts = batch[chosen_text]
                labels = batch[chosen_label]

                tokenized = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=effective_max_length,
                )
                tokenized = {
                    key: value.to(self.device) for key, value in tokenized.items()
                }

                with torch.inference_mode():
                    logits = self.model(**tokenized).logits
                    batch_predictions = torch.argmax(logits, dim=-1).cpu().tolist()

                predictions.extend(int(p) for p in batch_predictions)
                references.extend(int(l) for l in labels)

            metrics = _compute_binary_metrics(predictions, references)
            summary[split_name] = metrics

            correct = metrics["accuracy"] * metrics["samples"]
            print(f"\n{split_name.upper()} SUMMARY")
            print("-" * 60)
            print(
                f"Samples: {int(metrics['samples'])} | Accuracy: {metrics['accuracy']:.4f}"
            )
            print(
                f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
            )
            print(f"Macro F1: {metrics['f1_macro']:.4f}")
            print(
                f"Correct predictions: {int(correct)} / {int(metrics['samples'])}"
            )

        return summary



if __name__ == "__main__":
    fire.Fire(PromptInjectionEvaluator)