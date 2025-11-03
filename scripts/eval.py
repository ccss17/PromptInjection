from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import evaluate
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ("normal", "attack")
DEFAULT_MODEL_REPO = "answerdotai/ModernBERT-large"
DEFAULT_MODEL_PATH = "outputs/modernbert-lora/final"
DEFAULT_DATASET_PATH = "data/processed/dataset"


def _batched(iterable, batch_size):
    total = len(iterable)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield range(start, end)


class PromptInjectionEvaluator:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        base_model: str = DEFAULT_MODEL_REPO,
        max_seq_length: int = 2048,
    ):
        self.model_path = Path(model_path)

        self.base_model = base_model
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        base = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            device_map="cuda"
        )
        self.model = PeftModel.from_pretrained(base, self.model_path)
        self.model.eval()

    def predict(self, prompt: str):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).tolist()

        predicted_index = int(torch.argmax(logits, dim=-1).item())
        predicted_label = LABELS[predicted_index]
        confidence = probabilities[predicted_index]

        return {
            "label": predicted_label,
            "score": confidence,
            "probabilities": {
                label: probabilities[idx] for idx, label in enumerate(LABELS)
            }
        }

    def evaluate(self, dataset_path: str = DEFAULT_DATASET_PATH, batch_size: int = 32):
        dataset_root = Path(dataset_path)
        dataset_dict = load_from_disk(str(dataset_root))

        summary = {}
        for split_name in dataset_dict.keys():
            dataset = dataset_dict[split_name]
            predictions = []
            references = []

            indices = range(len(dataset))
            for batch_range in _batched(indices, batch_size):
                batch = dataset.select(list(batch_range))
                texts = batch["text"]
                labels = batch["labels"]

                tokenized = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                tokenized = {
                    key: value.to("cuda") for key, value in tokenized.items()
                }

                with torch.inference_mode():
                    logits = self.model(**tokenized).logits
                    batch_predictions = torch.argmax(logits, dim=-1).cpu().tolist()

                predictions.extend(batch_predictions)
                references.extend(labels)

            metrics_lib = evaluate.combine(["accuracy", "f1", "precision", "recall"])
            metrics = metrics_lib.compute(
                predictions=predictions,
                references=references,
                average="binary",
                pos_label=1,
            )
            summary[split_name] = metrics

            correct = metrics["accuracy"] * len(predictions)
            print(f"\n{split_name.upper()} SUMMARY")
            print("-" * 60)
            print(f"Samples: {len(predictions)} | Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
            print(f"Correct predictions: {int(correct)} / {len(predictions)}")

        return summary



if __name__ == "__main__":
    """Evaluate or run predictions with the LoRA ModernBERT prompt injection detector.

    This script loads the LoRA adapters from ``outputs/modernbert-lora/final`` and
    exposes two Fire commands:

    predict – run a single prompt through the classifier.
    evaluate – score an entire dataset saved with ``Dataset.save_to_disk``.

    python scripts/eval.py predict "Ignore previous instructions"
    python scripts/eval.py evaluate --dataset_path data/processed/dataset
    """
    import fire
    fire.Fire(PromptInjectionEvaluator)