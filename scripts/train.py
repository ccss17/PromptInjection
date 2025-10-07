#!/usr/bin/env python3
"""
Training script for Prompt Injection Detection.
Uses ModernBERT with LoRA for sequence classification.

Usage:
    # Quick test run (minimal resources)
    python scripts/train.py --batch-size 2 --gradient-accumulation-steps 2 --max-seq-length 256

    # Balanced training
    python scripts/train.py --batch-size 8 --max-seq-length 512 --num-epochs 3

    # Production training
    python scripts/train.py --batch-size 16 --max-seq-length 768 --num-epochs 5 --lora-r 32
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# Disable fast generation for BERT models
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

from unsloth import FastModel, is_bfloat16_supported
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate
import numpy as np
import torch
import fire

import wandb

# Load HuggingFace metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

MODEL_NAME = "answerdotai/ModernBERT-large"
ID2LABEL = {0: "normal", 1: "attack"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}
TARGET_MODULE_CHOICES = {
    "qv": ["Wqkv", "Wo"],
    "qkv": ["Wqkv", "Wo", "Wi"],
}


def compute_metrics(eval_pred):
    """Compute accuracy, macro/positive F1, precision, recall for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    positive_average = {"average": "binary", "pos_label": 1}

    accuracy = accuracy_metric.compute(
        predictions=predictions, references=labels
    )["accuracy"]
    f1_positive = f1_metric.compute(
        predictions=predictions, references=labels, **positive_average
    )["f1"]
    f1_macro = f1_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )["f1"]
    precision_positive = precision_metric.compute(
        predictions=predictions, references=labels, **positive_average
    )["precision"]
    recall_positive = recall_metric.compute(
        predictions=predictions, references=labels, **positive_average
    )["recall"]

    return {
        "accuracy": accuracy,
        "f1": f1_positive,
        "f1_positive": f1_positive,
        "f1_macro": f1_macro,
        "precision": precision_positive,
        "recall": recall_positive,
    }


def print_section(title: str, leading_newline: bool = True) -> None:
    prefix = "\n" if leading_newline else ""
    line = "=" * 60
    print(f"{prefix}{line}")
    print(title)
    print(line)


def log_eval_metrics(
    title: str, metrics: Dict[str, float], prefix: str = "eval_"
) -> None:
    print(title)
    metric_preferences = (
        ("accuracy", "Accuracy", None),
        ("f1_positive", "F1 (Positive)", "f1"),
        ("f1_macro", "F1 (Macro)", None),
        ("precision", "Precision", None),
        ("recall", "Recall", None),
    )
    for primary, label, fallback in metric_preferences:
        candidates = [primary]
        if fallback:
            candidates.append(fallback)

        for candidate in candidates:
            key = f"{prefix}{candidate}" if prefix else candidate
            if key in metrics:
                print(f"    {label}: {metrics[key]:.4f}")
                break


def summarize_lengths(name: str, lengths: List[int], max_length: int) -> None:
    count = len(lengths)
    truncated = sum(1 for value in lengths if value == max_length)
    average = sum(lengths) / count if count else 0
    print(f"  {name}: {count} sequences")
    print(f"    Avg length: {average:.1f} tokens")
    if count:
        print(f"    Truncated: {truncated} ({(truncated / count) * 100:.1f}%)")


def tokenize_splits(dataset, tokenizer, max_seq_length: int):
    """Tokenize train, validation, and test splits consistently."""

    def tokenize_function(examples):
        """Tokenizer wrapper with head truncation to mirror inference."""

        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized_train = dataset["train"].map(
        tokenize_function,
        batched=True,
        desc="Tokenizing training set",
    )
    tokenized_val = dataset["validation"].map(
        tokenize_function,
        batched=True,
        desc="Tokenizing validation set",
    )
    tokenized_test = dataset["test"].map(
        tokenize_function,
        batched=True,
        desc="Tokenizing test set (balanced)",
    )

    return tokenized_train, tokenized_val, tokenized_test


def get_device_descriptor() -> str:
    try:
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{device_name} / {total_memory:.1f}GB"
    except Exception:  # pragma: no cover - defensive
        return "cuda"


def load_classification_model(
    max_seq_length: int,
    *,
    load_in_4bit: bool = False,
):
    """Load ModernBERT classification model and tokenizer."""

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        auto_model=AutoModelForSequenceClassification,
        max_seq_length=max_seq_length,
        dtype=None,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        load_in_4bit=load_in_4bit,
    )

    return model, tokenizer


def apply_lora_adapters(
    model: torch.nn.Module,
    *,
    target_modules: List[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    seed: int,
) -> torch.nn.Module:
    """Attach LoRA adapters using Unsloth helper."""

    return FastModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
        task_type="SEQ_CLS",
    )


def _normalize_best_hparams(
    raw: object,
) -> tuple[Dict[str, object], Dict[str, object]]:
    """Convert Optuna export formats into a flat parameter dict."""

    if not isinstance(raw, dict):
        return {}, {}

    params = dict(raw)
    metadata: Dict[str, object] = {}

    if "params" in raw and isinstance(raw["params"], dict):
        params = dict(raw["params"])
        metadata = {k: v for k, v in raw.items() if k != "params"}
        user_attrs = metadata.get("user_attrs")
        if isinstance(user_attrs, dict):
            if "lora_alpha" in user_attrs and "lora_alpha" not in params:
                params["lora_alpha"] = user_attrs["lora_alpha"]
            if (
                "lora_alpha_multiplier" in params
                and "lora_r" in params
                and "lora_alpha" not in params
            ):
                try:
                    params["lora_alpha"] = int(params["lora_r"]) * int(
                        params["lora_alpha_multiplier"]
                    )
                except (TypeError, ValueError):
                    pass
    else:
        metadata = {}

    # Ensure numeric types are plain Python numbers (not NumPy / strings)
    for key, value in list(params.items()):
        if isinstance(value, str):
            try:
                if value.isdigit():
                    params[key] = int(value)
                else:
                    params[key] = float(value)
            except ValueError:
                continue

    return params, metadata


def load_best_hparams(
    output_dir: str, override_path: Optional[str] = None
) -> tuple[Optional[Path], Dict[str, object], Dict[str, object]]:
    """Locate and load Optuna best hyperparameters if available."""

    candidates = []
    if override_path:
        candidates.append(Path(override_path))

    base_dir = Path(output_dir)
    candidates.append(base_dir / "best_hparams.json")
    candidates.append(base_dir / "optuna" / "best_hparams.json")

    workspace_candidate = Path("best_params.json")
    if not override_path:
        candidates.append(workspace_candidate)

    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                params, metadata = _normalize_best_hparams(raw)
                return candidate, params, metadata
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: Failed to parse hyperparameter file at {candidate}: {exc}"
                )

    return None, {}, {}


def log_wandb_examples(dataset, tokenized_train, max_seq_length: int) -> None:
    print("\nLogging example data to W&B...")

    try:
        sample_indices = [0, 10, 50, 100, 200]
        table = wandb.Table(
            columns=["Index", "Text (Preview)", "Label", "Tokens", "Truncated"]
        )

        for idx in sample_indices:
            if idx >= len(dataset["train"]):
                continue

            sample = dataset["train"][idx]
            text = sample["text"]
            text_preview = text[:200] + ("..." if len(text) > 200 else "")
            label = "attack" if sample["labels"] == 1 else "normal"
            token_count = len(tokenized_train[idx]["input_ids"])
            truncated = "Yes" if token_count == max_seq_length else "No"

            table.add_data(idx, text_preview, label, token_count, truncated)

        wandb.log({"training_examples": table})
    except Exception as exc:  # pragma: no cover - logging only
        print(f"  Could not log examples: {exc}")


def initialize_wandb_session(
    enabled: bool,
    run_name: str,
    tags: List[str],
    config: Dict[str, object],
    dataset,
    tokenized_train,
    max_seq_length: int,
) -> bool:
    if not enabled:
        return False

    wandb.init(
        project="prompt-injection-detector",
        name=run_name,
        tags=tags,
        config=config,
        notes=f"Prompt Injection Detection with {MODEL_NAME}",
    )

    log_wandb_examples(dataset, tokenized_train, max_seq_length)
    return True


def log_wandb_baselines(
    enabled: bool,
    baseline_val: Dict[str, float],
    baseline_test: Optional[Dict[str, float]] = None,
) -> None:
    if not enabled:
        return

    metrics = {
        "baseline/val_accuracy": baseline_val["eval_accuracy"],
        "baseline/val_precision": baseline_val["eval_precision"],
        "baseline/val_recall": baseline_val["eval_recall"],
    }

    if "eval_f1_positive" in baseline_val:
        metrics["baseline/val_f1_positive"] = baseline_val["eval_f1_positive"]

    if "eval_f1" in baseline_val:
        metrics["baseline/val_f1"] = baseline_val["eval_f1"]

    if baseline_test:
        metrics.update(
            {
                "baseline/test_accuracy": baseline_test["eval_accuracy"],
                "baseline/test_precision": baseline_test["eval_precision"],
                "baseline/test_recall": baseline_test["eval_recall"],
            }
        )

        if "eval_f1_positive" in baseline_test:
            metrics["baseline/test_f1_positive"] = baseline_test[
                "eval_f1_positive"
            ]

        if "eval_f1" in baseline_test:
            metrics["baseline/test_f1"] = baseline_test["eval_f1"]

    wandb.log(metrics)
    print("\n  Baseline metrics logged to W&B")


def log_wandb_final_metrics(enabled: bool, metrics: Dict[str, float]) -> None:
    if not enabled or not metrics:
        return

    wandb.log(metrics)
    print("\n  Final test metrics logged to W&B")


def log_wandb_artifact(
    enabled: bool,
    run_name: str,
    final_dir: Path,
    best_metric: Optional[float],
    lora_r: int,
    max_seq_length: int,
    train_size: int,
) -> None:
    if not enabled:
        return

    print("\nLogging model artifact to W&B...")

    try:
        artifact = wandb.Artifact(
            name=f"model-{run_name}",
            type="model",
            description=f"Prompt Injection Detector - {MODEL_NAME}",
            metadata={
                "best_f1": best_metric,
                "lora_r": lora_r,
                "max_seq_length": max_seq_length,
                "training_samples": train_size,
            },
        )

        artifact.add_dir(str(final_dir))
        wandb.log_artifact(artifact)
        print("  Model artifact logged successfully")
    except Exception as exc:  # pragma: no cover - logging only
        print(f"  Could not log artifact: {exc}")


def main(
    # Data
    data_dir: str = "data/processed/dataset",
    output_dir: str = "outputs/modernbert-lora",
    max_seq_length: int = 512,  # OVERRIDE
    # LoRA
    lora_r: int = 16,  # OVERRIDE
    lora_alpha: int = 32,  # OVERRIDE
    lora_dropout: float = 0.0,  # OVERRIDE
    # Training
    batch_size: int = 16,  # OVERRIDE
    gradient_accumulation_steps: int = 1,  # OVERRIDE
    learning_rate: float = 2e-4,  # OVERRIDE
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,  # OVERRIDE
    weight_decay: float = 0.01,  # OVERRIDE
    max_grad_norm: float = 1.0,
    optimizer: str = "adamw_torch",  # OVERRIDE
    lr_scheduler_type: str = "cosine",  # OVERRIDE
    # Evaluation
    eval_steps: int = 4,
    save_steps: int = 4,
    logging_steps: int = 5,
    early_stopping_patience: int = 3,
    # Resume
    resume: bool = True,
    # Misc
    seed: int = 42,
    use_wandb: bool = False,
    optuna_hparams_path: Optional[str] = "best_params.json",
):
    """
    Train ModernBERT with Unsloth + LoRA for prompt injection detection.

    Args:
        data_dir: Path to processed dataset.
        output_dir: Where to save model checkpoints (auto-named when empty).
        max_seq_length: Maximum sequence length (256/512/768/1024).
        lora_r: LoRA rank (8/16/32/64).
        lora_alpha: LoRA alpha (typically 2*r).
        lora_dropout: LoRA dropout (0.0-0.2).
        batch_size: Per-device batch size (2/4/8/16).
        gradient_accumulation_steps: Accumulate gradients (1/2/4/8).
        learning_rate: Learning rate (1e-5 to 5e-4).
        num_epochs: Number of training epochs (1-10).
        warmup_ratio: Warmup ratio (0.0-0.2).
        weight_decay: Weight decay (0.0-0.1).
        max_grad_norm: Maximum gradient norm for gradient clipping (0.0 to disable).
        optimizer: Optimizer passed to HuggingFace trainer (default adamw_torch).
        lr_scheduler_type: Learning rate scheduler passed to Trainer.
        eval_steps: Evaluate every N steps.
        save_steps: Save checkpoint every N steps.
        logging_steps: Log metrics every N steps.
        early_stopping_patience: Stop if no improvement for N evaluations (0 to disable).
        resume: Auto-resume from latest checkpoint if exists.
        seed: Random seed.
        use_wandb: Enable Weights & Biases logging.
    optuna_hparams_path: Optional explicit path to Optuna best_hparams.json file (defaults to best_params.json).
    """

    target_modules_key = "qkv"
    eval_batch_size = None

    (
        best_hparams_path,
        best_hparams,
        best_hparams_meta,
    ) = load_best_hparams(output_dir, optuna_hparams_path)

    if best_hparams:
        print_section(
            "Applying Optuna best hyperparameters", leading_newline=False
        )
        print(f"Source: {best_hparams_path}")

        best_trial_number = best_hparams_meta.get("number")
        if best_trial_number is not None:
            print(f"  Optuna trial #: {best_trial_number}")

        best_value = best_hparams_meta.get("value")
        if best_value is not None:
            try:
                print(f"  Optuna best value: {float(best_value):.4f}")
            except (TypeError, ValueError):
                pass

        user_attrs = best_hparams_meta.get("user_attrs")
        if isinstance(user_attrs, dict):
            effective = user_attrs.get("effective_batch_size")
            if effective is not None:
                print(f"  Optuna effective batch size: {effective}")

            eval_metrics = user_attrs.get("eval_metrics")
            if isinstance(eval_metrics, dict):
                log_eval_metrics("  Optuna validation metrics:", eval_metrics)

        def override_scalar(
            current_value,
            key: str,
            cast_type,
        ):
            if key not in best_hparams:
                return current_value, None
            new_value = cast_type(best_hparams[key])
            if cast_type is float:
                new_value = float(best_hparams[key])
            if cast_type is int:
                new_value = int(best_hparams[key])
            if cast_type is str:
                new_value = str(best_hparams[key])
            changed = current_value != new_value
            return new_value, (current_value, new_value) if changed else None

        batch_size, change = override_scalar(
            batch_size, "per_device_train_batch_size", int
        )
        if change:
            print(f"  batch_size: {change[0]} -> {change[1]}")

        gradient_accumulation_steps, change = override_scalar(
            gradient_accumulation_steps, "gradient_accumulation_steps", int
        )
        if change:
            print(f"  gradient_accumulation_steps: {change[0]} -> {change[1]}")

        learning_rate, change = override_scalar(
            learning_rate, "learning_rate", float
        )
        if change:
            print(f"  learning_rate: {change[0]} -> {change[1]}")

        warmup_ratio, change = override_scalar(
            warmup_ratio, "warmup_ratio", float
        )
        if change:
            print(f"  warmup_ratio: {change[0]} -> {change[1]}")

        weight_decay, change = override_scalar(
            weight_decay, "weight_decay", float
        )
        if change:
            print(f"  weight_decay: {change[0]} -> {change[1]}")

        optimizer, change = override_scalar(optimizer, "optim", str)
        if change:
            print(f"  optimizer: {change[0]} -> {change[1]}")

        lr_scheduler_type, change = override_scalar(
            lr_scheduler_type, "lr_scheduler_type", str
        )
        if change:
            print(f"  lr_scheduler_type: {change[0]} -> {change[1]}")

        lora_r, change = override_scalar(lora_r, "lora_r", int)
        if change:
            print(f"  lora_r: {change[0]} -> {change[1]}")

        lora_alpha, change = override_scalar(lora_alpha, "lora_alpha", int)
        if change:
            print(f"  lora_alpha: {change[0]} -> {change[1]}")

        lora_dropout, change = override_scalar(
            lora_dropout, "lora_dropout", float
        )
        if change:
            print(f"  lora_dropout: {change[0]} -> {change[1]}")

        max_seq_length, change = override_scalar(
            max_seq_length, "max_seq_length", int
        )
        if change:
            print(f"  max_seq_length: {change[0]} -> {change[1]}")

        if "per_device_eval_batch_size" in best_hparams:
            eval_batch_size = int(best_hparams["per_device_eval_batch_size"])
            print(f"  per_device_eval_batch_size set to {eval_batch_size}")

        if "target_modules_set" in best_hparams:
            candidate_key = str(best_hparams["target_modules_set"]).lower()
            if candidate_key in TARGET_MODULE_CHOICES:
                if candidate_key != target_modules_key:
                    print(
                        f"  target_modules_set: {target_modules_key} -> {candidate_key}"
                    )
                target_modules_key = candidate_key
            else:
                print(
                    f"  Warning: Unknown target_modules_set '{candidate_key}', using default."
                )

    effective_batch_size = batch_size * gradient_accumulation_steps
    if eval_batch_size is None:
        eval_batch_size = min(batch_size * 2, 128)

    target_modules = TARGET_MODULE_CHOICES.get(
        target_modules_key, TARGET_MODULE_CHOICES["qkv"]
    )

    print_section(
        "Training Configuration", leading_newline=not bool(best_hparams)
    )
    print(f"Model: {MODEL_NAME}")
    print(f"Max Sequence Length: {max_seq_length}")
    print(
        f"Training Mode: LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})"
    )
    print(f"Batch Size: {batch_size} (effective: {effective_batch_size})")
    print(f"Gradient Accumulation: {gradient_accumulation_steps}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Eval Batch Size: {eval_batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Warmup Ratio: {warmup_ratio}")
    print(f"Weight Decay: {weight_decay}")
    print(f"LR Scheduler: {lr_scheduler_type}")
    print(f"Target Modules: {target_modules_key} -> {target_modules}")
    print(f"Seed: {seed}")
    print("=" * 60)

    print("\nLoading model with Unsloth...")

    model, tokenizer = load_classification_model(max_seq_length)

    print("Applying LoRA adapters...")

    # ModernBERT uses different module names than standard BERT:
    # - Wqkv: Combined Q,K,V projection in attention
    # - Wo: Output projection (both attention and MLP)
    # - Wi: Input projection in MLP
    model = apply_lora_adapters(
        model,
        target_modules=target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        seed=seed,
    )

    # Print trainable parameters
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    all_params = sum(p.numel() for p in model.parameters())
    print_section("Model Parameters")
    print(
        f"  Trainable: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)"
    )
    print(f"  Total: {all_params:,}")

    print("\nLoading datasets...")
    dataset = load_from_disk(data_dir)

    # Verify we have all expected splits
    expected_splits = ["train", "validation", "test"]
    available_splits = list(dataset.keys())
    print(f"  Available splits: {available_splits}")

    for split in expected_splits:
        if split not in available_splits:
            print(f"  Warning: Missing '{split}' split")

    train_size = len(dataset["train"])
    val_size = len(dataset["validation"])
    test_size = len(dataset["test"])

    print(f"  Train: {train_size} samples")
    print(f"  Validation: {val_size} samples")
    print(f"  Test: {test_size} samples")

    # Tokenize datasets
    print("\nTokenizing datasets...")
    tokenized_train, tokenized_val, tokenized_test = tokenize_splits(
        dataset, tokenizer, max_seq_length
    )

    # Calculate steps
    steps_per_epoch = train_size // effective_batch_size
    total_steps = steps_per_epoch * num_epochs

    print_section("Training Stats")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {int(total_steps * warmup_ratio)}")

    # Analyze tokenization statistics
    print_section("Tokenization Statistics")
    train_lengths = [len(x) for x in tokenized_train["input_ids"]]
    val_lengths = [len(x) for x in tokenized_val["input_ids"]]

    test_lengths = [len(x) for x in tokenized_test["input_ids"]]

    train_truncated = sum(1 for x in train_lengths if x == max_seq_length)
    val_truncated = sum(1 for x in val_lengths if x == max_seq_length)
    test_truncated = sum(1 for x in test_lengths if x == max_seq_length)
    summarize_lengths("Train", train_lengths, max_seq_length)
    summarize_lengths("Validation", val_lengths, max_seq_length)
    summarize_lengths("Test", test_lengths, max_seq_length)

    device_descriptor = get_device_descriptor()
    run_name = (
        f"mode=lora;sq={max_seq_length};r={lora_r};a={lora_alpha};"
        f"d={lora_dropout};bs={effective_batch_size};ga={gradient_accumulation_steps};"
        f"lr={learning_rate:.0e};e={num_epochs};w={warmup_ratio};wd={weight_decay};"
        f"opt={optimizer};sched={lr_scheduler_type};tm={target_modules_key}"
        f" [{device_descriptor}]"
    )

    wandb_tags = [
        f"lora_r_{lora_r}",
        f"seq_len_{max_seq_length}",
        f"bs_{effective_batch_size}",
        f"tm_{target_modules_key}",
        f"sched_{lr_scheduler_type}",
        "modernbert",
        "lora",
    ]

    wandb_config = {
        "model": MODEL_NAME,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "eval_batch_size": eval_batch_size,
        "num_epochs": num_epochs,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler_type,
        "train_size": train_size,
        "val_size": val_size,
        "target_modules_set": target_modules_key,
        "train_truncated_pct": train_truncated / len(train_lengths) * 100,
        "val_truncated_pct": val_truncated / len(val_lengths) * 100,
        "test_truncated_pct": test_truncated / len(test_lengths) * 100,
        "avg_train_length": sum(train_lengths) / len(train_lengths),
        "trainable_params": trainable_params,
        "total_params": all_params,
        "trainable_pct": 100 * trainable_params / all_params,
        "seed": seed,
        "output_dir": output_dir,
    }

    wandb_enabled = initialize_wandb_session(
        use_wandb,
        run_name,
        wandb_tags,
        wandb_config,
        dataset,
        tokenized_train,
        max_seq_length,
    )

    report_to = "wandb" if wandb_enabled else "none"

    print("\nStarting training...")
    print(f"Output directory: {output_dir}")
    if early_stopping_patience > 0:
        print(
            f"Early stopping enabled (patience: {early_stopping_patience} evaluations)"
        )

    model = model.cuda()

    # Setup callbacks
    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        )

    # Check for existing checkpoints to resume from
    checkpoint_to_resume = None
    if resume and os.path.exists(output_dir):
        # Find all checkpoint directories
        checkpoints = [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoints:
            # Get the latest checkpoint by number
            latest_checkpoint = max(
                checkpoints, key=lambda x: int(x.split("-")[1])
            )
            checkpoint_to_resume = os.path.join(output_dir, latest_checkpoint)
            print(f"Resuming from checkpoint: {checkpoint_to_resume}")
        else:
            print("No existing checkpoints found, starting fresh training")
    elif resume:
        print("No output directory found, starting fresh training")

    # Use standard Trainer for sequence classification
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,  # Use processing_class (official Unsloth style)
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        args=TrainingArguments(
            output_dir=output_dir,
            # Training
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # Optimization
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,  # Gradient clipping
            optim=optimizer,  # Use the optimizer parameter
            lr_scheduler_type=lr_scheduler_type,
            # Precision
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            # Logging
            logging_steps=logging_steps,
            logging_first_step=True,
            report_to=report_to,
            # Evaluation
            eval_strategy="steps",
            eval_steps=eval_steps,
            # Saving
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",  # Use F1 for classification
            greater_is_better=True,
            # Misc
            seed=seed,
            data_seed=seed,
            push_to_hub=False,
        ),
    )

    # Compute baseline performance (before training)
    if not checkpoint_to_resume:
        print_section("Computing baseline performance (untrained model)")

        # Baseline on validation set
        baseline_val = trainer.evaluate(eval_dataset=tokenized_val)
        log_eval_metrics("  Validation baseline:", baseline_val)

        # Baseline on balanced test set
        baseline_test = trainer.evaluate(eval_dataset=tokenized_test)
        log_eval_metrics("  Test (balanced) baseline:", baseline_test)

        log_wandb_baselines(wandb_enabled, baseline_val, baseline_test)

    # Train
    print("\n" + "=" * 60)
    trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    print("=" * 60)

    print("\nSaving final model...")
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("\nTraining complete!")
    print(f"Model saved to: {final_dir}")
    print(f"Checkpoints saved to: {output_dir}")

    # Print best metrics
    if (
        hasattr(trainer.state, "best_metric")
        and trainer.state.best_metric is not None
    ):
        print(f"\nBest validation F1: {trainer.state.best_metric:.4f}")

    # Final evaluation on test sets
    print_section("Final evaluation on test sets")

    final_metrics = {}

    # Evaluate on balanced test set
    final_test = trainer.evaluate(eval_dataset=tokenized_test)
    log_eval_metrics("  Test (balanced) - Final Results:", final_test)

    final_metrics.update(
        {
            "final/test_accuracy": final_test["eval_accuracy"],
            "final/test_precision": final_test["eval_precision"],
            "final/test_recall": final_test["eval_recall"],
        }
    )

    if "eval_f1_positive" in final_test:
        final_metrics["final/test_f1_positive"] = final_test[
            "eval_f1_positive"
        ]

    if "eval_f1" in final_test:
        final_metrics["final/test_f1"] = final_test["eval_f1"]

    # Log all final metrics to WandB
    log_wandb_final_metrics(wandb_enabled, final_metrics)

    # Log model artifact to W&B
    best_metric = (
        trainer.state.best_metric
        if hasattr(trainer.state, "best_metric")
        else None
    )

    log_wandb_artifact(
        wandb_enabled,
        run_name,
        final_dir,
        best_metric,
        lora_r,
        max_seq_length,
        train_size,
    )

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
