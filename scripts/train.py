"""https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/bert_classification.ipynb"""
import json
import os
from pathlib import Path
from typing import Dict, Optional

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
import wandb


MODEL_NAME = "answerdotai/ModernBERT-large"
ID2LABEL = {0: "normal", 1: "attack"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj", 
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    return metrics.compute(
        predictions=predictions,
        references=labels,
        average="binary",
        pos_label=1,
    )


def tokenize_dataset(dataset, tokenizer, max_seq_length: int):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
    
    return dataset.map(tokenize_function, batched=True)


def load_best_hparams(hparams_path: str):
    with open(hparams_path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)
    params = raw["params"]
    params["lora_alpha"] = params["lora_r"] * params["lora_alpha_multiplier"]
    return params


def load_model(max_seq_length: int):
    """Load classification model and tokenizer."""
    return FastModel.from_pretrained(
        model_name=MODEL_NAME,
        auto_model=AutoModelForSequenceClassification,
        max_seq_length=max_seq_length,
        dtype=None,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        load_in_4bit=False,
    )


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
    optuna_hparams_path: Optional[str] = None,
):
    if optuna_hparams_path:
        print(f"Applying Optuna hyperparameters from: {optuna_hparams_path}")
        best_hparams = load_best_hparams(optuna_hparams_path)
        batch_size = best_hparams["per_device_train_batch_size"]
        gradient_accumulation_steps = best_hparams["gradient_accumulation_steps"]
        learning_rate = best_hparams["learning_rate"]
        warmup_ratio = best_hparams["warmup_ratio"]
        weight_decay = best_hparams["weight_decay"]
        optimizer = best_hparams["optim"]
        lr_scheduler_type = best_hparams["lr_scheduler_type"]
        lora_r = best_hparams["lora_r"]
        lora_alpha = best_hparams["lora_alpha"]
        lora_dropout = best_hparams["lora_dropout"]
        max_seq_length = best_hparams["max_seq_length"]

    effective_batch_size = batch_size * gradient_accumulation_steps
    eval_batch_size = min(batch_size * 2, 128)

    if use_wandb:
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
            "lora_target_modules": LORA_TARGET_MODULES,
            "seed": seed,
            "output_dir": output_dir,
            "device": torch.cuda.get_device_name(0),
        }
        wandb.init(
            project="prompt-injection-detector",
            config=wandb_config,
        )

    model, tokenizer = load_model(max_seq_length)

    model = FastModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
        task_type="SEQ_CLS",
    )

    dataset = load_from_disk(data_dir)
    tokenized_train = tokenize_dataset(dataset["train"], tokenizer, max_seq_length)
    tokenized_val = tokenize_dataset(dataset["validation"], tokenizer, max_seq_length)

    training_args = TrainingArguments(
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
        report_to="wandb" if use_wandb else "none",
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
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        args=training_args
    )

    # Check for existing checkpoints to resume from
    checkpoint_to_resume = None
    output_path = Path(output_dir)
    if resume and output_path.exists():
        checkpoints = [
            d.name
            for d in output_path.iterdir()
            if d.name.startswith("checkpoint-")
        ]
        if checkpoints:
            latest_checkpoint = max(
                checkpoints, key=lambda x: int(x.split("-")[1])
            )
            checkpoint_to_resume = str(output_path / latest_checkpoint)
            print(f"Resuming from checkpoint: {checkpoint_to_resume}")

    tokenized_test = None
    # Compute baseline performance before training
    if not checkpoint_to_resume:
        tokenized_test = tokenize_dataset(dataset["test"], tokenizer, max_seq_length)
        baseline_val = trainer.evaluate(eval_dataset=tokenized_val)
        baseline_test = trainer.evaluate(eval_dataset=tokenized_test)
        print("  Validation baseline:", baseline_val)
        print("  Test (balanced) baseline:", baseline_test)
        
        if use_wandb:
            wandb.log({
                "baseline/val_accuracy": baseline_val["eval_accuracy"],
                "baseline/val_precision": baseline_val["eval_precision"],
                "baseline/val_recall": baseline_val["eval_recall"],
                "baseline/val_f1": baseline_val["eval_f1"],
                "baseline/test_accuracy": baseline_test["eval_accuracy"],
                "baseline/test_precision": baseline_test["eval_precision"],
                "baseline/test_recall": baseline_test["eval_recall"],
                "baseline/test_f1": baseline_test["eval_f1"],
            })

    # Train
    trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    # Save model file
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nModel saved to: {final_dir}")

    # Final evaluation on test sets
    print(f"\n{'=' * 60}\nFinal evaluation on test sets\n{'=' * 60}")

    if tokenized_test is None:
        tokenized_test = tokenize_dataset(dataset["test"], tokenizer, max_seq_length)

    # Evaluate on balanced test set
    final_test = trainer.evaluate(eval_dataset=tokenized_test)
    print("  Test (balanced) - Final Results:", final_test)

    final_metrics = {
        "final/test_accuracy": final_test["eval_accuracy"],
        "final/test_precision": final_test["eval_precision"],
        "final/test_recall": final_test["eval_recall"],
        "final/test_f1": final_test["eval_f1"],
    }

    if use_wandb:
        wandb.log(final_metrics)
        artifact = wandb.Artifact(
            name="model",
            type="model",
            description=f"Prompt Injection Detector - {MODEL_NAME}",
            metadata={
                "best_f1": trainer.state.best_metric,
                "lora_r": lora_r,
                "max_seq_length": max_seq_length,
            },
        )
        artifact.add_dir(str(final_dir))
        wandb.log_artifact(artifact)
        print("  Model artifact logged successfully")
        
        wandb.finish()


if __name__ == "__main__":
    """
    # Quick test run 
    python scripts/train.py --batch-size 2 --gradient-accumulation-steps 2 --max-seq-length 256

    # Balanced training
    python scripts/train.py --batch-size 8 --max-seq-length 512 --num-epochs 3

    # Production training
    python scripts/train.py --batch-size 16 --max-seq-length 768 --num-epochs 5 --lora-r 32
    """

    import fire
    fire.Fire(main)
