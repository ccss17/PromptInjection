#!/usr/bin/env python3
"""Optuna hyperparameter search for ModernBERT + LoRA."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Dict, Optional

# Ensure consistent Unsloth behaviour across scripts
os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")

from unsloth import is_bfloat16_supported
import fire
import numpy as np
import optuna
from datasets import load_from_disk
from optuna.exceptions import TrialPruned
from optuna.integration import WeightsAndBiasesCallback
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

import torch

from train import (
    TARGET_MODULE_CHOICES,
    apply_lora_adapters,
    compute_metrics,
    get_device_descriptor,
    log_eval_metrics,
    print_section,
    load_classification_model,
    summarize_lengths,
    tokenize_splits,
)

SEQ_LENGTH_CHOICES = [768, 1024, 1536, 2048]


class TruncatingDataset:
    """Wraps a tokenized dataset to allow dynamic max sequence length."""

    def __init__(self, dataset, max_seq_length: int):
        self._dataset = dataset
        self.max_seq_length = max_seq_length

    def set_max_seq_length(self, value: int) -> None:
        self.max_seq_length = int(value)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        max_len = self.max_seq_length
        if max_len is None:
            return item

        result = dict(item)
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in result:
                result[key] = result[key][:max_len]
        return result


def _hp_space_1(trial: optuna.Trial) -> Dict[str, object]:
    """Define Optuna search space following HF/Unsloth guidance."""

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size", [16, 32, 64, 128, 256]
    )
    gradient_accumulation_steps = trial.suggest_categorical(
        "gradient_accumulation_steps", [1, 2, 4]
    )
    warmup_ratio = trial.suggest_categorical(
        "warmup_ratio", [0.03, 0.05, 0.1, 0.2]
    )
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-2, log=True)
    optim = trial.suggest_categorical("optim", ["adamw_torch", "lion_32bit"])
    lr_scheduler_type = trial.suggest_categorical(
        "lr_scheduler_type", ["linear", "cosine"]
    )

    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha_multiplier = trial.suggest_categorical(
        "lora_alpha_multiplier", [1, 2, 4]
    )
    lora_alpha = lora_r * lora_alpha_multiplier
    lora_dropout = trial.suggest_categorical("lora_dropout", [0.0, 0.05, 0.1])
    target_modules_set = trial.suggest_categorical(
        "target_modules_set", list(TARGET_MODULE_CHOICES.keys())
    )
    max_seq_length = trial.suggest_categorical(
        "max_seq_length", SEQ_LENGTH_CHOICES
    )

    trial.set_user_attr(
        "effective_batch_size",
        per_device_train_batch_size * gradient_accumulation_steps,
    )

    return {
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": min(
            per_device_train_batch_size * 2, 128
        ),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "optim": optim,
        "lr_scheduler_type": lr_scheduler_type,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules_set": target_modules_set,
        "max_seq_length": max_seq_length,
    }


def _hp_space_2(trial: optuna.Trial) -> Dict[str, object]:
    """Define Optuna search space following HF/Unsloth guidance."""

    learning_rate = trial.suggest_float(
        "learning_rate", 3e-5, 5.5e-5, log=True
    )
    per_device_train_batch_size = trial.suggest_categorical(
        "per_device_train_batch_size", [16, 32]
    )
    gradient_accumulation_steps = trial.suggest_categorical(
        "gradient_accumulation_steps", [1, 2]
    )
    warmup_ratio = trial.suggest_categorical(
        "warmup_ratio", [0.03, 0.05, 0.08]
    )
    weight_decay = trial.suggest_float("weight_decay", 2e-3, 2e-2, log=True)
    optim = "lion_32bit"
    lr_scheduler_type = "cosine"
    lora_r = trial.suggest_categorical("lora_r", [24, 32, 40])
    lora_alpha_multiplier = trial.suggest_categorical(
        "lora_alpha_multiplier", [2, 4, 6]
    )
    lora_alpha = lora_r * lora_alpha_multiplier
    lora_dropout = trial.suggest_categorical("lora_dropout", [0.0, 0.05])
    target_modules_set = "qkv"
    max_seq_length = trial.suggest_categorical(
        "max_seq_length", [1024, 1536, 2048]
    )

    trial.set_user_attr(
        "effective_batch_size",
        per_device_train_batch_size * gradient_accumulation_steps,
    )

    return {
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": min(
            per_device_train_batch_size * 2, 128
        ),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "optim": optim,
        "lr_scheduler_type": lr_scheduler_type,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules_set": target_modules_set,
        "max_seq_length": max_seq_length,
    }


def _create_model_loader(dataset_setter, default_max_seq_length: int):
    """Factory returning callable that loads a base classification model."""

    def load_model(trial: Optional[optuna.Trial] = None):
        trial_params = trial.params if trial is not None else {}
        trial_max_seq = int(
            trial_params.get("max_seq_length", default_max_seq_length)
        )
        dataset_setter(trial_max_seq)
        model, _ = load_classification_model(trial_max_seq)
        return model

    return load_model


def _compute_objective(metrics: Dict[str, float]) -> float:
    """Optuna objective: maximise positive-class F1."""

    return metrics.get("eval_f1_positive", 0.0)


def _save_best_params(output_dir: Path, study: optuna.Study) -> Path:
    best_params = study.best_trial.params.copy()
    if (
        "lora_alpha" not in best_params
        and "lora_r" in best_params
        and "lora_alpha_multiplier" in best_params
    ):
        best_params["lora_alpha"] = (
            best_params["lora_r"] * best_params["lora_alpha_multiplier"]
        )
    best_params["effective_batch_size"] = study.best_trial.user_attrs.get(
        "effective_batch_size"
    )
    best_params["best_eval_f1_positive"] = study.best_value
    best_params["trial_number"] = study.best_trial.number

    best_file = output_dir / "best_hparams.json"
    with best_file.open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2)
    return best_file


def main(
    data_dir: str = "data/processed/dataset",
    output_dir: str = "outputs/modernbert-lora/optuna",
    max_seq_length: int = 512,
    n_trials: int = 25,
    timeout: Optional[int] = None,
    study_name: str = "modernbert-lora-optuna",
    seed: int = 42,
    num_train_epochs: float = 3.0,
    eval_steps: int = 100,
    logging_steps: int = 50,
    save_strategy: str = "no",
    early_stopping_patience: int = 0,
    use_wandb: bool = True,
    wandb_project: str = "prompt-injection-detector-hpo",
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
):
    """Run Optuna HPO with Hugging Face Trainer backend."""

    storage = f"sqlite:///{output_dir}/study.db"
    np.random.seed(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print_section("Optuna Configuration", leading_newline=False)
    print(f"Study name: {study_name}")
    print(f"Storage: {storage}")
    print(f"Trials: {n_trials}")
    print(f"Timeout: {timeout}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"W&B enabled: {use_wandb}")
    print(f"Device: {get_device_descriptor()}")
    print("=" * 60)

    print_section("Loading dataset")
    dataset = load_from_disk(data_dir)
    print(f"  Splits: {list(dataset.keys())}")

    tokenize_length = max([max_seq_length, *SEQ_LENGTH_CHOICES])
    _, tokenizer = load_classification_model(tokenize_length)

    tokenized_train, tokenized_val, tokenized_test = tokenize_splits(
        dataset, tokenizer, tokenize_length
    )

    train_dataset = TruncatingDataset(tokenized_train, max_seq_length)
    val_dataset = TruncatingDataset(tokenized_val, max_seq_length)

    def set_dataset_max_length(length: int) -> None:
        train_dataset.set_max_seq_length(length)
        val_dataset.set_max_seq_length(length)

    set_dataset_max_length(max_seq_length)

    print_section("Tokenization statistics")
    summarize_lengths(
        "Train",
        [len(ids) for ids in tokenized_train["input_ids"]],
        max_seq_length,
    )
    summarize_lengths(
        "Validation",
        [len(ids) for ids in tokenized_val["input_ids"]],
        max_seq_length,
    )
    summarize_lengths(
        "Test",
        [len(ids) for ids in tokenized_test["input_ids"]],
        max_seq_length,
    )

    sampler = TPESampler(multivariate=True, seed=seed)
    pruner = SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )  # ASHA pruner

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    optuna_callbacks = []
    wandb_module = None
    wandb_callback: Optional[WeightsAndBiasesCallback] = None
    if use_wandb:
        try:
            import wandb as wandb_module  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "use_wandb=True requires the 'wandb' package to be installed."
            ) from exc
        wandb_kwargs = {"project": wandb_project, "reinit": True}
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity
        if wandb_group:
            wandb_kwargs["group"] = wandb_group
        wandb_callback = WeightsAndBiasesCallback(
            metric_name="eval_f1_positive", wandb_kwargs=wandb_kwargs
        )
        optuna_callbacks.append(wandb_callback)

    base_model_loader = _create_model_loader(
        set_dataset_max_length, max_seq_length
    )

    def objective(trial: optuna.Trial) -> float:
        model = None
        trainer = None

        try:
            # params = _hp_space_1(trial)
            params = _hp_space_2(trial)
            target_modules = TARGET_MODULE_CHOICES[
                params["target_modules_set"]
            ]

            trial_output_dir = output_path / f"trial-{trial.number}"
            trial_output_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(trial_output_dir),
                eval_strategy="steps",
                eval_steps=eval_steps,
                logging_steps=logging_steps,
                logging_first_step=True,
                save_strategy=save_strategy,
                learning_rate=float(params["learning_rate"]),
                per_device_train_batch_size=int(
                    params["per_device_train_batch_size"]
                ),
                per_device_eval_batch_size=int(
                    params["per_device_eval_batch_size"]
                ),
                gradient_accumulation_steps=int(
                    params["gradient_accumulation_steps"]
                ),
                warmup_ratio=float(params["warmup_ratio"]),
                weight_decay=float(params["weight_decay"]),
                optim=str(params["optim"]),
                lr_scheduler_type=str(params["lr_scheduler_type"]),
                num_train_epochs=num_train_epochs,
                max_grad_norm=1.0,
                bf16=is_bfloat16_supported(),
                fp16=not is_bfloat16_supported(),
                load_best_model_at_end=False,
                metric_for_best_model="f1_positive",
                greater_is_better=True,
                report_to="none",
                seed=seed,
                data_seed=seed,
                push_to_hub=False,
            )

            trial_callbacks = []
            if early_stopping_patience > 0:
                trial_callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=early_stopping_patience
                    )
                )

            model = apply_lora_adapters(
                base_model_loader(trial),
                target_modules=target_modules,
                lora_r=params["lora_r"],
                lora_alpha=params["lora_alpha"],
                lora_dropout=params["lora_dropout"],
                seed=seed,
            )

            trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=trial_callbacks,
            )

            trainer.train()
            metrics = trainer.evaluate(eval_dataset=val_dataset)
            objective_value = _compute_objective(metrics)

            if np.isnan(objective_value):
                raise TrialPruned("Objective returned NaN")

            trial.set_user_attr(
                "effective_batch_size",
                params["per_device_train_batch_size"]
                * params["gradient_accumulation_steps"],
            )
            trial.set_user_attr("lora_alpha", params["lora_alpha"])
            trial.set_user_attr("eval_metrics", metrics)

            return float(objective_value)

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                trial.set_user_attr("oom", True)
                raise TrialPruned("CUDA OOM") from exc
            raise

        finally:
            if trainer is not None:
                del trainer
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print_section("Starting Optuna search")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=optuna_callbacks,
        n_jobs=1,
    )

    if wandb_callback is not None and wandb_module is not None:
        active_run = wandb_module.run
        if active_run is not None:
            active_run.summary["best_eval_f1_positive"] = study.best_value
            active_run.summary["best_trial"] = study.best_trial.number
            active_run.summary["best_effective_batch_size"] = (
                study.best_trial.user_attrs.get("effective_batch_size")
            )
            wandb_module.log(
                {
                    "best/trial": study.best_trial.number,
                    "best/effective_batch_size": study.best_trial.user_attrs.get(
                        "effective_batch_size"
                    ),
                    "best/eval_f1_positive": study.best_value,
                }
            )
        wandb_module.finish()

    print_section("Best trial summary")
    print(f"  Trial number: {study.best_trial.number}")
    print(f"  Best eval F1 (positive): {study.best_value:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    best_file = _save_best_params(output_path, study)
    print(f"\nBest hyperparameters written to: {best_file}")

    best_metrics = study.best_trial.user_attrs.get("eval_metrics", {})
    print_section("Validation metrics")
    log_eval_metrics(
        "  Best trial validation:",
        {f"eval_{k}": v for k, v in best_metrics.items()},
    )

    print(
        "\nNext: train with scripts/train.py using the saved hyperparameters "
        "for a full-fidelity run."
    )


if __name__ == "__main__":
    fire.Fire(main)
