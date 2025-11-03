import gc
import json
import os
from pathlib import Path
from typing import Dict, Optional

# Disable fast generation for BERT models
os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")

from unsloth import is_bfloat16_supported, FastModel
import torch
from torch.utils.data import Dataset
import numpy as np
import optuna
import wandb
from datasets import load_from_disk
from optuna.exceptions import TrialPruned
from optuna.integration import WeightsAndBiasesCallback
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from train import (
    LORA_TARGET_MODULES,
    compute_metrics,
    tokenize_dataset,
    load_model,
)

SEQ_LENGTH_CHOICES = [768, 1024, 1536, 2048]


class TruncatingDataset(Dataset):
    """Wraps a tokenized dataset to allow dynamic max sequence length."""

    def __init__(self, dataset, max_seq_length: int):
        self._dataset = dataset
        self.max_seq_length = max_seq_length

    def set_max_seq_length(self, value: int) -> None:
        self.max_seq_length = value

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        item["input_ids"] = item["input_ids"][:self.max_seq_length]
        item["attention_mask"] = item["attention_mask"][:self.max_seq_length]
        return item


def hpo_space(trial: optuna.Trial) -> Dict[str, object]:
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    bs = trial.suggest_categorical(
        "per_device_train_batch_size", [16, 32, 64, 128, 256]
    )
    grad_accum = trial.suggest_categorical(
        "gradient_accumulation_steps", [1, 2, 4]
    )
    warmup = trial.suggest_categorical(
        "warmup_ratio", [0.03, 0.05, 0.1, 0.2]
    )
    wd = trial.suggest_float("weight_decay", 1e-6, 5e-2, log=True)
    optim = trial.suggest_categorical("optim", ["adamw_torch", "lion_32bit"])
    sched = trial.suggest_categorical(
        "lr_scheduler_type", ["linear", "cosine"]
    )

    r = trial.suggest_categorical("lora_r", [8, 16, 32])
    alpha_mult = trial.suggest_categorical(
        "lora_alpha_multiplier", [1, 2, 4]
    )
    alpha = r * alpha_mult
    dropout = trial.suggest_categorical("lora_dropout", [0.0, 0.05, 0.1])
    seq_len = trial.suggest_categorical(
        "max_seq_length", SEQ_LENGTH_CHOICES
    )

    trial.set_user_attr(
        "effective_batch_size",
        bs * grad_accum,
    )

    return {
        "learning_rate": lr,
        "per_device_train_batch_size": bs,
        "per_device_eval_batch_size": min(bs * 2, 128),
        "gradient_accumulation_steps": grad_accum,
        "warmup_ratio": warmup,
        "weight_decay": wd,
        "optim": optim,
        "lr_scheduler_type": sched,
        "lora_r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "max_seq_length": seq_len,
    }


def save_best_params(output_dir: Path, study: optuna.Study) -> Path:
    best_params = study.best_trial.params.copy()
    best_params["effective_batch_size"] = study.best_trial.user_attrs.get("effective_batch_size")
    best_params["best_eval_f1"] = study.best_value
    best_params["trial_number"] = study.best_trial.number

    best_file = output_dir / "best_hparams.json"
    with best_file.open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2)
    return best_file


def objective(
    trial: optuna.Trial,
    train_dataset: TruncatingDataset,
    val_dataset: TruncatingDataset,
    tokenizer,
    output_path: Path,
    seed: int,
    num_train_epochs: float,
    eval_steps: int,
    logging_steps: int,
    save_strategy: str,
    early_stopping_patience: int,
) -> float:
    model = None
    trainer = None

    try:
        params = hpo_space(trial)

        trial_output_dir = output_path / f"trial-{trial.number}"
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(trial_output_dir),
            eval_strategy="steps",
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            logging_first_step=True,
            save_strategy=save_strategy,
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            per_device_eval_batch_size=params["per_device_eval_batch_size"],
            gradient_accumulation_steps=params["gradient_accumulation_steps"],
            warmup_ratio=params["warmup_ratio"],
            weight_decay=params["weight_decay"],
            optim=params["optim"],
            lr_scheduler_type=params["lr_scheduler_type"],
            num_train_epochs=num_train_epochs,
            max_grad_norm=1.0,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            load_best_model_at_end=False,
            metric_for_best_model="f1",
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

        # Update dataset max length for this trial
        train_dataset.set_max_seq_length(params["max_seq_length"])
        val_dataset.set_max_seq_length(params["max_seq_length"])

        # Load base model
        base_model, _ = load_model(params["max_seq_length"])
        model = FastModel.get_peft_model(
            base_model,
            r=params["lora_r"],
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
            use_rslora=False,
            loftq_config=None,
            task_type="SEQ_CLS",
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
        objective_value = metrics["eval_f1"]

        trial.set_user_attr(
            "effective_batch_size",
            params["per_device_train_batch_size"]
            * params["gradient_accumulation_steps"],
        )
        trial.set_user_attr("lora_alpha", params["lora_alpha"])
        trial.set_user_attr("eval_metrics", metrics)

        return objective_value

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
):
    storage = f"sqlite:///{output_dir}/study.db"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_dir}")
    dataset = load_from_disk(data_dir)

    tokenize_length = max([max_seq_length, *SEQ_LENGTH_CHOICES])
    
    # Load tokenizer 
    _, tokenizer = load_model(tokenize_length)

    tokenized_train = tokenize_dataset(dataset["train"], tokenizer, tokenize_length)
    tokenized_val = tokenize_dataset(dataset["validation"], tokenizer, tokenize_length)
    train_dataset = TruncatingDataset(tokenized_train, max_seq_length)
    val_dataset = TruncatingDataset(tokenized_val, max_seq_length)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(multivariate=True, seed=seed),
        pruner=SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0,
        ),
        storage=storage,
        load_if_exists=True,
    )

    optuna_callbacks = []
    if use_wandb:
        wandb_callback = WeightsAndBiasesCallback(
            metric_name="eval_f1", 
            wandb_kwargs={"project": wandb_project, "reinit": True}
        )
        optuna_callbacks.append(wandb_callback)

    print(f"Starting Optuna search: {n_trials} trials")
    study.optimize(
        lambda trial: objective(
            trial,
            train_dataset,
            val_dataset,
            tokenizer,
            output_path,
            seed,
            num_train_epochs,
            eval_steps,
            logging_steps,
            save_strategy,
            early_stopping_patience,
        ),
        n_trials=n_trials,
        timeout=timeout,
        callbacks=optuna_callbacks,
        n_jobs=1,
    )

    if use_wandb:
        wandb.run.summary["best_eval_f1"] = study.best_value
        wandb.run.summary["best_trial"] = study.best_trial.number
        wandb.run.summary["best_effective_batch_size"] = (
            study.best_trial.user_attrs.get("effective_batch_size")
        )
        best_metrics = {
            "best/trial": study.best_trial.number,
            "best/effective_batch_size": study.best_trial.user_attrs.get(
                "effective_batch_size"
            ),
            "best/eval_f1": study.best_value,
        }
        wandb.log(best_metrics)
        wandb.finish()

    print(f"\nBest trial: {study.best_trial.number}, F1: {study.best_value:.4f}")
    best_file = save_best_params(output_path, study)
    print(f"Best hyperparameters saved to: {best_file}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
