# Prompt Injection Detector – Agent Guide

## 🧭 Architecture focus
- ModernBERT is fine-tuned with LoRA via `scripts/train.py`, using Unsloth `FastModel` to load `answerdotai/ModernBERT-large`, attach adapters defined in `TARGET_MODULE_CHOICES`, and compute metrics in `compute_metrics`.
- Processed data lives in `data/processed/dataset/` (Hugging Face `save_to_disk` layout) with `text`, `labels`, and `source` columns; preserve this schema so `load_from_disk` calls across training/eval keep working.
- Hyperparameter search artifacts populate `best_params.json` (repo root) and `outputs/modernbert-lora/optuna/best_hparams.json`; the trainer auto-loads these overrides, so update the loader before renaming keys or paths.
- Final adapters and checkpoints are written to `outputs/modernbert-lora/` (`checkpoint-*` plus `final/`), which downstream scripts and deploy steps expect.

## ⚙️ Core workflows
- Environment and commands are managed by Pixi (`pyproject.toml`); run `pixi install` once, then use `pixi run <task>` for repeatable automation.
- Data refresh: `pixi run prepare-data` triggers `scripts/prepare_data.py`, rebuilding the dataset, `length_statistics.json`, and `dataset_info.json`; pass `--force` to regenerate even if outputs exist.
- Hyperparameter search: `pixi run optuna-search` executes `scripts/optuna_search.py`, writing best trials to `outputs/modernbert-lora/optuna/`; its `TruncatingDataset` wrapper dynamically enforces per-trial sequence lengths.
- Training: `pixi run train-best` or custom CLI flags call `scripts/train.py`, which resumes from the latest `checkpoint-*` when `--resume` is true, applies Optuna overrides, and logs to Weights & Biases if `--use-wandb` is set.
- Evaluation: `pixi run eval3` / `pixi run eval4` rely on `scripts/eval.py`; the Fire CLI exposes `predict` and `evaluate` subcommands that assume logits ordered `[normal, attack]` and return custom accuracy/F1 aggregates.
- Local Gradio demo: `pixi run demo` launches root `app.py`, a mirror of `space_deployment/app.py` that hits the published repo `ccss17/modernbert-prompt-injection-detector` unless you point it at `outputs/modernbert-lora/final`.
- Deployment: `pixi run deploy-hub` and `pixi run deploy-space` use Hugging Face Hub APIs to publish the adapter folder and the Gradio Space defined under `space_deployment/`.

## 💡 Implementation patterns
- Training always keeps ModernBERT logit order `{0: "normal", 1: "attack"}`; when extending heads or metrics, keep `ID2LABEL` / `LABEL2ID` in sync with downstream consumers.
- Tokenization mirrors inference (`truncation=True`, no padding) via `tokenize_splits`; reuse it instead of ad-hoc tokenization to maintain length stats and truncation reporting.
- `scripts/prepare_data.py` guarantees every attack prompt and all NotInject hard negatives are included before sampling other normal data—preserve that invariant when tweaking balance logic.
- Environment variables like `UNSLOTH_DISABLE_FAST_GENERATION=1` are set in training/search scripts; keep them when spawning new entry points so Unsloth remains deterministic.
- Optuna trials reuse base models via `_create_model_loader`; if you introduce new search spaces, ensure they still call `apply_lora_adapters` with modules from `TARGET_MODULE_CHOICES`.

## 🔍 Observability & validation
- W&B logging is wired through both training (`use_wandb`) and Optuna (`WeightsAndBiasesCallback`) and defaults to the projects referenced in `README.md`; expect new runs to appear there unless disabled.
- Length statistics, label balance, and source provenance are persisted in `data/processed/length_statistics.json` and `dataset_info.json`; inspect these before changing sequence lengths or class ratios.
- `verify_model.py` provides a quick regression test against the published Hugging Face weights; run it after deployments or when adapters are refreshed to confirm inference still matches expectations.
