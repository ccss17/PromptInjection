# Prompt Injection Detector (ModernBERT LoRA)

End-to-end pipeline for training, evaluating, and deploying a ModernBERT-based
prompt injection detector. The adapters are trained with LoRA, searched with
Optuna, logged to Weights & Biases, and packaged for Hugging Face Hub/Spaces.

---

## Dataset

Saved under `data/processed/` (see `dataset_info.json`).

- **Total samples:** 2,450 (train 1,960 • validation 244 • test 246)
- **Label balance:** ⟨normal = 50%⟩, ⟨attack = 50%⟩ in every split
- **Sources:** chatgpt_jailbreak, deepset, harmbench variants, jailbreakbench,
  notinject, openorca, ultrachat
- **Storage format:** `Dataset.save_to_disk()` (arrow shards) and Parquet

### Test examples surfaced in Spaces

The Gradio demo now showcases real samples from the **test** split (see
`space_deployment/app.py`): 4 normal prompts and 5 attack prompts, including
social-engineering, exploit, and malware requests.

---

## Hyperparameter Search (Optuna)

Results logged in [`best_params.json`](best_params.json); best trial (#16):

| Hyperparameter                  | Value |
|---------------------------------|-------|
| Learning rate                   | 4.439e-05 |
| Optimizer                       | lion_32bit |
| Per-device train batch size     | 16 |
| Gradient accumulation steps     | 1 |
| Effective batch size            | 16 |
| Max sequence length             | 2048 |
| LoRA rank (`lora_r`)            | 32 |
| LoRA alpha                      | 128 (multiplier 4) |
| LoRA dropout                    | 0.0 |
| Target modules                  | qkv |
| Warmup ratio                    | 0.05 |
| Weight decay                    | 0.0058467 |
| LR scheduler                    | cosine |

Best validation metrics from Optuna trial:

| Metric              | Score |
|---------------------|-------|
| Accuracy            | 0.9754 |
| Precision           | 0.9603 |
| Recall              | 0.9918 |
| F1 (positive class) | 0.9758 |

---

## Training Run (W&B: `latest-run`)

Configuration (`scripts/train.py`):

```bash
pixi run train-best
```

Additional CLI arguments recorded in
[`wandb-metadata.json`](wandb/latest-run/files/wandb-metadata.json):

```
--num_epochs 30
--early_stopping_patience 5
--logging_steps 15
--save_steps 10
--eval_steps 10
--use-wandb
--resume False
```

Hardware/Environment:

- OS: Ubuntu 22.04 (kernel 6.8)
- Python: 3.13.7 (pixi-managed)
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER (17.17 GB)
- CUDA: 13.0 (driver stack) / toolkit 12.9
- CPU: 44 physical (88 logical)
- Runtime: 7.98 minutes (train runtime 459.9 s, end-to-end 478 s)

Key training statistics (from
[`wandb-summary.json`](wandb/latest-run/files/wandb-summary.json)):

- Global steps: 340
- Train loss (final): 0.1164 (running) / 0.2391 (logged `train_loss`)
- Train samples/sec: 127.86 • steps/sec: 8.02
- Gradient norm: 1.976e-04
- Total FLOPs: 8.46e15

Baseline (pre-LoRA) metrics logged for comparison:

| Metric              | Baseline | LoRA (final/test) |
|---------------------|----------|-------------------|
| Accuracy            | 0.6138   | 0.9797 |
| Precision           | 0.7333   | 0.9758 |
| Recall              | 0.3577   | 0.9837 |
| F1 (positive class) | 0.4809   | 0.9798 |

---

## Evaluation

Command line helpers (see `[tool.pixi.tasks]` in `pyproject.toml`):

```bash
# Single-prompt predictions (test samples)
pixi run eval-normal1
pixi run eval-attack3

# Full dataset evaluation
pixi run eval3             # all splits
pixi run eval4             # test split, batch size 16
```

`scripts/eval.py` now loads LoRA adapters from
`outputs/modernbert-lora/final`, exposes `predict` and `evaluate` via Fire, and
prints accuracy, precision, recall, F1, and macro F1 per split.

---

## Deployment Targets

- **Model repo (default):** `ccss17/modernbert-prompt-injection-detector`
- **Space repo (default):** `ccss17/prompt-injection-detector`
- Use `pixi run deploy-hub` or `pixi run deploy-space` to publish the latest
  adapters and Gradio app.

`space_deployment/app.py` loads the Hugging Face model directly and mirrors the
metrics/explanations exposed by `scripts/eval.py`.

---

## Reproducing the Pipeline

1. **Environment**

	```bash
	pixi install
	```

2. **Prepare data (optional refresh)**

	```bash
	pixi run prepare-data
	```

3. **Run Optuna search**

	```bash
	pixi run optuna-search
	```

4. **Train with best settings**

	```bash
	pixi run train-best
	```

5. **Evaluate** — choose from the tasks listed above (`eval3`, `eval4`, etc.).

6. **Deploy**

	```bash
	pixi run deploy-hub
	pixi run deploy-space
	```

---

## Artifacts

- LoRA adapters: `outputs/modernbert-lora/final/`
- Model card template: generated during `deploy_to_hub.py`
- W&B run: `PromptInjectionDetector/latest-run`
- Config logs: see the JSON files under `wandb/latest-run/files/`

---

## License

Apache-2.0 (see repository LICENSE).
