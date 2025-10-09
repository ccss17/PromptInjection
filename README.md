- https://wandb.ai/cccsss17-xxx/prompt-injection-detector
- https://wandb.ai/cccsss17-xxx/prompt-injection-detector-hpo
- https://huggingface.co/ccss17/modernbert-prompt-injection-detector
- https://huggingface.co/spaces/ccss17/prompt-injection-detector

# Prompt Injection Detector (ModernBERT LoRA)

End-to-end pipeline for training, evaluating, and deploying a ModernBERT-based
prompt injection detector. The adapters are trained with LoRA, searched with
Optuna, logged to Weights & Biases, and packaged for Hugging Face Hub/Spaces.

---

## Development Environment

Set up the Pixi-managed toolchain and dependencies:

```bash
pixi install
```

This creates the locked Python 3.13 environment, installs GPU-enabled PyTorch, and makes all project tasks (e.g., `pixi run train-best`) available.

---

## Dataset

Saved under `data/processed/` (see `dataset_info.json`).

```bash
pixi run prepare-data [--force]
```

Use `--force` to regenerate the Hugging Face dataset and derived statistics even when cached outputs exist.

- **Split sizes:** train 1,960 • validation 244 • test 246 (2,450 total)
- **Label balance:** perfectly stratified at 980/980 (train), 122/122 (val), 123/123 (test)
- **Sources:** chatgpt_jailbreak, deepset, jailbreakbench, harmbench contextual/copyright/standard,
  notinject, openorca, ultrachat
- **Token lengths:** median 25 tokens; p95 ≈ 495, p98 ≈ 940; 95% of prompts fit within 512 tokens,
  and 98% within 1,024 tokens (see `length_statistics.json` for per-class coverage)
- **Storage format:** `Dataset.save_to_disk()` (arrow shards) plus split Parquet exports

### Test examples surfaced in Spaces

The Gradio demo now showcases real samples from the **test** split (see
`space_deployment/app.py`): 4 normal prompts and 5 attack prompts, including
social-engineering, exploit, and malware requests.

---

## Hyperparameter Search (Optuna)

Candidate space explored via `scripts/optuna_search.py`:

- Learning rate: log-uniform $[1\times10^{-5}, 5\times10^{-5}]$
- Per-device batch size: {16, 32, 64, 128, 256} with eval batch size `min(train*2, 128)`
- Gradient accumulation: {1, 2, 4} (effective batch = train * accumulation)
- Warmup ratio: {0.03, 0.05, 0.1, 0.2}
- Weight decay: log-uniform $[1\times10^{-6}, 5\times10^{-2}]$
- Optimizer / LR scheduler: {`adamw_torch`, `lion_32bit`} × {`linear`, `cosine`}
- LoRA configs: rank {8, 16, 32}, alpha multiplier {1, 2, 4}, dropout {0.0, 0.05, 0.1},
  target modules in `TARGET_MODULE_CHOICES`
- Max sequence length: {768, 1024, 1536, 2048}

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

### Batch-size tuning journey

Running on a Vast.ai H200 instance (150.1 GB VRAM) gave plenty of headroom, but the large-batch experiments were counter‑productive:

- [`batch_size=700`](https://wandb.ai/cccsss17-xxx/prompt-injection-detector/runs/va9vasa1?nw=nwusercccsss17) – throughput was great, accuracy cratered.
- [`batch_size=384`](https://wandb.ai/cccsss17-xxx/prompt-injection-detector/runs/br9gw3xb?nw=nwusercccsss17) – still unstable, gradients appeared too “polished.”
- [`batch_size=128`](https://wandb.ai/cccsss17-xxx/prompt-injection-detector/runs/cleul7sr?nw=nwusercccsss17) – marginally better, but convergence lagged behind expectations.

Letting Optuna co‑tune the entire space ultimately picked `per_device_train_batch_size=16` with no gradient accumulation. The smaller batch injected exactly the right amount of noise, and the validation F1 jumped to ~0.976—confirming that ModernBERT preferred stochasticity over massive batches despite the abundant GPU memory.

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

- Python: 3.13.7 (pixi-managed)
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER (17.17 GB)
- CUDA: 13.0 (driver stack) / toolkit 12.9

Key training statistics:

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


## Limitations and Future Improvements

### Current Limitations
The model achieves high F1 scores (0.97 on test set) but exhibits over-defense in production, misclassifying benign prompts as injections. For example, the harmless query "What’s the weather like in Seoul?" is flagged as injection with 98.5% confidence, due to:
- **Imbalanced training data**: Trained on ~1:1 attack:normal ratio (train: 980/980; val: 122/122; test: 123/123), not reflecting real-world distributions where normal prompts vastly outnumber attacks (50:1 to 1000:1).
- **Distribution shift**: Normal samples from OpenOrca, ultrachat_200k, and NotInject differ from production inputs, causing false positives on out-of-distribution benign prompts.
- **Limited diversity**: ~2,000 normal samples fail to capture wide legitimate patterns, leading to high false positive rates (FPR) despite strong test performance.
- **Threshold miscalibration**: Default 0.5 threshold ignores operational FPR constraints; evaluation lacks TNR (True Negative Rate) metrics.

### Future Improvement Plan
To address these, implement a phased approach:

- Retrain with 50,000 normal samples (50:1 ratio) from OpenOrca and ultrachat_200k.
- Apply class weighting (attack weight = 50) and Focal Loss (γ=2.0).
- Evaluate TNR on NotInject (339 benign samples); calibrate threshold to FPR < 5% using ROC analysis.
- Expected: 50-70% FPR reduction, TNR > 85%.



## License

MIT
