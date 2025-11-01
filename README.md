
- https://wandb.ai/cccsss17-xxx/prompt-injection-detector
- https://wandb.ai/cccsss17-xxx/prompt-injection-detector-hpo
- https://huggingface.co/ccss17/modernbert-prompt-injection-detector
- https://huggingface.co/spaces/ccss17/prompt-injection-detector

# Prompt Injection Detection — ModernBERT SFT


* **Training**: PyTorch, Transformers, Unsloth, PEFT, Flash-Attention
* **Optimization**: LoRA, Optuna HPO

**Prompt dataset construction (total 2,450 samples)**:

* **Benign:** [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), [Open-Orca/OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca), [leolee99/NotInject](https://huggingface.co/datasets/leolee99/NotInject)
* **Attack:** [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), [walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench), [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections), [rubend18/ChatGPT-Jailbreak-Prompts](https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts)

**Optuna-based HPO** — `HPO Logging (Wandb)`: [https://wandb.ai/cccsss17-xxx/prompt-injection-detector-hpo/runs/ayhldqbw?nw=nwusercccsss17](https://wandb.ai/cccsss17-xxx/prompt-injection-detector-hpo/runs/ayhldqbw?nw=nwusercccsss17)

| Learning Rate | Batch Size | LoRA Rank | LoRA Alpha |
| ------------: | :--------: | :-------: | :--------: |
|     4.439e-05 |     16     |     32    |     128    |

**Training (25 epochs, early stopping `patience=5`)** — `Train Logging (Wandb)`: [https://wandb.ai/cccsss17-xxx/prompt-injection-detector](https://wandb.ai/cccsss17-xxx/prompt-injection-detector)

* **Model**: ModernBERT-large (0.4B)
* **Total training steps**: 340 steps

## Results

Hugging Face model: [ccss17/modernbert-prompt-injection-detector](https://huggingface.co/ccss17/modernbert-prompt-injection-detector)

Hugging Face demo (Spaces): [ccss17/prompt-injection-detector Demo Spaces](https://huggingface.co/spaces/ccss17/prompt-injection-detector)

| Metric        | Baseline | LoRA Fine-tuned |
| ------------- | -------: | --------------: |
| **Accuracy**  |    61.4% |           98.0% |
| **Precision** |    73.3% |           97.6% |
| **Recall**    |    35.8% |           98.4% |
| **F1 Score**  |    48.1% |           98.0% |

**Notes — Observed issue**: The model sometimes falsely flags entirely novel benign prompts (e.g., "What’s the weather like in Seoul?") as attacks. Possible causes:

1. **Dataset imbalance**: Although trained with a 1:1 ratio, the real-world distribution contains far fewer attack samples.
2. **Distribution gap**: Benign samples derived from OpenOrca and ultrachat differ from real production inputs, causing domain shift.

# License

MIT
