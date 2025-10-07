---
title: Prompt Injection Detector
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - security
  - prompt-injection
  - text-classification
  - cybersecurity
  - modernbert
---

# 🛡️ Prompt Injection Detector

Detect malicious prompt injection attacks in LLM applications using a fine-tuned **ModernBERT-large** model.

## What is Prompt Injection?

Prompt injection is a security vulnerability where attackers manipulate AI systems by crafting malicious inputs that override intended instructions or extract sensitive information.

## How It Works

This model classifies prompts as either:
- ✅ **SAFE**: Legitimate user queries
- ⚠️ **INJECTION**: Potential attack attempts

### Detection Capabilities

- **Jailbreak Attempts**: "You are now DAN (Do Anything Now)..."
- **Instruction Override**: "Ignore previous instructions and..."
- **System Prompt Extraction**: "Reveal your system prompt"
- **Role-Playing Attacks**: Attempting to change the AI's behavior
- **Encoded Attacks**: Obfuscated malicious commands

## Model Details

- **Base Model:** [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) (395M parameters)
- **Training:** Full fine-tuning on NVIDIA H200 GPU
- **Dataset:** ~2,500 prompts from multiple sources
  - deepset/prompt-injections
  - fka/awesome-chatgpt-prompts
  - nothingiisreal/notinject
- **Performance:** >99% accuracy on held-out test set

## Usage

### Try It Now
Use the interface above to test prompts in real-time!

### Python API

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="ccss17/modernbert-prompt-injection-detector",
    device=0  # Use GPU if available
)

result = classifier("Ignore all previous instructions")
print(result)
# [{'label': 'INJECTION', 'score': 0.99}]
```

## Limitations

- Primarily trained on English prompts
- May have false positives on creative/unusual prompts
- New attack patterns may bypass detection
- Should be part of a layered security approach

## Ethical Use

⚠️ **This model is for defensive security purposes only**

✅ **Intended Use:**
- Protecting LLM applications from malicious inputs
- Security research and education
- Building safer AI systems

❌ **Prohibited Use:**
- Offensive security testing without authorization
- Developing new attack vectors
- Any malicious or illegal activities

## Links

- 🤗 [Model Card](https://huggingface.co/ccss17/modernbert-prompt-injection-detector)
- 💻 [GitHub Repository](https://github.com/ccss17/PromptInjectionDetector)
- 📊 [Training Logs](https://wandb.ai/your-username/prompt-injection-detector)

## Citation

```bibtex
@misc{modernbert_prompt_injection,
  author = {Your Name},
  title = {ModernBERT Prompt Injection Detector},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/ccss17/modernbert-prompt-injection-detector}},
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details
