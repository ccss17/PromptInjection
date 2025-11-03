from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk
)
from tqdm import tqdm
from transformers import AutoTokenizer
from langdetect import detect_langs, LangDetectException


LABEL_NORMAL = 0
LABEL_ATTACK = 1


def calculate_length_statistics(
    dataset: Dataset,
    tokenizer_name: str = "answerdotai/ModernBERT-large",
    percentiles: List[int] = [50, 75, 90, 95, 96, 97, 98, 99],
) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    lengths = []
    for example in tqdm(dataset, desc="Tokenizing"):
        tokens = tokenizer(
            example["text"], truncation=False, add_special_tokens=True
        )
        lengths.append(len(tokens["input_ids"]))

    lengths = np.array(lengths)

    stats = {
        "count": len(lengths),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "median": float(np.median(lengths)),
    }

    for p in percentiles:
        stats[f"p{p}"] = float(np.percentile(lengths, p))

    return stats


def print_dataset_info(
    dataset_path: str,
    tokenizer_name: str = "answerdotai/ModernBERT-large"
):
    print(f"Loading dataset from {dataset_path}")
    dataset_dict = load_from_disk(dataset_path)
    
    print(f"{'=' * 60}\nDataset Summary\n{'=' * 60}")
    print(f"  Training: {len(dataset_dict['train'])} samples")
    
    train_labels = Counter(dataset_dict['train']['labels'])
    print(f"    - Normal: {train_labels.get(LABEL_NORMAL, 0)}")
    print(f"    - Attack: {train_labels.get(LABEL_ATTACK, 0)}")
    
    print(f"  Validation: {len(dataset_dict['validation'])} samples")
    val_labels = Counter(dataset_dict['validation']['labels'])
    print(f"    - Normal: {val_labels.get(LABEL_NORMAL, 0)}")
    print(f"    - Attack: {val_labels.get(LABEL_ATTACK, 0)}")
    
    print(f"  Test: {len(dataset_dict['test'])} samples")
    test_labels = Counter(dataset_dict['test']['labels'])
    print(f"    - Normal: {test_labels.get(LABEL_NORMAL, 0)}")
    print(f"    - Attack: {test_labels.get(LABEL_ATTACK, 0)}")
    
    sources = sorted(set(dataset_dict['train']['source']))
    print(f"  Sources: {', '.join(sources)}")

    # Calculate statistics on combined train+validation
    print(f"\n{'=' * 60}\nToken Length Statistics (Overall)\n{'=' * 60}")
    combined_train_val = concatenate_datasets(
        [dataset_dict["train"], dataset_dict["validation"]]
    )
    stats = calculate_length_statistics(combined_train_val, tokenizer_name=tokenizer_name)
    
    print(f"  Count: {stats['count']}")
    print(f"  Mean: {stats['mean']:.1f}")
    print(f"  Median: {stats['median']:.1f}")
    print(f"  Std: {stats['std']:.1f}")
    print(f"  Min: {stats['min']}")
    print(f"  Max: {stats['max']}")

    print("\n  Percentiles:")
    for p in [50, 75, 90, 95, 96, 97, 98, 99]:
        if f"p{p}" in stats:
            print(f"    p{p}: {stats[f'p{p}']:.1f}")

    # Per-Class Statistics
    print(f"\n{'=' * 60}\nPer-Class Statistics\n{'=' * 60}")
    
    normal_samples = combined_train_val.filter(lambda x: x["labels"] == LABEL_NORMAL)
    attack_samples = combined_train_val.filter(lambda x: x["labels"] == LABEL_ATTACK)

    normal_stats = calculate_length_statistics(
        normal_samples, tokenizer_name=tokenizer_name, percentiles=[95, 98]
    )
    print(f"\n  Normal Prompts ({normal_stats['count']} samples):")
    print(f"    Mean: {normal_stats['mean']:.1f}")
    print(f"    Median: {normal_stats['median']:.1f}")
    print(f"    p95: {normal_stats['p95']:.1f}")
    print(f"    p98: {normal_stats['p98']:.1f}")

    attack_stats = calculate_length_statistics(
        attack_samples, tokenizer_name=tokenizer_name, percentiles=[95, 98]
    )
    print(f"\n  Attack Prompts ({attack_stats['count']} samples):")
    print(f"    Mean: {attack_stats['mean']:.1f}")
    print(f"    Median: {attack_stats['median']:.1f}")
    print(f"    p95: {attack_stats['p95']:.1f}")
    print(f"    p98: {attack_stats['p98']:.1f}")



def _is_english(text: str) -> bool:
    try:
        if not text or len(text.strip()) == 0:
            return False
        langs = detect_langs(text)
        return langs[0].lang == "en" and langs[0].prob > 0.9
    except (LangDetectException, Exception):
        return False


def load_notinject() -> Dataset:
    print("Loading NotInject")
    splits = ["NotInject_one", "NotInject_two", "NotInject_three"]
    datasets = []
    for split in splits:
        ds = load_dataset("leolee99/NotInject", split=split)
        datasets.append(ds)
    
    combined = concatenate_datasets(datasets)
    combined = combined.map(lambda x: {"text": x["prompt"], "source": "notinject", "labels": LABEL_NORMAL})
    combined = combined.filter(lambda x: _is_english(x["text"]))
    print(f"  Total: {len(combined):,} NotInject samples\n")
    return combined


def load_openorca() -> Dataset:
    ds = load_dataset("Open-Orca/OpenOrca", split="train")
    ds = ds.map(lambda x: {"text": x["question"], "source": "openorca", "labels": LABEL_NORMAL})
    print(f"  Total: {len(ds):,} OpenOrca samples\n")
    return ds


def load_ultrachat() -> Dataset:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.map(lambda x: {"text": x["prompt"], "source": "ultrachat", "labels": LABEL_NORMAL})
    print(f"  Total: {len(ds):,} UltraChat samples\n")
    return ds


def load_jbb() -> Dataset:
    # Load "behaviors" subset (harmful and benign splits)
    behaviors_splits = ["harmful", "benign"]
    datasets = []
    
    for split in behaviors_splits:
        ds = load_dataset(
            "JailbreakBench/JBB-Behaviors",
            name="behaviors",
            split=split,
        )
        ds = ds.map(lambda x: {"text": x["Goal"], "source": "jailbreakbench", "labels": LABEL_ATTACK})
        datasets.append(ds)
    
    # Load "judge_comparison" subset
    judge_ds = load_dataset(
        "JailbreakBench/JBB-Behaviors",
        name="judge_comparison",
        split="test",
    )
    judge_ds = judge_ds.map(lambda x: {"text": x["prompt"], "source": "jailbreakbench", "labels": LABEL_ATTACK})
    datasets.append(judge_ds)
    
    combined = concatenate_datasets(datasets)
    
    print(f"  Total: {len(combined):,} JailbreakBench samples\n")
    return combined


def load_harmbench() -> Dataset:
    configs = ["contextual", "copyright", "standard"]
    datasets = []
    
    for cfg in configs:
        ds = load_dataset("walledai/HarmBench", name=cfg, split="train")
        datasets.append(ds)
        print(f"  Loaded {len(ds):,} from {cfg}")
    
    combined = concatenate_datasets(datasets)
    combined = combined.map(lambda x: {"text": x["prompt"], "source": "harmbench", "labels": LABEL_ATTACK})
    
    print(f"  Total: {len(combined):,} HarmBench samples\n")
    return combined


def load_deepset() -> Dataset:
    train_ds = load_dataset("deepset/prompt-injections", split="train")
    test_ds = load_dataset("deepset/prompt-injections", split="test")
    
    combined = concatenate_datasets([train_ds, test_ds])
    combined = combined.map(lambda x: {"source": "deepset", "labels": LABEL_ATTACK})
    
    print(f"  Total: {len(combined):,} deepset samples\n")
    return combined


def load_jailbreak() -> Dataset:
    ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
    ds = ds.map(lambda x: {"text": x["Prompt"], "source": "jailbreak", "labels": LABEL_ATTACK})
    print(f"  Total: {len(ds):,} ChatGPT-Jailbreak samples\n")
    return ds


class PromptInjectionDatasetBuilder:
    def __init__(
        self,
        output_dir: str = "data/processed",
        normal_sample_size: int = 50000,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normal_sample_size = normal_sample_size
        self.seed = seed

    def load_normal_prompts(self) -> Dataset:
        print("Loading Normal Prompts\n")
        
        notinject_ds = load_notinject()
        notinject_count = len(notinject_ds) 
        
        ds_list = [notinject_ds] 
        remaining_budget = self.normal_sample_size - notinject_count
        per_dataset_budget = remaining_budget // 2
        
        # Load and sample OpenOrca
        openorca_ds = load_openorca()
        openorca_sample = openorca_ds.select(range(per_dataset_budget))
        print(f"  Sampled: {len(openorca_sample):,} OpenOrca samples\n")
        ds_list.append(openorca_sample)
        
        # Load and sample UltraChat
        ultrachat_budget = remaining_budget - per_dataset_budget
        ultrachat_ds = load_ultrachat()
        ultrachat_sample = ultrachat_ds.select(range(ultrachat_budget))
        print(f"  Sampled: {len(ultrachat_sample):,} UltraChat samples\n")
        ds_list.append(ultrachat_sample)
    
        ds = concatenate_datasets(ds_list)
        print(f"{'=' * 60}\nTotal normal prompts: {len(ds):,}\n{'=' * 60}\n")
        return ds

    def load_attack_prompts(self) -> Dataset:
        print("Loading Attack Prompts\n")
        datasets_to_combine = [
            load_jbb(),
            load_harmbench(),
            load_deepset(),
            load_jailbreak(),
        ]
        combined = concatenate_datasets(datasets_to_combine)
        print(f"{'=' * 60}\nTotal attack prompts: {len(combined):,}\n{'=' * 60}\n")
        return combined

    def balance_datasets(self, ratio: float = 0.5) -> Tuple[Dataset, Dataset]:
        normal_ds = self.load_normal_prompts()
        attack_ds = self.load_attack_prompts()
        
        # Separate NotInject samples to guarantee they are always included
        notinject_ds = normal_ds.filter(lambda x: x["source"] == "notinject")
        other_normal_ds = normal_ds.filter(lambda x: x["source"] != "notinject")
        notinject_count = len(notinject_ds)

        # Calculate target number of normal samples for desired ratio
        n_attack = len(attack_ds)
        target_normal_size = int(n_attack * (1 - ratio) / ratio)
        needed_other = target_normal_size - notinject_count
        other_sample = other_normal_ds.select(range(needed_other))
        normal_balanced = concatenate_datasets([notinject_ds, other_sample])

        print(f"  {len(normal_balanced):,} normal, {len(attack_ds):,} attack")
        return normal_balanced, attack_ds

    def _split_ds(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        train_test = dataset.train_test_split(test_size=0.2, seed=self.seed, shuffle=True)
        val_test = train_test["test"].train_test_split(test_size=0.5, seed=self.seed, shuffle=True)
        return train_test["train"], val_test["train"], val_test["test"]

    def build_dataset(self, balance_ratio: float = 0.5) -> DatasetDict:
        normal_balanced, attack_balanced = self.balance_datasets(balance_ratio)
        normal_splits = self._split_ds(normal_balanced)
        attack_splits = self._split_ds(attack_balanced)

        dataset_dict = DatasetDict()
        for split_name, normal_split, attack_split in zip(
            ["train", "validation", "test"], normal_splits, attack_splits
        ):
            combined = concatenate_datasets([normal_split, attack_split])
            dataset_dict[split_name] = combined.shuffle(seed=self.seed)

        return dataset_dict

    def save_dataset(self, dataset_dict: DatasetDict):
        """Save dataset to disk."""
        print(f"\nSaving dataset to {self.output_dir}")
        dataset_dict.save_to_disk(str(self.output_dir / "dataset"))
        print(f"  Dataset saved to {self.output_dir / 'dataset'}")


def main(
    output_dir: str = "data/processed",
    normal_size: int = 50000,
    balance_ratio: float = 0.5,
    seed: int = 42,
    tokenizer: str = "answerdotai/ModernBERT-large",
    force: bool = False,
):
    output_path = Path(output_dir)
    dataset_path = output_path / "dataset"

    # Check if dataset already exists
    if dataset_path.exists() and not force:
        print("Dataset already exists! Loading existing data\n")
        from datasets import load_from_disk
        dataset_dict = load_from_disk(str(dataset_path))
        print_dataset_info(dataset_dict, tokenizer_name=tokenizer)
        return

    # Create new dataset
    print("Starting dataset preparation")
    print(f"   Output directory: {output_dir}")
    print(f"   Normal sample size: {normal_size:,}")
    print(f"   Balance ratio: {balance_ratio} (attack samples)")
    print(f"   Random seed: {seed}")
    print(f"   Tokenizer: {tokenizer}\n")

    # Build dataset
    builder = PromptInjectionDatasetBuilder(
        output_dir=output_dir,
        normal_sample_size=normal_size,
        seed=seed,
    )

    # Load and build datasets
    dataset_dict = builder.build_dataset(balance_ratio=balance_ratio)

    # Save dataset
    builder.save_dataset(dataset_dict)

    # Print comprehensive dataset info with all statistics
    print("\nDataset preparation complete!\n")
    print_dataset_info(output_dir, tokenizer_name=tokenizer)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
