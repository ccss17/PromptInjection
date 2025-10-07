#!/usr/bin/env python3
"""
Script to prepare datasets for prompt injection detection.

Usage:
    python scripts/prepare_data.py --output_dir=data/processed --normal_size=50000

This script will:
1. Load ALL NotInject samples (hard negatives) - never truncated
2. Load normal prompts from OpenOrca and UltraChat to fill remaining budget
3. Load ALL attack prompts from JailbreakBench, HarmBench, deepset, and ChatGPT-Jailbreak
4. Balance datasets and create train/validation/test splits (80/10/10)
5. Calculate token length statistics (p50-p99, per-class)
6. Suggest optimal sequence lengths
7. Save processed datasets and statistics

NOTE: Attack samples and NotInject samples are ALWAYS fully used regardless of parameters.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    ClassLabel,
    DownloadMode,
    concatenate_datasets,
    load_dataset,
)
from tqdm import tqdm
from transformers import AutoTokenizer
from langdetect import detect_langs, LangDetectException
import fire


# Global dataset configurations
# NOTE: Attack datasets have NO max_samples - we ALWAYS use ALL attack samples
ATTACK_DATASETS_CONFIG = [
    {
        "name": "JailbreakBench/JBB-Behaviors",
        "hf_path": "JailbreakBench/JBB-Behaviors",
        "config_name": "behaviors",
        "splits_to_load": [
            "harmful",
            "benign",
        ],  # JBB has harmful/benign splits, not train
        "transform": lambda example: {
            "text": example.get(
                "Goal", example.get("Behavior", example.get("behavior", ""))
            ),
            "source": "jailbreakbench",
        },
        "source_name": "jailbreakbench",
    },
    {
        "name": "walledai/HarmBench",
        "hf_path": "walledai/HarmBench",
        "configs": ["contextual", "copyright", "standard"],
        "transform": lambda example, cfg_name: {
            "text": example.get(
                "prompt",
                example.get("text", example.get("behavior", "")),
            ),
            "source": f"harmbench_{cfg_name}",
        },
        "source_name": "harmbench",
    },
    {
        "name": "deepset/prompt-injections",
        "hf_path": "deepset/prompt-injections",
        "split": "train",
        "text_fields": ["text"],
        "source_name": "deepset",
    },
    {
        "name": "rubend18/ChatGPT-Jailbreak-Prompts",
        "hf_path": "rubend18/ChatGPT-Jailbreak-Prompts",
        "split": "train",
        "text_fields": ["Prompt", "prompt", "text"],
        "source_name": "chatgpt_jailbreak",
    },
]


def calculate_length_statistics(
    dataset: Dataset,
    tokenizer_name: str = "answerdotai/ModernBERT-large",
    percentiles: List[int] = [50, 75, 90, 95, 96, 97, 98, 99],
) -> Dict:
    """Calculate token length statistics for sequence length determination."""
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

    # Add percentiles
    for p in percentiles:
        stats[f"p{p}"] = float(np.percentile(lengths, p))

    # Suggest sequence lengths
    suggested_lengths = []
    for p in [95, 96, 97, 98]:
        if f"p{p}" in stats:
            p_value = stats[f"p{p}"]
        else:
            p_value = float(np.percentile(lengths, p))
            stats[f"p{p}"] = p_value

        common_lengths = [128, 256, 512, 1024, 2048, 4096]
        for length in common_lengths:
            if p_value <= length:
                suggested_lengths.append((p, length))
                break

    stats["suggested_seq_lengths"] = suggested_lengths

    # Store custom length coverage in stats
    custom_lengths = [256, 384, 512, 768, 1024, 1536, 2048]
    stats["custom_length_coverage"] = {
        str(length): {
            "coverage_percent": float(
                (lengths <= length).sum() / len(lengths) * 100
            ),
            "samples_fit": int((lengths <= length).sum()),
            "samples_truncated": int((lengths > length).sum()),
            "avg_truncation": float(np.mean(np.maximum(0, lengths - length))),
            "max_truncation": int(np.maximum(0, lengths - length).max()),
        }
        for length in custom_lengths
    }

    return stats


def print_dataset_info(
    stats_path: str, info_path: str, output_dir: str = None
):
    """
    Comprehensive function to print all dataset information including:
    - Dataset summary (train/val/test splits, label distributions)
    - Token length statistics (overall and per-class)
    - Training recommendations
    - File locations and next steps
    """
    # Load existing stats
    with open(stats_path, "r") as f:
        stats = json.load(f)

    # Load dataset info
    with open(info_path, "r") as f:
        info = json.load(f)

    # 1. Dataset Summary
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"  Training: {info['train_size']} samples")
    print(f"    - Normal: {info['label_distribution_train']['normal']}")
    print(f"    - Attack: {info['label_distribution_train']['attack']}")
    print(f"  Validation: {info['validation_size']} samples")
    print(f"    - Normal: {info['label_distribution_validation']['normal']}")
    print(f"    - Attack: {info['label_distribution_validation']['attack']}")
    print(f"  Test: {info['test_size']} samples")
    print(f"    - Normal: {info['label_distribution_test']['normal']}")
    print(f"    - Attack: {info['label_distribution_test']['attack']}")
    print(f"  Sources: {', '.join(info['sources'])}")

    # 2. Overall Token Length Statistics
    print("\n" + "=" * 60)
    print("Token Length Statistics (Overall)")
    print("=" * 60)
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

    # 3. Coverage for common sequence lengths
    if "custom_length_coverage" in stats:
        print("\n  Coverage for common sequence lengths:")
        for length_str, coverage_info in sorted(
            stats["custom_length_coverage"].items(), key=lambda x: int(x[0])
        ):
            length = int(length_str)
            coverage_pct = coverage_info["coverage_percent"]
            avg_trunc = coverage_info["avg_truncation"]
            print(
                f"    {length:4d} tokens: {coverage_pct:5.1f}% samples fit, "
                f"avg truncation: {avg_trunc:6.1f} tokens"
            )

    # 4. Per-Class Statistics
    if "per_class" in stats:
        print("\n" + "=" * 60)
        print("Per-Class Statistics")
        print("=" * 60)

        if "normal" in stats["per_class"]:
            normal_stats = stats["per_class"]["normal"]
            print(
                f"\n  Normal Prompts ({normal_stats.get('count', 'N/A')} samples):"
            )
            print(f"    Mean: {normal_stats['mean']:.1f}")
            print(f"    Median: {normal_stats.get('median', 'N/A')}")
            print(f"    p95: {normal_stats.get('p95', 'N/A')}")
            print(f"    p98: {normal_stats.get('p98', 'N/A')}")

        if "attack" in stats["per_class"]:
            attack_stats = stats["per_class"]["attack"]
            print(
                f"\n  Attack Prompts ({attack_stats.get('count', 'N/A')} samples):"
            )
            print(f"    Mean: {attack_stats['mean']:.1f}")
            print(f"    Median: {attack_stats.get('median', 'N/A')}")
            print(f"    p95: {attack_stats.get('p95', 'N/A')}")
            print(f"    p98: {attack_stats.get('p98', 'N/A')}")

    # 5. Training Recommendations
    print("\n" + "=" * 60)
    print("Training Recommendations")
    print("=" * 60)

    if stats.get("suggested_seq_lengths"):
        print("\nRecommended sequence lengths based on p95-p98 coverage:")
        for p, length in stats["suggested_seq_lengths"]:
            print(f"  * seq_length={length} (covers p{p})")

        print("\nSuggested starting point:")
        suggested = [
            s for s in stats["suggested_seq_lengths"] if s[0] in [96, 97]
        ]
        if suggested:
            p, length = suggested[0]
            print(
                f"  Use seq_length={length} for a good balance of coverage and efficiency"
            )
        else:
            print(f"  Use seq_length={stats['suggested_seq_lengths'][0][1]}")

    # 6. File locations and next steps
    if output_dir:
        print(f"\nFiles saved to: {output_dir}/")
        print("  - dataset/           (HuggingFace Dataset format)")
        print("  - train.parquet      (Training data)")
        print("  - validation.parquet (Validation data)")
        print("  - test.parquet       (Test data)")
        print("  - length_statistics.json")
        print("  - dataset_info.json")

        print("\nNext steps:")
        print("  1. Review length statistics in length_statistics.json")
        print("  2. Choose sequence length for training")
        print("  3. Run training script with chosen configuration")

        suggested_seq_len = (
            stats["suggested_seq_lengths"][0][1]
            if stats.get("suggested_seq_lengths")
            else 512
        )
        print(f"     python scripts/train.py --seq-length {suggested_seq_len}")


class PromptInjectionDatasetBuilder:
    """Build balanced dataset for prompt injection detection."""

    def __init__(
        self,
        output_dir: str = "data/processed",
        normal_sample_size: int = 50000,
        seed: int = 42,
    ):
        """
        Initialize dataset builder.

        Args:
            output_dir: Directory to save processed datasets
            normal_sample_size: Number of normal prompts to sample (will be balanced with attacks)
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normal_sample_size = normal_sample_size
        self.seed = seed

    def _load_prompts(
        self,
        datasets_config: List[Dict],
        label: int,
        category_name: str,
    ) -> Dataset:
        """
        Utility function to load and combine multiple datasets.

        Args:
            datasets_config: List of dataset configurations
            label: Label to assign (0 for normal, 1 for attack)
            category_name: Name of the category for logging

        Returns:
            Combined dataset with label and source columns
        """
        print(f"Loading {category_name} prompts...")
        combined_datasets = []

        for config in datasets_config:
            dataset_name = config["name"]
            print(f"  - Loading {dataset_name}...")

            try:
                transform_applied = False
                if config.get("streaming", False):
                    # For streaming datasets (e.g., OpenOrca)
                    ds = load_dataset(
                        config["hf_path"],
                        split=config.get("split", "train"),
                        streaming=True,
                    )
                    samples = []
                    max_samples = config.get("max_samples", None)

                    for i, example in enumerate(
                        tqdm(ds, desc=dataset_name, total=max_samples)
                    ):
                        if max_samples and i >= max_samples:
                            break

                        # Extract text using field mapping
                        text = None
                        for field in config.get("text_fields", ["text"]):
                            text = example.get(field)
                            if text:
                                break

                        if text:
                            samples.append(
                                {
                                    "text": text,
                                    "source": config.get(
                                        "source_name", dataset_name.lower()
                                    ),
                                }
                            )

                    ds = Dataset.from_list(samples)

                else:
                    # For regular datasets (single or multi-config)
                    if "configs" in config:
                        # Load multiple configs and combine
                        config_datasets = []
                        for cfg_name in config["configs"]:
                            cfg_ds = load_dataset(
                                config["hf_path"],
                                cfg_name,
                                split=config.get("split", "train"),
                            )

                            # Apply transform if specified
                            if "transform" in config:
                                cfg_ds = cfg_ds.map(
                                    lambda x: config["transform"](x, cfg_name)
                                )
                                transform_applied = True

                            config_datasets.append(cfg_ds)
                            print(f"    Loaded {len(cfg_ds)} from {cfg_name}")

                        ds = concatenate_datasets(config_datasets)

                    elif "splits_to_load" in config:
                        # Load multiple splits and combine (e.g., JailbreakBench with harmful/benign)
                        split_datasets = []
                        for split_name in config["splits_to_load"]:
                            load_args = {
                                "path": config["hf_path"],
                                "split": split_name,
                                "trust_remote_code": config.get(
                                    "trust_remote_code", True
                                ),
                            }
                            if config.get("config_name"):
                                load_args["name"] = config["config_name"]
                            if config.get("download_mode"):
                                load_args["download_mode"] = config[
                                    "download_mode"
                                ]

                            split_ds = load_dataset(**load_args)

                            # Apply transform if specified
                            if "transform" in config:
                                split_ds = split_ds.map(config["transform"])
                                transform_applied = True

                            split_datasets.append(split_ds)
                            print(
                                f"    Loaded {len(split_ds)} from {split_name}"
                            )

                        ds = concatenate_datasets(split_datasets)

                    else:
                        # Single config dataset
                        load_kwargs = {
                            "path": config["hf_path"],
                            "split": config.get("split", "train"),
                            "trust_remote_code": config.get(
                                "trust_remote_code", True
                            ),
                        }

                        if config.get("config_name"):
                            load_kwargs["name"] = config["config_name"]

                        # Add download_mode if specified (for problematic datasets like NotInject)
                        if "download_mode" in config:
                            load_kwargs["download_mode"] = config[
                                "download_mode"
                            ]

                        ds = load_dataset(**load_kwargs)

                    # Apply transform or default text extraction
                    if "transform" in config and not transform_applied:
                        ds = ds.map(config["transform"])
                    elif "text_fields" in config:
                        # Default text extraction
                        def extract_text(example):
                            text = None
                            for field in config.get("text_fields", ["text"]):
                                text = example.get(field)
                                if text:
                                    break
                            return {
                                "text": text if text else "",
                                "source": config.get(
                                    "source_name", dataset_name.lower()
                                ),
                            }

                        ds = ds.map(extract_text)

                # Apply post-processing if specified
                if "filter" in config:
                    ds = ds.filter(config["filter"])

                if "max_samples" in config and not config.get(
                    "streaming", False
                ):
                    ds = ds.shuffle(seed=self.seed).select(
                        range(min(config["max_samples"], len(ds)))
                    )

                combined_datasets.append(ds)
                print(f"    Loaded {len(ds)} samples")

            except Exception as e:
                print(f"    Error: Could not load {dataset_name}: {e}")
                import traceback

                traceback.print_exc()

        # Combine all datasets
        if not combined_datasets:
            raise ValueError(f"No {category_name} prompts loaded!")

        result_ds = concatenate_datasets(combined_datasets)

        # Filter out empty texts
        result_ds = result_ds.filter(
            lambda x: x.get("text") and len(x["text"].strip()) > 0
        )

        # Add label
        result_ds = result_ds.map(lambda x: {"labels": label})

        print(f"  Loaded {len(result_ds)} {category_name} prompts")
        return result_ds

    def load_normal_prompts(self) -> Dataset:
        """Load and combine normal instruction prompts, including NotInject hard negatives."""
        print("Loading normal prompts...")

        notinject_config = {
            "name": "NotInject (hard negatives)",
            "hf_path": "leolee99/NotInject",
            "splits_to_load": [
                "NotInject_one",
                "NotInject_two",
                "NotInject_three",
            ],
            "transform": lambda example: {
                "text": example.get("text", example.get("prompt", "")),
                "source": "notinject",
            },
            "filter": self._is_english_filter(),
            "source_name": "notinject",
            "download_mode": DownloadMode.FORCE_REDOWNLOAD,
        }

        try:
            notinject_ds = self._load_prompts(
                [notinject_config],
                label=0,
                category_name="NotInject (hard negatives)",
            )
        except ValueError:
            print("  Warning: No NotInject samples loaded")
            notinject_ds = None

        if notinject_ds:
            notinject_count = len(notinject_ds)
            print(f"  Keeping ALL {notinject_count:,} NotInject samples")
        else:
            notinject_count = 0

        remaining_budget = max(0, self.normal_sample_size - notinject_count)
        print(
            f"  Remaining budget for other normal samples: {remaining_budget:,}"
        )

        other_ds = None
        if remaining_budget > 0:
            openorca_budget = remaining_budget // 2
            ultrachat_budget = remaining_budget - openorca_budget

            other_config = [
                {
                    "name": "Open-Orca/OpenOrca",
                    "hf_path": "Open-Orca/OpenOrca",
                    "split": "train",
                    "streaming": True,
                    "max_samples": openorca_budget,
                    "text_fields": ["question", "instruction"],
                    "source_name": "openorca",
                },
                {
                    "name": "HuggingFaceH4/ultrachat_200k",
                    "hf_path": "HuggingFaceH4/ultrachat_200k",
                    "split": "train_sft",
                    "max_samples": ultrachat_budget,
                    "transform": lambda example: {
                        "text": next(
                            (
                                msg["content"]
                                for msg in example.get("messages", [])
                                if msg.get("role") == "user"
                            ),
                            "",
                        ),
                        "source": "ultrachat",
                    },
                    "source_name": "ultrachat",
                },
            ]

            other_ds = self._load_prompts(
                other_config, label=0, category_name="other normal"
            )

        if notinject_ds and other_ds:
            return concatenate_datasets([notinject_ds, other_ds])
        if notinject_ds:
            if remaining_budget == 0:
                print(
                    "  Warning: NotInject samples exceed normal_sample_size; using all NotInject samples"
                )
            return notinject_ds
        if other_ds:
            return other_ds

        raise ValueError("No normal prompts loaded!")

    def _is_english_filter(self):
        """Return English filter function for NotInject."""

        def is_english(example):
            try:
                text = example.get("text", example.get("prompt", ""))
                if not text or len(text.strip()) == 0:
                    return False
                langs = detect_langs(text)
                return langs[0].lang == "en" and langs[0].prob > 0.9
            except (LangDetectException, Exception):
                return False

        return is_english

    def load_attack_prompts(self) -> Dataset:
        """Load and combine attack/jailbreak prompts."""
        return self._load_prompts(
            ATTACK_DATASETS_CONFIG, label=1, category_name="attack"
        )

    def balance_datasets(
        self, normal_ds: Dataset, attack_ds: Dataset, ratio: float = 0.5
    ) -> Tuple[Dataset, Dataset]:
        """
        Balance normal and attack datasets.

        IMPORTANT: Since attack samples are limited, we USE ALL available attack samples
        and then sample normal prompts to match the desired ratio.

        Args:
            normal_ds: Normal prompts dataset
            attack_ds: Attack prompts dataset
            ratio: Target ratio of attack samples (0.5 = balanced)

        Returns:
            Tuple of (balanced_normal, balanced_attack)
        """
        print(f"\nBalancing datasets (target attack ratio: {ratio})...")

        n_normal = len(normal_ds)
        n_attack = len(attack_ds)

        print(f"  Before: {n_normal:,} normal, {n_attack:,} attack")

        # USE ALL ATTACK SAMPLES (they are limited and valuable)
        attack_balanced = attack_ds
        attack_size = n_attack

        # Separate NotInject samples to guarantee they are always included
        notinject_ds = normal_ds.filter(
            lambda x: x.get("source") == "notinject"
        )
        other_normal_ds = normal_ds.filter(
            lambda x: x.get("source") != "notinject"
        )

        notinject_count = len(notinject_ds)
        other_normal_count = len(other_normal_ds)

        print(
            f"  NotInject samples available: {notinject_count:,} (will be kept in full)"
        )

        # Calculate target number of normal samples for desired ratio
        target_normal_size = int(attack_size * (1 - ratio) / ratio)

        if target_normal_size < notinject_count:
            print(
                "  Requested ratio would drop some NotInject samples; keeping them all instead"
            )
            normal_size = notinject_count
            needed_other = 0
        else:
            normal_size = target_normal_size
            needed_other = target_normal_size - notinject_count

        if needed_other > other_normal_count:
            print(
                "  Warning: Not enough additional normal samples to meet target ratio"
            )
            print(
                f"     Need {needed_other:,} from other sources but only have {other_normal_count:,}"
            )
            needed_other = other_normal_count
            normal_size = notinject_count + other_normal_count

        if normal_size > n_normal:
            normal_size = n_normal

        # Build the balanced normal dataset with all NotInject samples plus additional normal prompts
        if needed_other > 0:
            other_sample = other_normal_ds.shuffle(seed=self.seed).select(
                range(needed_other)
            )
            normal_balanced = concatenate_datasets(
                [notinject_ds, other_sample]
            )
        else:
            normal_balanced = notinject_ds

        # Shuffle to avoid grouped sources
        normal_balanced = normal_balanced.shuffle(seed=self.seed)

        print(
            f"  After: {len(normal_balanced):,} normal, {len(attack_balanced):,} attack"
        )
        final_ratio = len(attack_balanced) / (
            len(normal_balanced) + len(attack_balanced)
        )
        print(f"  Attack ratio: {final_ratio:.2%}")
        print(f"  Using ALL {len(attack_balanced):,} attack samples!")
        if abs(final_ratio - ratio) > 1e-6:
            print(
                f"  Note: Actual ratio differs from target {ratio:.0%} due to keeping all NotInject samples"
            )

        return normal_balanced, attack_balanced

    def _stratified_split(
        self, dataset: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset into train/validation/test with 80/10/10 ratio."""

        train_test = dataset.train_test_split(
            test_size=0.2, seed=self.seed, shuffle=True
        )
        val_test = train_test["test"].train_test_split(
            test_size=0.5, seed=self.seed, shuffle=True
        )
        return train_test["train"], val_test["train"], val_test["test"]

    @staticmethod
    def _label_distribution(dataset: Dataset) -> Dict[str, int]:
        counts = Counter(dataset["labels"])
        return {"normal": counts.get(0, 0), "attack": counts.get(1, 0)}

    def build_dataset(self, balance_ratio: float = 0.5) -> DatasetDict:
        """
        Build complete train/validation/test dataset.

        Args:
            balance_ratio: Ratio of attack samples (0.5 = balanced, 0.4 = 40% attack)

        Returns:
            DatasetDict with train/validation/test splits (80/10/10)
        """
        print("Building Prompt Injection Detection Dataset\n")

        # Load datasets
        normal_ds = self.load_normal_prompts()
        attack_ds = self.load_attack_prompts()

        # Balance datasets
        normal_balanced, attack_balanced = self.balance_datasets(
            normal_ds, attack_ds, balance_ratio
        )

        # Combine and split with stratification
        print("\nCreating train/validation/test splits (80/10/10)...")

        # Ensure label column has proper type for stratification
        normal_balanced = normal_balanced.cast_column(
            "labels", ClassLabel(names=["normal", "attack"])
        )
        attack_balanced = attack_balanced.cast_column(
            "labels", ClassLabel(names=["normal", "attack"])
        )

        normal_splits = self._stratified_split(normal_balanced)
        attack_splits = self._stratified_split(attack_balanced)

        dataset_dict = DatasetDict()
        for split_name, normal_split, attack_split in zip(
            ["train", "validation", "test"], normal_splits, attack_splits
        ):
            combined = concatenate_datasets([normal_split, attack_split])
            dataset_dict[split_name] = combined.shuffle(seed=self.seed)

        # Remove legacy single-label column in favor of `labels`
        for split_name, split_ds in dataset_dict.items():
            if "label" in split_ds.column_names:
                dataset_dict[split_name] = split_ds.remove_columns("label")

        return dataset_dict

    def save_dataset(self, dataset_dict: DatasetDict, stats: Dict = None):
        """Save dataset and statistics to disk."""
        print(f"\nSaving dataset to {self.output_dir}...")

        empty_splits = [
            name for name, split in dataset_dict.items() if len(split) == 0
        ]
        for split_name in empty_splits:
            print(f"  Warning: Skipping empty split '{split_name}' (0 rows)")

        if empty_splits:
            dataset_dict = DatasetDict(
                {
                    name: split
                    for name, split in dataset_dict.items()
                    if name not in empty_splits
                }
            )

        # Save dataset
        dataset_dict.save_to_disk(str(self.output_dir / "dataset"))

        # Save to parquet
        print("  - Saving in HuggingFace format...")
        for split_name in ["train", "validation", "test"]:
            if (
                split_name in dataset_dict
                and len(dataset_dict[split_name]) > 0
            ):
                dataset_dict[split_name].to_parquet(
                    str(self.output_dir / f"{split_name}.parquet")
                )
                print(
                    f"    * {split_name}.parquet ({len(dataset_dict[split_name])} rows)"
                )
            elif split_name in ["train", "validation", "test"]:
                print(f"    Warning: '{split_name}' split is empty!")

        # Save statistics
        if stats:
            stats_file = self.output_dir / "length_statistics.json"
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"  - Statistics saved to {stats_file}")

        # Create dataset info
        info = {
            "train_size": len(dataset_dict["train"]),
            "validation_size": len(dataset_dict["validation"]),
            "test_size": len(dataset_dict["test"]),
            "label_distribution_train": self._label_distribution(
                dataset_dict["train"]
            ),
            "label_distribution_validation": self._label_distribution(
                dataset_dict["validation"]
            ),
            "label_distribution_test": self._label_distribution(
                dataset_dict["test"]
            ),
            "sources": sorted(set(dataset_dict["train"]["source"])),
        }

        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)

        print("  Dataset saved successfully!")


def main(
    output_dir: str = "data/processed",
    normal_size: int = 50000,
    balance_ratio: float = 0.5,
    seed: int = 42,
    tokenizer: str = "answerdotai/ModernBERT-large",
    force: bool = False,
):
    """
    Prepare datasets for prompt injection detection.

    Args:
        output_dir: Output directory for processed datasets
        normal_size: Number of normal prompts to sample
        balance_ratio: Ratio of attack samples (0.5 = balanced, 0.4 = 40% attack)
        seed: Random seed for reproducibility
        tokenizer: Tokenizer for length statistics
        force: Force re-creation even if dataset exists
        custom_seq_lengths: Comma-separated custom sequence lengths to analyze (e.g., "768,896,1536")
    """
    output_path = Path(output_dir)
    dataset_path = output_path / "dataset"
    stats_path = output_path / "length_statistics.json"
    info_path = output_path / "dataset_info.json"

    # Check if dataset already exists
    dataset_exists = (
        dataset_path.exists() and stats_path.exists() and info_path.exists()
    )

    if dataset_exists and not force:
        print("Dataset already exists! Loading existing data...\n")
        print_dataset_info(stats_path, info_path, output_dir)
        return

    # Create new dataset
    print("Starting dataset preparation...")
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

    # Calculate statistics on combined train+validation
    print("\n" + "=" * 60)
    print("Calculating Token Length Statistics")
    print("=" * 60)
    combined_train_val = concatenate_datasets(
        [dataset_dict["train"], dataset_dict["validation"]]
    )
    stats = calculate_length_statistics(
        combined_train_val,
        tokenizer_name=tokenizer,
        percentiles=[50, 75, 90, 95, 96, 97, 98, 99],
    )

    # Calculate per-class statistics
    print("Calculating per-class statistics...")
    normal_samples = combined_train_val.filter(lambda x: x["labels"] == 0)
    attack_samples = combined_train_val.filter(lambda x: x["labels"] == 1)

    normal_stats = calculate_length_statistics(
        normal_samples, tokenizer_name=tokenizer, percentiles=[95, 98]
    )
    attack_stats = calculate_length_statistics(
        attack_samples, tokenizer_name=tokenizer, percentiles=[95, 98]
    )

    # Save everything
    stats["per_class"] = {"normal": normal_stats, "attack": attack_stats}
    builder.save_dataset(dataset_dict, stats)

    # Print comprehensive dataset info with all statistics and recommendations
    print("\nDataset preparation complete!\n")
    print_dataset_info(stats_path, info_path, output_dir)


if __name__ == "__main__":
    fire.Fire(main)
