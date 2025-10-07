#!/usr/bin/env python3
"""Quick check of specific samples."""

from datasets import load_from_disk
import pandas as pd

dataset = load_from_disk("data/processed/dataset")

indices_to_check = [0, 1, 1000, 1500, 2000]

normal_sources = {"notinject", "openorca", "ultrachat"}
attack_sources = {
    "deepset",
    "jailbreakbench",
    "harmbench_contextual",
    "harmbench_copyright",
    "harmbench_standard",
    "chatgpt_jailbreak",
}


def analyze_split(split_name: str):
    split_dataset = dataset[split_name]
    print("\n" + "=" * 100)
    print(f"{split_name.upper()} SPLIT - Specific Samples")
    print("=" * 100)

    for idx in indices_to_check:
        if idx < len(split_dataset):
            sample = split_dataset[idx]
            print(f"\nIndex {idx}:")
            print(f"  Source: {sample.get('source', 'N/A')}")
            print(
                f"  Label: {sample['label']} ({'normal' if sample['label'] == 0 else 'attack'})"
            )
            text = sample.get("text", "")
            preview = text[:100] + ("..." if len(text) > 100 else "")
            print(f"  Text: {preview}")

    print("\n" + "=" * 100)
    print(f"LABEL DISTRIBUTION BY SOURCE ({split_name.upper()} SPLIT)")
    print("=" * 100)

    df_list = [
        {"source": sample.get("source", "unknown"), "label": sample["label"]}
        for sample in split_dataset
    ]
    if not df_list:
        print("(split is empty)")
        return

    df = pd.DataFrame(df_list)
    summary = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    summary.columns = [f"label_{col}" for col in summary.columns]
    summary["total"] = summary.sum(axis=1)
    print(summary)

    print("\n" + "=" * 100)
    print(f"CHECKING FOR LABEL ERRORS ({split_name.upper()} SPLIT)")
    print("=" * 100)

    errors_found = False

    for source in normal_sources:
        source_df = df[df["source"] == source]
        if len(source_df) > 0:
            attack_count = (source_df["label"] == 1).sum()
            if attack_count > 0:
                print(
                    f"❌ ERROR: '{source}' (should be normal/0) has {attack_count} samples with label=1"
                )
                errors_found = True

    for source in attack_sources:
        source_df = df[df["source"] == source]
        if len(source_df) > 0:
            normal_count = (source_df["label"] == 0).sum()
            if normal_count > 0:
                print(
                    f"❌ ERROR: '{source}' (should be attack/1) has {normal_count} samples with label=0"
                )
                errors_found = True

    if not errors_found:
        print("✅ No label errors found! All sources have correct labels.")


splits_to_check = [
    split for split in ["train", "validation", "test"] if split in dataset
]

for split in splits_to_check:
    analyze_split(split)
