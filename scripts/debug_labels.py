#!/usr/bin/env python3
"""Debug script to check dataset labels."""

from datasets import load_from_disk

print("Loading dataset...")
dataset = load_from_disk("data/processed/dataset")

print("\n" + "=" * 80)
print("CHECKING TRAIN SPLIT LABELS")
print("=" * 80)

# Check first 10 samples
for i in range(min(10, len(dataset["train"]))):
    sample = dataset["train"][i]
    source = sample.get("source", "unknown")
    label = sample["label"]
    text_preview = sample["text"][:80]

    # Expected label
    if source in ["notinject", "openorca", "ultrachat"]:
        expected = 0  # normal
    else:
        expected = 1  # attack

    status = "✅" if label == expected else "❌ WRONG"

    print(f"\nSample {i}:")
    print(f"  Source: {source}")
    print(f"  Label: {label} (expected: {expected}) {status}")
    print(f"  Text: {text_preview}...")

print("\n" + "=" * 80)
print("CHECKING SAMPLES FROM DIFFERENT SOURCES")
print("=" * 80)

# Find examples from each source
sources_to_check = [
    "notinject",
    "openorca",
    "ultrachat",
    "deepset",
    "jailbreakbench_harmful",
]

for source in sources_to_check:
    print(f"\nSearching for '{source}' samples...")
    found = False
    for i, sample in enumerate(dataset["train"]):
        if sample.get("source", "") == source:
            label = sample["label"]

            # Expected label
            if source in ["notinject", "openorca", "ultrachat"]:
                expected = 0  # normal
            else:
                expected = 1  # attack

            status = "✅" if label == expected else "❌ WRONG"

            print(f"  Index {i}: label={label} (expected={expected}) {status}")
            print(f"  Text: {sample['text'][:80]}...")
            found = True
            break

    if not found:
        print(f"  ⚠️  No samples found with source '{source}'")

# Count label distribution
print("\n" + "=" * 80)
print("LABEL DISTRIBUTION BY SOURCE")
print("=" * 80)

from collections import defaultdict

train_sources = defaultdict(lambda: {"label_0": 0, "label_1": 0})

for sample in dataset["train"]:
    source = sample.get("source", "unknown")
    label = sample["label"]

    if label == 0:
        train_sources[source]["label_0"] += 1
    else:
        train_sources[source]["label_1"] += 1

for source in sorted(train_sources.keys()):
    counts = train_sources[source]
    total = counts["label_0"] + counts["label_1"]
    print(f"\n{source}:")
    print(
        f"  Label 0 (normal):  {counts['label_0']:4d} ({counts['label_0'] / total * 100:.1f}%)"
    )
    print(
        f"  Label 1 (attack):  {counts['label_1']:4d} ({counts['label_1'] / total * 100:.1f}%)"
    )

    # Check if labels make sense
    if source in ["notinject", "openorca", "ultrachat"]:
        if counts["label_1"] > 0:
            print(f"  ❌ ERROR: Normal source has attack labels!")
    else:
        if counts["label_0"] > 0:
            print(f"  ❌ ERROR: Attack source has normal labels!")
