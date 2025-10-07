#!/usr/bin/env python3
"""Check how many attack samples are available from each source."""

from datasets import load_dataset, concatenate_datasets

print("🔍 Checking available attack samples from each source...")
print("=" * 80)

attack_datasets = []
total_attack = 0

# JailbreakBench/JBB-Behaviors
print("\n📊 JailbreakBench/JBB-Behaviors")
try:
    jbb_harmful = load_dataset(
        "JailbreakBench/JBB-Behaviors", "behaviors", split="harmful"
    )
    jbb_judge = load_dataset(
        "JailbreakBench/JBB-Behaviors", "judge_comparison", split="test"
    )
    jbb_total = len(jbb_harmful) + len(jbb_judge)
    print(f"  ✅ harmful: {len(jbb_harmful):,}")
    print(f"  ✅ judge: {len(jbb_judge):,}")
    print(f"  Total: {jbb_total:,}")
    total_attack += jbb_total
except Exception as e:
    print(f"  ❌ Error: {e}")

# walledai/HarmBench
print("\n📊 walledai/HarmBench")
try:
    harmbench_contextual = load_dataset(
        "walledai/HarmBench", "contextual", split="train"
    )
    harmbench_copyright = load_dataset(
        "walledai/HarmBench", "copyright", split="train"
    )
    harmbench_standard = load_dataset(
        "walledai/HarmBench", "standard", split="train"
    )
    harmbench_total = (
        len(harmbench_contextual)
        + len(harmbench_copyright)
        + len(harmbench_standard)
    )
    print(f"  ✅ contextual: {len(harmbench_contextual):,}")
    print(f"  ✅ copyright: {len(harmbench_copyright):,}")
    print(f"  ✅ standard: {len(harmbench_standard):,}")
    print(f"  Total: {harmbench_total:,}")
    total_attack += harmbench_total
except Exception as e:
    print(f"  ❌ Error: {e}")

# deepset/prompt-injections
print("\n📊 deepset/prompt-injections")
try:
    prompt_inj = load_dataset("deepset/prompt-injections", split="train")
    print(f"  ✅ Total: {len(prompt_inj):,}")
    total_attack += len(prompt_inj)
except Exception as e:
    print(f"  ❌ Error: {e}")

# rubend18/ChatGPT-Jailbreak-Prompts
print("\n📊 rubend18/ChatGPT-Jailbreak-Prompts")
try:
    jailbreak = load_dataset(
        "rubend18/ChatGPT-Jailbreak-Prompts", split="train"
    )
    print(f"  ✅ Total: {len(jailbreak):,}")
    total_attack += len(jailbreak)
except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n" + "=" * 80)
print(f"📈 TOTAL ATTACK SAMPLES AVAILABLE: {total_attack:,}")
print("=" * 80)

print("\n💡 Recommendation:")
print(f"   Current dataset uses: ~919 attack samples (40% ratio)")
print(f"   Available: {total_attack:,} attack samples")
if total_attack > 919:
    print(f"   You can use ALL {total_attack:,} attack samples!")
    print(
        f"   This would give you {total_attack} attacks + {total_attack} normals = {total_attack * 2:,} total samples"
    )
else:
    print(f"   Using all available attack samples!")
