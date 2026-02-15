"""Validate the full fact-checking pipeline on held-out data.

Runs each validation claim through the ENTIRE pipeline:
  sentence → triplet extraction → entity linking → KB query → neural → verdict

Then compares pipeline verdicts with ground truth labels.
"""

import json
import random
import time
import argparse
from collections import Counter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.fact_checker import FactChecker


def load_validation_data(
    val_path: str = "data/validation.json",
    sample_size: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load validation data, optionally sampling a subset."""
    with open(val_path) as f:
        data = json.load(f)

    if sample_size and sample_size < len(data):
        random.seed(seed)
        # Stratified sample
        by_label: dict[str, list] = {}
        for item in data:
            by_label.setdefault(item["label"], []).append(item)

        sampled = []
        for label, items in by_label.items():
            n = max(1, int(sample_size * len(items) / len(data)))
            sampled.extend(random.sample(items, min(n, len(items))))
        data = sampled

    return data


def validate(
    val_path: str = "data/validation.json",
    model_path: str = "models/fact_checker",
    sample_size: int | None = None,
):
    """Run full pipeline validation."""
    print("Loading validation data...")
    data = load_validation_data(val_path, sample_size)
    print(f"Validation set: {len(data)} claims")

    label_counts = Counter(d["label"] for d in data)
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

    print("\nLoading pipeline...")
    checker = FactChecker(model_path=model_path, use_neural=True)

    print("\nRunning pipeline on validation claims...")
    expected = []
    predicted = []
    errors = []
    start = time.time()

    for i, item in enumerate(data):
        claim = item["claim"]
        true_label = item["label"]

        try:
            result = checker.check(claim)
            pred_label = result["verdict"]
        except Exception as e:
            pred_label = "NOT ENOUGH INFO"
            errors.append((i, claim, str(e)))

        expected.append(true_label)
        predicted.append(pred_label)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(data) - i - 1) / rate
            acc_so_far = accuracy_score(expected, predicted)
            print(f"  [{i+1}/{len(data)}] acc={acc_so_far:.2%} "
                  f"({rate:.1f} claims/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s ({len(data)/elapsed:.1f} claims/s)")

    if errors:
        print(f"\n{len(errors)} errors during evaluation:")
        for idx, claim, err in errors[:5]:
            print(f"  [{idx}] {claim[:60]}... -> {err}")

    # Metrics
    labels = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
    acc = accuracy_score(expected, predicted)

    print(f"\n{'=' * 60}")
    print(f"PIPELINE VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Claims evaluated: {len(data)}")
    print(f"Accuracy: {acc:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(expected, predicted, labels=labels, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(expected, predicted, labels=labels)
    print("Confusion Matrix:")
    print(f"{'':20s} {'SUPPORTED':>12s} {'REFUTED':>12s} {'NEI':>12s}")
    for i, label in enumerate(labels):
        row = "  ".join(f"{cm[i][j]:>10d}" for j in range(3))
        print(f"  {label:18s} {row}")

    # Per-verdict analysis
    print(f"\nPer-verdict breakdown:")
    for label in labels:
        indices = [i for i, e in enumerate(expected) if e == label]
        if not indices:
            continue
        correct = sum(1 for i in indices if predicted[i] == label)
        total = len(indices)
        print(f"  {label:18s}: {correct}/{total} correct ({correct/total:.1%})")

    # Save detailed results
    results_path = "data/validation_results.json"
    results = []
    for i, item in enumerate(data):
        results.append({
            "claim": item["claim"],
            "expected": expected[i],
            "predicted": predicted[i],
            "correct": expected[i] == predicted[i],
        })
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate fact-checking pipeline")
    parser.add_argument("--data", default="data/validation.json", help="Validation data path")
    parser.add_argument("--model", default="models/fact_checker", help="Model path")
    parser.add_argument("--sample", type=int, default=None, help="Sample size (None=all)")
    args = parser.parse_args()

    validate(val_path=args.data, model_path=args.model, sample_size=args.sample)
