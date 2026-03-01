"""Evaluate the full fact-checking pipeline on a sample of claims.

Usage
-----
    python -m src.evaluate
    python -m src.evaluate --n 300 --data data/validation.json
"""

import argparse
import json
import logging
import random
import time
from collections import Counter

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.fact_checker import FactChecker

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def sample_balanced(data: list[dict], n: int) -> list[dict]:
    """Sample n items balanced across labels."""
    by_label: dict[str, list[dict]] = {}
    for item in data:
        by_label.setdefault(item["label"], []).append(item)

    per_label = n // len(by_label)
    remainder = n % len(by_label)

    sampled = []
    for i, (label, items) in enumerate(sorted(by_label.items())):
        count = per_label + (1 if i < remainder else 0)
        sampled.extend(random.sample(items, min(count, len(items))))

    random.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Evaluate fact-checking pipeline")
    parser.add_argument("--data", type=str, default="data/validation.json")
    parser.add_argument("--n", type=int, default=300, help="Number of claims to test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.data) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {args.data}")

    sample = sample_balanced(data, args.n)
    print(f"Sampled {len(sample)} claims (balanced across labels)")
    label_dist = Counter(item["label"] for item in sample)
    for label, count in sorted(label_dist.items()):
        print(f"  {label}: {count}")

    print("\nLoading pipeline...")
    checker = FactChecker(use_neural=True, use_gan=False, use_explainer=False)
    print("Pipeline ready.\n")

    y_true = []
    y_pred = []
    errors = 0
    start = time.time()

    for i, item in enumerate(sample):
        claim = item["claim"]
        true_label = item["label"]

        try:
            result = checker.check(claim)
            pred_label = result["verdict"]
        except Exception as e:
            logger.error(f"Error on claim {i}: {e}")
            pred_label = "NOT ENOUGH INFO"
            errors += 1

        y_true.append(true_label)
        y_pred.append(pred_label)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            acc_so_far = accuracy_score(y_true, y_pred)
            rate = (i + 1) / elapsed
            eta = (len(sample) - i - 1) / rate
            print(
                f"  [{i+1}/{len(sample)}] "
                f"acc={acc_so_far:.3f} | "
                f"{rate:.1f} claims/s | "
                f"ETA {eta:.0f}s"
            )

    elapsed = time.time() - start

    # --- Results ---
    labels = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({len(sample)} claims, {elapsed:.1f}s)")
    print("=" * 60)
    print(f"\nOverall Accuracy: {acc:.4f} ({int(acc * len(sample))}/{len(sample)})")
    if errors:
        print(f"Errors: {errors}")

    print(f"\n{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    for i, label in enumerate(labels):
        print(f"{label:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")

    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>20} {'SUPPORTED':>12} {'REFUTED':>12} {'NEI':>12}")
    for i, label in enumerate(labels):
        short = label if label != "NOT ENOUGH INFO" else "NEI"
        print(f"{short:>20} {cm[i][0]:>12} {cm[i][1]:>12} {cm[i][2]:>12}")


if __name__ == "__main__":
    main()
