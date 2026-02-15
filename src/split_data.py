"""Split training data into train/validation sets with stratified sampling."""

import json
import random
from collections import defaultdict

SEED = 42
VAL_RATIO = 0.2


def split_data(
    input_path: str = "data/bert_training_data.json",
    train_path: str = "data/train.json",
    val_path: str = "data/validation.json",
    val_ratio: float = VAL_RATIO,
):
    """Split data into train/validation with stratified sampling by label."""
    random.seed(SEED)

    with open(input_path) as f:
        data = json.load(f)

    # Group by label for stratified split
    by_label: dict[str, list] = defaultdict(list)
    for item in data:
        by_label[item["label"]].append(item)

    train_data = []
    val_data = []

    for label, items in by_label.items():
        random.shuffle(items)
        n_val = int(len(items) * val_ratio)
        val_data.extend(items[:n_val])
        train_data.extend(items[n_val:])

    # Shuffle both sets
    random.shuffle(train_data)
    random.shuffle(val_data)

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"Total: {len(data)}")
    print(f"Train: {len(train_data)} ({len(train_data)/len(data):.1%})")
    print(f"Val:   {len(val_data)} ({len(val_data)/len(data):.1%})")

    # Distribution check
    from collections import Counter
    for name, split in [("Train", train_data), ("Val", val_data)]:
        counts = Counter(d["label"] for d in split)
        print(f"\n{name} distribution:")
        for label, count in counts.most_common():
            print(f"  {label}: {count} ({count/len(split):.1%})")


if __name__ == "__main__":
    split_data()
