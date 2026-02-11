"""Training script for the fact-checking BERT classifier."""

import json
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.model import LABEL_TO_ID, LABEL_MAP, NUM_LABELS


class FactCheckDataset(Dataset):
    """Dataset for fact-checking claims."""

    def __init__(self, data: list[dict], tokenizer: BertTokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        claim = item["claim"]
        evidence = item.get("evidence", "")
        label = item["label"]

        text = f"{claim} [SEP] {evidence}" if evidence else claim

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(LABEL_TO_ID[label], dtype=torch.long),
        }


def generate_synthetic_data() -> list[dict]:
    """Generate synthetic training data for demonstration."""
    data = [
        # SUPPORTED claims
        {"claim": "Paris is the capital of France", "evidence": "Paris is the capital and most populous city of France.", "label": "SUPPORTED"},
        {"claim": "The Earth orbits the Sun", "evidence": "Earth orbits the Sun at an average distance of about 150 million km.", "label": "SUPPORTED"},
        {"claim": "Water boils at 100 degrees Celsius", "evidence": "At standard atmospheric pressure, water boils at 100 degrees Celsius.", "label": "SUPPORTED"},
        {"claim": "Barack Obama was the 44th president of the United States", "evidence": "Barack Obama served as the 44th president of the United States from 2009 to 2017.", "label": "SUPPORTED"},
        {"claim": "The Amazon is the largest river by volume", "evidence": "The Amazon River is the largest river by discharge volume of water in the world.", "label": "SUPPORTED"},
        {"claim": "Tokyo is the capital of Japan", "evidence": "Tokyo is the capital and most populous city of Japan.", "label": "SUPPORTED"},
        {"claim": "Albert Einstein developed the theory of relativity", "evidence": "Einstein is best known for developing the theory of relativity.", "label": "SUPPORTED"},
        {"claim": "The Great Wall of China is visible from space", "evidence": "The Great Wall of China is a series of fortifications made of stone, brick and other materials.", "label": "REFUTED"},
        {"claim": "Humans have 206 bones in their body", "evidence": "The adult human skeleton consists of 206 bones.", "label": "SUPPORTED"},
        {"claim": "The speed of light is approximately 300000 km per second", "evidence": "The speed of light in vacuum is 299792 kilometers per second.", "label": "SUPPORTED"},
        # REFUTED claims
        {"claim": "The Earth is flat", "evidence": "The Earth is an oblate spheroid, slightly flattened at the poles.", "label": "REFUTED"},
        {"claim": "The Sun revolves around the Earth", "evidence": "Earth orbits the Sun at an average distance of about 150 million km.", "label": "REFUTED"},
        {"claim": "Napoleon was born in England", "evidence": "Napoleon Bonaparte was born on 15 August 1769 in Corsica, France.", "label": "REFUTED"},
        {"claim": "Mount Everest is in Africa", "evidence": "Mount Everest is located in the Mahalangur Himal sub-range of the Himalayas, on the border of Nepal and Tibet.", "label": "REFUTED"},
        {"claim": "The Pacific Ocean is the smallest ocean", "evidence": "The Pacific Ocean is the largest and deepest ocean on Earth.", "label": "REFUTED"},
        {"claim": "Shakespeare was born in France", "evidence": "William Shakespeare was born and raised in Stratford-upon-Avon, England.", "label": "REFUTED"},
        {"claim": "Gold is lighter than aluminum", "evidence": "Gold has a density of 19.3 g/cm3 while aluminum has a density of 2.7 g/cm3.", "label": "REFUTED"},
        {"claim": "Mars is the largest planet in the solar system", "evidence": "Jupiter is the largest planet in the solar system.", "label": "REFUTED"},
        {"claim": "The Amazon River is in Europe", "evidence": "The Amazon River flows through South America.", "label": "REFUTED"},
        {"claim": "Penguins can fly", "evidence": "Penguins are flightless seabirds.", "label": "REFUTED"},
        # NOT ENOUGH INFO claims
        {"claim": "There is life on other planets", "evidence": "", "label": "NOT ENOUGH INFO"},
        {"claim": "Aliens have visited Earth", "evidence": "", "label": "NOT ENOUGH INFO"},
        {"claim": "Chocolate causes acne", "evidence": "Studies on the relationship between chocolate and acne are inconclusive.", "label": "NOT ENOUGH INFO"},
        {"claim": "Reading in dim light damages your eyes permanently", "evidence": "Reading in dim light can cause eye strain but evidence on permanent damage is limited.", "label": "NOT ENOUGH INFO"},
        {"claim": "Coffee stunts growth", "evidence": "There is no conclusive evidence that coffee stunts growth.", "label": "NOT ENOUGH INFO"},
        {"claim": "Cracking knuckles causes arthritis", "evidence": "Studies have not found a definitive link between knuckle cracking and arthritis.", "label": "NOT ENOUGH INFO"},
        {"claim": "Eating carrots gives you night vision", "evidence": "Carrots contain vitamin A which is good for eye health but claims about night vision are exaggerated.", "label": "NOT ENOUGH INFO"},
        {"claim": "The number of stars in the universe is exactly 1 trillion", "evidence": "", "label": "NOT ENOUGH INFO"},
        {"claim": "Dogs can sense earthquakes before they happen", "evidence": "Some anecdotal evidence suggests dogs may sense earthquakes but scientific evidence is limited.", "label": "NOT ENOUGH INFO"},
        {"claim": "Listening to classical music makes you smarter", "evidence": "The Mozart effect has been debated in research with mixed results.", "label": "NOT ENOUGH INFO"},
    ]

    # Duplicate with variations to increase dataset size
    augmented = []
    for item in data:
        augmented.append(item)
        # Add claim-only version (no evidence)
        augmented.append({
            "claim": item["claim"],
            "evidence": "",
            "label": item["label"],
        })

    return augmented


def load_data(data_path: str | None = None) -> list[dict]:
    """Load training data from a JSON file or generate synthetic data."""
    if data_path and os.path.exists(data_path):
        with open(data_path) as f:
            return json.load(f)
    print("No dataset file found, using synthetic training data.")
    return generate_synthetic_data()


def train(
    data_path: str | None = None,
    output_dir: str = "models/fact_checker",
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    val_split: float = 0.2,
):
    """Train the BERT fact classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=NUM_LABELS
    )
    model.to(device)

    data = load_data(data_path)
    dataset = FactCheckDataset(data, tokenizer)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_val_acc = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {acc:.4f} | "
            f"P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}"
        )

        if acc > best_val_acc:
            best_val_acc = acc
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  -> Saved best model (acc={acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fact-checking classifier")
    parser.add_argument("--data", type=str, default=None, help="Path to training data JSON")
    parser.add_argument("--output", type=str, default="models/fact_checker", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
