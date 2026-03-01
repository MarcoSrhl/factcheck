"""Training script for the fact-checking BERT classifier.

Trains on triplet-formatted inputs (not raw claims) with early stopping.
Supports logging to Neon DB and pushing to MLflow.
"""

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
from tqdm import tqdm

from factcheck.model import LABEL_TO_ID, LABEL_MAP, NUM_LABELS
from factcheck.database import NeonDB
from factcheck.triplet_extractor import TripletExtractor


class FactCheckDataset(Dataset):
    """Dataset for fact-checking using triplet text + evidence."""

    def __init__(self, data: list[dict], tokenizer: BertTokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        triplet_text = item["triplet_text"]
        evidence = item.get("evidence", "") or None
        label = item["label"]

        encoding = self.tokenizer(
            triplet_text,
            evidence,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(LABEL_TO_ID[label], dtype=torch.long),
        }


def preprocess_triplets(data: list[dict], cache_path: str | None = None) -> list[dict]:
    """Extract triplets from claims and add triplet_text field.

    Uses a cache file to avoid re-extracting on subsequent runs.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached triplets from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print("Extracting triplets from all claims (this may take a few minutes)...")
    extractor = TripletExtractor()

    for i, item in enumerate(tqdm(data, desc="Extracting triplets", unit="claim")):
        triplets = extractor.extract(item["claim"])
        if triplets:
            item["triplet_text"] = ". ".join(f"{s} {p} {o}" for s, p, o in triplets)
        else:
            # Fallback: use raw claim if no triplets extracted
            item["triplet_text"] = item["claim"]

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(data)} claims")

    # Report stats
    fallback_count = sum(1 for d in data if d["triplet_text"] == d["claim"])
    print(f"Triplet extraction done: {len(data) - fallback_count}/{len(data)} extracted, {fallback_count} fallbacks")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f)
        print(f"Cached triplets to {cache_path}")

    return data


def generate_synthetic_data() -> list[dict]:
    """Generate synthetic training data for demonstration."""
    data = [
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

    augmented = []
    for item in data:
        augmented.append(item)
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
    max_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    val_split: float = 0.2,
    patience: int = 3,
    save_to_db: bool = False,
    push_to_mlflow: bool = False,
):
    """Train the BERT fact classifier with triplet inputs and early stopping.

    Args:
        data_path: Path to training data JSON file
        output_dir: Directory to save the trained model
        max_epochs: Maximum number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        val_split: Validation split ratio
        patience: Early stopping patience (epochs without improvement)
        save_to_db: If True, saves training data and metadata to Neon database
        push_to_mlflow: If True, pushes trained model to MLflow after training
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=NUM_LABELS
    )
    model.to(device)

    data = load_data(data_path)
    print(f"Loaded {len(data)} training examples")

    # Preprocess: extract triplets from claims
    cache_path = None
    if data_path:
        cache_path = data_path.replace(".json", "_triplets.json")
    data = preprocess_triplets(data, cache_path=cache_path)

    # Database tracking (optional)
    db = None
    db_run_id = None
    if save_to_db:
        try:
            from dotenv import load_dotenv
            load_dotenv()

            db = NeonDB()
            db.initialize_schema()

            db_run_id = db.create_training_run(
                model_type="bert",
                model_name="bert-base-uncased",
                hyperparameters={
                    "max_epochs": max_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "val_split": val_split,
                    "patience": patience,
                    "input_format": "triplet",
                    "dataset": data_path or "synthetic_data",
                    "num_examples": len(data),
                },
                notes="Triplet-based training with early stopping",
            )
            print(f"Created database run_id: {db_run_id}")

            print("Saving training data to database...")
            db.save_training_data(db_run_id, data)
            print(f"Saved {len(data)} training examples to database")

        except Exception as e:
            print(f"Warning: Database saving failed: {e}")
            print("Continuing training without database tracking...")
            db = None
            db_run_id = None

    dataset = FactCheckDataset(data, tokenizer)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    num_workers = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=num_workers, persistent_workers=True,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    final_metrics = {}

    for epoch in range(max_epochs):
        # Training
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        print(
            f"  Epoch {epoch+1}/{max_epochs} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {acc:.4f} | "
            f"P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}",
            flush=True,
        )

        if acc > best_val_acc:
            best_val_acc = acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            final_metrics = {
                "best_val_accuracy": best_val_acc,
                "best_val_f1": f1,
                "best_val_precision": precision,
                "best_val_recall": recall,
                "best_epoch": best_epoch,
                "train_loss_at_best": avg_train_loss,
            }
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  -> Saved best model (acc={acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement ({epochs_without_improvement}/{patience})")

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    final_metrics["total_epochs"] = epoch + 1
    final_metrics["final_train_loss"] = avg_train_loss

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Model saved to: {output_dir}")

    # Update database with final metrics
    if save_to_db and db and db_run_id:
        try:
            db.update_training_run(
                run_id=db_run_id,
                status="completed",
                num_training_examples=train_size,
                num_validation_examples=val_size,
                metrics=final_metrics,
            )
            print(f"Updated database run_id {db_run_id} with final metrics")
        except Exception as e:
            print(f"Warning: Failed to update database: {e}")

    # Push to MLflow
    mlflow_run_id = None
    if push_to_mlflow:
        try:
            from src.push_to_mlflow import push
            print("\nPushing model to MLflow...")
            mlflow_run_id = push(
                model_path=output_dir,
                run_name=f"triplet_bert_ep{best_epoch}_acc{best_val_acc:.3f}",
                accuracy=best_val_acc,
                notes=f"Triplet-based BERT. Early stopped at epoch {epoch+1}/{max_epochs}. Best epoch: {best_epoch}.",
            )
            print(f"MLflow run_id: {mlflow_run_id}")

            # Link MLflow run_id in DB
            if db and db_run_id and mlflow_run_id:
                db.update_training_run(
                    run_id=db_run_id,
                    mlflow_run_id=mlflow_run_id,
                )
                print(f"Linked DB run_id {db_run_id} <-> MLflow run_id {mlflow_run_id}")

        except Exception as e:
            print(f"Warning: MLflow push failed: {e}")

    print("\n=== Summary ===")
    print(f"Best accuracy: {best_val_acc:.4f} (epoch {best_epoch}/{epoch+1})")
    print(f"Model: {output_dir}")
    if db_run_id:
        print(f"DB run_id: {db_run_id}")
    if mlflow_run_id:
        print(f"MLflow run_id: {mlflow_run_id}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fact-checking classifier")
    parser.add_argument("--data", type=str, default=None, help="Path to training data JSON")
    parser.add_argument("--output", type=str, default="models/fact_checker", help="Output directory")
    parser.add_argument("--max-epochs", type=int, default=20, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--save-to-db", action="store_true", help="Save training data to Neon database")
    parser.add_argument("--push-to-mlflow", action="store_true", help="Push model to MLflow after training")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        save_to_db=args.save_to_db,
        push_to_mlflow=args.push_to_mlflow,
    )
