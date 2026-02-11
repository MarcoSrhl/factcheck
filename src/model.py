"""Neural fact-checking classifier based on BERT."""

import torch
from transformers import BertTokenizer, BertForSequenceClassification


LABEL_MAP = {0: "SUPPORTED", 1: "REFUTED", 2: "NOT ENOUGH INFO"}
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)


class FactClassifier:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        model_path: str | None = None,
        device: str | None = None,
    ):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        if model_path:
            self.model = BertForSequenceClassification.from_pretrained(
                model_path, num_labels=NUM_LABELS
            )
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=NUM_LABELS
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, claim: str, evidence: str = "") -> dict:
        """Predict the label for a claim given optional evidence.

        Returns dict with 'label', 'confidence', and 'probabilities'.
        """
        if evidence:
            text = f"{claim} [SEP] {evidence}"
        else:
            text = claim

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        predicted_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_id].item()

        return {
            "label": LABEL_MAP[predicted_id],
            "confidence": confidence,
            "probabilities": {
                LABEL_MAP[i]: probs[0][i].item() for i in range(NUM_LABELS)
            },
        }

    def predict_batch(self, claims: list[str], evidences: list[str] | None = None) -> list[dict]:
        """Predict labels for a batch of claims."""
        if evidences is None:
            evidences = [""] * len(claims)

        texts = []
        for claim, evidence in zip(claims, evidences):
            if evidence:
                texts.append(f"{claim} [SEP] {evidence}")
            else:
                texts.append(claim)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        results = []
        for i in range(len(texts)):
            predicted_id = torch.argmax(probs[i]).item()
            results.append({
                "label": LABEL_MAP[predicted_id],
                "confidence": probs[i][predicted_id].item(),
                "probabilities": {
                    LABEL_MAP[j]: probs[i][j].item() for j in range(NUM_LABELS)
                },
            })
        return results

    def save(self, path: str):
        """Save model and tokenizer to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "FactClassifier":
        """Load a fine-tuned model from disk."""
        return cls(model_name=path, model_path=None, device=device)


if __name__ == "__main__":
    print("Loading BERT model (this may take a moment)...")
    classifier = FactClassifier()
    print(f"Device: {classifier.device}")

    claims = [
        "Paris is the capital of France",
        "The Earth is flat",
        "Barack Obama was born in Hawaii",
    ]

    for claim in claims:
        result = classifier.predict(claim)
        print(f"\nClaim: {claim}")
        print(f"  Label: {result['label']} (confidence: {result['confidence']:.3f})")
