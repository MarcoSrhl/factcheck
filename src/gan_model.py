"""BERT-based GAN for knowledge-graph fact verification.

Components
----------
SwapGenerator (G)
    Creates fake triplets by swapping the subject or object with another
    entity from the training data.  Produces plausible but factually
    incorrect triplets (e.g. "London is capital of France").

BERTDiscriminator (D)
    BERT + classification head.  Classifies triplets as real (1) or fake (0).
    Must learn *factual* distinctions since fakes are syntactically identical.

FactGAN
    High-level wrapper: ``discriminate_triplets``, ``train_step``, ``save``,
    ``load``.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

logger = logging.getLogger(__name__)

BERT_MODEL_NAME = "bert-base-uncased"
BERT_HIDDEN_SIZE = 768
TRIPLET_SEP = " [REL] "


def _detect_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _format_triplet(subject: str, predicate: str, obj: str) -> str:
    return f"{subject}{TRIPLET_SEP}{predicate}{TRIPLET_SEP}{obj}"


# =========================================================================
# Generator — entity swap
# =========================================================================


class SwapGenerator:
    """Creates fake triplets by swapping subject or object with another
    entity from the training data.

    Pools entities by predicate so swaps stay within the same relation type
    (e.g. a capital city is replaced by another capital city, not by a person).
    """

    def __init__(self, triplets: list[tuple[str, str, str]]) -> None:
        self.subject_pool: dict[str, list[str]] = {}
        self.object_pool: dict[str, list[str]] = {}
        self.all_subjects: list[str] = []
        self.all_objects: list[str] = []

        for s, p, o in triplets:
            self.subject_pool.setdefault(p, []).append(s)
            self.object_pool.setdefault(p, []).append(o)

        # Deduplicate
        self.all_subjects = list({s for s, _, _ in triplets})
        self.all_objects = list({o for _, _, o in triplets})
        for p in self.subject_pool:
            self.subject_pool[p] = list(set(self.subject_pool[p]))
            self.object_pool[p] = list(set(self.object_pool[p]))

    def generate_fakes(
        self, triplets: list[tuple[str, str, str]]
    ) -> list[tuple[str, str, str]]:
        """Generate fake triplets by swapping subject or object."""
        fakes = []
        for s, p, o in triplets:
            if random.random() < 0.5:
                # Swap subject
                pool = self.subject_pool.get(p, self.all_subjects)
                candidates = [x for x in pool if x != s]
                if not candidates:
                    candidates = [x for x in self.all_subjects if x != s]
                new_s = random.choice(candidates) if candidates else s
                fakes.append((new_s, p, o))
            else:
                # Swap object
                pool = self.object_pool.get(p, self.all_objects)
                candidates = [x for x in pool if x != o]
                if not candidates:
                    candidates = [x for x in self.all_objects if x != o]
                new_o = random.choice(candidates) if candidates else o
                fakes.append((s, p, new_o))
        return fakes


# =========================================================================
# Discriminator
# =========================================================================


class BERTDiscriminator(nn.Module):
    """BERT-based discriminator with a classification head."""

    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._device = device or _detect_device()

        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.to(self._device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score triplets. Returns (batch, 1) in [0, 1]."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [CLS]
        return self.classifier(cls_hidden)

    def encode_triplets(
        self, triplets: list[tuple[str, str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize triplets and return (input_ids, attention_mask) on device."""
        texts = [_format_triplet(s, p, o) for s, p, o in triplets]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        return enc["input_ids"].to(self._device), enc["attention_mask"].to(self._device)


# =========================================================================
# FactGAN wrapper
# =========================================================================


class FactGAN:
    """High-level wrapper around SwapGenerator + BERTDiscriminator.

    Parameters
    ----------
    triplets : list of (subject, predicate, object)
        Training triplets used to build the swap pools.
    model_name : str
        HuggingFace BERT model identifier.
    device : str or None
        ``'mps'``, ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        triplets: list[tuple[str, str, str]] | None = None,
        model_name: str = BERT_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device) if device else _detect_device()

        self.generator = SwapGenerator(triplets or [])
        self.discriminator = BERTDiscriminator(
            model_name=model_name,
            device=self.device,
        )

        self.criterion = nn.BCELoss()

        logger.info(
            "FactGAN (swap) initialised on %s (model=%s, pool=%d triplets).",
            self.device,
            model_name,
            len(triplets or []),
        )

    def discriminate_triplets(
        self, triplets: list[tuple[str, str, str]]
    ) -> torch.Tensor:
        """Score triplets through the discriminator. Returns (n, 1) in [0,1]."""
        self.discriminator.eval()
        input_ids, attention_mask = self.discriminator.encode_triplets(triplets)
        with torch.no_grad():
            return self.discriminator(input_ids, attention_mask)

    def train_step(
        self,
        real_triplets: list[tuple[str, str, str]],
        optimizer_d: torch.optim.Optimizer,
        label_smoothing: float = 0.9,
    ) -> dict[str, float]:
        """One training step: real triplets vs swap-generated fakes.

        Returns dict with d_loss, d_real_score, d_fake_score.
        """
        batch_size = len(real_triplets)
        real_labels = torch.full((batch_size, 1), label_smoothing, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # Generate fakes by swapping entities
        fake_triplets = self.generator.generate_fakes(real_triplets)

        # Tokenize both
        real_ids, real_mask = self.discriminator.encode_triplets(real_triplets)
        fake_ids, fake_mask = self.discriminator.encode_triplets(fake_triplets)

        # Train discriminator
        self.discriminator.train()
        optimizer_d.zero_grad()

        d_real_out = self.discriminator(real_ids, real_mask)
        d_loss_real = self.criterion(d_real_out, real_labels)

        d_fake_out = self.discriminator(fake_ids, fake_mask)
        d_loss_fake = self.criterion(d_fake_out, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        return {
            "d_loss": d_loss.item(),
            "d_real_score": d_real_out.mean().item(),
            "d_fake_score": d_fake_out.mean().item(),
            "fake_examples": fake_triplets[:3],
        }

    # ----- persistence -----------------------------------------------------

    def save(self, directory: str) -> None:
        """Save discriminator state_dict."""
        os.makedirs(directory, exist_ok=True)
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(directory, "discriminator.pt"),
        )
        meta = {"architecture": "bert-gan-swap"}
        torch.save(meta, os.path.join(directory, "gan_meta.pt"))
        logger.info("FactGAN (swap) saved to %s", directory)

    def load(self, directory: str) -> None:
        """Load previously saved discriminator state_dict."""
        disc_path = os.path.join(directory, "discriminator.pt")
        if os.path.isfile(disc_path):
            self.discriminator.load_state_dict(
                torch.load(disc_path, map_location=self.device, weights_only=True)
            )
        logger.info("FactGAN (swap) loaded from %s", directory)


# =========================================================================
# Quick smoke test
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sample_triplets = [
        ("Paris", "is capital of", "France"),
        ("Berlin", "is capital of", "Germany"),
        ("Tokyo", "is capital of", "Japan"),
        ("Barack Obama", "was born in", "Hawaii"),
        ("Albert Einstein", "was born in", "Ulm"),
    ]

    gan = FactGAN(triplets=sample_triplets)

    # Generate fakes
    fakes = gan.generator.generate_fakes(sample_triplets[:3])
    for real, fake in zip(sample_triplets[:3], fakes):
        print(f"  REAL: {real}")
        print(f"  FAKE: {fake}")
        print()

    # Discriminate
    scores = gan.discriminate_triplets(sample_triplets)
    print(f"Discriminator scores: {scores.squeeze().tolist()}")

    # One training step
    optimizer_d = torch.optim.AdamW(gan.discriminator.parameters(), lr=2e-5)
    metrics = gan.train_step(sample_triplets, optimizer_d)
    print(f"Train step metrics: {metrics}")
