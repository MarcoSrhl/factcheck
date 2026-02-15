"""GAN training pipeline for knowledge-graph fact verification.

This module fetches real triplets from DBpedia (via
:mod:`src.sparql_queries`), encodes them with the :class:`TripletEncoder`,
and trains the :class:`FactGAN` using the standard adversarial loop:

1. Train D on real triplets (label=1) and generated fakes (label=0).
2. Train G to fool D (maximise D's output on generated fakes).

Usage
-----
From the command line::

    python -m src.gan_trainer --epochs 50 --batch-size 64 --output models/gan

Programmatically::

    from src.gan_trainer import GANTrainer
    trainer = GANTrainer()
    trainer.train(epochs=50)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from src.gan_model import FactGAN
from src.sparql_queries import fetch_mixed_triplets

logger = logging.getLogger(__name__)


# =========================================================================
# Training configuration
# =========================================================================


@dataclass
class GANTrainingConfig:
    """Hyper-parameters and settings for GAN training."""

    # GAN architecture
    noise_dim: int = 128
    embedding_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.3

    # Optimiser (standard GAN hyper-params)
    lr: float = 0.0002
    betas: tuple[float, float] = (0.5, 0.999)

    # Training loop
    epochs: int = 50
    batch_size: int = 64
    log_every: int = 5

    # Data
    triplets_per_category: int = 200
    categories: Optional[list[str]] = None
    use_synthetic_fallback: bool = True

    # Encoder
    use_sentence_transformer: bool = True
    sentence_model_name: str = "all-MiniLM-L6-v2"

    # Output
    output_dir: str = "models/gan"

    # Device
    device: Optional[str] = None


# =========================================================================
# Synthetic fallback data
# =========================================================================

_SYNTHETIC_TRIPLETS: list[tuple[str, str, str]] = [
    ("Paris", "is capital of", "France"),
    ("Berlin", "is capital of", "Germany"),
    ("Tokyo", "is capital of", "Japan"),
    ("London", "is capital of", "United Kingdom"),
    ("Madrid", "is capital of", "Spain"),
    ("Rome", "is capital of", "Italy"),
    ("Ottawa", "is capital of", "Canada"),
    ("Canberra", "is capital of", "Australia"),
    ("Washington D.C.", "is capital of", "United States"),
    ("Beijing", "is capital of", "China"),
    ("Moscow", "is capital of", "Russia"),
    ("Brasilia", "is capital of", "Brazil"),
    ("Barack Obama", "was born in", "Honolulu"),
    ("Albert Einstein", "was born in", "Ulm"),
    ("Isaac Newton", "was born in", "Woolsthorpe"),
    ("Marie Curie", "was born in", "Warsaw"),
    ("Leonardo da Vinci", "was born in", "Vinci"),
    ("Wolfgang Amadeus Mozart", "was born in", "Salzburg"),
    ("Mahatma Gandhi", "was born in", "Porbandar"),
    ("Nelson Mandela", "was born in", "Mvezo"),
    ("Charles Darwin", "was born in", "Shrewsbury"),
    ("Nikola Tesla", "was born in", "Smiljan"),
    ("Albert Einstein", "has occupation", "Physicist"),
    ("Marie Curie", "has occupation", "Physicist"),
    ("William Shakespeare", "has occupation", "Playwright"),
    ("Leonardo da Vinci", "has occupation", "Painter"),
    ("Isaac Newton", "has occupation", "Mathematician"),
    ("Aristotle", "has occupation", "Philosopher"),
    ("Charles Darwin", "has occupation", "Naturalist"),
    ("Sigmund Freud", "has occupation", "Psychologist"),
    ("Eiffel Tower", "is located in", "Paris"),
    ("Statue of Liberty", "is located in", "New York"),
    ("Big Ben", "is located in", "London"),
    ("Colosseum", "is located in", "Rome"),
    ("Taj Mahal", "is located in", "Agra"),
    ("Great Wall", "is located in", "China"),
    ("Sydney Opera House", "is located in", "Sydney"),
    ("Machu Picchu", "is located in", "Peru"),
    ("Apple", "was founded in", "1976"),
    ("Microsoft", "was founded in", "1975"),
    ("Google", "was founded in", "1998"),
    ("Amazon", "was founded in", "1994"),
    ("Tesla", "was founded in", "2003"),
    ("Facebook", "was founded in", "2004"),
    ("William Shakespeare", "wrote", "Hamlet"),
    ("Charles Dickens", "wrote", "Oliver Twist"),
    ("Leo Tolstoy", "wrote", "War and Peace"),
    ("Jane Austen", "wrote", "Pride and Prejudice"),
    ("Mark Twain", "wrote", "Adventures of Huckleberry Finn"),
    ("Homer", "wrote", "Iliad"),
    ("J.K. Rowling", "wrote", "Harry Potter"),
    ("George Orwell", "wrote", "1984"),
    ("Albert Einstein", "studied at", "ETH Zurich"),
    ("Barack Obama", "studied at", "Harvard University"),
    ("Isaac Newton", "studied at", "Trinity College"),
    ("Marie Curie", "studied at", "University of Paris"),
    ("Alan Turing", "studied at", "Princeton University"),
    ("Stephen Hawking", "studied at", "University of Oxford"),
]


def _get_synthetic_triplets(n: int = 0) -> list[tuple[str, str, str]]:
    """Return synthetic triplets, optionally limited to *n* items.

    If *n* is 0 or exceeds the available count, all triplets are returned.
    """
    triplets = list(_SYNTHETIC_TRIPLETS)
    random.shuffle(triplets)
    if 0 < n < len(triplets):
        return triplets[:n]
    return triplets


# =========================================================================
# Training metrics logger
# =========================================================================


@dataclass
class EpochMetrics:
    """Accumulated metrics for a single epoch."""

    d_loss: float = 0.0
    g_loss: float = 0.0
    d_real_acc: float = 0.0
    d_fake_acc: float = 0.0
    num_steps: int = 0

    def update(self, step_metrics: dict[str, float]) -> None:
        self.d_loss += step_metrics["d_loss"]
        self.g_loss += step_metrics["g_loss"]
        self.d_real_acc += step_metrics["d_real_acc"]
        self.d_fake_acc += step_metrics["d_fake_acc"]
        self.num_steps += 1

    def average(self) -> dict[str, float]:
        n = max(self.num_steps, 1)
        return {
            "d_loss": self.d_loss / n,
            "g_loss": self.g_loss / n,
            "d_real_acc": self.d_real_acc / n,
            "d_fake_acc": self.d_fake_acc / n,
        }


@dataclass
class TrainingHistory:
    """Stores per-epoch metrics across the full training run."""

    epochs: list[dict[str, float]] = field(default_factory=list)

    def append(self, epoch_avg: dict[str, float]) -> None:
        self.epochs.append(epoch_avg)

    def summary(self) -> dict[str, float]:
        """Return metrics from the last recorded epoch."""
        if not self.epochs:
            return {}
        return self.epochs[-1]


# =========================================================================
# GANTrainer
# =========================================================================


class GANTrainer:
    """End-to-end GAN training pipeline.

    Parameters
    ----------
    config : GANTrainingConfig or None
        Configuration object.  If *None*, defaults are used.
    """

    def __init__(self, config: Optional[GANTrainingConfig] = None) -> None:
        self.config = config or GANTrainingConfig()

        self.gan = FactGAN(
            noise_dim=self.config.noise_dim,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            use_sentence_transformer=self.config.use_sentence_transformer,
            sentence_model_name=self.config.sentence_model_name,
            device=self.config.device,
        )

        self.optimizer_g = torch.optim.Adam(
            self.gan.generator.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
        )
        self.optimizer_d = torch.optim.Adam(
            self.gan.discriminator.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
        )

        self.history = TrainingHistory()

    # ----- data acquisition -----------------------------------------------

    def fetch_training_triplets(self) -> list[tuple[str, str, str]]:
        """Fetch real triplets from DBpedia for training.

        If the DBpedia queries fail or return too few results and
        ``config.use_synthetic_fallback`` is True, synthetic triplets
        are used as a fallback.

        Returns
        -------
        list[tuple[str, str, str]]
            List of ``(subject, predicate, object)`` text triplets.
        """
        triplets: list[tuple[str, str, str]] = []

        logger.info("Fetching training triplets from DBpedia...")
        try:
            triplets = fetch_mixed_triplets(
                per_category=self.config.triplets_per_category,
                categories=self.config.categories,
            )
        except Exception as exc:
            logger.warning("Failed to fetch from DBpedia: %s", exc)

        if len(triplets) < self.config.batch_size and self.config.use_synthetic_fallback:
            logger.warning(
                "Only %d triplets fetched; augmenting with synthetic data.",
                len(triplets),
            )
            synthetic = _get_synthetic_triplets()
            # Avoid duplicates
            existing = set(triplets)
            for t in synthetic:
                if t not in existing:
                    triplets.append(t)

        logger.info("Total training triplets: %d", len(triplets))
        return triplets

    # ----- encoding -------------------------------------------------------

    def encode_all(
        self, triplets: list[tuple[str, str, str]], batch_size: int = 128
    ) -> torch.Tensor:
        """Encode all triplets in batches and concatenate.

        Parameters
        ----------
        triplets : list[tuple[str, str, str]]
            Text triplets.
        batch_size : int
            Encoding batch size (to limit memory usage).

        Returns
        -------
        torch.Tensor
            Shape ``(len(triplets), embedding_dim)``.
        """
        all_embeddings: list[torch.Tensor] = []
        for i in range(0, len(triplets), batch_size):
            batch = triplets[i : i + batch_size]
            emb = self.gan.encode_triplets(batch)
            all_embeddings.append(emb)
        return torch.cat(all_embeddings, dim=0)

    # ----- training loop --------------------------------------------------

    def train(
        self,
        epochs: Optional[int] = None,
        triplets: Optional[list[tuple[str, str, str]]] = None,
        save: bool = True,
    ) -> TrainingHistory:
        """Run the full GAN training loop.

        Parameters
        ----------
        epochs : int or None
            Override ``config.epochs`` if provided.
        triplets : list or None
            Pre-fetched triplets.  If *None*, :meth:`fetch_training_triplets`
            is called automatically.
        save : bool
            Whether to save the trained model at the end.

        Returns
        -------
        TrainingHistory
            Per-epoch averaged metrics.
        """
        num_epochs = epochs or self.config.epochs
        batch_size = self.config.batch_size

        # ----- Data -------------------------------------------------------
        if triplets is None:
            triplets = self.fetch_training_triplets()

        if not triplets:
            logger.error("No training triplets available. Aborting.")
            return self.history

        logger.info("Encoding %d triplets...", len(triplets))
        encoded = self.encode_all(triplets)
        logger.info("Encoded tensor shape: %s", encoded.shape)

        num_batches = math.ceil(encoded.size(0) / batch_size)

        # ----- Training ---------------------------------------------------
        logger.info(
            "Starting GAN training: %d epochs, %d batches/epoch, batch_size=%d",
            num_epochs,
            num_batches,
            batch_size,
        )
        t_start = time.time()

        for epoch in range(1, num_epochs + 1):
            # Shuffle data each epoch
            perm = torch.randperm(encoded.size(0))
            encoded_shuffled = encoded[perm]

            epoch_metrics = EpochMetrics()

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, encoded_shuffled.size(0))
                real_batch = encoded_shuffled[start_idx:end_idx]

                if real_batch.size(0) < 2:
                    # BatchNorm requires at least 2 samples
                    continue

                step_metrics = self.gan.train_step(
                    real_embeddings=real_batch,
                    optimizer_g=self.optimizer_g,
                    optimizer_d=self.optimizer_d,
                )
                epoch_metrics.update(step_metrics)

            avg = epoch_metrics.average()
            self.history.append(avg)

            if epoch % self.config.log_every == 0 or epoch == 1:
                elapsed = time.time() - t_start
                logger.info(
                    "Epoch %3d/%d | D_loss: %.4f | G_loss: %.4f | "
                    "D_real_acc: %.3f | D_fake_acc: %.3f | elapsed: %.1fs",
                    epoch,
                    num_epochs,
                    avg["d_loss"],
                    avg["g_loss"],
                    avg["d_real_acc"],
                    avg["d_fake_acc"],
                    elapsed,
                )

        total_time = time.time() - t_start
        logger.info("Training complete in %.1f seconds.", total_time)

        # ----- Save -------------------------------------------------------
        if save:
            self.gan.save(self.config.output_dir)
            logger.info("Model saved to %s", self.config.output_dir)

        return self.history

    # ----- evaluation helpers ---------------------------------------------

    def evaluate_triplets(
        self, triplets: list[tuple[str, str, str]]
    ) -> list[dict[str, object]]:
        """Score a list of triplets through the discriminator.

        Returns a list of dicts, one per triplet, with:
          - ``triplet``: the original ``(s, p, o)`` tuple
          - ``score``: discriminator output (higher = more realistic)
        """
        embeddings = self.encode_all(triplets)
        scores = self.gan.discriminate(embeddings)  # (n, 1)
        results: list[dict[str, object]] = []
        for idx, (s, p, o) in enumerate(triplets):
            results.append({
                "triplet": (s, p, o),
                "score": scores[idx].item(),
            })
        return results


# =========================================================================
# CLI entry point
# =========================================================================


def main() -> None:
    """Parse arguments and run GAN training."""
    parser = argparse.ArgumentParser(
        description="Train the FactGAN on DBpedia triplets."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/gan",
        help="Directory to save trained model.",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=200,
        help="Max triplets per SPARQL category.",
    )
    parser.add_argument(
        "--no-sentence-transformer",
        action="store_true",
        help="Disable sentence-transformers (use hash fallback).",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip DBpedia and use only synthetic triplets.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config = GANTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output,
        triplets_per_category=args.per_category,
        use_sentence_transformer=not args.no_sentence_transformer,
    )

    trainer = GANTrainer(config=config)

    if args.synthetic_only:
        triplets = _get_synthetic_triplets()
        logger.info("Using %d synthetic triplets (--synthetic-only).", len(triplets))
        trainer.train(triplets=triplets)
    else:
        trainer.train()

    # Quick evaluation on a few examples
    test_triplets = [
        ("Paris", "is capital of", "France"),
        ("The Moon", "is made of", "green cheese"),
        ("Albert Einstein", "was born in", "Ulm"),
    ]
    print("\n--- Discriminator evaluation on test triplets ---")
    for result in trainer.evaluate_triplets(test_triplets):
        s, p, o = result["triplet"]
        print(f"  ({s}, {p}, {o}) -> score={result['score']:.4f}")


if __name__ == "__main__":
    main()
