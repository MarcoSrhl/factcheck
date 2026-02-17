"""BERT-based GAN training pipeline for knowledge-graph fact verification.

This module fetches real triplets from DBpedia (via
:mod:`src.sparql_queries`), and trains the BERT-based :class:`FactGAN`
using the standard adversarial loop directly on raw text triplets
(no pre-encoding step):

1. Train D on real triplets (label=1) and Generator fakes (label=0).
2. Train G to fool D (maximise D's output on generated fakes).
3. Anneal Gumbel-Softmax temperature from high (smooth) to low (discrete).

Usage
-----
From the command line::

    python -m src.gan_trainer --epochs 50 --batch-size 16 --output models/gan

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
    """Hyper-parameters and settings for BERT-based GAN training."""

    # GAN architecture
    noise_dim: int = 128

    # Optimiser — separate LRs so G can learn faster than D
    lr_g: float = 5e-5
    lr_d: float = 1e-5
    weight_decay: float = 0.01

    # Training loop
    epochs: int = 50
    batch_size: int = 16
    log_every: int = 5

    # G/D balance
    g_steps: int = 3           # G updates per D update
    label_smoothing: float = 0.9  # real labels for D (< 1.0 slows D)

    # Anti-collapse
    d_noise_std: float = 0.1       # instance noise on D inputs
    feat_match_weight: float = 10.0  # feature matching loss weight
    diversity_weight: float = 1.0    # diversity loss weight
    mlm_weight: float = 1.0         # MLM reconstruction loss weight

    # Gumbel-Softmax temperature annealing
    gumbel_temp_start: float = 1.0
    gumbel_temp_end: float = 0.5
    temp_anneal_epochs: int = 15

    # Data
    triplets_per_category: int = 200
    categories: Optional[list[str]] = None
    use_synthetic_fallback: bool = True

    # Output
    output_dir: str = "models/gan"

    # Device
    device: Optional[str] = None


# =========================================================================
# Synthetic fallback data
# =========================================================================

_SYNTHETIC_TRIPLETS: list[tuple[str, str, str]] = [
    # --- Capitals ---
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
    ("New Delhi", "is capital of", "India"),
    ("Cairo", "is capital of", "Egypt"),
    ("Buenos Aires", "is capital of", "Argentina"),
    ("Ankara", "is capital of", "Turkey"),
    ("Bangkok", "is capital of", "Thailand"),
    ("Lisbon", "is capital of", "Portugal"),
    ("Vienna", "is capital of", "Austria"),
    ("Stockholm", "is capital of", "Sweden"),
    ("Oslo", "is capital of", "Norway"),
    ("Helsinki", "is capital of", "Finland"),
    ("Athens", "is capital of", "Greece"),
    ("Warsaw", "is capital of", "Poland"),
    ("Dublin", "is capital of", "Ireland"),
    ("Bern", "is capital of", "Switzerland"),
    ("Seoul", "is capital of", "South Korea"),
    ("Nairobi", "is capital of", "Kenya"),
    # --- Birth places ---
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
    ("Napoleon Bonaparte", "was born in", "Ajaccio"),
    ("Galileo Galilei", "was born in", "Pisa"),
    ("Confucius", "was born in", "Qufu"),
    ("Pablo Picasso", "was born in", "Malaga"),
    ("Vincent van Gogh", "was born in", "Zundert"),
    ("Frederic Chopin", "was born in", "Zelazowa Wola"),
    ("Ludwig van Beethoven", "was born in", "Bonn"),
    ("Socrates", "was born in", "Athens"),
    ("Marco Polo", "was born in", "Venice"),
    ("Cleopatra", "was born in", "Alexandria"),
    # --- Occupations ---
    ("Albert Einstein", "has occupation", "Physicist"),
    ("Marie Curie", "has occupation", "Physicist"),
    ("William Shakespeare", "has occupation", "Playwright"),
    ("Leonardo da Vinci", "has occupation", "Painter"),
    ("Isaac Newton", "has occupation", "Mathematician"),
    ("Aristotle", "has occupation", "Philosopher"),
    ("Charles Darwin", "has occupation", "Naturalist"),
    ("Sigmund Freud", "has occupation", "Psychologist"),
    ("Nikola Tesla", "has occupation", "Inventor"),
    ("Pablo Picasso", "has occupation", "Painter"),
    ("Ludwig van Beethoven", "has occupation", "Composer"),
    ("Hippocrates", "has occupation", "Physician"),
    ("Archimedes", "has occupation", "Mathematician"),
    ("Plato", "has occupation", "Philosopher"),
    ("Galileo Galilei", "has occupation", "Astronomer"),
    ("Ada Lovelace", "has occupation", "Mathematician"),
    ("Florence Nightingale", "has occupation", "Nurse"),
    ("Alexander Fleming", "has occupation", "Biologist"),
    # --- Locations ---
    ("Eiffel Tower", "is located in", "Paris"),
    ("Statue of Liberty", "is located in", "New York"),
    ("Big Ben", "is located in", "London"),
    ("Colosseum", "is located in", "Rome"),
    ("Taj Mahal", "is located in", "Agra"),
    ("Great Wall", "is located in", "China"),
    ("Sydney Opera House", "is located in", "Sydney"),
    ("Machu Picchu", "is located in", "Peru"),
    ("Pyramids of Giza", "is located in", "Egypt"),
    ("Parthenon", "is located in", "Athens"),
    ("Sagrada Familia", "is located in", "Barcelona"),
    ("Christ the Redeemer", "is located in", "Rio de Janeiro"),
    ("Buckingham Palace", "is located in", "London"),
    ("Kremlin", "is located in", "Moscow"),
    ("Forbidden City", "is located in", "Beijing"),
    ("Louvre Museum", "is located in", "Paris"),
    ("Vatican City", "is located in", "Rome"),
    ("Acropolis", "is located in", "Athens"),
    ("Tower of London", "is located in", "London"),
    ("Mount Fuji", "is located in", "Japan"),
    # --- Founded ---
    ("Apple", "was founded in", "1976"),
    ("Microsoft", "was founded in", "1975"),
    ("Google", "was founded in", "1998"),
    ("Amazon", "was founded in", "1994"),
    ("Tesla", "was founded in", "2003"),
    ("Facebook", "was founded in", "2004"),
    ("IBM", "was founded in", "1911"),
    ("Intel", "was founded in", "1968"),
    ("Samsung", "was founded in", "1938"),
    ("Toyota", "was founded in", "1937"),
    ("Netflix", "was founded in", "1997"),
    ("Twitter", "was founded in", "2006"),
    ("SpaceX", "was founded in", "2002"),
    ("Wikipedia", "was founded in", "2001"),
    ("Coca-Cola", "was founded in", "1886"),
    ("Nike", "was founded in", "1964"),
    # --- Wrote ---
    ("William Shakespeare", "wrote", "Hamlet"),
    ("Charles Dickens", "wrote", "Oliver Twist"),
    ("Leo Tolstoy", "wrote", "War and Peace"),
    ("Jane Austen", "wrote", "Pride and Prejudice"),
    ("Mark Twain", "wrote", "Adventures of Huckleberry Finn"),
    ("Homer", "wrote", "Iliad"),
    ("J.K. Rowling", "wrote", "Harry Potter"),
    ("George Orwell", "wrote", "1984"),
    ("Victor Hugo", "wrote", "Les Miserables"),
    ("Fyodor Dostoevsky", "wrote", "Crime and Punishment"),
    ("Gabriel Garcia Marquez", "wrote", "One Hundred Years of Solitude"),
    ("Franz Kafka", "wrote", "The Metamorphosis"),
    ("Herman Melville", "wrote", "Moby Dick"),
    ("Miguel de Cervantes", "wrote", "Don Quixote"),
    ("Dante Alighieri", "wrote", "Divine Comedy"),
    ("James Joyce", "wrote", "Ulysses"),
    ("Ernest Hemingway", "wrote", "The Old Man and the Sea"),
    ("F. Scott Fitzgerald", "wrote", "The Great Gatsby"),
    # --- Studied at ---
    ("Albert Einstein", "studied at", "ETH Zurich"),
    ("Barack Obama", "studied at", "Harvard University"),
    ("Isaac Newton", "studied at", "Trinity College"),
    ("Marie Curie", "studied at", "University of Paris"),
    ("Alan Turing", "studied at", "Princeton University"),
    ("Stephen Hawking", "studied at", "University of Oxford"),
    ("Niels Bohr", "studied at", "University of Copenhagen"),
    ("John Maynard Keynes", "studied at", "University of Cambridge"),
    ("Noam Chomsky", "studied at", "University of Pennsylvania"),
    ("Tim Berners-Lee", "studied at", "University of Oxford"),
    # --- Invented / discovered ---
    ("Alexander Graham Bell", "invented", "telephone"),
    ("Thomas Edison", "invented", "light bulb"),
    ("Gutenberg", "invented", "printing press"),
    ("Tim Berners-Lee", "invented", "World Wide Web"),
    ("Alexander Fleming", "discovered", "penicillin"),
    ("Marie Curie", "discovered", "radium"),
    ("Isaac Newton", "discovered", "gravity"),
    ("Charles Darwin", "discovered", "natural selection"),
    ("Galileo Galilei", "discovered", "moons of Jupiter"),
    ("Albert Einstein", "developed", "theory of relativity"),
    # --- Nationality ---
    ("Albert Einstein", "has nationality", "German"),
    ("Marie Curie", "has nationality", "Polish"),
    ("Nikola Tesla", "has nationality", "Serbian"),
    ("Leonardo da Vinci", "has nationality", "Italian"),
    ("Napoleon Bonaparte", "has nationality", "French"),
    ("Mahatma Gandhi", "has nationality", "Indian"),
    ("Pablo Picasso", "has nationality", "Spanish"),
    ("Confucius", "has nationality", "Chinese"),
    ("Socrates", "has nationality", "Greek"),
    ("William Shakespeare", "has nationality", "English"),
    # --- Languages ---
    ("France", "has official language", "French"),
    ("Germany", "has official language", "German"),
    ("Japan", "has official language", "Japanese"),
    ("China", "has official language", "Mandarin"),
    ("Brazil", "has official language", "Portuguese"),
    ("Russia", "has official language", "Russian"),
    ("Spain", "has official language", "Spanish"),
    ("Italy", "has official language", "Italian"),
    ("Egypt", "has official language", "Arabic"),
    ("South Korea", "has official language", "Korean"),
    # --- Currencies ---
    ("United States", "has currency", "Dollar"),
    ("United Kingdom", "has currency", "Pound"),
    ("Japan", "has currency", "Yen"),
    ("India", "has currency", "Rupee"),
    ("China", "has currency", "Yuan"),
    ("Switzerland", "has currency", "Franc"),
    ("Russia", "has currency", "Ruble"),
    ("South Korea", "has currency", "Won"),
    ("Brazil", "has currency", "Real"),
    ("Mexico", "has currency", "Peso"),
]


def _get_synthetic_triplets(n: int = 0) -> list[tuple[str, str, str]]:
    """Return synthetic triplets, optionally limited to *n* items."""
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
    d_real_score: float = 0.0
    d_fake_score: float = 0.0
    feat_match: float = 0.0
    mlm_loss: float = 0.0
    diversity: float = 0.0
    num_steps: int = 0

    def update(self, step_metrics: dict[str, float]) -> None:
        self.d_loss += step_metrics["d_loss"]
        self.g_loss += step_metrics["g_loss"]
        self.d_real_score += step_metrics["d_real_score"]
        self.d_fake_score += step_metrics["d_fake_score"]
        self.feat_match += step_metrics.get("feat_match", 0.0)
        self.mlm_loss += step_metrics.get("mlm_loss", 0.0)
        self.diversity += step_metrics.get("diversity", 0.0)
        self.num_steps += 1

    def average(self) -> dict[str, float]:
        n = max(self.num_steps, 1)
        return {
            "d_loss": self.d_loss / n,
            "g_loss": self.g_loss / n,
            "d_real_score": self.d_real_score / n,
            "d_fake_score": self.d_fake_score / n,
            "feat_match": self.feat_match / n,
            "mlm_loss": self.mlm_loss / n,
            "diversity": self.diversity / n,
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
    """End-to-end BERT-based GAN training pipeline.

    Works directly on raw text triplets — no pre-encoding step.
    """

    def __init__(self, config: Optional[GANTrainingConfig] = None) -> None:
        self.config = config or GANTrainingConfig()

        self.gan = FactGAN(
            noise_dim=self.config.noise_dim,
            device=self.config.device,
        )

        # Separate LRs: G learns faster so it can catch up with D
        self.optimizer_g = torch.optim.AdamW(
            self.gan.generator.parameters(),
            lr=self.config.lr_g,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer_d = torch.optim.AdamW(
            self.gan.discriminator.parameters(),
            lr=self.config.lr_d,
            weight_decay=self.config.weight_decay,
        )

        self.history = TrainingHistory()

    # ----- data acquisition -----------------------------------------------

    def fetch_training_triplets(self) -> list[tuple[str, str, str]]:
        """Fetch real triplets from DBpedia for training.

        Falls back to synthetic triplets if DBpedia queries fail or
        return too few results.
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
            existing = set(triplets)
            for t in synthetic:
                if t not in existing:
                    triplets.append(t)

        logger.info("Total training triplets: %d", len(triplets))
        return triplets

    # ----- temperature annealing ------------------------------------------

    def _compute_temperature(self, epoch: int) -> float:
        """Linearly anneal Gumbel-Softmax temperature."""
        cfg = self.config
        if epoch >= cfg.temp_anneal_epochs:
            return cfg.gumbel_temp_end
        progress = epoch / max(cfg.temp_anneal_epochs, 1)
        return cfg.gumbel_temp_start + progress * (cfg.gumbel_temp_end - cfg.gumbel_temp_start)

    # ----- training loop --------------------------------------------------

    def train(
        self,
        epochs: Optional[int] = None,
        triplets: Optional[list[tuple[str, str, str]]] = None,
        save: bool = True,
    ) -> TrainingHistory:
        """Run the full BERT-GAN training loop.

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
        """
        num_epochs = epochs or self.config.epochs
        batch_size = self.config.batch_size

        # ----- Data -------------------------------------------------------
        if triplets is None:
            triplets = self.fetch_training_triplets()

        if not triplets:
            logger.error("No training triplets available. Aborting.")
            return self.history

        num_batches = math.ceil(len(triplets) / batch_size)

        # ----- Training ---------------------------------------------------
        logger.info(
            "Starting BERT-GAN training: %d epochs, %d batches/epoch, "
            "batch_size=%d, %d triplets",
            num_epochs,
            num_batches,
            batch_size,
            len(triplets),
        )
        t_start = time.time()

        for epoch in range(1, num_epochs + 1):
            # Shuffle triplets each epoch
            shuffled = list(triplets)
            random.shuffle(shuffled)

            temperature = self._compute_temperature(epoch - 1)
            epoch_metrics = EpochMetrics()

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(shuffled))
                batch = shuffled[start_idx:end_idx]

                if len(batch) < 2:
                    continue

                step_metrics = self.gan.train_step(
                    real_triplets=batch,
                    optimizer_g=self.optimizer_g,
                    optimizer_d=self.optimizer_d,
                    temperature=temperature,
                    g_steps=self.config.g_steps,
                    label_smoothing=self.config.label_smoothing,
                    d_noise_std=self.config.d_noise_std,
                    feat_match_weight=self.config.feat_match_weight,
                    diversity_weight=self.config.diversity_weight,
                    mlm_weight=self.config.mlm_weight,
                )
                epoch_metrics.update(step_metrics)

            avg = epoch_metrics.average()
            self.history.append(avg)

            if epoch % self.config.log_every == 0 or epoch == 1:
                elapsed = time.time() - t_start
                logger.info(
                    "Epoch %3d/%d | D_loss: %.4f | G_adv: %.4f | "
                    "D_real: %.3f | D_fake: %.3f | FM: %.4f | MLM: %.4f | "
                    "Div: %.4f | temp: %.3f | elapsed: %.1fs",
                    epoch,
                    num_epochs,
                    avg["d_loss"],
                    avg["g_loss"],
                    avg["d_real_score"],
                    avg["d_fake_score"],
                    avg["feat_match"],
                    avg["mlm_loss"],
                    avg["diversity"],
                    temperature,
                    elapsed,
                )

                # Log a few generated triplets to see what G produces
                sample = shuffled[:3]
                generated = self.gan.generator.decode_generated(sample, temperature)
                originals = [f"{s} [REL] {p} [REL] {o}" for s, p, o in sample]
                for orig, gen in zip(originals, generated):
                    logger.info("  REAL: %s", orig)
                    logger.info("  FAKE: %s", gen)

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
        """Score a list of triplets through the discriminator."""
        scores = self.gan.discriminate_triplets(triplets)  # (n, 1)
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
        description="Train the BERT-based FactGAN on DBpedia triplets."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size."
    )
    parser.add_argument(
        "--lr-g", type=float, default=5e-5, help="Generator learning rate."
    )
    parser.add_argument(
        "--lr-d", type=float, default=1e-5, help="Discriminator learning rate."
    )
    parser.add_argument(
        "--g-steps", type=int, default=3, help="G updates per D update."
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
        "--synthetic-only",
        action="store_true",
        help="Skip DBpedia and use only synthetic triplets.",
    )
    parser.add_argument(
        "--gumbel-temp-start",
        type=float,
        default=1.0,
        help="Initial Gumbel-Softmax temperature.",
    )
    parser.add_argument(
        "--gumbel-temp-end",
        type=float,
        default=0.1,
        help="Final Gumbel-Softmax temperature.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config = GANTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        g_steps=args.g_steps,
        output_dir=args.output,
        triplets_per_category=args.per_category,
        gumbel_temp_start=args.gumbel_temp_start,
        gumbel_temp_end=args.gumbel_temp_end,
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
