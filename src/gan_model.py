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

    def _swap_one(self, s: str, p: str, o: str) -> tuple[str, str, str]:
        """Generate a single fake by swapping subject or object."""
        if random.random() < 0.5:
            pool = self.subject_pool.get(p, self.all_subjects)
            candidates = [x for x in pool if x != s]
            if not candidates:
                candidates = [x for x in self.all_subjects if x != s]
            new_s = random.choice(candidates) if candidates else s
            return (new_s, p, o)
        else:
            pool = self.object_pool.get(p, self.all_objects)
            candidates = [x for x in pool if x != o]
            if not candidates:
                candidates = [x for x in self.all_objects if x != o]
            new_o = random.choice(candidates) if candidates else o
            return (s, p, new_o)

    def generate_fakes(
        self, triplets: list[tuple[str, str, str]]
    ) -> list[tuple[str, str, str]]:
        """Generate fake triplets by swapping subject or object."""
        return [self._swap_one(s, p, o) for s, p, o in triplets]

    def generate_candidates(
        self, triplets: list[tuple[str, str, str]], k: int = 5
    ) -> list[list[tuple[str, str, str]]]:
        """Generate K candidate fakes per triplet for hard negative mining.

        Returns list of length len(triplets), each element is a list of K fakes.
        """
        return [
            [self._swap_one(s, p, o) for _ in range(k)]
            for s, p, o in triplets
        ]


# =========================================================================
# Policy Generator — REINFORCE-based learned swap selector
# =========================================================================


class PolicyGenerator(nn.Module):
    """Neural generator that learns to select hard negative swaps.

    Uses the discriminator's BERT [CLS] embeddings (detached) to score
    K random swap candidates per real triplet.  Trained with REINFORCE:
    the discriminator's "real" score on the selected fake is the reward.
    """

    def __init__(
        self,
        hidden_size: int = BERT_HIDDEN_SIZE,
        mlp_hidden: int = 256,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._device = device or _detect_device()
        # Scores each candidate relative to the real triplet
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 1),
        )
        # Running average baseline for REINFORCE variance reduction
        self.baseline: float = 0.0
        self.baseline_decay: float = 0.99
        self.to(self._device)

    def forward(
        self,
        real_cls: torch.Tensor,
        candidate_cls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score candidates and sample one per real triplet.

        Parameters
        ----------
        real_cls : (batch, hidden)
            [CLS] embeddings of real triplets.
        candidate_cls : (batch, K, hidden)
            [CLS] embeddings of K swap candidates per triplet.

        Returns
        -------
        log_probs : (batch, K)
            Log-probabilities over candidates.
        selected : (batch,)
            Sampled candidate index per triplet.
        """
        batch_size, k, hidden = candidate_cls.shape
        real_expanded = real_cls.unsqueeze(1).expand(-1, k, -1)
        combined = torch.cat([real_expanded, candidate_cls], dim=-1)
        scores = self.scorer(combined).squeeze(-1)  # (batch, K)
        log_probs = torch.log_softmax(scores, dim=-1)

        with torch.no_grad():
            selected = torch.multinomial(log_probs.exp(), 1).squeeze(-1)

        return log_probs, selected

    def update_baseline(self, reward_mean: float) -> None:
        self.baseline = (
            self.baseline_decay * self.baseline
            + (1 - self.baseline_decay) * reward_mean
        )


# =========================================================================
# Discriminator
# =========================================================================


class BERTDiscriminator(nn.Module):
    """BERT-based discriminator with a classification head."""

    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        device: Optional[torch.device] = None,
        freeze_bert_layers: int = 10,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self._device = device or _detect_device()

        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Freeze embeddings + first N encoder layers (BERT has 12 layers)
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert.encoder.layer[:freeze_bert_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.cls_dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
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
        cls_hidden = self.cls_dropout(cls_hidden)
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
        freeze_bert_layers: int = 10,
        dropout: float = 0.4,
        generator_mode: str = "random",
    ) -> None:
        self.device = torch.device(device) if device else _detect_device()
        self.generator_mode = generator_mode

        self.generator = SwapGenerator(triplets or [])
        self.discriminator = BERTDiscriminator(
            model_name=model_name,
            device=self.device,
            freeze_bert_layers=freeze_bert_layers,
            dropout=dropout,
        )

        self.criterion = nn.BCELoss()

        # REINFORCE mode: create a policy generator
        self.policy_generator: Optional[PolicyGenerator] = None
        if generator_mode == "reinforce":
            self.policy_generator = PolicyGenerator(device=self.device)
            logger.info("PolicyGenerator created (%d params).",
                        sum(p.numel() for p in self.policy_generator.parameters()))

        logger.info(
            "FactGAN (swap) initialised on %s (model=%s, pool=%d triplets, mode=%s).",
            self.device,
            model_name,
            len(triplets or []),
            generator_mode,
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

    def train_step_hard_negative(
        self,
        real_triplets: list[tuple[str, str, str]],
        optimizer_d: torch.optim.Optimizer,
        label_smoothing: float = 0.9,
        k: int = 5,
    ) -> dict[str, float]:
        """Training step with hard negative mining.

        Generates K candidate fakes per real triplet, scores them all,
        and keeps the hardest (highest discriminator score) for training.
        """
        batch_size = len(real_triplets)

        # Generate K candidates per triplet
        candidates_per = self.generator.generate_candidates(real_triplets, k=k)
        flat_candidates = [c for group in candidates_per for c in group]

        # Score all candidates (no grad) to find hardest
        self.discriminator.eval()
        with torch.no_grad():
            cand_ids, cand_mask = self.discriminator.encode_triplets(flat_candidates)
            cand_scores = self.discriminator(cand_ids, cand_mask).squeeze(-1)

        # Select hardest fake per triplet (highest D score = most convincing)
        cand_scores = cand_scores.view(batch_size, k)
        hardest_idx = cand_scores.argmax(dim=1)
        fake_triplets = [
            candidates_per[i][hardest_idx[i].item()] for i in range(batch_size)
        ]

        # Train discriminator on real vs hard fakes
        real_labels = torch.full((batch_size, 1), label_smoothing, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        real_ids, real_mask = self.discriminator.encode_triplets(real_triplets)
        fake_ids, fake_mask = self.discriminator.encode_triplets(fake_triplets)

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
            "hard_neg_score": cand_scores.max(dim=1).values.mean().item(),
            "fake_examples": fake_triplets[:3],
        }

    def train_step_reinforce(
        self,
        real_triplets: list[tuple[str, str, str]],
        optimizer_d: torch.optim.Optimizer,
        optimizer_g: torch.optim.Optimizer,
        label_smoothing: float = 0.9,
        k: int = 5,
    ) -> dict[str, float]:
        """Training step with REINFORCE policy gradient generator.

        The PolicyGenerator learns to select swap candidates that fool
        the discriminator, using D's score as reward signal.
        """
        assert self.policy_generator is not None, "PolicyGenerator required for reinforce mode"
        batch_size = len(real_triplets)

        # 1. Generate K random candidates per real triplet
        candidates_per = self.generator.generate_candidates(real_triplets, k=k)
        flat_candidates = [c for group in candidates_per for c in group]

        # 2. Get BERT [CLS] embeddings (detached — only PolicyGenerator MLP gets gradients)
        self.discriminator.eval()
        real_ids, real_mask = self.discriminator.encode_triplets(real_triplets)
        cand_ids, cand_mask = self.discriminator.encode_triplets(flat_candidates)

        with torch.no_grad():
            real_cls = self.discriminator.bert(
                input_ids=real_ids, attention_mask=real_mask
            ).last_hidden_state[:, 0, :]  # (batch, 768)
            cand_cls = self.discriminator.bert(
                input_ids=cand_ids, attention_mask=cand_mask
            ).last_hidden_state[:, 0, :]  # (batch*K, 768)

        cand_cls = cand_cls.view(batch_size, k, -1)  # (batch, K, 768)

        # 3. PolicyGenerator scores candidates and samples one
        self.policy_generator.train()
        log_probs, selected_idx = self.policy_generator(real_cls, cand_cls)

        fake_triplets = [
            candidates_per[i][selected_idx[i].item()] for i in range(batch_size)
        ]

        # 4. Discriminator scores selected fakes → reward
        with torch.no_grad():
            fake_ids_r, fake_mask_r = self.discriminator.encode_triplets(fake_triplets)
            reward = self.discriminator(fake_ids_r, fake_mask_r).squeeze(-1)  # (batch,)

        reward_mean = reward.mean().item()

        # 5. Generator loss: REINFORCE with baseline
        advantage = reward - self.policy_generator.baseline
        selected_log_probs = log_probs[torch.arange(batch_size, device=self.device), selected_idx]
        g_loss = -(selected_log_probs * advantage.detach()).mean()

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()
        self.policy_generator.update_baseline(reward_mean)

        # 6. Train discriminator on real vs generator-selected fakes
        real_labels = torch.full((batch_size, 1), label_smoothing, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        fake_ids, fake_mask = self.discriminator.encode_triplets(fake_triplets)

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
            "g_loss": g_loss.item(),
            "d_real_score": d_real_out.mean().item(),
            "d_fake_score": d_fake_out.mean().item(),
            "g_reward": reward_mean,
            "g_baseline": self.policy_generator.baseline,
            "fake_examples": fake_triplets[:3],
        }

    # ----- persistence -----------------------------------------------------

    def save(self, directory: str) -> None:
        """Save discriminator (and policy generator if present) state_dict."""
        os.makedirs(directory, exist_ok=True)
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(directory, "discriminator.pt"),
        )
        if self.policy_generator is not None:
            torch.save(
                self.policy_generator.state_dict(),
                os.path.join(directory, "policy_generator.pt"),
            )
        meta = {"architecture": "bert-gan-swap", "generator_mode": self.generator_mode}
        torch.save(meta, os.path.join(directory, "gan_meta.pt"))
        logger.info("FactGAN (swap) saved to %s", directory)

    def load(self, directory: str) -> None:
        """Load previously saved discriminator (and policy generator) state_dict."""
        disc_path = os.path.join(directory, "discriminator.pt")
        if os.path.isfile(disc_path):
            self.discriminator.load_state_dict(
                torch.load(disc_path, map_location=self.device, weights_only=True)
            )
        pg_path = os.path.join(directory, "policy_generator.pt")
        if self.policy_generator is not None and os.path.isfile(pg_path):
            self.policy_generator.load_state_dict(
                torch.load(pg_path, map_location=self.device, weights_only=True)
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
