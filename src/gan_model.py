"""GAN architecture for knowledge-graph fact verification.

Components
----------
TripletEncoder
    Encodes text triplets (subject, predicate, object) into fixed-size
    embedding vectors using a pre-trained sentence transformer.

FactGenerator (G)
    Takes random noise ``z`` (optionally concatenated with entity
    embeddings) and produces *fake* triplet embeddings.

FactDiscriminator (D)
    Takes a triplet embedding and outputs a probability that the
    embedding represents a *real* fact (1) vs. a *fake* fact (0).

FactGAN
    Wrapper that bundles generator, discriminator, and encoder and
    exposes high-level training / inference helpers.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default hyper-parameters (can be overridden via constructor kwargs)
# ---------------------------------------------------------------------------
DEFAULT_NOISE_DIM = 128
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_HIDDEN_DIM = 512
DEFAULT_DROPOUT = 0.3


# =========================================================================
# Triplet Encoder
# =========================================================================


class TripletEncoder(nn.Module):
    """Encode a text triplet into a fixed-size embedding vector.

    The encoder concatenates the three parts of the triplet with a
    separator token and passes the result through a pre-trained sentence
    transformer.  The output is then projected to ``embedding_dim``
    dimensions via a trainable linear layer.

    When ``use_sentence_transformer=False`` a lightweight bag-of-words
    fallback is used instead (useful when ``sentence-transformers`` is
    not installed or for fast unit tests).
    """

    SEPARATOR = " [REL] "

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        sentence_model_name: str = "all-MiniLM-L6-v2",
        use_sentence_transformer: bool = True,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._use_st = use_sentence_transformer
        self._st_model = None
        self._st_dim: int = 0

        if self._use_st:
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(
                    sentence_model_name, device=str(self._device)
                )
                # Determine the native output dimension of the model
                self._st_dim = self._st_model.get_sentence_embedding_dimension()
                logger.info(
                    "TripletEncoder: using SentenceTransformer '%s' (dim=%d).",
                    sentence_model_name,
                    self._st_dim,
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Falling back to simple hash-based encoder."
                )
                self._use_st = False

        if not self._use_st:
            # Lightweight fallback: deterministic hash-based embeddings
            self._st_dim = embedding_dim
            logger.info(
                "TripletEncoder: using hash-based fallback (dim=%d).",
                self._st_dim,
            )

        # Projection from sentence-transformer dimension to embedding_dim
        self.projection = nn.Linear(self._st_dim, embedding_dim)
        self.to(self._device)

    # ----- public helpers ------------------------------------------------

    def format_triplet(
        self, subject: str, predicate: str, obj: str
    ) -> str:
        """Join a triplet into a single string suitable for encoding."""
        return f"{subject}{self.SEPARATOR}{predicate}{self.SEPARATOR}{obj}"

    def format_triplets(
        self, triplets: list[tuple[str, str, str]]
    ) -> list[str]:
        """Format a batch of triplets into strings."""
        return [self.format_triplet(s, p, o) for s, p, o in triplets]

    # ----- forward -------------------------------------------------------

    def forward(
        self, triplets: list[tuple[str, str, str]]
    ) -> torch.Tensor:
        """Encode a list of text triplets into a ``(batch, embedding_dim)`` tensor.

        Parameters
        ----------
        triplets : list[tuple[str, str, str]]
            Each element is ``(subject, predicate, object)`` as strings.

        Returns
        -------
        torch.Tensor
            Shape ``(len(triplets), embedding_dim)``.
        """
        texts = self.format_triplets(triplets)

        if self._use_st and self._st_model is not None:
            # sentence-transformers returns a numpy array by default
            raw_embeddings = self._st_model.encode(
                texts, convert_to_tensor=True, show_progress_bar=False
            )
            raw_embeddings = raw_embeddings.to(self._device).float()
        else:
            raw_embeddings = self._hash_encode(texts)

        projected = self.projection(raw_embeddings)
        return projected

    def encode(
        self, triplets: list[tuple[str, str, str]]
    ) -> torch.Tensor:
        """Convenience alias for ``forward`` (no-grad context)."""
        self.eval()
        with torch.no_grad():
            return self.forward(triplets)

    # ----- fallback encoder -----------------------------------------------

    def _hash_encode(self, texts: list[str]) -> torch.Tensor:
        """Deterministic hash-based encoding used as a fallback.

        This is NOT a learned embedding; it produces a fixed pseudo-random
        vector for each input string so that training can proceed without
        ``sentence-transformers``.
        """
        embeddings: list[torch.Tensor] = []
        for text in texts:
            # Use Python's hash seeded with each character position to
            # produce a reproducible vector.
            gen = torch.Generator()
            gen.manual_seed(abs(hash(text)) % (2**31))
            vec = torch.randn(self._st_dim, generator=gen)
            embeddings.append(vec)
        return torch.stack(embeddings).to(self._device)


# =========================================================================
# Generator
# =========================================================================


class FactGenerator(nn.Module):
    """Generator network that produces fake triplet embeddings.

    Architecture
    ------------
    Input : ``noise_dim`` (+ optional ``entity_embedding_dim``)
    Hidden: two fully-connected layers with BatchNorm + LeakyReLU
    Output: ``embedding_dim`` with Tanh activation
    """

    def __init__(
        self,
        noise_dim: int = DEFAULT_NOISE_DIM,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        entity_embedding_dim: int = 0,
    ) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.entity_embedding_dim = entity_embedding_dim

        input_dim = noise_dim + entity_embedding_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        noise: torch.Tensor,
        entity_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate fake triplet embeddings.

        Parameters
        ----------
        noise : torch.Tensor
            Random noise of shape ``(batch, noise_dim)``.
        entity_embedding : torch.Tensor or None
            Optional entity conditioning of shape ``(batch, entity_embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Fake embeddings of shape ``(batch, embedding_dim)``.
        """
        if entity_embedding is not None:
            x = torch.cat([noise, entity_embedding], dim=1)
        else:
            x = noise
        return self.net(x)


# =========================================================================
# Discriminator
# =========================================================================


class FactDiscriminator(nn.Module):
    """Discriminator network that classifies embeddings as real / fake.

    Architecture
    ------------
    Input : ``embedding_dim``
    Hidden: two fully-connected layers with LeakyReLU + Dropout
    Output: single logit passed through Sigmoid
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return the probability that *embedding* is a real fact.

        Parameters
        ----------
        embedding : torch.Tensor
            Shape ``(batch, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` with values in ``[0, 1]``.
        """
        return self.net(embedding)


# =========================================================================
# FactGAN wrapper
# =========================================================================


class FactGAN:
    """High-level wrapper around the GAN components.

    Bundles the :class:`FactGenerator`, :class:`FactDiscriminator`, and
    :class:`TripletEncoder` and provides convenience methods for
    training, generation, and discrimination.

    Parameters
    ----------
    noise_dim : int
        Dimensionality of the generator input noise.
    embedding_dim : int
        Dimensionality of triplet embeddings (output of encoder and
        generator, input of discriminator).
    hidden_dim : int
        Width of hidden layers in G and D.
    dropout : float
        Dropout rate for the discriminator.
    entity_embedding_dim : int
        If > 0 the generator accepts additional entity conditioning.
    use_sentence_transformer : bool
        Whether to use ``sentence-transformers`` for encoding.
    sentence_model_name : str
        HuggingFace model identifier for the sentence transformer.
    device : str or None
        ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        noise_dim: int = DEFAULT_NOISE_DIM,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        entity_embedding_dim: int = 0,
        use_sentence_transformer: bool = True,
        sentence_model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim

        # Components
        self.encoder = TripletEncoder(
            embedding_dim=embedding_dim,
            sentence_model_name=sentence_model_name,
            use_sentence_transformer=use_sentence_transformer,
            device=str(self.device),
        )

        self.generator = FactGenerator(
            noise_dim=noise_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            entity_embedding_dim=entity_embedding_dim,
        ).to(self.device)

        self.discriminator = FactDiscriminator(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)

        # Loss function
        self.criterion = nn.BCELoss()

        logger.info(
            "FactGAN initialised on %s (noise=%d, emb=%d, hidden=%d).",
            self.device,
            noise_dim,
            embedding_dim,
            hidden_dim,
        )

    # ----- noise sampling ------------------------------------------------

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        """Sample a batch of random noise vectors for the generator."""
        return torch.randn(batch_size, self.noise_dim, device=self.device)

    # ----- high-level methods --------------------------------------------

    def encode_triplets(
        self, triplets: list[tuple[str, str, str]]
    ) -> torch.Tensor:
        """Encode text triplets into embedding vectors.

        Returns a ``(batch, embedding_dim)`` tensor on ``self.device``.
        """
        return self.encoder.encode(triplets).to(self.device)

    def generate_fake_facts(
        self,
        n: int,
        entity_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate *n* fake triplet embeddings using the generator.

        Parameters
        ----------
        n : int
            Number of fake embeddings to produce.
        entity_embedding : torch.Tensor or None
            Optional conditioning tensor of shape ``(n, entity_embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Fake embeddings of shape ``(n, embedding_dim)``.
        """
        self.generator.eval()
        with torch.no_grad():
            noise = self.sample_noise(n)
            fakes = self.generator(noise, entity_embedding)
        return fakes

    def discriminate(self, triplet_embedding: torch.Tensor) -> torch.Tensor:
        """Return the discriminator's "realness" score for each embedding.

        Parameters
        ----------
        triplet_embedding : torch.Tensor
            Shape ``(batch, embedding_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` with values in ``[0, 1]``.
        """
        self.discriminator.eval()
        with torch.no_grad():
            return self.discriminator(triplet_embedding.to(self.device))

    def discriminate_triplets(
        self, triplets: list[tuple[str, str, str]]
    ) -> torch.Tensor:
        """Encode text triplets and return the discriminator scores.

        Convenience method that chains :meth:`encode_triplets` and
        :meth:`discriminate`.
        """
        embeddings = self.encode_triplets(triplets)
        return self.discriminate(embeddings)

    def train_step(
        self,
        real_embeddings: torch.Tensor,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        entity_embedding: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Execute one GAN training step (update D then G).

        Parameters
        ----------
        real_embeddings : torch.Tensor
            A batch of real triplet embeddings ``(batch, embedding_dim)``.
        optimizer_g : torch.optim.Optimizer
            Optimizer for the generator parameters.
        optimizer_d : torch.optim.Optimizer
            Optimizer for the discriminator parameters.
        entity_embedding : torch.Tensor or None
            Optional generator conditioning.

        Returns
        -------
        dict[str, float]
            Training metrics: ``d_loss``, ``g_loss``, ``d_real_acc``,
            ``d_fake_acc``.
        """
        batch_size = real_embeddings.size(0)
        real_embeddings = real_embeddings.to(self.device)

        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # ----- Train Discriminator ---------------------------------------
        self.discriminator.train()
        self.generator.eval()

        optimizer_d.zero_grad()

        # Real examples
        d_real_out = self.discriminator(real_embeddings)
        d_loss_real = self.criterion(d_real_out, real_labels)

        # Fake examples
        noise = self.sample_noise(batch_size)
        fake_embeddings = self.generator(noise, entity_embedding).detach()
        d_fake_out = self.discriminator(fake_embeddings)
        d_loss_fake = self.criterion(d_fake_out, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # ----- Train Generator -------------------------------------------
        self.generator.train()
        self.discriminator.eval()

        optimizer_g.zero_grad()

        noise = self.sample_noise(batch_size)
        fake_embeddings = self.generator(noise, entity_embedding)
        g_out = self.discriminator(fake_embeddings)
        g_loss = self.criterion(g_out, real_labels)  # G wants D to say "real"

        g_loss.backward()
        optimizer_g.step()

        # ----- Metrics ----------------------------------------------------
        d_real_acc = (d_real_out > 0.5).float().mean().item()
        d_fake_acc = (d_fake_out < 0.5).float().mean().item()

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "d_real_acc": d_real_acc,
            "d_fake_acc": d_fake_acc,
        }

    # ----- persistence ----------------------------------------------------

    def save(self, directory: str) -> None:
        """Save generator, discriminator, and encoder projection weights."""
        import os

        os.makedirs(directory, exist_ok=True)
        torch.save(
            self.generator.state_dict(),
            os.path.join(directory, "generator.pt"),
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(directory, "discriminator.pt"),
        )
        torch.save(
            self.encoder.projection.state_dict(),
            os.path.join(directory, "encoder_projection.pt"),
        )
        # Save hyper-parameters so we can reconstruct the model later
        meta = {
            "noise_dim": self.noise_dim,
            "embedding_dim": self.embedding_dim,
            "generator": {
                "hidden_dim": self.generator.net[0].in_features
                if hasattr(self.generator.net[0], "in_features")
                else DEFAULT_HIDDEN_DIM,
            },
            "discriminator": {
                "hidden_dim": self.discriminator.net[0].out_features,
            },
        }
        torch.save(meta, os.path.join(directory, "gan_meta.pt"))
        logger.info("FactGAN saved to %s", directory)

    def load(self, directory: str) -> None:
        """Load previously saved weights into the current model."""
        import os

        gen_path = os.path.join(directory, "generator.pt")
        disc_path = os.path.join(directory, "discriminator.pt")
        proj_path = os.path.join(directory, "encoder_projection.pt")

        if os.path.isfile(gen_path):
            self.generator.load_state_dict(
                torch.load(gen_path, map_location=self.device, weights_only=True)
            )
        if os.path.isfile(disc_path):
            self.discriminator.load_state_dict(
                torch.load(disc_path, map_location=self.device, weights_only=True)
            )
        if os.path.isfile(proj_path):
            self.encoder.projection.load_state_dict(
                torch.load(proj_path, map_location=self.device, weights_only=True)
            )
        logger.info("FactGAN loaded from %s", directory)


# =========================================================================
# Quick smoke test
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    gan = FactGAN(use_sentence_transformer=False)

    sample_triplets = [
        ("Paris", "is capital of", "France"),
        ("Barack Obama", "was born in", "Hawaii"),
        ("Albert Einstein", "developed", "theory of relativity"),
    ]

    # Encode
    embs = gan.encode_triplets(sample_triplets)
    print(f"Encoded shape: {embs.shape}")  # (3, 256)

    # Generate fakes
    fakes = gan.generate_fake_facts(5)
    print(f"Fake shape: {fakes.shape}")  # (5, 256)

    # Discriminate
    scores = gan.discriminate(embs)
    print(f"Discriminator scores: {scores.squeeze().tolist()}")

    # One training step
    optimizer_g = torch.optim.Adam(
        gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    metrics = gan.train_step(embs, optimizer_g, optimizer_d)
    print(f"Train step metrics: {metrics}")
