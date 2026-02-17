"""BERT-based GAN architecture for knowledge-graph fact verification.

Components
----------
BERTGenerator (G)
    Based on BertForMaskedLM.  Takes a real triplet, masks the subject or
    object, injects noise, and uses Gumbel-Softmax to produce differentiable
    soft token embeddings that can fool the discriminator.

BERTDiscriminator (D)
    Based on BertModel.  Classifies triplet representations as real (1) or
    fake (0).  Supports two forward paths: ``forward_text`` for real tokenised
    triplets and ``forward_soft`` for the Generator's soft embeddings.

FactGAN
    High-level wrapper that bundles G and D and exposes the same public
    interface as the previous MLP-based implementation (``discriminate_triplets``,
    ``train_step``, ``save``, ``load``).
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertModel, BertTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BERT_MODEL_NAME = "bert-base-uncased"
BERT_HIDDEN_SIZE = 768
DEFAULT_NOISE_DIM = 128
TRIPLET_SEP = " [REL] "


def _detect_device() -> torch.device:
    """Select the best available device (mps > cuda > cpu)."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _format_triplet(subject: str, predicate: str, obj: str) -> str:
    return f"{subject}{TRIPLET_SEP}{predicate}{TRIPLET_SEP}{obj}"


# =========================================================================
# Generator
# =========================================================================


class BERTGenerator(nn.Module):
    """MLM-corruption generator built on ``BertForMaskedLM``.

    Strategy:
    1. Receive a real triplet as text ``"subject [REL] predicate [REL] object"``.
    2. Tokenise and randomly mask the *subject* or *object* tokens.
    3. Add projected noise to the input embeddings.
    4. Forward through BERT-MLM to get logits over the vocabulary.
    5. Apply **Gumbel-Softmax** on the masked positions to obtain
       differentiable *soft* one-hot distributions.
    6. Convert to *soft embeddings* via ``distributions @ embedding_matrix``.
    7. Return the full sequence of soft embeddings (real positions use the
       normal embedding; masked positions use the Gumbel output).
    """

    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        noise_dim: int = DEFAULT_NOISE_DIM,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._device = device or _detect_device()
        self.noise_dim = noise_dim

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_mlm = BertForMaskedLM.from_pretrained(model_name)

        # Project noise into BERT hidden space and add to embeddings
        self.noise_proj = nn.Linear(noise_dim, BERT_HIDDEN_SIZE)

        self.to(self._device)

    # ------------------------------------------------------------------

    def _tokenize_and_mask(
        self, triplets: list[tuple[str, str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenise triplets and mask either subject or object.

        Returns
        -------
        input_ids : (batch, seq_len)
        attention_mask : (batch, seq_len)
        mask_positions : (batch, seq_len) bool — True where tokens were masked
        """
        texts = [_format_triplet(s, p, o) for s, p, o in triplets]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)

        for i, (s, p, o) in enumerate(triplets):
            # Decide whether to mask subject or object
            target = s if random.random() < 0.5 else o
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            seq = input_ids[i].tolist()
            # Find the target subsequence in the tokenised sequence
            for start in range(len(seq) - len(target_ids) + 1):
                if seq[start : start + len(target_ids)] == target_ids:
                    for j in range(start, start + len(target_ids)):
                        input_ids[i, j] = mask_token_id
                        mask_positions[i, j] = True
                    break

        return input_ids, attention_mask, mask_positions

    # ------------------------------------------------------------------

    def forward(
        self,
        triplets: list[tuple[str, str, str]],
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate soft embeddings from masked triplets.

        Parameters
        ----------
        triplets : list of (subject, predicate, object) strings
        temperature : float
            Gumbel-Softmax temperature (lower → harder, more discrete).

        Returns
        -------
        soft_embeddings : (batch, seq_len, hidden_size)
            Differentiable token embeddings mixing real and Gumbel-sampled.
        attention_mask : (batch, seq_len)
        token_ids : (batch, seq_len)
            Token IDs with masked positions replaced by generated argmax IDs
            (for decoding).
        mask_positions : (batch, seq_len) bool
            Which positions were masked/generated.
        logits : (batch, seq_len, vocab_size)
            Raw MLM logits (for MLM reconstruction loss).
        target_ids : (batch, seq_len)
            Original token IDs before masking (ground truth for MLM loss).
        """
        input_ids, attention_mask, mask_positions = self._tokenize_and_mask(triplets)
        batch_size, seq_len = input_ids.shape

        # Keep a copy of original IDs before masking for decoding
        # (input_ids already has [MASK] tokens, so reconstruct from triplets)
        texts = [_format_triplet(s, p, o) for s, p, o in triplets]
        orig_enc = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=64,
        )
        original_ids = orig_enc["input_ids"].to(self._device)

        # Save target IDs (ground truth) before we overwrite with generated IDs
        target_ids = original_ids.clone()

        # --- Get the normal BERT word embeddings for the (partially masked) input
        word_embeddings = self.bert_mlm.bert.embeddings.word_embeddings
        embedding_matrix = word_embeddings.weight  # (vocab_size, hidden_size)

        # Get base embeddings for all positions
        base_embeds = word_embeddings(input_ids)  # (batch, seq_len, hidden_size)

        # --- Inject noise: project z and broadcast-add to all positions
        noise = torch.randn(batch_size, self.noise_dim, device=self._device)
        noise_embed = self.noise_proj(noise).unsqueeze(1)  # (batch, 1, hidden_size)
        noisy_embeds = base_embeds + noise_embed

        # --- Forward through BERT-MLM with noisy embeddings
        outputs = self.bert_mlm(
            inputs_embeds=noisy_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # --- Gumbel-Softmax on masked positions
        soft_embeddings = base_embeds.clone()

        if mask_positions.any():
            masked_logits = logits[mask_positions]  # (num_masked, vocab_size)
            gumbel_dist = F.gumbel_softmax(
                masked_logits, tau=temperature, hard=False
            )  # (num_masked, vocab_size)

            # Soft embeddings = weighted sum over vocabulary embeddings
            soft_tokens = gumbel_dist @ embedding_matrix  # (num_masked, hidden_size)
            soft_embeddings[mask_positions] = soft_tokens

            # Store the argmax token IDs for decoding
            generated_ids = gumbel_dist.argmax(dim=-1)
            original_ids[mask_positions] = generated_ids

        return soft_embeddings, attention_mask, original_ids, mask_positions, logits, target_ids

    # ------------------------------------------------------------------

    def decode_generated(
        self,
        triplets: list[tuple[str, str, str]],
        temperature: float = 1.0,
    ) -> list[str]:
        """Generate fake triplets and decode them to readable text.

        Returns a list of strings showing the generated triplets with
        masked positions replaced by the Generator's predictions.
        """
        self.eval()
        with torch.no_grad():
            _, _, token_ids, mask_pos, _, _ = self.forward(triplets, temperature)

        results = []
        for i in range(token_ids.shape[0]):
            ids = token_ids[i].tolist()
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            results.append(decoded)
        return results


# =========================================================================
# Discriminator
# =========================================================================


class BERTDiscriminator(nn.Module):
    """BERT-based discriminator with a classification head.

    Supports two forward paths:
    - ``forward_text``: takes tokenised input_ids (for real triplets)
    - ``forward_soft``: takes soft embeddings directly (for Generator output)

    The classifier is split into a feature extractor and an output head
    so that intermediate features can be used for feature matching loss.
    """

    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._device = device or _detect_device()

        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Split classifier into feature layer + output layer so we can
        # extract intermediate features for feature matching loss.
        self.feature_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(BERT_HIDDEN_SIZE, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.output_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(256, 1)),
            nn.Sigmoid(),
        )

        self.to(self._device)

    # ------------------------------------------------------------------

    def _get_cls(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract [CLS] hidden state from BERT."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        return outputs.last_hidden_state[:, 0, :]

    def _score_from_cls(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(cls_hidden)
        return self.output_layer(features)

    # ------------------------------------------------------------------

    def forward_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score real tokenised triplets. Returns (batch, 1) in [0, 1]."""
        cls_hidden = self._get_cls(input_ids=input_ids, attention_mask=attention_mask)
        return self._score_from_cls(cls_hidden)

    def forward_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score + return intermediate features for feature matching.

        Returns (scores (batch,1), features (batch, 256)).
        """
        cls_hidden = self._get_cls(input_ids=input_ids, attention_mask=attention_mask)
        features = self.feature_layer(cls_hidden)
        scores = self.output_layer(features)
        return scores, features

    # ------------------------------------------------------------------

    def forward_soft(
        self,
        soft_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score Generator's soft embeddings. Returns (batch, 1) in [0, 1]."""
        cls_hidden = self._get_cls(inputs_embeds=soft_embeddings, attention_mask=attention_mask)
        return self._score_from_cls(cls_hidden)

    def forward_soft_features(
        self,
        soft_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score + return intermediate features for feature matching.

        Returns (scores (batch,1), features (batch, 256)).
        """
        cls_hidden = self._get_cls(inputs_embeds=soft_embeddings, attention_mask=attention_mask)
        features = self.feature_layer(cls_hidden)
        scores = self.output_layer(features)
        return scores, features


# =========================================================================
# FactGAN wrapper
# =========================================================================


class FactGAN:
    """High-level wrapper around the BERT-based GAN components.

    Preserves the same public interface as the previous MLP-based
    implementation: ``discriminate_triplets``, ``train_step``, ``save``,
    ``load``.

    Parameters
    ----------
    noise_dim : int
        Dimensionality of the Generator noise input.
    model_name : str
        HuggingFace BERT model identifier (shared by G and D).
    device : str or None
        ``'mps'``, ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        noise_dim: int = DEFAULT_NOISE_DIM,
        model_name: str = BERT_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device) if device else _detect_device()
        self.noise_dim = noise_dim

        self.generator = BERTGenerator(
            model_name=model_name,
            noise_dim=noise_dim,
            device=self.device,
        )
        self.discriminator = BERTDiscriminator(
            model_name=model_name,
            device=self.device,
        )

        self.criterion = nn.BCELoss()

        logger.info(
            "FactGAN (BERT) initialised on %s (noise_dim=%d, model=%s).",
            self.device,
            noise_dim,
            model_name,
        )

    # ----- public interface ------------------------------------------------

    def discriminate_triplets(
        self, triplets: list[tuple[str, str, str]]
    ) -> torch.Tensor:
        """Tokenise text triplets and return discriminator scores.

        Parameters
        ----------
        triplets : list of (subject, predicate, object)

        Returns
        -------
        torch.Tensor
            Shape ``(len(triplets), 1)`` with values in ``[0, 1]``.
        """
        self.discriminator.eval()
        texts = [_format_triplet(s, p, o) for s, p, o in triplets]
        enc = self.discriminator.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            scores = self.discriminator.forward_text(input_ids, attention_mask)
        return scores

    # ----- training --------------------------------------------------------

    def train_step(
        self,
        real_triplets: list[tuple[str, str, str]],
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        temperature: float = 1.0,
        g_steps: int = 3,
        label_smoothing: float = 0.9,
        d_noise_std: float = 0.1,
        feat_match_weight: float = 10.0,
        diversity_weight: float = 1.0,
        mlm_weight: float = 1.0,
    ) -> dict[str, float]:
        """Execute one adversarial training step with anti-collapse mechanisms.

        Anti-collapse techniques:
        1. **Feature matching**: G minimises the L2 distance between the mean
           of D's intermediate features on real vs. fake data, instead of
           only trying to maximise D's score.  This gives G a smoother,
           more informative gradient signal.
        2. **Instance noise**: Gaussian noise is added to D's input embeddings
           so that D cannot easily memorise the difference between real and
           fake distributions.
        3. **Diversity loss**: Penalises G when different inputs in a batch
           produce similar soft embeddings at the masked positions, forcing
           variety in the generated tokens.
        4. **MLM reconstruction loss**: Cross-entropy between G's logits at
           masked positions and the true token IDs, guiding G toward
           producing real, readable words rather than gibberish.

        Parameters
        ----------
        real_triplets : list of (subject, predicate, object)
        optimizer_g, optimizer_d : Optimizers.
        temperature : Gumbel-Softmax temperature.
        g_steps : Generator updates per Discriminator update.
        label_smoothing : Real label for D (< 1.0).
        d_noise_std : Std of Gaussian noise injected into D's inputs.
        feat_match_weight : Weight of the feature matching loss for G.
        diversity_weight : Weight of the diversity loss for G.
        mlm_weight : Weight of the MLM reconstruction loss for G.

        Returns
        -------
        dict with d_loss, g_loss, d_real_score, d_fake_score, feat_match, diversity, mlm_loss
        """
        batch_size = len(real_triplets)
        real_labels = torch.full((batch_size, 1), label_smoothing, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # --- Tokenise real triplets for D --------------------------------
        texts = [_format_triplet(s, p, o) for s, p, o in real_triplets]
        enc = self.discriminator.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        real_input_ids = enc["input_ids"].to(self.device)
        real_attention_mask = enc["attention_mask"].to(self.device)

        # =================================================================
        # Train Discriminator (1 step) — with instance noise
        # =================================================================
        self.discriminator.train()
        self.generator.eval()
        optimizer_d.zero_grad()

        # Real — add instance noise to BERT embeddings before D scores
        real_embeds = self.discriminator.bert.embeddings.word_embeddings(real_input_ids)
        if d_noise_std > 0:
            real_embeds = real_embeds + torch.randn_like(real_embeds) * d_noise_std
        d_real_out = self.discriminator.forward_soft(real_embeds, real_attention_mask)
        d_loss_real = self.criterion(d_real_out, real_labels)

        # Fake — also add noise to G's output before D scores
        with torch.no_grad():
            fake_embeds, fake_attn, _, _, _, _ = self.generator(real_triplets, temperature)
        noisy_fake = fake_embeds.detach()
        if d_noise_std > 0:
            noisy_fake = noisy_fake + torch.randn_like(noisy_fake) * d_noise_std
        d_fake_out = self.discriminator.forward_soft(noisy_fake, fake_attn)
        d_loss_fake = self.criterion(d_fake_out, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # =================================================================
        # Compute real features for feature matching (detached from D)
        # =================================================================
        self.discriminator.eval()
        with torch.no_grad():
            _, real_features = self.discriminator.forward_text_features(
                real_input_ids, real_attention_mask
            )
            real_feat_mean = real_features.mean(dim=0)  # (256,)

        # =================================================================
        # Train Generator (multiple steps) — feature matching + diversity
        # =================================================================
        self.generator.train()

        g_loss_total = 0.0
        feat_match_total = 0.0
        diversity_total = 0.0
        mlm_loss_total = 0.0
        g_real_labels = torch.ones(batch_size, 1, device=self.device)

        for _ in range(g_steps):
            optimizer_g.zero_grad()

            fake_embeds_g, fake_attn_g, _, mask_pos, logits, target_ids = self.generator(
                real_triplets, temperature
            )

            # --- Adversarial loss (G wants D to say "real")
            g_out, fake_features = self.discriminator.forward_soft_features(
                fake_embeds_g, fake_attn_g
            )
            g_adv_loss = self.criterion(g_out, g_real_labels)

            # --- Feature matching loss: match mean features of real data
            fake_feat_mean = fake_features.mean(dim=0)
            feat_match_loss = F.mse_loss(fake_feat_mean, real_feat_mean)

            # --- MLM reconstruction loss: cross-entropy between G's logits
            #     at masked positions and the true token IDs.  This guides G
            #     toward producing real, readable words.
            mlm_loss = torch.tensor(0.0, device=self.device)
            if mask_pos.any():
                mlm_logits = logits[mask_pos]       # (num_masked, vocab_size)
                mlm_targets = target_ids[mask_pos]   # (num_masked,)
                mlm_loss = F.cross_entropy(mlm_logits, mlm_targets)

            # --- Diversity loss: penalise when masked positions across
            #     different samples produce identical embeddings.
            div_loss = torch.tensor(0.0, device=self.device)
            if batch_size > 1 and mask_pos.any():
                # Average the soft embeddings over masked positions per sample
                masked_means = []
                for i in range(batch_size):
                    if mask_pos[i].any():
                        masked_means.append(fake_embeds_g[i][mask_pos[i]].mean(dim=0))
                if len(masked_means) > 1:
                    stacked = torch.stack(masked_means)  # (N, hidden)
                    # Compute pairwise L2 distances without torch.cdist
                    # (cdist backward is not supported on MPS)
                    diff = stacked.unsqueeze(0) - stacked.unsqueeze(1)  # (N, N, hidden)
                    pairwise_dist = diff.norm(dim=-1)  # (N, N)
                    # Exclude diagonal (self-distance = 0)
                    n = pairwise_dist.shape[0]
                    mask = ~torch.eye(n, dtype=torch.bool, device=self.device)
                    mean_dist = pairwise_dist[mask].mean()
                    # Diversity loss = negative distance (we want to maximise distance)
                    div_loss = -mean_dist

            g_loss = (
                g_adv_loss
                + feat_match_weight * feat_match_loss
                + mlm_weight * mlm_loss
                + diversity_weight * div_loss
            )
            g_loss.backward()
            optimizer_g.step()

            g_loss_total += g_adv_loss.item()
            feat_match_total += feat_match_loss.item()
            mlm_loss_total += mlm_loss.item()
            diversity_total += div_loss.item()

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss_total / g_steps,
            "d_real_score": d_real_out.mean().item(),
            "d_fake_score": d_fake_out.mean().item(),
            "feat_match": feat_match_total / g_steps,
            "mlm_loss": mlm_loss_total / g_steps,
            "diversity": diversity_total / g_steps,
        }

    # ----- persistence -----------------------------------------------------

    def save(self, directory: str) -> None:
        """Save Generator and Discriminator state_dicts."""
        os.makedirs(directory, exist_ok=True)
        torch.save(
            self.generator.state_dict(),
            os.path.join(directory, "generator.pt"),
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(directory, "discriminator.pt"),
        )
        meta = {
            "architecture": "bert-gan",
            "noise_dim": self.noise_dim,
        }
        torch.save(meta, os.path.join(directory, "gan_meta.pt"))
        logger.info("FactGAN (BERT) saved to %s", directory)

    def load(self, directory: str) -> None:
        """Load previously saved state_dicts."""
        gen_path = os.path.join(directory, "generator.pt")
        disc_path = os.path.join(directory, "discriminator.pt")

        if os.path.isfile(gen_path):
            self.generator.load_state_dict(
                torch.load(gen_path, map_location=self.device, weights_only=True)
            )
        if os.path.isfile(disc_path):
            self.discriminator.load_state_dict(
                torch.load(disc_path, map_location=self.device, weights_only=True)
            )
        logger.info("FactGAN (BERT) loaded from %s", directory)


# =========================================================================
# Quick smoke test
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    gan = FactGAN()

    sample_triplets = [
        ("Paris", "is capital of", "France"),
        ("Barack Obama", "was born in", "Hawaii"),
        ("Albert Einstein", "developed", "theory of relativity"),
    ]

    # Discriminate
    scores = gan.discriminate_triplets(sample_triplets)
    print(f"Discriminator scores: {scores.squeeze().tolist()}")

    # One training step
    optimizer_g = torch.optim.AdamW(gan.generator.parameters(), lr=2e-5)
    optimizer_d = torch.optim.AdamW(gan.discriminator.parameters(), lr=2e-5)
    metrics = gan.train_step(sample_triplets, optimizer_g, optimizer_d)
    print(f"Train step metrics: {metrics}")
