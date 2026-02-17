"""Unit tests for the BERT-based GAN architecture (src/gan_model.py)."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from src.gan_model import (
    BERT_HIDDEN_SIZE,
    BERTDiscriminator,
    BERTGenerator,
    FactGAN,
    _format_triplet,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TRIPLETS: list[tuple[str, str, str]] = [
    ("Paris", "is capital of", "France"),
    ("Barack Obama", "was born in", "Hawaii"),
]

DEVICE = "cpu"


@pytest.fixture(scope="module")
def generator() -> BERTGenerator:
    g = BERTGenerator(device=torch.device(DEVICE))
    return g


@pytest.fixture(scope="module")
def discriminator() -> BERTDiscriminator:
    d = BERTDiscriminator(device=torch.device(DEVICE))
    return d


@pytest.fixture(scope="module")
def fact_gan() -> FactGAN:
    return FactGAN(device=DEVICE)


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


class TestBERTGenerator:
    def test_generator_output_shape(self, generator: BERTGenerator) -> None:
        """Generator returns soft embeddings of shape (batch, seq_len, hidden_size)."""
        soft_embeds, attn_mask, token_ids, mask_pos, logits, target_ids = generator(
            SAMPLE_TRIPLETS, temperature=1.0
        )

        batch_size = len(SAMPLE_TRIPLETS)
        seq_len = soft_embeds.shape[1]
        assert soft_embeds.dim() == 3
        assert soft_embeds.shape[0] == batch_size
        assert soft_embeds.shape[2] == BERT_HIDDEN_SIZE
        # attention mask matches
        assert attn_mask.shape[0] == batch_size
        assert attn_mask.shape[1] == seq_len
        # token_ids and mask_pos have correct shapes
        assert token_ids.shape == (batch_size, seq_len)
        assert mask_pos.shape == (batch_size, seq_len)
        # logits and target_ids shapes
        assert logits.shape[0] == batch_size
        assert logits.shape[1] == seq_len
        assert logits.dim() == 3  # (batch, seq_len, vocab_size)
        assert target_ids.shape == (batch_size, seq_len)

    def test_generator_output_requires_grad(self, generator: BERTGenerator) -> None:
        """Generator soft embeddings must be differentiable (requires_grad)."""
        generator.train()
        soft_embeds, _, _, _, _, _ = generator(SAMPLE_TRIPLETS, temperature=1.0)
        assert soft_embeds.requires_grad

    def test_gumbel_softmax_differentiable(self, generator: BERTGenerator) -> None:
        """Gumbel-Softmax output preserves gradient flow through the Generator."""
        generator.train()
        soft_embeds, _, _, _, _, _ = generator(SAMPLE_TRIPLETS, temperature=0.5)

        # Check we can compute gradients back to generator parameters
        loss = soft_embeds.sum()
        loss.backward()

        # At least the noise_proj layer should have gradients
        assert generator.noise_proj.weight.grad is not None
        generator.zero_grad()


# ---------------------------------------------------------------------------
# Discriminator tests
# ---------------------------------------------------------------------------


class TestBERTDiscriminator:
    def test_forward_text(self, discriminator: BERTDiscriminator) -> None:
        """D.forward_text accepts tokenised input and returns scores in [0, 1]."""
        texts = [_format_triplet(s, p, o) for s, p, o in SAMPLE_TRIPLETS]
        enc = discriminator.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        discriminator.eval()
        with torch.no_grad():
            scores = discriminator.forward_text(input_ids, attention_mask)

        assert scores.shape == (len(SAMPLE_TRIPLETS), 1)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_forward_soft(
        self, discriminator: BERTDiscriminator, generator: BERTGenerator
    ) -> None:
        """D.forward_soft accepts soft embeddings and returns scores in [0, 1]."""
        generator.eval()
        with torch.no_grad():
            soft_embeds, attn_mask, _, _, _, _ = generator(SAMPLE_TRIPLETS, temperature=1.0)

        discriminator.eval()
        with torch.no_grad():
            scores = discriminator.forward_soft(soft_embeds, attn_mask)

        assert scores.shape == (len(SAMPLE_TRIPLETS), 1)
        assert (scores >= 0).all() and (scores <= 1).all()


# ---------------------------------------------------------------------------
# FactGAN wrapper tests
# ---------------------------------------------------------------------------


class TestFactGAN:
    def test_discriminate_triplets_interface(self, fact_gan: FactGAN) -> None:
        """discriminate_triplets returns tensor of correct shape."""
        scores = fact_gan.discriminate_triplets(SAMPLE_TRIPLETS)

        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (len(SAMPLE_TRIPLETS), 1)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_train_step_runs(self, fact_gan: FactGAN) -> None:
        """A complete train_step (G + D) runs without error."""
        optimizer_g = torch.optim.AdamW(fact_gan.generator.parameters(), lr=2e-5)
        optimizer_d = torch.optim.AdamW(fact_gan.discriminator.parameters(), lr=2e-5)

        metrics = fact_gan.train_step(
            real_triplets=SAMPLE_TRIPLETS,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            temperature=1.0,
        )

        assert "d_loss" in metrics
        assert "g_loss" in metrics
        assert "d_real_score" in metrics
        assert "d_fake_score" in metrics
        assert "mlm_loss" in metrics
        # Losses should be finite
        assert all(
            isinstance(v, float) and not torch.tensor(v).isnan()
            for v in metrics.values()
        )

    def test_mlm_loss_computable(self, fact_gan: FactGAN) -> None:
        """MLM cross-entropy loss can be computed from Generator outputs."""
        import torch.nn.functional as F

        fact_gan.generator.train()
        _, _, _, mask_pos, logits, target_ids = fact_gan.generator(
            SAMPLE_TRIPLETS, temperature=1.0
        )

        assert mask_pos.any(), "At least some positions should be masked"
        mlm_logits = logits[mask_pos]       # (num_masked, vocab_size)
        mlm_targets = target_ids[mask_pos]  # (num_masked,)
        loss = F.cross_entropy(mlm_logits, mlm_targets)
        assert loss.isfinite()
        assert loss.requires_grad

    def test_decode_generated(self, fact_gan: FactGAN) -> None:
        """Generator can decode its output to readable text."""
        generated = fact_gan.generator.decode_generated(SAMPLE_TRIPLETS, temperature=1.0)
        assert len(generated) == len(SAMPLE_TRIPLETS)
        for text in generated:
            assert isinstance(text, str)
            assert len(text) > 0

    def test_save_load_roundtrip(self, fact_gan: FactGAN) -> None:
        """Save then reload produces identical discriminator scores."""
        scores_before = fact_gan.discriminate_triplets(SAMPLE_TRIPLETS)

        with tempfile.TemporaryDirectory() as tmpdir:
            fact_gan.save(tmpdir)

            gan2 = FactGAN(device=DEVICE)
            gan2.load(tmpdir)

        scores_after = gan2.discriminate_triplets(SAMPLE_TRIPLETS)
        torch.testing.assert_close(scores_before, scores_after)
