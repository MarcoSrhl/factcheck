"""Tests for the explainability module."""

import pytest
from unittest.mock import patch, MagicMock

from src.explainer import (
    KBReasoningExplainer,
    ConfidenceDecomposer,
    AttentionAnalyzer,
    FactExplainer,
    _predicate_to_human,
    PREDICATE_EXPLANATIONS,
)


# ---------------------------------------------------------------------------
# Helpers: reusable fake fact-check results
# ---------------------------------------------------------------------------

def _make_result(
    claim="Paris is the capital of France",
    triplets=None,
    entities=None,
    kb_evidence=None,
    neural_prediction=None,
    gan_score=None,
    verdict="SUPPORTED",
    confidence=0.85,
):
    if triplets is None:
        triplets = [("Paris", "is the capital of", "France")]
    if entities is None:
        entities = {
            "Paris": "http://dbpedia.org/resource/Paris",
            "France": "http://dbpedia.org/resource/France",
        }
    if kb_evidence is None:
        kb_evidence = [{
            "triplet": ("Paris", "is the capital of", "France"),
            "subject_uri": "http://dbpedia.org/resource/Paris",
            "object_uri": "http://dbpedia.org/resource/France",
            "found": True,
            "predicates": [
                "http://dbpedia.org/ontology/capital",
                "http://dbpedia.org/ontology/country",
            ],
            "method": "sparql",
        }]
    return {
        "claim": claim,
        "triplets": triplets,
        "entities": entities,
        "kb_evidence": kb_evidence,
        "neural_prediction": neural_prediction,
        "gan_score": gan_score,
        "verdict": verdict,
        "confidence": confidence,
    }


def _make_not_found_result():
    return _make_result(
        claim="The Earth is flat",
        triplets=[("Earth", "is", "flat")],
        entities={"Earth": "http://dbpedia.org/resource/Earth"},
        kb_evidence=[{
            "triplet": ("Earth", "is", "flat"),
            "subject_uri": "http://dbpedia.org/resource/Earth",
            "object_uri": None,
            "found": False,
            "predicates": [],
            "method": "none",
        }],
        verdict="NOT ENOUGH INFO",
        confidence=0.3,
    )


# =========================================================================
# Tests: _predicate_to_human
# =========================================================================

class TestPredicateToHuman:
    def test_known_predicate(self):
        result = _predicate_to_human("http://dbpedia.org/ontology/capital")
        assert result == "is the capital of"

    def test_unknown_predicate_camelcase(self):
        result = _predicate_to_human("http://example.org/birthPlace")
        assert "birth" in result.lower()
        assert "place" in result.lower()

    def test_unknown_predicate_simple(self):
        result = _predicate_to_human("http://example.org/name")
        assert result == "name"


# =========================================================================
# Tests: KBReasoningExplainer
# =========================================================================

class TestKBReasoningExplainer:
    def setup_method(self):
        self.explainer = KBReasoningExplainer()

    def test_explain_returns_steps_and_summary(self):
        result = _make_result()
        explanation = self.explainer.explain(result)
        assert "steps" in explanation
        assert "summary" in explanation
        assert len(explanation["steps"]) > 0

    def test_explain_found_relation(self):
        result = _make_result()
        explanation = self.explainer.explain(result)
        # Should mention that DBpedia confirms a relation
        steps_text = " ".join(explanation["steps"])
        assert "DBpedia confirms" in steps_text

    def test_explain_not_found(self):
        result = _make_not_found_result()
        explanation = self.explainer.explain(result)
        steps_text = " ".join(explanation["steps"])
        assert "No direct relation found" in steps_text or "could not be linked" in steps_text

    def test_explain_no_triplets(self):
        result = _make_result(triplets=[], kb_evidence=[])
        explanation = self.explainer.explain(result)
        assert "Could not extract" in explanation["steps"][0]

    def test_summary_all_found(self):
        result = _make_result()
        explanation = self.explainer.explain(result)
        assert "confirmed" in explanation["summary"].lower()

    def test_summary_none_found(self):
        result = _make_not_found_result()
        explanation = self.explainer.explain(result)
        assert "none" in explanation["summary"].lower() or "could not" in explanation["summary"].lower()


# =========================================================================
# Tests: ConfidenceDecomposer
# =========================================================================

class TestConfidenceDecomposer:
    def setup_method(self):
        self.decomposer = ConfidenceDecomposer()

    def test_decompose_returns_components(self):
        result = _make_result()
        decomposition = self.decomposer.decompose(result)
        assert "components" in decomposition
        assert "combined_verdict" in decomposition
        assert "combined_confidence" in decomposition
        assert "formatted" in decomposition

    def test_three_components(self):
        result = _make_result()
        decomposition = self.decomposer.decompose(result)
        names = [c["name"] for c in decomposition["components"]]
        assert "Knowledge Base" in names
        assert "Neural Model" in names
        assert "GAN Discriminator" in names

    def test_kb_found(self):
        result = _make_result()
        decomposition = self.decomposer.decompose(result)
        kb = next(c for c in decomposition["components"] if c["name"] == "Knowledge Base")
        assert "FOUND" in kb["status"]

    def test_kb_not_found(self):
        result = _make_not_found_result()
        decomposition = self.decomposer.decompose(result)
        kb = next(c for c in decomposition["components"] if c["name"] == "Knowledge Base")
        assert "NOT FOUND" in kb["status"]

    def test_neural_disabled(self):
        result = _make_result(neural_prediction=None)
        decomposition = self.decomposer.decompose(result)
        neural = next(c for c in decomposition["components"] if c["name"] == "Neural Model")
        assert neural["status"] == "DISABLED"

    def test_neural_enabled(self):
        result = _make_result(
            neural_prediction={"label": "SUPPORTED", "confidence": 0.9, "probabilities": {}}
        )
        decomposition = self.decomposer.decompose(result)
        neural = next(c for c in decomposition["components"] if c["name"] == "Neural Model")
        assert "SUPPORTED" in neural["status"]

    def test_gan_disabled(self):
        result = _make_result(gan_score=None)
        decomposition = self.decomposer.decompose(result)
        gan = next(c for c in decomposition["components"] if c["name"] == "GAN Discriminator")
        assert gan["status"] == "DISABLED"

    def test_gan_high_score(self):
        result = _make_result(gan_score=0.85)
        decomposition = self.decomposer.decompose(result)
        gan = next(c for c in decomposition["components"] if c["name"] == "GAN Discriminator")
        assert "real" in gan["status"]

    def test_gan_low_score(self):
        result = _make_result(gan_score=0.2)
        decomposition = self.decomposer.decompose(result)
        gan = next(c for c in decomposition["components"] if c["name"] == "GAN Discriminator")
        assert "fabricated" in gan["status"]

    def test_formatted_output(self):
        result = _make_result()
        decomposition = self.decomposer.decompose(result)
        assert "Confidence Breakdown" in decomposition["formatted"]


# =========================================================================
# Tests: AttentionAnalyzer
# =========================================================================

class TestAttentionAnalyzer:
    def setup_method(self):
        self.analyzer = AttentionAnalyzer()

    def test_no_model_returns_empty(self):
        result = self.analyzer.analyze("test claim", "test evidence", None, None)
        assert result["top_tokens"] == []
        assert "unavailable" in result["attention_summary"].lower()

    def test_returns_correct_keys(self):
        result = self.analyzer.analyze("test", "test", None, None)
        assert "top_tokens" in result
        assert "attention_summary" in result
        assert "all_token_scores" in result


# =========================================================================
# Tests: FactExplainer (master class)
# =========================================================================

class TestFactExplainer:
    def test_init_without_t5(self):
        explainer = FactExplainer(use_t5=False, use_attention=False)
        assert explainer.t5_generator is None
        assert explainer.attention_analyzer is None
        assert explainer.kb_reasoner is not None
        assert explainer.confidence_decomposer is not None

    def test_explain_without_t5(self):
        explainer = FactExplainer(use_t5=False, use_attention=False)
        result = _make_result()
        explanation = explainer.explain(result)
        assert "reasoning_chain" in explanation
        assert "confidence_breakdown" in explanation
        assert "summary" in explanation
        assert explanation["natural_explanation"] is None
        assert explanation["attention_analysis"] is None

    def test_explain_has_reasoning_steps(self):
        explainer = FactExplainer(use_t5=False, use_attention=False)
        result = _make_result()
        explanation = explainer.explain(result)
        steps = explanation["reasoning_chain"]["steps"]
        assert len(steps) > 0

    def test_format_explanation(self):
        explainer = FactExplainer(use_t5=False, use_attention=False)
        result = _make_result()
        explanation = explainer.explain(result)
        formatted = explainer.format_explanation(explanation)
        assert "FACT-CHECK EXPLANATION" in formatted
        assert "Knowledge Base Reasoning" in formatted

    def test_explain_summary_contains_verdict(self):
        explainer = FactExplainer(use_t5=False, use_attention=False)
        result = _make_result(verdict="SUPPORTED")
        explanation = explainer.explain(result)
        assert "SUPPORTED" in explanation["summary"]
