"""Tests for the main fact-checking pipeline."""

import pytest
from unittest.mock import patch, MagicMock
from src.fact_checker import FactChecker, format_result


@pytest.fixture
def checker():
    """Create a FactChecker with neural model disabled for fast unit tests."""
    return FactChecker(use_neural=False)


class TestFactChecker:
    def test_check_returns_dict(self, checker):
        """Test that check returns a proper result dict."""
        result = checker.check("Paris is the capital of France")
        assert isinstance(result, dict)
        assert "claim" in result
        assert "triplets" in result
        assert "entities" in result
        assert "kb_evidence" in result
        assert "verdict" in result
        assert "confidence" in result

    def test_check_claim_preserved(self, checker):
        """Test that the original claim is in the result."""
        claim = "Paris is the capital of France"
        result = checker.check(claim)
        assert result["claim"] == claim

    def test_triplets_extracted(self, checker):
        """Test that triplets are extracted."""
        result = checker.check("Paris is the capital of France")
        assert len(result["triplets"]) >= 1

    def test_verdict_values(self, checker):
        """Test that verdict is one of the expected values."""
        result = checker.check("Paris is the capital of France")
        assert result["verdict"] in ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")

    def test_confidence_range(self, checker):
        """Test that confidence is between 0 and 1."""
        result = checker.check("Paris is the capital of France")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_check_batch(self, checker):
        """Test batch checking."""
        claims = ["Paris is the capital of France", "The Earth is flat"]
        results = checker.check_batch(claims)
        assert len(results) == 2
        for r in results:
            assert "verdict" in r

    def test_format_result(self, checker):
        """Test result formatting."""
        result = {
            "claim": "Test claim",
            "triplets": [("A", "is", "B")],
            "entities": {"A": "http://dbpedia.org/resource/A"},
            "kb_evidence": [{"triplet": ("A", "is", "B"), "found": True, "predicates": ["http://p"]}],
            "neural_prediction": None,
            "verdict": "SUPPORTED",
            "confidence": 0.8,
        }
        formatted = format_result(result)
        assert "Test claim" in formatted
        assert "SUPPORTED" in formatted

    def test_combine_verdicts_kb_only(self, checker):
        """Test verdict combining with KB evidence only (no neural)."""
        kb_results = [{"found": True, "predicates": ["http://p"], "method": "sparql"}]
        verdict, conf = checker._combine_verdicts(kb_results, None)
        assert verdict == "SUPPORTED"

    def test_combine_verdicts_kb_not_found(self, checker):
        """Test verdict when KB finds nothing."""
        kb_results = [{"found": False, "predicates": [], "method": "none"}]
        verdict, conf = checker._combine_verdicts(kb_results, None)
        assert verdict == "NOT ENOUGH INFO"

    def test_combine_verdicts_with_neural(self, checker):
        """Test verdict combining with neural predictions."""
        kb_results = [{"found": True, "predicates": ["http://p"], "method": "sparql"}]
        neural = {"label": "SUPPORTED", "confidence": 0.9, "probabilities": {}}
        verdict, conf = checker._combine_verdicts(kb_results, neural)
        assert verdict == "SUPPORTED"
        assert conf > 0.5

    @pytest.mark.integration
    def test_full_pipeline(self, checker):
        """Integration test: run full pipeline on a real claim."""
        result = checker.check("Paris is the capital of France")
        assert result["verdict"] in ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")
        assert len(result["entities"]) > 0
