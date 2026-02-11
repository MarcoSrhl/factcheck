"""Tests for the triplet extractor module."""

import pytest
from src.triplet_extractor import TripletExtractor


@pytest.fixture(scope="module")
def extractor():
    return TripletExtractor()


class TestTripletExtractor:
    def test_copular_sentence(self, extractor):
        """Test extraction from copular sentences like 'X is Y'."""
        triplets = extractor.extract("Paris is the capital of France")
        assert len(triplets) >= 1
        s, p, o = triplets[0]
        assert "Paris" in s
        assert "capital" in o
        assert "France" in o

    def test_passive_verbal(self, extractor):
        """Test extraction from passive verbal sentences."""
        triplets = extractor.extract("Barack Obama was born in Hawaii")
        assert len(triplets) >= 1
        s, _, o = triplets[0]
        assert "Obama" in s
        assert "Hawaii" in o

    def test_active_verbal(self, extractor):
        """Test extraction from active verbal sentences."""
        triplets = extractor.extract("Albert Einstein developed the theory of relativity")
        assert len(triplets) >= 1
        s, p, o = triplets[0]
        assert "Einstein" in s
        assert "develop" in p
        assert "theory" in o

    def test_located_in(self, extractor):
        """Test 'located in' type sentences."""
        triplets = extractor.extract("The Eiffel Tower is located in Paris")
        assert len(triplets) >= 1
        s, _, o = triplets[0]
        assert "Eiffel Tower" in s or "Tower" in s
        assert "Paris" in o

    def test_empty_string(self, extractor):
        """Test with empty string returns no triplets."""
        triplets = extractor.extract("")
        assert triplets == []

    def test_no_triplet_sentence(self, extractor):
        """Test with a sentence that may not yield clear triplets."""
        triplets = extractor.extract("Hello world")
        # Should return empty or some fallback
        assert isinstance(triplets, list)

    def test_return_type(self, extractor):
        """Test that return type is correct."""
        triplets = extractor.extract("Water boils at 100 degrees")
        assert isinstance(triplets, list)
        for t in triplets:
            assert isinstance(t, tuple)
            assert len(t) == 3

    def test_multiple_facts(self, extractor):
        """Test extraction from a simple claim."""
        triplets = extractor.extract("Tokyo is the capital of Japan")
        assert len(triplets) >= 1
        s, _, o = triplets[0]
        assert "Tokyo" in s
        assert "Japan" in o
