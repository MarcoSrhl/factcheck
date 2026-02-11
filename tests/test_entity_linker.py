"""Tests for the entity linker module."""

import pytest
from unittest.mock import patch, MagicMock
from src.entity_linker import EntityLinker


@pytest.fixture
def linker():
    return EntityLinker()


class TestEntityLinker:
    def test_clean_entity_removes_determiners(self, linker):
        """Test that determiners are removed from entity text."""
        assert linker._clean_entity("the Eiffel Tower") == "Eiffel Tower"
        assert linker._clean_entity("a dog") == "dog"
        assert linker._clean_entity("an apple") == "apple"
        assert linker._clean_entity("Paris") == "Paris"

    def test_clean_entity_strips_whitespace(self, linker):
        """Test whitespace stripping."""
        assert linker._clean_entity("  Paris  ") == "Paris"

    def test_caching(self, linker):
        """Test that results are cached."""
        with patch.object(linker, "_lookup", return_value="http://dbpedia.org/resource/Paris") as mock:
            linker.link("Paris")
            linker.link("Paris")
            mock.assert_called_once()

    def test_link_returns_none_for_unknown(self, linker):
        """Test that unknown entities return None."""
        with patch.object(linker, "_lookup", return_value=None):
            result = linker.link("xyznonexistent123")
            assert result is None

    def test_link_triplet(self, linker):
        """Test linking a full triplet."""
        with patch.object(linker, "link") as mock_link:
            mock_link.side_effect = lambda x: {
                "Paris": "http://dbpedia.org/resource/Paris",
                "the capital of France": None,
            }.get(x)

            subj_uri, pred, obj_uri = linker.link_triplet(
                ("Paris", "is", "the capital of France")
            )
            assert subj_uri == "http://dbpedia.org/resource/Paris"
            assert pred == "is"
            assert obj_uri is None

    def test_lookup_handles_request_error(self, linker):
        """Test that request errors are handled gracefully."""
        import requests
        with patch("src.entity_linker.requests.get", side_effect=requests.RequestException("timeout")):
            result = linker._lookup("Paris")
            assert result is None

    def test_lookup_handles_empty_response(self, linker):
        """Test handling of empty API response."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"docs": []}

        with patch("src.entity_linker.requests.get", return_value=mock_resp):
            result = linker._lookup("nonexistent")
            assert result is None

    @pytest.mark.integration
    def test_link_real_entity(self, linker):
        """Integration test: link a real entity to DBpedia."""
        uri = linker.link("Paris")
        assert uri is not None
        assert "dbpedia.org" in uri
        assert "Paris" in uri
