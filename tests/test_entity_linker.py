"""Tests for the entity linker module."""

import pytest
from unittest.mock import patch, MagicMock
from src.entity_linker import EntityLinker, _BAD_URI_PATTERNS


@pytest.fixture
def linker():
    return EntityLinker()


def _make_doc(label, resource, ref_count=100):
    """Helper to create a mock DBpedia Lookup API doc."""
    return {
        "label": [label],
        "resource": [resource],
        "refCount": [ref_count],
    }


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

    def test_clean_entity_possessives(self, linker):
        """Test that possessives are removed."""
        assert linker._clean_entity("Einstein's theory") == "Einstein theory"
        assert linker._clean_entity("Napoleon's army") == "Napoleon army"

    def test_clean_entity_parentheticals(self, linker):
        """Test that parenthetical expressions are removed."""
        assert linker._clean_entity("Mars (planet)") == "Mars"
        assert linker._clean_entity("Paris (city)") == "Paris"

    def test_clean_entity_trailing_punctuation(self, linker):
        """Test trailing punctuation removal."""
        assert linker._clean_entity("Paris.") == "Paris"
        assert linker._clean_entity("Earth,") == "Earth"

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
            with patch.object(linker, "_fallback_lookup", return_value=None):
                result = linker._lookup("Paris")
                assert result is None

    def test_lookup_handles_empty_response(self, linker):
        """Test handling of empty API response."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"docs": []}

        with patch("src.entity_linker.requests.get", return_value=mock_resp):
            with patch.object(linker, "_fallback_lookup", return_value=None):
                result = linker._lookup("nonexistent")
                assert result is None

    # --- Scoring tests ---

    def test_score_candidate_exact_match(self, linker):
        """Exact label match should score high."""
        doc = _make_doc("Napoleon", "http://dbpedia.org/resource/Napoleon", 5000)
        score = linker._score_candidate(doc, "Napoleon")
        assert score > 0.7

    def test_score_candidate_penalizes_list_uris(self, linker):
        """URIs with List_of_ should get URI simplicity = 0."""
        doc = _make_doc("Mars craters", "http://dbpedia.org/resource/List_of_craters_on_Mars", 100)
        score = linker._score_candidate(doc, "Mars")
        # The URI simplicity component should be 0
        doc_good = _make_doc("Mars", "http://dbpedia.org/resource/Mars", 100)
        score_good = linker._score_candidate(doc_good, "Mars")
        assert score_good > score

    def test_score_candidate_penalizes_disambiguation(self, linker):
        """URIs with _(disambiguation) should get URI simplicity = 0."""
        doc = _make_doc("Earth", "http://dbpedia.org/resource/Earth_(disambiguation)", 50)
        score = linker._score_candidate(doc, "Earth")
        doc_good = _make_doc("Earth", "http://dbpedia.org/resource/Earth", 50)
        score_good = linker._score_candidate(doc_good, "Earth")
        assert score_good > score

    def test_score_candidate_no_resource_returns_zero(self, linker):
        """Doc without resource should score 0."""
        doc = {"label": ["Paris"], "resource": [], "refCount": [100]}
        score = linker._score_candidate(doc, "Paris")
        assert score == 0.0

    # --- Selection tests ---

    def test_select_best_candidate_picks_best(self, linker):
        """Should pick the candidate with the highest score."""
        docs = [
            _make_doc("Napoleonic Wars", "http://dbpedia.org/resource/Napoleonic_Wars", 3000),
            _make_doc("Napoleon", "http://dbpedia.org/resource/Napoleon", 5000),
            _make_doc("Napoleon III", "http://dbpedia.org/resource/Napoleon_III", 2000),
        ]
        best = linker._select_best_candidate(docs, "Napoleon")
        assert best == "http://dbpedia.org/resource/Napoleon"

    def test_select_best_candidate_returns_none_below_threshold(self, linker):
        """Should return None if no candidate scores above 0.3."""
        docs = [
            _make_doc("Completely Unrelated", "http://dbpedia.org/resource/List_of_something", 0),
        ]
        best = linker._select_best_candidate(docs, "xyzabc")
        assert best is None

    def test_select_best_candidate_mars(self, linker):
        """Mars should resolve to Mars, not List_of_craters_on_Mars."""
        docs = [
            _make_doc("List of craters on Mars", "http://dbpedia.org/resource/List_of_craters_on_Mars", 200),
            _make_doc("Mars", "http://dbpedia.org/resource/Mars", 8000),
            _make_doc("Mars (chocolate bar)", "http://dbpedia.org/resource/Mars_(chocolate_bar)", 500),
        ]
        best = linker._select_best_candidate(docs, "Mars")
        assert best == "http://dbpedia.org/resource/Mars"

    # --- Fallback tests ---

    def test_fallback_lookup_constructs_uri(self, linker):
        """Fallback should construct URI and verify it exists."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("src.entity_linker.requests.head", return_value=mock_resp):
            result = linker._fallback_lookup("Napoleon")
            assert result == "http://dbpedia.org/resource/Napoleon"

    def test_fallback_lookup_spaces_to_underscores(self, linker):
        """Fallback should replace spaces with underscores."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("src.entity_linker.requests.head", return_value=mock_resp):
            result = linker._fallback_lookup("Barack Obama")
            assert result == "http://dbpedia.org/resource/Barack_Obama"

    def test_fallback_lookup_returns_none_on_404(self, linker):
        """Fallback should return None for non-existent URIs."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("src.entity_linker.requests.head", return_value=mock_resp):
            result = linker._fallback_lookup("xyznonexistent123")
            assert result is None

    def test_fallback_lookup_returns_none_on_error(self, linker):
        """Fallback should return None on network errors."""
        import requests
        with patch("src.entity_linker.requests.head", side_effect=requests.RequestException):
            result = linker._fallback_lookup("Paris")
            assert result is None

    # --- Integration tests ---

    @pytest.mark.integration
    def test_link_real_entity(self, linker):
        """Integration test: link a real entity to DBpedia."""
        uri = linker.link("Paris")
        assert uri is not None
        assert "dbpedia.org" in uri
        assert "Paris" in uri

    @pytest.mark.integration
    def test_disambiguation_napoleon(self, linker):
        """Integration test: Napoleon should resolve to Napoleon, not Napoleonic_Wars."""
        uri = linker.link("Napoleon")
        assert uri is not None
        uri_name = uri.split("/")[-1]
        assert "List_of_" not in uri_name
        assert "Napoleonic" not in uri_name

    @pytest.mark.integration
    def test_disambiguation_mars(self, linker):
        """Integration test: Mars should resolve to Mars, not List_of_craters."""
        uri = linker.link("Mars")
        assert uri is not None
        uri_name = uri.split("/")[-1]
        assert "List_of_" not in uri_name

    @pytest.mark.integration
    def test_disambiguation_earth(self, linker):
        """Integration test: Earth should resolve to Earth."""
        uri = linker.link("Earth")
        assert uri is not None
        assert "Earth" in uri
        uri_name = uri.split("/")[-1]
        assert "Moon" not in uri_name
