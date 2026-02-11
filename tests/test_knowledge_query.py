"""Tests for the knowledge query module."""

import pytest
from unittest.mock import patch, MagicMock
from src.knowledge_query import KnowledgeQuery


@pytest.fixture
def kq():
    return KnowledgeQuery()


class TestKnowledgeQuery:
    def test_verify_triplet_no_uris(self, kq):
        """Test that missing URIs return not found."""
        result = kq.verify_triplet(None, None)
        assert result["found"] is False
        assert result["method"] == "none"

    def test_verify_triplet_one_uri_missing(self, kq):
        """Test with one URI missing."""
        result = kq.verify_triplet("http://dbpedia.org/resource/Paris", None)
        assert result["found"] is False

    def test_sparql_check_relation_handles_error(self, kq):
        """Test SPARQL error handling."""
        with patch.object(kq.sparql, "query", side_effect=Exception("Connection error")):
            result = kq.sparql_check_relation(
                "http://dbpedia.org/resource/Paris",
                "http://dbpedia.org/resource/France",
            )
            assert result == []

    def test_json_get_entity_data_caching(self, kq):
        """Test that JSON data is cached."""
        mock_data = {"http://dbpedia.org/resource/Paris": {"prop": [{"value": "val"}]}}
        kq._json_cache["http://dbpedia.org/resource/Paris"] = mock_data

        result = kq.json_get_entity_data("http://dbpedia.org/resource/Paris")
        assert result == mock_data

    def test_json_check_relation(self, kq):
        """Test JSON relation check with mock data."""
        subject_uri = "http://dbpedia.org/resource/Paris"
        object_uri = "http://dbpedia.org/resource/France"
        mock_data = {
            subject_uri: {
                "http://dbpedia.org/ontology/country": [
                    {"value": object_uri, "type": "uri"}
                ]
            }
        }
        kq._json_cache[subject_uri] = mock_data

        result = kq.json_check_relation(subject_uri, object_uri)
        assert "http://dbpedia.org/ontology/country" in result

    def test_json_get_property_values(self, kq):
        """Test getting property values from JSON."""
        subject_uri = "http://dbpedia.org/resource/Paris"
        pred = "http://dbpedia.org/ontology/country"
        mock_data = {
            subject_uri: {
                pred: [{"value": "http://dbpedia.org/resource/France"}]
            }
        }
        kq._json_cache[subject_uri] = mock_data

        values = kq.json_get_property_values(subject_uri, pred)
        assert "http://dbpedia.org/resource/France" in values

    def test_sparql_ask_handles_error(self, kq):
        """Test SPARQL ASK error handling."""
        with patch.object(kq.sparql, "query", side_effect=Exception("Error")):
            result = kq.sparql_ask("http://a", "http://b", "http://c")
            assert result is False

    def test_json_get_entity_data_handles_error(self, kq):
        """Test JSON fetch error handling."""
        import requests
        with patch("src.knowledge_query.requests.get", side_effect=requests.RequestException("timeout")):
            kq._json_cache.clear()
            result = kq.json_get_entity_data("http://dbpedia.org/resource/NonExistent")
            assert result == {}

    @pytest.mark.integration
    def test_sparql_real_query(self, kq):
        """Integration test: verify Paris-France relation via SPARQL."""
        preds = kq.sparql_check_relation(
            "http://dbpedia.org/resource/Paris",
            "http://dbpedia.org/resource/France",
        )
        assert len(preds) > 0

    @pytest.mark.integration
    def test_verify_triplet_real(self, kq):
        """Integration test: verify full triplet."""
        result = kq.verify_triplet(
            "http://dbpedia.org/resource/Paris",
            "http://dbpedia.org/resource/France",
        )
        assert result["found"] is True
