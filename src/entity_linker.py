"""Entity linking: map text entities to DBpedia URIs via DBpedia Lookup API."""

import logging
import re
import math
from difflib import SequenceMatcher

import requests

logger = logging.getLogger(__name__)

DBPEDIA_LOOKUP_URL = "https://lookup.dbpedia.org/api/search"

# URI patterns that indicate disambiguation / aggregate pages (not real entities)
_BAD_URI_PATTERNS = ["List_of_", "Category:", "_(disambiguation)", "_in_", "Template:"]


class EntityLinker:
    def __init__(self):
        self._cache: dict[str, str | None] = {}

    def link(self, entity_text: str) -> str | None:
        """Map an entity string to a DBpedia URI.

        Returns the DBpedia resource URI or None if not found.
        """
        if entity_text in self._cache:
            return self._cache[entity_text]

        uri = self._lookup(entity_text)
        self._cache[entity_text] = uri
        return uri

    def _lookup(self, entity_text: str) -> str | None:
        """Query the DBpedia Lookup API with disambiguation scoring."""
        clean = self._clean_entity(entity_text)

        params = {
            "query": clean,
            "maxResults": 10,
            "format": "json",
        }
        headers = {"Accept": "application/json"}

        try:
            resp = requests.get(
                DBPEDIA_LOOKUP_URL,
                params=params,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            docs = data.get("docs", [])
            if docs:
                best = self._select_best_candidate(docs, clean)
                if best:
                    logger.info(f"Linked '{clean}' -> {best}")
                    return best

            # Fallback: try constructing the URI directly
            fallback = self._fallback_lookup(clean)
            if fallback:
                logger.info(f"Linked '{clean}' -> {fallback} (fallback)")
                return fallback

            logger.warning(f"No DBpedia results for: {clean}")
            return None

        except requests.RequestException as e:
            logger.error(f"DBpedia Lookup failed for '{clean}': {e}")
            return self._fallback_lookup(clean)
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse DBpedia response for '{clean}': {e}")
            return self._fallback_lookup(clean)

    def _score_candidate(self, doc: dict, query: str) -> float:
        """Score a candidate document for disambiguation.

        Weighted scoring:
          - Label similarity  (0.40)
          - Exact match bonus (0.20)
          - URI simplicity    (0.25)
          - Popularity        (0.15)
        """
        # Extract label from doc
        label = doc.get("label", [""])[0] if isinstance(doc.get("label"), list) else doc.get("label", "")
        resource = doc.get("resource", [None])
        if isinstance(resource, list):
            resource = resource[0] if resource else None
        if not resource:
            return 0.0

        query_lower = query.lower().strip()
        label_lower = label.lower().strip()

        # 1. Label similarity (0.40)
        similarity = SequenceMatcher(None, query_lower, label_lower).ratio()

        # 2. Exact match bonus (0.20)
        exact_match = 1.0 if query_lower == label_lower else 0.0

        # 3. URI simplicity (0.25)
        uri_name = resource.split("/")[-1]
        # Penalize bad URI patterns
        if any(pat in uri_name for pat in _BAD_URI_PATTERNS):
            uri_simplicity = 0.0
        else:
            # Favor simple URIs (fewer underscores = simpler)
            n_underscores = uri_name.count("_")
            uri_simplicity = 1.0 / (1.0 + n_underscores * 0.3)

        # 4. Popularity via refCount (0.15)
        ref_count = doc.get("refCount", [0])
        if isinstance(ref_count, list):
            ref_count = ref_count[0] if ref_count else 0
        try:
            ref_count = int(ref_count)
        except (ValueError, TypeError):
            ref_count = 0
        popularity = math.log(1 + ref_count) / 20.0  # normalize roughly to 0-1
        popularity = min(popularity, 1.0)

        score = (
            0.40 * similarity
            + 0.20 * exact_match
            + 0.25 * uri_simplicity
            + 0.15 * popularity
        )
        return score

    def _select_best_candidate(self, docs: list[dict], query: str) -> str | None:
        """Pick the best candidate from API results using scoring."""
        best_score = 0.0
        best_uri = None

        for doc in docs:
            score = self._score_candidate(doc, query)
            resource = doc.get("resource", [None])
            if isinstance(resource, list):
                resource = resource[0] if resource else None
            if resource and score > best_score:
                best_score = score
                best_uri = resource

        if best_score >= 0.3:
            return best_uri
        return None

    def _fallback_lookup(self, entity_text: str) -> str | None:
        """Construct a DBpedia URI directly and verify it exists."""
        uri = f"http://dbpedia.org/resource/{entity_text.replace(' ', '_')}"
        try:
            resp = requests.head(uri, timeout=5, allow_redirects=True)
            if resp.status_code == 200:
                return uri
        except requests.RequestException:
            pass
        return None

    def _clean_entity(self, text: str) -> str:
        """Remove leading determiners, possessives, parentheticals, extra whitespace."""
        text = text.strip()
        # Remove parenthetical expressions
        text = re.sub(r"\s*\([^)]*\)", "", text)
        # Remove possessives
        text = re.sub(r"'s\b", "", text)
        # Remove trailing punctuation
        text = re.sub(r"[.,;:!?]+$", "", text)
        # Remove leading determiners
        stop_words = {"the", "a", "an"}
        words = text.split()
        if words and words[0].lower() in stop_words:
            words = words[1:]
        return " ".join(words).strip()

    def link_triplet(
        self, triplet: tuple[str, str, str]
    ) -> tuple[str | None, str, str | None]:
        """Link subject and object of a triplet to DBpedia URIs.

        Returns (subject_uri, predicate, object_uri).
        """
        subject, predicate, obj = triplet
        subject_uri = self.link(subject)
        object_uri = self.link(obj)
        return (subject_uri, predicate, object_uri)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    linker = EntityLinker()

    entities = ["Paris", "France", "Barack Obama", "Hawaii", "Eiffel Tower",
                "Napoleon", "Earth", "Mars", "Albert Einstein"]
    for entity in entities:
        uri = linker.link(entity)
        print(f"  {entity} -> {uri}")

    # Test caching - second call should be instant
    print("\n--- Cached call ---")
    uri = linker.link("Paris")
    print(f"  Paris (cached) -> {uri}")
