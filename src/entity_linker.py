"""Entity linking: map text entities to DBpedia URIs via DBpedia Lookup API."""

import logging
import requests

logger = logging.getLogger(__name__)

DBPEDIA_LOOKUP_URL = "https://lookup.dbpedia.org/api/search"


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
        """Query the DBpedia Lookup API."""
        # Clean entity text: remove determiners
        clean = self._clean_entity(entity_text)

        params = {
            "query": clean,
            "maxResults": 5,
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
            if not docs:
                logger.warning(f"No DBpedia results for: {clean}")
                return None

            # Return the first result's resource URI
            for doc in docs:
                resource = doc.get("resource", [None])
                if isinstance(resource, list):
                    resource = resource[0] if resource else None
                if resource:
                    logger.info(f"Linked '{clean}' -> {resource}")
                    return resource

            return None

        except requests.RequestException as e:
            logger.error(f"DBpedia Lookup failed for '{clean}': {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse DBpedia response for '{clean}': {e}")
            return None

    def _clean_entity(self, text: str) -> str:
        """Remove leading determiners and extra whitespace."""
        stop_words = {"the", "a", "an", "The", "A", "An"}
        words = text.strip().split()
        if words and words[0] in stop_words:
            words = words[1:]
        return " ".join(words)

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

    entities = ["Paris", "France", "Barack Obama", "Hawaii", "Eiffel Tower"]
    for entity in entities:
        uri = linker.link(entity)
        print(f"  {entity} -> {uri}")

    # Test caching - second call should be instant
    print("\n--- Cached call ---")
    uri = linker.link("Paris")
    print(f"  Paris (cached) -> {uri}")
