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
        """Query the DBpedia Lookup API with disambiguation scoring.

        Multi-phase approach: try with parentheticals first, then without,
        then extract proper noun from descriptive phrases.
        """
        # Phase 1: try with parentheticals preserved
        clean_with_parens = self._clean_entity(entity_text, keep_parens=True)
        result = self._try_api_lookup(clean_with_parens)
        if result:
            return result

        # Phase 2: try without parentheticals
        clean_no_parens = self._clean_entity(entity_text, keep_parens=False)
        if clean_no_parens != clean_with_parens:
            result = self._try_api_lookup(clean_no_parens)
            if result:
                return result

        # Phase 3: extract proper noun from descriptive phrases
        # "capital of Japan" → try "Japan", "birthplace of X" → try "X"
        extracted = self._extract_entity_from_phrase(clean_no_parens)
        if extracted and extracted != clean_no_parens:
            result = self._try_api_lookup(extracted)
            if result:
                logger.info(f"Linked '{entity_text}' -> {result} (phrase extraction)")
                return result

        # Phase 4: Wikipedia API search → DBpedia URI
        for clean in dict.fromkeys([clean_with_parens, clean_no_parens]):
            result = self._wikipedia_lookup(clean)
            if result:
                logger.info(f"Linked '{entity_text}' -> {result} (wikipedia)")
                return result

        # Fallback: construct URI directly
        for clean in dict.fromkeys([clean_with_parens, clean_no_parens]):
            fallback = self._fallback_lookup(clean)
            if fallback:
                logger.info(f"Linked '{clean}' -> {fallback} (fallback)")
                return fallback

        logger.warning(f"No DBpedia results for: {entity_text}")
        return None

    def _extract_entity_from_phrase(self, text: str) -> str | None:
        """Extract the likely entity from descriptive phrases.

        Handles patterns like:
          - "capital of Japan" → "Japan"
          - "birthplace of Einstein" → "Einstein"
          - "author of Hamlet" → "Hamlet"
        """
        if " of " not in text:
            return None
        parts = text.split(" of ", 1)
        after_of = parts[1].strip()
        # Only extract if the part after "of" looks like an entity
        # (starts with uppercase or is multi-word)
        if after_of and (after_of[0].isupper() or len(after_of.split()) > 1):
            return after_of
        return None

    def _try_api_lookup(self, clean: str) -> str | None:
        """Try DBpedia Lookup API for a cleaned entity string."""
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
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            docs = data.get("docs", [])
            if docs:
                best = self._select_best_candidate(docs, clean)
                if best:
                    logger.info(f"Linked '{clean}' -> {best}")
                    return best
            return None

        except requests.RequestException as e:
            logger.error(f"DBpedia Lookup failed for '{clean}': {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse DBpedia response for '{clean}': {e}")
            return None

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

        # Hard reject: if label is very different from query, don't consider it
        if similarity < 0.4:
            return 0.0

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

        # 5. Word overlap bonus: reward candidates sharing key words with query
        query_words = set(query_lower.split())
        label_words_set = set(label_lower.split())
        if query_words and label_words_set:
            overlap = len(query_words & label_words_set) / max(len(query_words), len(label_words_set))
        else:
            overlap = 0.0

        score = (
            0.30 * similarity
            + 0.20 * exact_match
            + 0.15 * uri_simplicity
            + 0.10 * popularity
            + 0.25 * overlap
        )

        # Penalize when a short single-word query matches a multi-word label
        # (e.g., "capital" matching "Capitol Records")
        label_words = label_lower.split()
        if len(query_words) == 1 and len(label_words) > 2 and exact_match == 0.0:
            score *= 0.5

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

        if best_score >= 0.55:
            return best_uri
        return None

    def _wikipedia_lookup(self, entity_text: str) -> str | None:
        """Search Wikipedia API and convert the top result to a DBpedia URI."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": entity_text,
            "srlimit": 3,
            "format": "json",
        }
        headers = {
            "User-Agent": "FactChecker/1.0 (academic project; entity linking)",
        }
        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
            if not results:
                return None

            # Pick the best title match — require reasonable similarity
            query_lower = entity_text.lower()
            for hit in results:
                title = hit.get("title", "")
                title_lower = title.lower()
                sim = SequenceMatcher(None, query_lower, title_lower).ratio()
                # Accept if title closely matches the query
                if title_lower == query_lower or sim >= 0.7:
                    return f"http://dbpedia.org/resource/{title.replace(' ', '_')}"

            # Don't use a non-matching first result — better to return None
            return None

        except requests.RequestException as e:
            logger.error(f"Wikipedia API failed for '{entity_text}': {e}")
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

    def _clean_entity(self, text: str, keep_parens: bool = False) -> str:
        """Remove leading determiners, possessives, parentheticals, extra whitespace."""
        text = text.strip()
        if not keep_parens:
            # Remove parenthetical expressions
            text = re.sub(r"\s*\([^)]*\)", "", text)
        # Remove possessives
        text = re.sub(r"'s\b", "", text)
        # Remove trailing punctuation
        text = re.sub(r"[.,;:!?]+$", "", text)
        # Remove leading determiners
        stop_words = {"the", "a", "an", "its"}
        words = text.split()
        if words and words[0].lower() in stop_words:
            words = words[1:]
        # Strip trailing generic words: "a Country music work" → "Country music"
        trailing_generic = {"work", "genre", "season", "nationality", "type"}
        while words and words[-1].lower() in trailing_generic:
            words.pop()
        # Handle "headquarters in X" → "X"
        text = " ".join(words).strip()
        m = re.match(r"(?:headquarters?\s+in|based\s+in)\s+(.+)", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        return text

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
