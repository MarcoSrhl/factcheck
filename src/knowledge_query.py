"""Knowledge base querying: SPARQL and JSON endpoints on DBpedia.

Queries the public DBpedia SPARQL endpoint and JSON API for fact verification.
"""

import logging
import os
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)

# DBpedia endpoints
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
DBPEDIA_DATA_URL = "https://dbpedia.org/data/{entity}.json"


class KnowledgeQuery:
    """Query DBpedia knowledge base for fact verification.
    
    Uses both SPARQL and JSON endpoints for optimal performance.
    """

    def __init__(self):
        """Initialize DBpedia query client."""
        self.sparql = SPARQLWrapper(SPARQL_ENDPOINT)
        self.sparql.setReturnFormat(JSON)
        self._json_cache: dict[str, dict] = {}
        logger.info("Using DBpedia endpoint: %s", SPARQL_ENDPOINT)

    # --- SPARQL-based methods ---

    def sparql_check_relation(
        self, subject_uri: str, object_uri: str
    ) -> list[str]:
        """Check if any direct relation exists between subject and object in DBpedia.

        Returns list of predicate URIs connecting them.
        """
        query = f"""
        SELECT ?predicate WHERE {{
            <{subject_uri}> ?predicate <{object_uri}> .
        }}
        LIMIT 50
        """
        return self._run_sparql(query, "predicate")

    def sparql_get_property(
        self, subject_uri: str, predicate_uri: str
    ) -> list[str]:
        """Get all objects for a given subject and predicate."""
        query = f"""
        SELECT ?object WHERE {{
            <{subject_uri}> <{predicate_uri}> ?object .
        }}
        LIMIT 50
        """
        return self._run_sparql(query, "object")

    def sparql_ask(self, subject_uri: str, predicate_uri: str, object_uri: str) -> bool:
        """ASK if a specific triple exists."""
        query = f"""
        ASK WHERE {{
            <{subject_uri}> <{predicate_uri}> <{object_uri}> .
        }}
        """
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            return results.get("boolean", False)
        except Exception as e:
            logger.error(f"SPARQL ASK failed: {e}")
            return False

    def _run_sparql(self, query: str, var_name: str) -> list[str]:
        """Execute a SPARQL SELECT query and return values for the given variable."""
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            return [b[var_name]["value"] for b in bindings if var_name in b]
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []

    # --- JSON-based methods (remote DBpedia only) ---

    def json_get_entity_data(self, entity_uri: str) -> dict:
        """Fetch all triples for an entity via the DBpedia JSON endpoint."""
        if entity_uri in self._json_cache:
            return self._json_cache[entity_uri]

        entity_name = entity_uri.replace("http://dbpedia.org/resource/", "")
        url = DBPEDIA_DATA_URL.format(entity=entity_name)

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            self._json_cache[entity_uri] = data
            return data
        except requests.RequestException as e:
            logger.error(f"JSON fetch failed for {entity_uri}: {e}")
            return {}

    def json_check_relation(
        self, subject_uri: str, object_uri: str
    ) -> list[str]:
        """Check relations between subject and object using JSON data."""
        data = self.json_get_entity_data(subject_uri)
        subject_data = data.get(subject_uri, {})
        matching_predicates = []

        for predicate, objects in subject_data.items():
            for obj in objects:
                obj_value = obj.get("value", "")
                if obj_value == object_uri:
                    matching_predicates.append(predicate)

        return matching_predicates

    def json_get_property_values(
        self, subject_uri: str, predicate_uri: str
    ) -> list[str]:
        """Get all values for a given property of an entity via JSON."""
        data = self.json_get_entity_data(subject_uri)
        subject_data = data.get(subject_uri, {})
        objects = subject_data.get(predicate_uri, [])
        return [obj.get("value", "") for obj in objects]

    # --- Entity property extraction ---

    # Key predicates to fetch for evidence (ontology + property)
    _KEY_PREDICATES = [
        "http://dbpedia.org/ontology/birthPlace",
        "http://dbpedia.org/ontology/deathPlace",
        "http://dbpedia.org/ontology/country",
        "http://dbpedia.org/ontology/capital",
        "http://dbpedia.org/ontology/location",
        "http://dbpedia.org/ontology/nationality",
        "http://dbpedia.org/ontology/knownFor",
        "http://dbpedia.org/ontology/occupation",
        "http://dbpedia.org/ontology/genre",
        "http://dbpedia.org/ontology/largestCity",
        "http://dbpedia.org/ontology/officialLanguage",
        "http://dbpedia.org/ontology/continent",
        "http://dbpedia.org/property/birthPlace",
        "http://dbpedia.org/property/capital",
        "http://dbpedia.org/property/location",
        "http://dbpedia.org/property/country",
        "http://www.w3.org/2000/01/rdf-schema#comment",
    ]

    def get_entity_properties(self, entity_uri: str) -> dict[str, list[str]]:
        """Fetch key properties of an entity for evidence building.

        Uses the JSON API (faster than SPARQL) to retrieve entity data,
        then filters for key predicates.
        Returns a dict mapping human-readable property names to their values.
        """
        if not entity_uri:
            return {}

        # Use JSON API for speed (avoids SPARQL timeouts)
        data = self.json_get_entity_data(entity_uri)
        subject_data = data.get(entity_uri, {})

        if not subject_data:
            return {}

        # Build a set of key predicate URIs for fast lookup
        key_pred_set = set(self._KEY_PREDICATES)

        properties: dict[str, list[str]] = {}
        for pred_uri, objects in subject_data.items():
            if pred_uri not in key_pred_set:
                continue

            pred_name = pred_uri.split("/")[-1].split("#")[-1]
            readable_values = []
            for obj in objects[:3]:
                value = obj.get("value", "")
                if not value:
                    continue
                if value.startswith("http://dbpedia.org/resource/"):
                    readable_values.append(value.split("/")[-1].replace("_", " "))
                elif len(value) < 200:
                    readable_values.append(value)

            if readable_values:
                properties[pred_name] = readable_values

        return properties

    # --- High-level verification ---

    def verify_triplet(
        self, subject_uri: str | None, object_uri: str | None
    ) -> dict:
        """Verify whether a relation exists between two entities.

        Returns a dict with:
          - found: bool
          - predicates: list of matching predicate URIs
          - method: 'sparql', 'json', or 'none'
        """
        if not subject_uri or not object_uri:
            return {"found": False, "predicates": [], "method": "none"}

        # Try SPARQL first
        predicates = self.sparql_check_relation(subject_uri, object_uri)
        if predicates:
            return {"found": True, "predicates": predicates, "method": "sparql"}

        # Fallback to JSON API
        predicates = self.json_check_relation(subject_uri, object_uri)
        if predicates:
            return {"found": True, "predicates": predicates, "method": "json"}

        return {"found": False, "predicates": [], "method": "none"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    kq = KnowledgeQuery()

    print("=== DBpedia Knowledge Query Demo ===")

    print("=== SPARQL: relations between Paris and France ===")
    preds = kq.sparql_check_relation(
        "http://dbpedia.org/resource/Paris",
        "http://dbpedia.org/resource/France",
    )
    for p in preds:
        print(f"  {p}")

    print("\n=== Verify: Paris <-> France ===")
    result = kq.verify_triplet(
        "http://dbpedia.org/resource/Paris",
        "http://dbpedia.org/resource/France",
    )
    print(f"  Found: {result['found']}, Method: {result['method']}")
    for p in result["predicates"][:5]:
        print(f"  Predicate: {p}")

    print("\n=== Verify: Barack Obama <-> Hawaii ===")
    result = kq.verify_triplet(
        "http://dbpedia.org/resource/Barack_Obama",
        "http://dbpedia.org/resource/Hawaii",
    )
    print(f"  Found: {result['found']}, Method: {result['method']}")
    for p in result["predicates"][:5]:
        print(f"  Predicate: {p}")
