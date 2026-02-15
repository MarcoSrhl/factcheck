"""Dedicated SPARQL query module for extracting diverse facts from DBpedia.

Provides well-crafted SPARQL queries covering different relation types:
capital cities, birth places, occupations, locations, founding dates,
authored works, and more. Each query returns (subject, predicate, object)
triplet strings suitable for GAN training.
"""

import logging
import random
from typing import Callable, Optional

from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://dbpedia.org/sparql"

# ---------------------------------------------------------------------------
# Timeout (seconds) for each SPARQL request
# ---------------------------------------------------------------------------
_SPARQL_TIMEOUT = 30


def _create_sparql_client() -> SPARQLWrapper:
    """Create a configured SPARQLWrapper client."""
    client = SPARQLWrapper(SPARQL_ENDPOINT)
    client.setReturnFormat(JSON)
    client.setTimeout(_SPARQL_TIMEOUT)
    return client


def _run_query(query: str, variables: list[str]) -> list[tuple[str, ...]]:
    """Execute a SPARQL SELECT query and return rows as tuples of strings.

    Parameters
    ----------
    query : str
        The SPARQL SELECT query to execute.
    variables : list[str]
        The variable names (without ``?``) expected in each result binding.

    Returns
    -------
    list[tuple[str, ...]]
        Each tuple contains one string value per variable, in order.
    """
    client = _create_sparql_client()
    try:
        client.setQuery(query)
        results = client.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        rows: list[tuple[str, ...]] = []
        for binding in bindings:
            values = []
            skip = False
            for var in variables:
                if var not in binding:
                    skip = True
                    break
                values.append(binding[var]["value"])
            if not skip:
                rows.append(tuple(values))
        return rows
    except Exception as exc:
        logger.error("SPARQL query failed: %s", exc)
        return []


def _uri_to_label(uri: str) -> str:
    """Convert a DBpedia resource URI to a human-readable label.

    ``http://dbpedia.org/resource/Barack_Obama`` becomes ``Barack Obama``.
    """
    if "/" in uri:
        name = uri.rsplit("/", 1)[-1]
    else:
        name = uri
    return name.replace("_", " ")


# =========================================================================
# Individual SPARQL query functions
# =========================================================================


def fetch_capital_cities(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (city, 'is capital of', country) triplets.

    Uses ``dbo:capital`` — the country's capital property — and inverts it
    so the triplet reads ``City is capital of Country``.
    """
    query = f"""
    SELECT DISTINCT ?country ?city WHERE {{
        ?country dbo:capital ?city .
        ?country a dbo:Country .
        ?city a dbo:City .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["country", "city"])
    triplets: list[tuple[str, str, str]] = []
    for country_uri, city_uri in rows:
        city = _uri_to_label(city_uri)
        country = _uri_to_label(country_uri)
        triplets.append((city, "is capital of", country))
    return triplets


def fetch_birth_places(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'was born in', place) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?place WHERE {{
        ?person dbo:birthPlace ?place .
        ?person a dbo:Person .
        ?place a dbo:Place .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "place"])
    return [
        (_uri_to_label(person), "was born in", _uri_to_label(place))
        for person, place in rows
    ]


def fetch_occupations(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'has occupation', occupation) triplets.

    Falls back to ``rdfs:label`` for the occupation resource when available,
    but uses the URI tail otherwise.
    """
    query = f"""
    SELECT DISTINCT ?person ?occupation WHERE {{
        ?person dbo:occupation ?occupation .
        ?person a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "occupation"])
    return [
        (_uri_to_label(person), "has occupation", _uri_to_label(occ))
        for person, occ in rows
    ]


def fetch_locations(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (entity, 'is located in', location) triplets.

    Covers buildings, organisations, and other entities with a ``dbo:location``.
    """
    query = f"""
    SELECT DISTINCT ?entity ?location WHERE {{
        ?entity dbo:location ?location .
        ?location a dbo:Place .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["entity", "location"])
    return [
        (_uri_to_label(entity), "is located in", _uri_to_label(loc))
        for entity, loc in rows
    ]


def fetch_founding_dates(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (organisation, 'was founded in', year) triplets."""
    query = f"""
    SELECT DISTINCT ?org ?date WHERE {{
        ?org dbo:foundingDate ?date .
        ?org a dbo:Organisation .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["org", "date"])
    triplets: list[tuple[str, str, str]] = []
    for org_uri, date_str in rows:
        # date_str may be a full xsd:date like "1976-04-01"; extract just the year
        year = date_str[:4] if len(date_str) >= 4 else date_str
        triplets.append((_uri_to_label(org_uri), "was founded in", year))
    return triplets


def fetch_authored_works(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (author, 'wrote', work) triplets."""
    query = f"""
    SELECT DISTINCT ?author ?work WHERE {{
        ?work dbo:author ?author .
        ?author a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["author", "work"])
    return [
        (_uri_to_label(author), "wrote", _uri_to_label(work))
        for author, work in rows
    ]


def fetch_country_leaders(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'is leader of', country) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?country WHERE {{
        ?country dbo:leader ?person .
        ?country a dbo:Country .
        ?person a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "country"])
    return [
        (_uri_to_label(person), "is leader of", _uri_to_label(country))
        for person, country in rows
    ]


def fetch_alma_maters(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'studied at', university) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?university WHERE {{
        ?person dbo:almaMater ?university .
        ?person a dbo:Person .
        ?university a dbo:University .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "university"])
    return [
        (_uri_to_label(person), "studied at", _uri_to_label(uni))
        for person, uni in rows
    ]


# =========================================================================
# Aggregation helpers
# =========================================================================

# Registry mapping a human-readable category name to its fetch function.
QUERY_REGISTRY: dict[str, Callable[[int], list[tuple[str, str, str]]]] = {
    "capital_cities": fetch_capital_cities,
    "birth_places": fetch_birth_places,
    "occupations": fetch_occupations,
    "locations": fetch_locations,
    "founding_dates": fetch_founding_dates,
    "authored_works": fetch_authored_works,
    "country_leaders": fetch_country_leaders,
    "alma_maters": fetch_alma_maters,
}


def fetch_mixed_triplets(
    per_category: int = 200,
    categories: Optional[list[str]] = None,
    shuffle: bool = True,
) -> list[tuple[str, str, str]]:
    """Fetch a large mixed batch of triplets from multiple categories.

    Parameters
    ----------
    per_category : int
        Maximum number of triplets to request per category.
    categories : list[str] or None
        Subset of category names from ``QUERY_REGISTRY``.  If *None*, all
        categories are used.
    shuffle : bool
        Whether to shuffle the combined result list.

    Returns
    -------
    list[tuple[str, str, str]]
        Combined list of ``(subject, predicate, object)`` text triplets.
    """
    if categories is None:
        categories = list(QUERY_REGISTRY.keys())

    all_triplets: list[tuple[str, str, str]] = []
    for cat_name in categories:
        fetch_fn = QUERY_REGISTRY.get(cat_name)
        if fetch_fn is None:
            logger.warning("Unknown category '%s', skipping.", cat_name)
            continue
        logger.info("Fetching category '%s' (limit=%d)...", cat_name, per_category)
        triplets = fetch_fn(limit=per_category)
        logger.info("  -> got %d triplets for '%s'.", len(triplets), cat_name)
        all_triplets.extend(triplets)

    if shuffle:
        random.shuffle(all_triplets)

    logger.info("Total mixed triplets fetched: %d", len(all_triplets))
    return all_triplets


# =========================================================================
# CLI quick-test
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Fetching a small mixed sample of triplets from DBpedia...\n")
    triplets = fetch_mixed_triplets(per_category=5)
    for subj, pred, obj in triplets:
        print(f"  ({subj}, {pred}, {obj})")
    print(f"\nTotal: {len(triplets)} triplets")
