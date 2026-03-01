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
# Migrated from generate_training_data.py EXTRA_QUERIES (12 categories)
# =========================================================================


def fetch_nationalities(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'has nationality', country) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?country WHERE {{
        ?person dbo:nationality ?country .
        ?person a dbo:Person .
        ?country a dbo:Country .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "country"])
    return [
        (_uri_to_label(person), "has nationality", _uri_to_label(country))
        for person, country in rows
    ]


def fetch_genres(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (work, 'belongs to genre', genre) triplets."""
    query = f"""
    SELECT DISTINCT ?work ?genre WHERE {{
        ?work dbo:genre ?genre .
        ?work a dbo:Work .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["work", "genre"])
    return [
        (_uri_to_label(work), "belongs to genre", _uri_to_label(genre))
        for work, genre in rows
    ]


def fetch_companies_founders(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (company, 'was founded by', founder) triplets."""
    query = f"""
    SELECT DISTINCT ?company ?founder WHERE {{
        ?company dbo:founder ?founder .
        ?founder a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["company", "founder"])
    return [
        (_uri_to_label(company), "was founded by", _uri_to_label(founder))
        for company, founder in rows
    ]


def fetch_spouses(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'is married to', spouse) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?spouse WHERE {{
        ?person dbo:spouse ?spouse .
        ?person a dbo:Person .
        ?spouse a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "spouse"])
    return [
        (_uri_to_label(person), "is married to", _uri_to_label(spouse))
        for person, spouse in rows
    ]


def fetch_country_of(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (place, 'is in', country) triplets."""
    query = f"""
    SELECT DISTINCT ?place ?country WHERE {{
        ?place dbo:country ?country .
        ?place a dbo:Place .
        ?country a dbo:Country .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["place", "country"])
    return [
        (_uri_to_label(place), "is in", _uri_to_label(country))
        for place, country in rows
    ]


def fetch_languages(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (country, 'has official language', language) triplets."""
    query = f"""
    SELECT DISTINCT ?country ?lang WHERE {{
        ?country dbo:language ?lang .
        ?country a dbo:Country .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["country", "lang"])
    return [
        (_uri_to_label(country), "has official language", _uri_to_label(lang))
        for country, lang in rows
    ]


def fetch_awards(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'received', award) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?award WHERE {{
        ?person dbo:award ?award .
        ?person a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "award"])
    return [
        (_uri_to_label(person), "received", _uri_to_label(award))
        for person, award in rows
    ]


def fetch_known_for(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'is known for', thing) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?thing WHERE {{
        ?person dbo:knownFor ?thing .
        ?person a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "thing"])
    return [
        (_uri_to_label(person), "is known for", _uri_to_label(thing))
        for person, thing in rows
    ]


def fetch_death_places(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'died in', place) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?place WHERE {{
        ?person dbo:deathPlace ?place .
        ?person a dbo:Person .
        ?place a dbo:Place .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "place"])
    return [
        (_uri_to_label(person), "died in", _uri_to_label(place))
        for person, place in rows
    ]


def fetch_headquarters(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (org, 'has headquarters in', place) triplets."""
    query = f"""
    SELECT DISTINCT ?org ?place WHERE {{
        ?org dbo:headquarter ?place .
        ?place a dbo:Place .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["org", "place"])
    return [
        (_uri_to_label(org), "has headquarters in", _uri_to_label(place))
        for org, place in rows
    ]


def fetch_developers(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (product, 'was developed by', developer) triplets."""
    query = f"""
    SELECT DISTINCT ?product ?dev WHERE {{
        ?product dbo:developer ?dev .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["product", "dev"])
    return [
        (_uri_to_label(product), "was developed by", _uri_to_label(dev))
        for product, dev in rows
    ]


def fetch_rivers_countries(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (river, 'flows through', country) triplets."""
    query = f"""
    SELECT DISTINCT ?river ?country WHERE {{
        ?river dbo:country ?country .
        ?river a dbo:River .
        ?country a dbo:Country .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["river", "country"])
    return [
        (_uri_to_label(river), "flows through", _uri_to_label(country))
        for river, country in rows
    ]


# =========================================================================
# New categories (21)
# =========================================================================


def fetch_cities(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (entity, 'is in the city of', city) triplets."""
    query = f"""
    SELECT DISTINCT ?entity ?city WHERE {{
        ?entity dbo:city ?city .
        ?city a dbo:City .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["entity", "city"])
    return [
        (_uri_to_label(entity), "is in the city of", _uri_to_label(city))
        for entity, city in rows
    ]


def fetch_teams(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'plays for', team) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?team WHERE {{
        ?person dbo:team ?team .
        ?person a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "team"])
    return [
        (_uri_to_label(person), "plays for", _uri_to_label(team))
        for person, team in rows
    ]


def fetch_managers(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (entity, 'is managed by', manager) triplets."""
    query = f"""
    SELECT DISTINCT ?entity ?manager WHERE {{
        ?entity dbo:manager ?manager .
        ?manager a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["entity", "manager"])
    return [
        (_uri_to_label(entity), "is managed by", _uri_to_label(manager))
        for entity, manager in rows
    ]


def fetch_directors(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (work, 'was directed by', director) triplets."""
    query = f"""
    SELECT DISTINCT ?work ?director WHERE {{
        ?work dbo:director ?director .
        ?director a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["work", "director"])
    return [
        (_uri_to_label(work), "was directed by", _uri_to_label(director))
        for work, director in rows
    ]


def fetch_producers(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (work, 'was produced by', producer) triplets."""
    query = f"""
    SELECT DISTINCT ?work ?producer WHERE {{
        ?work dbo:producer ?producer .
        ?producer a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["work", "producer"])
    return [
        (_uri_to_label(work), "was produced by", _uri_to_label(producer))
        for work, producer in rows
    ]


def fetch_starring(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (work, 'stars', actor) triplets."""
    query = f"""
    SELECT DISTINCT ?work ?actor WHERE {{
        ?work dbo:starring ?actor .
        ?actor a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["work", "actor"])
    return [
        (_uri_to_label(work), "stars", _uri_to_label(actor))
        for work, actor in rows
    ]


def fetch_parents(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'is child of', parent) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?parent WHERE {{
        ?person dbo:parent ?parent .
        ?person a dbo:Person .
        ?parent a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "parent"])
    return [
        (_uri_to_label(person), "is child of", _uri_to_label(parent))
        for person, parent in rows
    ]


def fetch_children(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'is parent of', child) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?child WHERE {{
        ?person dbo:child ?child .
        ?person a dbo:Person .
        ?child a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "child"])
    return [
        (_uri_to_label(person), "is parent of", _uri_to_label(child))
        for person, child in rows
    ]


def fetch_religions(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'follows', religion) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?religion WHERE {{
        ?person dbo:religion ?religion .
        ?person a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "religion"])
    return [
        (_uri_to_label(person), "follows", _uri_to_label(religion))
        for person, religion in rows
    ]


def fetch_parties(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'is member of', party) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?party WHERE {{
        ?person dbo:party ?party .
        ?person a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "party"])
    return [
        (_uri_to_label(person), "is member of", _uri_to_label(party))
        for person, party in rows
    ]


def fetch_general_languages(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (work, 'is in language', language) triplets."""
    query = f"""
    SELECT DISTINCT ?work ?lang WHERE {{
        ?work dbo:language ?lang .
        ?work a dbo:Work .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["work", "lang"])
    return [
        (_uri_to_label(work), "is in language", _uri_to_label(lang))
        for work, lang in rows
    ]


def fetch_industries(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (company, 'is in industry', industry) triplets."""
    query = f"""
    SELECT DISTINCT ?company ?industry WHERE {{
        ?company dbo:industry ?industry .
        ?company a dbo:Company .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["company", "industry"])
    return [
        (_uri_to_label(company), "is in industry", _uri_to_label(industry))
        for company, industry in rows
    ]


def fetch_products(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (company, 'produces', product) triplets."""
    query = f"""
    SELECT DISTINCT ?company ?product WHERE {{
        ?company dbo:product ?product .
        ?company a dbo:Company .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["company", "product"])
    return [
        (_uri_to_label(company), "produces", _uri_to_label(product))
        for company, product in rows
    ]


def fetch_currencies(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (country, 'uses currency', currency) triplets."""
    query = f"""
    SELECT DISTINCT ?country ?currency WHERE {{
        ?country dbo:currency ?currency .
        ?country a dbo:Country .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["country", "currency"])
    return [
        (_uri_to_label(country), "uses currency", _uri_to_label(currency))
        for country, currency in rows
    ]


def fetch_instruments(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'plays', instrument) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?instrument WHERE {{
        ?person dbo:instrument ?instrument .
        ?person a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "instrument"])
    return [
        (_uri_to_label(person), "plays", _uri_to_label(instrument))
        for person, instrument in rows
    ]


def fetch_record_labels(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (artist, 'is signed to', label) triplets."""
    query = f"""
    SELECT DISTINCT ?artist ?label WHERE {{
        ?artist dbo:recordLabel ?label .
        ?artist a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["artist", "label"])
    return [
        (_uri_to_label(artist), "is signed to", _uri_to_label(label))
        for artist, label in rows
    ]


def fetch_associated_bands(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (person, 'is associated with', band) triplets."""
    query = f"""
    SELECT DISTINCT ?person ?band WHERE {{
        ?person dbo:associatedBand ?band .
        ?person a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["person", "band"])
    return [
        (_uri_to_label(person), "is associated with", _uri_to_label(band))
        for person, band in rows
    ]


def fetch_distributors(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (work, 'is distributed by', distributor) triplets."""
    query = f"""
    SELECT DISTINCT ?work ?distributor WHERE {{
        ?work dbo:distributor ?distributor .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["work", "distributor"])
    return [
        (_uri_to_label(work), "is distributed by", _uri_to_label(dist))
        for work, dist in rows
    ]


def fetch_owners(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (entity, 'is owned by', owner) triplets."""
    query = f"""
    SELECT DISTINCT ?entity ?owner WHERE {{
        ?entity dbo:owner ?owner .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["entity", "owner"])
    return [
        (_uri_to_label(entity), "is owned by", _uri_to_label(owner))
        for entity, owner in rows
    ]


def fetch_chairmans(limit: int = 500) -> list[tuple[str, str, str]]:
    """Fetch (org, 'has chairman', person) triplets."""
    query = f"""
    SELECT DISTINCT ?org ?person WHERE {{
        ?org dbo:chairman ?person .
        ?person a dbo:Person .
    }}
    LIMIT {limit}
    """
    rows = _run_query(query, ["org", "person"])
    return [
        (_uri_to_label(org), "has chairman", _uri_to_label(person))
        for org, person in rows
    ]


# =========================================================================
# Aggregation helpers
# =========================================================================

# Registry mapping a human-readable category name to its fetch function.
QUERY_REGISTRY: dict[str, Callable[[int], list[tuple[str, str, str]]]] = {
    # Original categories (7)
    "capital_cities": fetch_capital_cities,
    "birth_places": fetch_birth_places,
    "occupations": fetch_occupations,
    "locations": fetch_locations,
    "founding_dates": fetch_founding_dates,
    "authored_works": fetch_authored_works,
    "alma_maters": fetch_alma_maters,
    # Migrated from generate_training_data.py (12)
    "nationalities": fetch_nationalities,
    "genres": fetch_genres,
    "companies_founders": fetch_companies_founders,
    "spouses": fetch_spouses,
    "country_of": fetch_country_of,
    "languages": fetch_languages,
    "awards": fetch_awards,
    "known_for": fetch_known_for,
    "death_places": fetch_death_places,
    "headquarters": fetch_headquarters,
    "developers": fetch_developers,
    "rivers_countries": fetch_rivers_countries,
    # New categories (21)
    "cities": fetch_cities,
    "teams": fetch_teams,
    "managers": fetch_managers,
    "directors": fetch_directors,
    "producers": fetch_producers,
    "starring": fetch_starring,
    "parents": fetch_parents,
    "children": fetch_children,
    "religions": fetch_religions,
    "parties": fetch_parties,
    "general_languages": fetch_general_languages,
    "industries": fetch_industries,
    "products": fetch_products,
    "currencies": fetch_currencies,
    "instruments": fetch_instruments,
    "record_labels": fetch_record_labels,
    "associated_bands": fetch_associated_bands,
    "distributors": fetch_distributors,
    "owners": fetch_owners,
    "chairmans": fetch_chairmans,
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
