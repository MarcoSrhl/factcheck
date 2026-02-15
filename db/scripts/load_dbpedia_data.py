"""Fetch RDF data from DBpedia's public SPARQL endpoint and load it into the
local GraphDB instance.

Categories of facts fetched:
  1. Capital cities and countries
  2. People and their birth places
  3. Organizations and their locations
  4. Historical events and dates
  5. Scientific facts (discoveries, inventions)

Usage:
    python -m db.scripts.load_dbpedia_data                # load all categories
    python -m db.scripts.load_dbpedia_data --category capitals
    python -m db.scripts.load_dbpedia_data --limit 500    # per-category limit
    python -m db.scripts.load_dbpedia_data --dry-run      # preview, don't load
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass

import requests
from SPARQLWrapper import SPARQLWrapper, N3, JSON

from db.config.graphdb_config import GraphDBConfig

logger = logging.getLogger(__name__)

DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"

# ---------------------------------------------------------------------------
# CONSTRUCT queries that pull self-contained RDF subgraphs from DBpedia.
# Each query uses CONSTRUCT so the result is directly loadable RDF.
# ---------------------------------------------------------------------------

CATEGORY_QUERIES: dict[str, str] = {
    "capitals": """
        CONSTRUCT {
            ?country dbo:capital ?capital .
            ?country rdfs:label ?countryLabel .
            ?capital rdfs:label ?capitalLabel .
            ?country rdf:type dbo:Country .
            ?capital rdf:type dbo:City .
        }
        WHERE {
            ?country a dbo:Country ;
                     dbo:capital ?capital ;
                     rdfs:label ?countryLabel .
            ?capital rdfs:label ?capitalLabel .
            FILTER (lang(?countryLabel) = "en")
            FILTER (lang(?capitalLabel) = "en")
        }
        LIMIT __LIMIT__
    """,
    "birthplaces": """
        CONSTRUCT {
            ?person dbo:birthPlace ?place .
            ?person rdfs:label ?personLabel .
            ?place rdfs:label ?placeLabel .
            ?person rdf:type dbo:Person .
            ?place rdf:type dbo:Place .
        }
        WHERE {
            ?person a dbo:Person ;
                    dbo:birthPlace ?place ;
                    rdfs:label ?personLabel .
            ?place rdfs:label ?placeLabel .
            FILTER (lang(?personLabel) = "en")
            FILTER (lang(?placeLabel) = "en")
        }
        LIMIT __LIMIT__
    """,
    "organizations": """
        CONSTRUCT {
            ?org dbo:location ?loc .
            ?org rdfs:label ?orgLabel .
            ?loc rdfs:label ?locLabel .
            ?org rdf:type dbo:Organisation .
            ?loc rdf:type dbo:Place .
        }
        WHERE {
            ?org a dbo:Organisation ;
                 dbo:location ?loc ;
                 rdfs:label ?orgLabel .
            ?loc rdfs:label ?locLabel .
            FILTER (lang(?orgLabel) = "en")
            FILTER (lang(?locLabel) = "en")
        }
        LIMIT __LIMIT__
    """,
    "events": """
        CONSTRUCT {
            ?event dbo:date ?date .
            ?event rdfs:label ?eventLabel .
            ?event dbo:place ?place .
            ?place rdfs:label ?placeLabel .
            ?event rdf:type dbo:Event .
        }
        WHERE {
            ?event a dbo:Event ;
                   rdfs:label ?eventLabel .
            OPTIONAL { ?event dbo:date ?date . }
            OPTIONAL {
                ?event dbo:place ?place .
                ?place rdfs:label ?placeLabel .
                FILTER (lang(?placeLabel) = "en")
            }
            FILTER (lang(?eventLabel) = "en")
        }
        LIMIT __LIMIT__
    """,
    "science": """
        CONSTRUCT {
            ?scientist dbo:knownFor ?contribution .
            ?scientist rdfs:label ?scientistLabel .
            ?contribution rdfs:label ?contribLabel .
            ?scientist rdf:type dbo:Scientist .
        }
        WHERE {
            ?scientist a dbo:Scientist ;
                       dbo:knownFor ?contribution ;
                       rdfs:label ?scientistLabel .
            ?contribution rdfs:label ?contribLabel .
            FILTER (lang(?scientistLabel) = "en")
            FILTER (lang(?contribLabel) = "en")
        }
        LIMIT __LIMIT__
    """,
}

# A SELECT-based count query to estimate the number of triples already loaded
# in the local repository for a given predicate.
COUNT_LOCAL_QUERY = """
SELECT (COUNT(*) AS ?cnt)
WHERE {{
    ?s <{predicate}> ?o .
}}
"""

# Predicates used in each category (for duplicate detection).
CATEGORY_PREDICATES: dict[str, str] = {
    "capitals": "http://dbpedia.org/ontology/capital",
    "birthplaces": "http://dbpedia.org/ontology/birthPlace",
    "organizations": "http://dbpedia.org/ontology/location",
    "events": "http://dbpedia.org/ontology/date",
    "science": "http://dbpedia.org/ontology/knownFor",
}


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

@dataclass
class LoadStats:
    """Tracks loading statistics for a single category."""
    category: str
    triples_fetched: int = 0
    triples_loaded: int = 0
    already_present: int = 0
    errors: int = 0
    duration_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"[{self.category}] fetched={self.triples_fetched}, "
            f"loaded={self.triples_loaded}, already_present={self.already_present}, "
            f"errors={self.errors}, duration={self.duration_seconds:.1f}s"
        )


class DBpediaDataLoader:
    """Fetch RDF data from DBpedia and load it into a local GraphDB instance."""

    def __init__(self, config: GraphDBConfig | None = None):
        self.config = config or GraphDBConfig()
        self.session = requests.Session()
        if self.config.auth:
            self.session.auth = self.config.auth

        # SPARQLWrapper for querying DBpedia
        self.dbpedia_sparql = SPARQLWrapper(DBPEDIA_SPARQL_ENDPOINT)

    # ------------------------------------------------------------------
    # Fetching from DBpedia
    # ------------------------------------------------------------------

    def fetch_rdf(self, category: str, limit: int = 1000) -> str | None:
        """Execute a CONSTRUCT query against DBpedia and return N3/Turtle text.

        Returns None on failure.
        """
        template = CATEGORY_QUERIES.get(category)
        if not template:
            logger.error("Unknown category: %s", category)
            return None

        query = template.replace("__LIMIT__", str(limit))
        logger.info("Fetching category '%s' (limit %d) from DBpedia ...", category, limit)

        try:
            self.dbpedia_sparql.setQuery(query)
            self.dbpedia_sparql.setReturnFormat(N3)
            results = self.dbpedia_sparql.query().convert()
            # results is bytes in N3/Turtle format
            if isinstance(results, bytes):
                rdf_text = results.decode("utf-8", errors="replace")
            else:
                rdf_text = str(results)

            triple_count = _estimate_triple_count(rdf_text)
            logger.info(
                "Fetched ~%d triples for category '%s'.", triple_count, category
            )
            return rdf_text
        except Exception as exc:
            logger.error("Failed to fetch '%s' from DBpedia: %s", category, exc)
            return None

    # ------------------------------------------------------------------
    # Loading into local GraphDB
    # ------------------------------------------------------------------

    def count_local_triples(self, predicate: str) -> int:
        """Return the number of triples in the local repo for *predicate*."""
        query = COUNT_LOCAL_QUERY.format(predicate=predicate)
        try:
            resp = self.session.get(
                self.config.repository_url,
                params={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            bindings = data.get("results", {}).get("bindings", [])
            if bindings:
                return int(bindings[0]["cnt"]["value"])
        except Exception as exc:
            logger.warning("Could not count local triples for %s: %s", predicate, exc)
        return 0

    def load_rdf(self, rdf_data: str, content_type: str = "text/turtle") -> bool:
        """Load RDF data (Turtle/N3) into the local GraphDB repository.

        Uses the SPARQL Graph Store HTTP Protocol (POST to statements endpoint).
        Returns True on success.
        """
        if not rdf_data or not rdf_data.strip():
            logger.warning("No RDF data to load.")
            return False

        logger.info("Loading RDF data into %s ...", self.config.statements_url)

        try:
            resp = self.session.post(
                self.config.statements_url,
                data=rdf_data.encode("utf-8"),
                headers={"Content-Type": content_type},
                timeout=self.config.timeout * 3,  # loading can be slow
            )
            if resp.status_code in (200, 201, 204):
                logger.info("Data loaded successfully (HTTP %d).", resp.status_code)
                return True
            else:
                logger.error(
                    "Failed to load data (HTTP %d): %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return False
        except requests.RequestException as exc:
            logger.error("Error loading RDF data: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Incremental loading for a single category
    # ------------------------------------------------------------------

    def load_category(
        self,
        category: str,
        limit: int = 1000,
        dry_run: bool = False,
    ) -> LoadStats:
        """Fetch and load a single category.

        With *dry_run=True* the data is fetched but not uploaded.
        """
        stats = LoadStats(category=category)
        start = time.time()

        # Check how many triples already exist locally for this predicate.
        predicate = CATEGORY_PREDICATES.get(category, "")
        if predicate:
            stats.already_present = self.count_local_triples(predicate)
            if stats.already_present > 0:
                logger.info(
                    "Category '%s' already has %d triple(s) locally.",
                    category,
                    stats.already_present,
                )

        rdf_text = self.fetch_rdf(category, limit=limit)
        if rdf_text is None:
            stats.errors += 1
            stats.duration_seconds = time.time() - start
            return stats

        stats.triples_fetched = _estimate_triple_count(rdf_text)

        if dry_run:
            logger.info("[DRY RUN] Would load %d triples for '%s'.", stats.triples_fetched, category)
        else:
            ok = self.load_rdf(rdf_text)
            if ok:
                stats.triples_loaded = stats.triples_fetched
            else:
                stats.errors += 1

        stats.duration_seconds = time.time() - start
        return stats

    # ------------------------------------------------------------------
    # Load all categories
    # ------------------------------------------------------------------

    def load_all(
        self,
        categories: list[str] | None = None,
        limit: int = 1000,
        dry_run: bool = False,
    ) -> list[LoadStats]:
        """Load multiple categories. Defaults to all available categories."""
        cats = categories or list(CATEGORY_QUERIES.keys())
        all_stats: list[LoadStats] = []

        for cat in cats:
            stats = self.load_category(cat, limit=limit, dry_run=dry_run)
            all_stats.append(stats)
            logger.info("  %s", stats)

        # Summary
        total_fetched = sum(s.triples_fetched for s in all_stats)
        total_loaded = sum(s.triples_loaded for s in all_stats)
        total_errors = sum(s.errors for s in all_stats)
        logger.info(
            "=== SUMMARY: fetched=%d, loaded=%d, errors=%d ===",
            total_fetched,
            total_loaded,
            total_errors,
        )
        return all_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_triple_count(rdf_text: str) -> int:
    """Rough estimate of the number of triples in a Turtle/N3 string.

    Counts non-blank lines that end with ' .' (simplified heuristic).
    """
    count = 0
    for line in rdf_text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("@") and not stripped.startswith("#"):
            # Each statement in N-Triples/Turtle ends with ' .'
            if stripped.endswith("."):
                count += 1
    return max(count, 1) if rdf_text.strip() else 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fetch RDF data from DBpedia and load into local GraphDB."
    )
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_QUERIES.keys()),
        default=None,
        help="Load only this category. Default: all categories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of results per CONSTRUCT query (default: 1000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but do not upload to GraphDB.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = GraphDBConfig()
    loader = DBpediaDataLoader(config)

    categories = [args.category] if args.category else None
    loader.load_all(categories=categories, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
