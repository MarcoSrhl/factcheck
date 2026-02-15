"""Reusable, parametric SPARQL query templates for the local GraphDB instance.

All templates are methods on the ``SPARQLTemplates`` class.  Each method returns
a ready-to-execute SPARQL string.  Parameters are safely interpolated.

Query types provided:
  - SELECT  : retrieve specific bindings
  - ASK     : boolean existence check (fact verification)
  - CONSTRUCT : extract a subgraph as RDF
  - INSERT  : add new triples
  - DELETE  : remove triples

Usage example::

    from db.queries.sparql_templates import SPARQLTemplates

    tpl = SPARQLTemplates()
    query = tpl.ask_triple_exists(
        "http://dbpedia.org/resource/Paris",
        "http://dbpedia.org/ontology/capital",
        "http://dbpedia.org/resource/France",
    )
    # ... execute query against local GraphDB ...
"""

from __future__ import annotations

import logging

from db.config.graphdb_config import GraphDBConfig

logger = logging.getLogger(__name__)

# Standard prefix block generated from the project's default config.
_DEFAULT_PREFIXES = GraphDBConfig().namespace_prefixes_sparql()


class SPARQLTemplates:
    """Factory for parametric SPARQL queries targeting the local GraphDB repo.

    Parameters
    ----------
    prefix_block : str, optional
        Custom SPARQL PREFIX declarations.  Defaults to the standard set
        defined in ``GraphDBConfig.namespaces``.
    """

    def __init__(self, prefix_block: str | None = None):
        self.prefixes = prefix_block or _DEFAULT_PREFIXES

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _prefixed(self, body: str) -> str:
        """Prepend the PREFIX block to a query body."""
        return f"{self.prefixes}\n\n{body}"

    @staticmethod
    def _uri(value: str) -> str:
        """Wrap a URI in angle brackets if not already wrapped.

        Also strips characters that could break out of a ``<URI>`` token
        and potentially allow SPARQL injection (``>``, ``{``, ``}``).
        """
        if value.startswith("<") and value.endswith(">"):
            value = value[1:-1]
        # Remove characters that could escape the URI or inject SPARQL.
        sanitized = value.replace(">", "").replace("{", "").replace("}", "")
        return f"<{sanitized}>"

    @staticmethod
    def _literal(value: str, lang: str | None = None, datatype: str | None = None) -> str:
        """Format a literal value for SPARQL."""
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        if lang:
            return f'"{escaped}"@{lang}'
        if datatype:
            return f'"{escaped}"^^<{datatype}>'
        return f'"{escaped}"'

    # ==================================================================
    # ASK queries  (fact checking / boolean verification)
    # ==================================================================

    def ask_triple_exists(
        self, subject: str, predicate: str, obj: str
    ) -> str:
        """ASK whether a specific <s, p, o> triple exists."""
        return self._prefixed(f"""\
ASK WHERE {{
    {self._uri(subject)} {self._uri(predicate)} {self._uri(obj)} .
}}""")

    def ask_relation_exists(self, subject: str, obj: str) -> str:
        """ASK whether *any* direct relation exists between two resources."""
        return self._prefixed(f"""\
ASK WHERE {{
    {self._uri(subject)} ?p {self._uri(obj)} .
}}""")

    # ==================================================================
    # SELECT queries
    # ==================================================================

    def select_predicates_between(
        self, subject: str, obj: str, limit: int = 50
    ) -> str:
        """Find all predicates linking *subject* to *obj*."""
        return self._prefixed(f"""\
SELECT DISTINCT ?predicate WHERE {{
    {self._uri(subject)} ?predicate {self._uri(obj)} .
}}
LIMIT {limit}""")

    def select_objects(
        self, subject: str, predicate: str, limit: int = 50
    ) -> str:
        """Get all objects for a given subject and predicate."""
        return self._prefixed(f"""\
SELECT ?object WHERE {{
    {self._uri(subject)} {self._uri(predicate)} ?object .
}}
LIMIT {limit}""")

    def select_subjects(
        self, predicate: str, obj: str, limit: int = 50
    ) -> str:
        """Get all subjects that have *predicate* pointing to *obj*."""
        return self._prefixed(f"""\
SELECT ?subject WHERE {{
    ?subject {self._uri(predicate)} {self._uri(obj)} .
}}
LIMIT {limit}""")

    def select_entity_properties(
        self, entity: str, limit: int = 100
    ) -> str:
        """Return all (predicate, object) pairs for an entity."""
        return self._prefixed(f"""\
SELECT ?predicate ?object WHERE {{
    {self._uri(entity)} ?predicate ?object .
}}
LIMIT {limit}""")

    def select_label(self, entity: str, lang: str = "en") -> str:
        """Get the rdfs:label for an entity in the given language."""
        return self._prefixed(f"""\
SELECT ?label WHERE {{
    {self._uri(entity)} rdfs:label ?label .
    FILTER (lang(?label) = "{lang}")
}}
LIMIT 1""")

    def select_count_triples(self) -> str:
        """Count the total number of triples in the repository."""
        return self._prefixed("""\
SELECT (COUNT(*) AS ?count) WHERE {
    ?s ?p ?o .
}""")

    def select_count_by_predicate(self, predicate: str) -> str:
        """Count triples with a specific predicate."""
        return self._prefixed(f"""\
SELECT (COUNT(*) AS ?count) WHERE {{
    ?s {self._uri(predicate)} ?o .
}}""")

    def select_types_of(self, entity: str) -> str:
        """Get all rdf:type values for an entity."""
        return self._prefixed(f"""\
SELECT ?type WHERE {{
    {self._uri(entity)} rdf:type ?type .
}}""")

    def select_search_by_label(
        self, label: str, lang: str = "en", limit: int = 20
    ) -> str:
        """Find entities whose rdfs:label contains *label* (case-insensitive)."""
        escaped = label.replace("\\", "\\\\").replace('"', '\\"')
        return self._prefixed(f"""\
SELECT ?entity ?label WHERE {{
    ?entity rdfs:label ?label .
    FILTER (lang(?label) = "{lang}")
    FILTER (CONTAINS(LCASE(?label), LCASE("{escaped}")))
}}
LIMIT {limit}""")

    # ==================================================================
    # CONSTRUCT queries  (graph extraction)
    # ==================================================================

    def construct_entity_neighbourhood(
        self, entity: str, limit: int = 200
    ) -> str:
        """Extract the immediate neighbourhood (1-hop) of an entity as RDF."""
        return self._prefixed(f"""\
CONSTRUCT {{
    {self._uri(entity)} ?p ?o .
}}
WHERE {{
    {self._uri(entity)} ?p ?o .
}}
LIMIT {limit}""")

    def construct_entity_pair(
        self, entity_a: str, entity_b: str, limit: int = 200
    ) -> str:
        """Extract all triples involving *entity_a* or *entity_b*."""
        ua = self._uri(entity_a)
        ub = self._uri(entity_b)
        return self._prefixed(f"""\
CONSTRUCT {{
    ?s ?p ?o .
}}
WHERE {{
    {{
        {ua} ?p ?o .
        BIND({ua} AS ?s)
    }}
    UNION
    {{
        {ub} ?p ?o .
        BIND({ub} AS ?s)
    }}
    UNION
    {{
        {ua} ?p {ub} .
        BIND({ua} AS ?s) BIND({ub} AS ?o)
    }}
}}
LIMIT {limit}""")

    # ==================================================================
    # INSERT queries  (adding new facts)
    # ==================================================================

    def insert_triple(
        self, subject: str, predicate: str, obj: str
    ) -> str:
        """Insert a single <s, p, o> triple (all URIs)."""
        return self._prefixed(f"""\
INSERT DATA {{
    {self._uri(subject)} {self._uri(predicate)} {self._uri(obj)} .
}}""")

    def insert_triple_with_literal(
        self,
        subject: str,
        predicate: str,
        value: str,
        lang: str | None = None,
        datatype: str | None = None,
    ) -> str:
        """Insert a triple whose object is a literal value."""
        lit = self._literal(value, lang=lang, datatype=datatype)
        return self._prefixed(f"""\
INSERT DATA {{
    {self._uri(subject)} {self._uri(predicate)} {lit} .
}}""")

    def insert_multiple(self, triples: list[tuple[str, str, str]]) -> str:
        """Bulk-insert multiple triples (all URIs).

        Parameters
        ----------
        triples : list of (subject, predicate, object) URI strings.
        """
        lines = [
            f"    {self._uri(s)} {self._uri(p)} {self._uri(o)} ."
            for s, p, o in triples
        ]
        body = "\n".join(lines)
        return self._prefixed(f"""\
INSERT DATA {{
{body}
}}""")

    def insert_entity_with_label(
        self,
        entity: str,
        label: str,
        rdf_type: str | None = None,
        lang: str = "en",
    ) -> str:
        """Insert an entity with its label and optional rdf:type."""
        stmts = [
            f"    {self._uri(entity)} rdfs:label {self._literal(label, lang=lang)} ."
        ]
        if rdf_type:
            stmts.append(
                f"    {self._uri(entity)} rdf:type {self._uri(rdf_type)} ."
            )
        body = "\n".join(stmts)
        return self._prefixed(f"""\
INSERT DATA {{
{body}
}}""")

    # ==================================================================
    # DELETE queries
    # ==================================================================

    def delete_triple(
        self, subject: str, predicate: str, obj: str
    ) -> str:
        """Delete a specific triple."""
        return self._prefixed(f"""\
DELETE DATA {{
    {self._uri(subject)} {self._uri(predicate)} {self._uri(obj)} .
}}""")

    def delete_entity(self, entity: str) -> str:
        """Delete all triples where *entity* appears as subject."""
        return self._prefixed(f"""\
DELETE WHERE {{
    {self._uri(entity)} ?p ?o .
}}""")

    def delete_all(self) -> str:
        """Delete every triple in the repository.  Use with caution."""
        return self._prefixed("""\
DELETE WHERE {
    ?s ?p ?o .
}""")
