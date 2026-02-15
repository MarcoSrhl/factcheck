"""GraphDB connection and repository configuration.

All settings can be overridden via environment variables prefixed with GRAPHDB_.
Example:
    export GRAPHDB_HOST=192.168.1.10
    export GRAPHDB_PORT=7200
    export GRAPHDB_REPOSITORY=factcheck
    export GRAPHDB_USERNAME=admin
    export GRAPHDB_PASSWORD=secret
"""

import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GraphDBConfig:
    """Configuration for connecting to a local GraphDB instance.

    Attributes:
        host: GraphDB server hostname or IP address.
        port: GraphDB server port (default 7200).
        repository: Name of the target RDF repository.
        username: Username for authentication (None for anonymous).
        password: Password for authentication (None for anonymous).
        timeout: Request timeout in seconds.
        max_retries: Number of retries for failed requests.
    """

    host: str = "localhost"
    port: int = 7200
    repository: str = "factcheck"
    username: str | None = None
    password: str | None = None
    timeout: int = 30
    max_retries: int = 3

    # Standard RDF namespace prefixes used throughout the project
    namespaces: dict[str, str] = field(default_factory=lambda: {
        "dbo": "http://dbpedia.org/ontology/",
        "dbr": "http://dbpedia.org/resource/",
        "dbp": "http://dbpedia.org/property/",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "foaf": "http://xmlns.com/foaf/0.1/",
        "dc": "http://purl.org/dc/elements/1.1/",
        "dct": "http://purl.org/dc/terms/",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    })

    def __post_init__(self):
        """Override fields from environment variables if present."""
        self.host = os.environ.get("GRAPHDB_HOST", self.host)
        self.port = int(os.environ.get("GRAPHDB_PORT", str(self.port)))
        self.repository = os.environ.get("GRAPHDB_REPOSITORY", self.repository)
        self.username = os.environ.get("GRAPHDB_USERNAME", self.username)
        self.password = os.environ.get("GRAPHDB_PASSWORD", self.password)
        self.timeout = int(os.environ.get("GRAPHDB_TIMEOUT", str(self.timeout)))
        self.max_retries = int(
            os.environ.get("GRAPHDB_MAX_RETRIES", str(self.max_retries))
        )

    # ----- Derived URLs -----

    @property
    def base_url(self) -> str:
        """Base URL for the GraphDB REST API (e.g. http://localhost:7200)."""
        return f"http://{self.host}:{self.port}"

    @property
    def repository_url(self) -> str:
        """SPARQL endpoint URL for the repository."""
        return f"{self.base_url}/repositories/{self.repository}"

    @property
    def statements_url(self) -> str:
        """URL for the statements endpoint (graph store protocol)."""
        return f"{self.repository_url}/statements"

    @property
    def rest_repositories_url(self) -> str:
        """REST API endpoint for repository management."""
        return f"{self.base_url}/rest/repositories"

    @property
    def rest_security_url(self) -> str:
        """REST API endpoint for user / security management."""
        return f"{self.base_url}/rest/security"

    @property
    def auth(self) -> tuple[str, str] | None:
        """Return (username, password) tuple for requests, or None."""
        if self.username and self.password:
            return (self.username, self.password)
        return None

    # ----- Helpers -----

    def namespace_prefixes_sparql(self) -> str:
        """Return all configured namespaces as a SPARQL PREFIX block.

        Example output:
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX dbr: <http://dbpedia.org/resource/>
            ...
        """
        lines = [f"PREFIX {k}: <{v}>" for k, v in self.namespaces.items()]
        return "\n".join(lines)

    def summary(self) -> str:
        """Return a human-readable configuration summary (masks password)."""
        auth_status = "enabled" if self.auth else "anonymous"
        return (
            f"GraphDB Config\n"
            f"  Base URL     : {self.base_url}\n"
            f"  Repository   : {self.repository}\n"
            f"  Endpoint     : {self.repository_url}\n"
            f"  Auth         : {auth_status}\n"
            f"  Timeout      : {self.timeout}s\n"
            f"  Max retries  : {self.max_retries}\n"
            f"  Namespaces   : {len(self.namespaces)}"
        )


# Module-level convenience: a shared default configuration instance.
default_config = GraphDBConfig()
