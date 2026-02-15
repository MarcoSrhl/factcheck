"""Setup and initialize a GraphDB repository for the fact-checking project.

This script:
  1. Checks that GraphDB is reachable.
  2. Creates a new repository (default name: 'factcheck') via the REST API.
  3. Registers standard RDF namespaces.
  4. Optionally verifies the repository with a simple SPARQL query.

Usage:
    python -m db.scripts.setup_graphdb          # create with defaults
    python -m db.scripts.setup_graphdb --drop    # drop and recreate
"""

import argparse
import json
import logging
import sys
import time

import requests

from db.config.graphdb_config import GraphDBConfig

logger = logging.getLogger(__name__)

# Repository configuration template understood by GraphDB's REST API.
# Type "free" works with the GraphDB Free edition.
REPO_CONFIG_TEMPLATE = {
    "id": "",  # filled at runtime
    "type": "free",
    "title": "",
    "params": {
        "ruleset": {"name": "ruleset", "value": "rdfsplus-optimized"},
        "disableSameAs": {"name": "disableSameAs", "value": "true"},
        "baseURL": {
            "name": "baseURL",
            "value": "http://example.org/owlim#",
        },
        "repositoryType": {"name": "repositoryType", "value": "file-repository"},
        "enableContextIndex": {"name": "enableContextIndex", "value": "true"},
        "entityIdSize": {"name": "entityIdSize", "value": "32"},
        "enablePredicateList": {"name": "enablePredicateList", "value": "true"},
        "inMemoryLiteralProperties": {
            "name": "inMemoryLiteralProperties",
            "value": "true",
        },
    },
}


class GraphDBSetup:
    """Manages the lifecycle of a GraphDB repository."""

    def __init__(self, config: GraphDBConfig | None = None):
        self.config = config or GraphDBConfig()
        self.session = requests.Session()
        if self.config.auth:
            self.session.auth = self.config.auth

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_graphdb_running(self) -> bool:
        """Return True if the GraphDB instance is reachable."""
        try:
            resp = self.session.get(
                f"{self.config.base_url}/rest/repositories",
                timeout=self.config.timeout,
            )
            return resp.status_code == 200
        except requests.ConnectionError:
            return False
        except requests.RequestException as exc:
            logger.warning("Unexpected error checking GraphDB status: %s", exc)
            return False

    def wait_for_graphdb(self, max_wait: int = 60, interval: int = 5) -> bool:
        """Block until GraphDB is reachable or *max_wait* seconds elapse.

        Returns True if the instance became available, False on timeout.
        """
        logger.info(
            "Waiting up to %ds for GraphDB at %s ...",
            max_wait,
            self.config.base_url,
        )
        elapsed = 0
        while elapsed < max_wait:
            if self.is_graphdb_running():
                logger.info("GraphDB is running.")
                return True
            time.sleep(interval)
            elapsed += interval
        logger.error("GraphDB did not become available within %ds.", max_wait)
        return False

    # ------------------------------------------------------------------
    # Repository CRUD
    # ------------------------------------------------------------------

    def repository_exists(self) -> bool:
        """Check whether the configured repository already exists."""
        try:
            resp = self.session.get(
                self.config.rest_repositories_url,
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            repos = resp.json()
            return any(r.get("id") == self.config.repository for r in repos)
        except requests.RequestException as exc:
            logger.error("Failed to list repositories: %s", exc)
            return False

    def create_repository(self) -> bool:
        """Create the repository using GraphDB's REST API.

        Returns True on success, False otherwise.
        """
        if self.repository_exists():
            logger.info(
                "Repository '%s' already exists -- skipping creation.",
                self.config.repository,
            )
            return True

        repo_config = _build_repo_config(
            repo_id=self.config.repository,
            title=f"Fact-Checker knowledge base ({self.config.repository})",
        )

        logger.info(
            "Creating repository '%s' on %s ...",
            self.config.repository,
            self.config.base_url,
        )

        try:
            resp = self.session.post(
                self.config.rest_repositories_url,
                json=repo_config,
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout,
            )
            if resp.status_code in (200, 201):
                logger.info("Repository '%s' created successfully.", self.config.repository)
                return True
            else:
                logger.error(
                    "Failed to create repository (HTTP %d): %s",
                    resp.status_code,
                    resp.text,
                )
                return False
        except requests.RequestException as exc:
            logger.error("Error creating repository: %s", exc)
            return False

    def drop_repository(self) -> bool:
        """Delete the repository. Returns True on success."""
        if not self.repository_exists():
            logger.info("Repository '%s' does not exist.", self.config.repository)
            return True

        logger.warning("Dropping repository '%s' ...", self.config.repository)
        try:
            resp = self.session.delete(
                f"{self.config.rest_repositories_url}/{self.config.repository}",
                timeout=self.config.timeout,
            )
            if resp.status_code in (200, 204):
                logger.info("Repository '%s' dropped.", self.config.repository)
                return True
            else:
                logger.error(
                    "Failed to drop repository (HTTP %d): %s",
                    resp.status_code,
                    resp.text,
                )
                return False
        except requests.RequestException as exc:
            logger.error("Error dropping repository: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Namespace registration
    # ------------------------------------------------------------------

    def register_namespaces(self) -> bool:
        """Register all configured namespace prefixes in the repository.

        Uses the repository namespaces REST endpoint
        PUT /repositories/{repo}/namespaces/{prefix}
        """
        logger.info("Registering %d namespace(s) ...", len(self.config.namespaces))
        all_ok = True
        for prefix, uri in self.config.namespaces.items():
            try:
                resp = self.session.put(
                    f"{self.config.repository_url}/namespaces/{prefix}",
                    data=uri,
                    headers={"Content-Type": "text/plain"},
                    timeout=self.config.timeout,
                )
                if resp.status_code in (200, 204):
                    logger.debug("  Registered %s: <%s>", prefix, uri)
                else:
                    logger.warning(
                        "  Failed to register %s (HTTP %d): %s",
                        prefix,
                        resp.status_code,
                        resp.text,
                    )
                    all_ok = False
            except requests.RequestException as exc:
                logger.warning("  Error registering namespace %s: %s", prefix, exc)
                all_ok = False

        if all_ok:
            logger.info("All namespaces registered successfully.")
        return all_ok

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_repository(self) -> bool:
        """Run a trivial SPARQL query to confirm the repo is queryable."""
        try:
            resp = self.session.get(
                self.config.repository_url,
                params={"query": "SELECT (1 AS ?ping) WHERE {}"},
                headers={"Accept": "application/sparql-results+json"},
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            bindings = data.get("results", {}).get("bindings", [])
            if bindings:
                logger.info("Repository '%s' is queryable.", self.config.repository)
                return True
            logger.warning("Repository responded but returned no bindings.")
            return False
        except requests.RequestException as exc:
            logger.error("Verification query failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Full setup orchestration
    # ------------------------------------------------------------------

    def full_setup(self, drop_first: bool = False) -> bool:
        """Run all setup steps in order.

        1. Check GraphDB is reachable.
        2. Optionally drop existing repository.
        3. Create repository.
        4. Register namespaces.
        5. Verify.

        Returns True if all steps succeed.
        """
        if not self.is_graphdb_running():
            logger.error(
                "GraphDB is not reachable at %s. "
                "Please start GraphDB and try again.",
                self.config.base_url,
            )
            return False

        if drop_first:
            if not self.drop_repository():
                return False

        if not self.create_repository():
            return False

        if not self.register_namespaces():
            return False

        if not self.verify_repository():
            return False

        logger.info("Setup complete.")
        return True


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_repo_config(repo_id: str, title: str) -> dict:
    """Return a repository configuration dict for the REST API."""
    config = json.loads(json.dumps(REPO_CONFIG_TEMPLATE))  # deep copy
    config["id"] = repo_id
    config["title"] = title
    return config


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Set up the GraphDB repository for the fact-checker project."
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop the existing repository before creating a new one.",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Wait up to N seconds for GraphDB to become available.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = GraphDBConfig()
    logger.info("\n%s", config.summary())

    setup = GraphDBSetup(config)

    if args.wait > 0:
        if not setup.wait_for_graphdb(max_wait=args.wait):
            sys.exit(1)

    success = setup.full_setup(drop_first=args.drop)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
