"""Manage users and roles in GraphDB for collaborative multi-user access.

This script provides a CLI to:
  - List existing users
  - Create a new user with a role
  - Delete a user
  - Update a user's role / repository permissions
  - Enable or disable security (free edition: basic access control)

GraphDB roles (Free edition):
  - ROLE_ADMIN  : full access
  - ROLE_REPO_MANAGER : create/delete repos
  - ROLE_USER   : read/write access to assigned repos

Usage:
    python -m db.scripts.manage_users list
    python -m db.scripts.manage_users create  alice --password s3cret --role writer
    python -m db.scripts.manage_users delete  alice
    python -m db.scripts.manage_users update  alice --role admin
    python -m db.scripts.manage_users security --enable
    python -m db.scripts.manage_users security --disable
"""

import argparse
import json
import logging
import sys

import requests

from db.config.graphdb_config import GraphDBConfig

logger = logging.getLogger(__name__)

# Mapping of friendly role names to GraphDB authority strings.
ROLE_MAP: dict[str, list[str]] = {
    "admin": ["ROLE_ADMIN"],
    "manager": ["ROLE_REPO_MANAGER"],
    "reader": ["ROLE_USER"],
    "writer": ["ROLE_USER"],
}


class UserManager:
    """CRUD operations for GraphDB users via the REST API."""

    def __init__(self, config: GraphDBConfig | None = None):
        self.config = config or GraphDBConfig()
        self.session = requests.Session()
        if self.config.auth:
            self.session.auth = self.config.auth

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _users_url(self) -> str:
        return f"{self.config.rest_security_url}/users"

    def _user_url(self, username: str) -> str:
        return f"{self._users_url}/{username}"

    def _make_granted_authorities(
        self, role: str, repository: str | None = None
    ) -> list[str]:
        """Build the grantedAuthorities list for a user.

        For 'reader' the user gets READ access; for 'writer' the user gets
        both READ and WRITE access to the configured repository.
        """
        authorities: list[str] = list(ROLE_MAP.get(role, ["ROLE_USER"]))

        repo = repository or self.config.repository
        if role == "reader":
            authorities.append(f"READ_REPO_{repo}")
        elif role in ("writer", "manager"):
            authorities.append(f"READ_REPO_{repo}")
            authorities.append(f"WRITE_REPO_{repo}")
        elif role == "admin":
            # Admin has access to everything; no per-repo entries needed.
            pass

        return authorities

    # ------------------------------------------------------------------
    # List users
    # ------------------------------------------------------------------

    def list_users(self) -> list[dict] | None:
        """Return the list of users from GraphDB, or None on error."""
        try:
            resp = self.session.get(self._users_url, timeout=self.config.timeout)
            if resp.status_code == 200:
                return resp.json()
            logger.error("Failed to list users (HTTP %d): %s", resp.status_code, resp.text)
            return None
        except requests.RequestException as exc:
            logger.error("Error listing users: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Create user
    # ------------------------------------------------------------------

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "reader",
        repository: str | None = None,
    ) -> bool:
        """Create a new GraphDB user.

        Parameters:
            username:   Unique login name.
            password:   Password for the new user.
            role:       One of 'admin', 'manager', 'reader', 'writer'.
            repository: Repository to grant access to (defaults to config).

        Returns True on success.
        """
        authorities = self._make_granted_authorities(role, repository)

        payload = {
            "username": username,
            "password": password,
            "grantedAuthorities": authorities,
            "appSettings": {
                "DEFAULT_INFERENCE": True,
                "DEFAULT_SAMEAS": True,
                "EXECUTE_COUNT": True,
                "IGNORE_SHARED_QUERIES": False,
            },
        }

        logger.info("Creating user '%s' with role '%s' ...", username, role)

        try:
            resp = self.session.post(
                self._users_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout,
            )
            if resp.status_code in (200, 201):
                logger.info("User '%s' created.", username)
                return True
            elif resp.status_code == 409:
                logger.warning("User '%s' already exists.", username)
                return False
            else:
                logger.error(
                    "Failed to create user '%s' (HTTP %d): %s",
                    username,
                    resp.status_code,
                    resp.text,
                )
                return False
        except requests.RequestException as exc:
            logger.error("Error creating user '%s': %s", username, exc)
            return False

    # ------------------------------------------------------------------
    # Delete user
    # ------------------------------------------------------------------

    def delete_user(self, username: str) -> bool:
        """Delete a GraphDB user. Returns True on success."""
        logger.info("Deleting user '%s' ...", username)
        try:
            resp = self.session.delete(
                self._user_url(username),
                timeout=self.config.timeout,
            )
            if resp.status_code in (200, 204):
                logger.info("User '%s' deleted.", username)
                return True
            elif resp.status_code == 404:
                logger.warning("User '%s' not found.", username)
                return False
            else:
                logger.error(
                    "Failed to delete user '%s' (HTTP %d): %s",
                    username,
                    resp.status_code,
                    resp.text,
                )
                return False
        except requests.RequestException as exc:
            logger.error("Error deleting user '%s': %s", username, exc)
            return False

    # ------------------------------------------------------------------
    # Update user role / permissions
    # ------------------------------------------------------------------

    def update_user(
        self,
        username: str,
        role: str | None = None,
        password: str | None = None,
        repository: str | None = None,
    ) -> bool:
        """Update an existing user's role and/or password.

        Only the fields provided will be changed.
        Returns True on success.
        """
        # First, fetch existing user data to merge.
        try:
            resp = self.session.get(
                self._user_url(username),
                timeout=self.config.timeout,
            )
            if resp.status_code == 404:
                logger.error("User '%s' not found.", username)
                return False
            resp.raise_for_status()
            user_data = resp.json()
        except requests.RequestException as exc:
            logger.error("Error fetching user '%s': %s", username, exc)
            return False

        # Build update payload.
        payload: dict = {}
        if role:
            payload["grantedAuthorities"] = self._make_granted_authorities(
                role, repository
            )
        else:
            payload["grantedAuthorities"] = user_data.get("grantedAuthorities", [])

        if password:
            payload["password"] = password

        logger.info("Updating user '%s' ...", username)
        try:
            resp = self.session.put(
                self._user_url(username),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout,
            )
            if resp.status_code in (200, 204):
                logger.info("User '%s' updated.", username)
                return True
            else:
                logger.error(
                    "Failed to update user '%s' (HTTP %d): %s",
                    username,
                    resp.status_code,
                    resp.text,
                )
                return False
        except requests.RequestException as exc:
            logger.error("Error updating user '%s': %s", username, exc)
            return False

    # ------------------------------------------------------------------
    # Security toggle
    # ------------------------------------------------------------------

    def set_security(self, enabled: bool) -> bool:
        """Enable or disable GraphDB access-control security.

        POST /rest/security with body "true" or "false".
        """
        value = "true" if enabled else "false"
        logger.info("Setting security to %s ...", value)

        try:
            resp = self.session.post(
                self.config.rest_security_url,
                data=value,
                headers={"Content-Type": "application/json"},
                timeout=self.config.timeout,
            )
            if resp.status_code in (200, 204):
                logger.info("Security %s.", "enabled" if enabled else "disabled")
                return True
            else:
                logger.error(
                    "Failed to set security (HTTP %d): %s",
                    resp.status_code,
                    resp.text,
                )
                return False
        except requests.RequestException as exc:
            logger.error("Error setting security: %s", exc)
            return False

    def is_security_enabled(self) -> bool | None:
        """Check if security is currently enabled. Returns None on error."""
        try:
            resp = self.session.get(
                self.config.rest_security_url,
                timeout=self.config.timeout,
            )
            if resp.status_code == 200:
                return resp.json() is True or resp.text.strip().lower() == "true"
            return None
        except requests.RequestException:
            return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_users(users: list[dict]):
    """Pretty-print a list of user dicts."""
    if not users:
        print("  (no users found)")
        return
    for u in users:
        name = u.get("username", "?")
        auths = u.get("grantedAuthorities", [])
        print(f"  {name:20s}  authorities={auths}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage GraphDB users for the fact-checker project."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- list --
    sub.add_parser("list", help="List all GraphDB users.")

    # -- create --
    p_create = sub.add_parser("create", help="Create a new user.")
    p_create.add_argument("username", help="Login name for the new user.")
    p_create.add_argument("--password", required=True, help="User password.")
    p_create.add_argument(
        "--role",
        choices=["admin", "manager", "reader", "writer"],
        default="reader",
        help="Role for the user (default: reader).",
    )

    # -- delete --
    p_delete = sub.add_parser("delete", help="Delete a user.")
    p_delete.add_argument("username", help="Username to delete.")

    # -- update --
    p_update = sub.add_parser("update", help="Update a user's role or password.")
    p_update.add_argument("username", help="Username to update.")
    p_update.add_argument(
        "--role",
        choices=["admin", "manager", "reader", "writer"],
        default=None,
        help="New role.",
    )
    p_update.add_argument("--password", default=None, help="New password.")

    # -- security --
    p_sec = sub.add_parser("security", help="Enable or disable security.")
    grp = p_sec.add_mutually_exclusive_group(required=True)
    grp.add_argument("--enable", action="store_true", help="Turn security on.")
    grp.add_argument("--disable", action="store_true", help="Turn security off.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mgr = UserManager()

    if args.command == "list":
        users = mgr.list_users()
        if users is not None:
            _print_users(users)
        else:
            print("Could not retrieve users. Is GraphDB running?")
            sys.exit(1)

    elif args.command == "create":
        ok = mgr.create_user(args.username, args.password, role=args.role)
        sys.exit(0 if ok else 1)

    elif args.command == "delete":
        ok = mgr.delete_user(args.username)
        sys.exit(0 if ok else 1)

    elif args.command == "update":
        if not args.role and not args.password:
            parser.error("Provide at least --role or --password to update.")
        ok = mgr.update_user(args.username, role=args.role, password=args.password)
        sys.exit(0 if ok else 1)

    elif args.command == "security":
        ok = mgr.set_security(enabled=args.enable)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
