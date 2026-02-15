# Database Setup -- Local GraphDB for Fact-Checking

This document covers how to install, configure, and operate the local GraphDB
instance used by the fact-checking system.

---

## Architecture overview

```
                    +------------------+
                    |   DBpedia (web)  |
                    |  SPARQL endpoint |
                    +--------+---------+
                             |
                 CONSTRUCT queries (bulk fetch)
                             |
                             v
+-----------+      +-------------------+      +------------------+
|  fact_    | ---> |   Local GraphDB   | <--- | manage_users.py  |
|  checker  |      |   (localhost:7200)|      | (REST API)       |
|  pipeline |      |   repo: factcheck |      +------------------+
+-----------+      +-------------------+
      |                     ^
      |  SPARQL SELECT/ASK  |
      +---------------------+
```

**Data flow:**
1. `load_dbpedia_data.py` fetches RDF triples from DBpedia's public endpoint.
2. The triples are loaded into GraphDB's `factcheck` repository.
3. The fact-checking pipeline queries the local GraphDB instance (fast, no rate
   limits) instead of hitting the public DBpedia endpoint every time.

---

## 1. Install GraphDB Free

GraphDB Free is available from Ontotext:

1. Go to <https://www.ontotext.com/products/graphdb/graphdb-free/> and download
   the installer for your platform (Linux, macOS, or Windows).
2. Follow the installation instructions for your OS.
3. After installation, start GraphDB:
   ```bash
   # Linux / macOS (if installed via standalone zip)
   cd /path/to/graphdb-free
   ./bin/graphdb

   # Or with Docker
   docker run -d --name graphdb \
       -p 7200:7200 \
       ontotext/graphdb:10.5.0 \
       --GDB_HEAP_SIZE=2g
   ```
4. Open the Workbench UI in a browser: <http://localhost:7200>.

---

## 2. Initialize the repository

Run the setup script from the project root:

```bash
python -m db.scripts.setup_graphdb
```

This will:
- Verify that GraphDB is running on `localhost:7200`.
- Create the `factcheck` repository with sensible defaults.
- Register all namespace prefixes (dbo, dbr, rdfs, owl, foaf, etc.).
- Run a verification query to confirm the repo is queryable.

**Options:**

| Flag | Description |
|------|-------------|
| `--drop` | Drop the existing repository first, then recreate it. |
| `--wait SECONDS` | Wait up to N seconds for GraphDB to become available. |

Example:
```bash
python -m db.scripts.setup_graphdb --drop --wait 30
```

---

## 3. Load data from DBpedia

Populate the local repository with RDF triples from DBpedia:

```bash
python -m db.scripts.load_dbpedia_data
```

**Categories loaded (by default all are fetched):**

| Category | Description | Example predicate |
|----------|-------------|-------------------|
| `capitals` | Countries and their capitals | `dbo:capital` |
| `birthplaces` | People and birth locations | `dbo:birthPlace` |
| `organizations` | Organisations and locations | `dbo:location` |
| `events` | Historical events and dates | `dbo:date` |
| `science` | Scientists and contributions | `dbo:knownFor` |

**Options:**

| Flag | Description |
|------|-------------|
| `--category NAME` | Load only one category. |
| `--limit N` | Max results per CONSTRUCT query (default: 1000). |
| `--dry-run` | Fetch from DBpedia but skip the upload to GraphDB. |

Examples:
```bash
# Load only capital-city data, up to 2000 results
python -m db.scripts.load_dbpedia_data --category capitals --limit 2000

# Preview what would be loaded without writing
python -m db.scripts.load_dbpedia_data --dry-run
```

The loader supports **incremental loading**: it checks how many triples already
exist for each predicate before deciding what to insert, so running the script
twice will not duplicate data.

---

## 4. Manage users

GraphDB supports multi-user access control. Manage users with:

```bash
# List users
python -m db.scripts.manage_users list

# Create a reader
python -m db.scripts.manage_users create alice --password s3cret --role reader

# Create a writer
python -m db.scripts.manage_users create bob --password p4ssw0rd --role writer

# Promote to admin
python -m db.scripts.manage_users update alice --role admin

# Delete a user
python -m db.scripts.manage_users delete bob

# Enable / disable security
python -m db.scripts.manage_users security --enable
python -m db.scripts.manage_users security --disable
```

**Available roles:**

| Role | Permissions |
|------|-------------|
| `reader` | Read-only access to the `factcheck` repository |
| `writer` | Read + write access to the `factcheck` repository |
| `manager` | Read + write + repository management |
| `admin` | Full access to everything |

> **Note:** Security must be enabled (`--enable`) for user authentication to
> take effect. With security disabled, all requests are anonymous.

---

## 5. Query the local instance

### From the fact-checker pipeline

The existing `KnowledgeQuery` class now accepts a `use_local` flag:

```python
from src.knowledge_query import KnowledgeQuery

# Query the local GraphDB instance
kq = KnowledgeQuery(use_local=True)
result = kq.verify_triplet(
    "http://dbpedia.org/resource/Paris",
    "http://dbpedia.org/resource/France",
)
```

### Using the SPARQL templates directly

```python
from db.queries.sparql_templates import SPARQLTemplates

tpl = SPARQLTemplates()

# Check if a triple exists
query = tpl.ask_triple_exists(
    "http://dbpedia.org/resource/Paris",
    "http://dbpedia.org/ontology/capital",
    "http://dbpedia.org/resource/France",
)

# Insert a new fact
query = tpl.insert_triple(
    "http://dbpedia.org/resource/Ottawa",
    "http://dbpedia.org/ontology/capital",
    "http://dbpedia.org/resource/Canada",
)
```

### From the GraphDB Workbench

Open <http://localhost:7200> and navigate to **SPARQL** in the left sidebar.
Select the `factcheck` repository and run queries interactively.

---

## 6. Environment variable overrides

All connection settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPHDB_HOST` | `localhost` | GraphDB hostname |
| `GRAPHDB_PORT` | `7200` | GraphDB port |
| `GRAPHDB_REPOSITORY` | `factcheck` | Repository name |
| `GRAPHDB_USERNAME` | *(none)* | Auth username |
| `GRAPHDB_PASSWORD` | *(none)* | Auth password |
| `GRAPHDB_TIMEOUT` | `30` | Request timeout in seconds |
| `GRAPHDB_MAX_RETRIES` | `3` | Max retries on failure |

Example:
```bash
export GRAPHDB_HOST=192.168.1.50
export GRAPHDB_USERNAME=admin
export GRAPHDB_PASSWORD=secret
python -m db.scripts.load_dbpedia_data
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "GraphDB is not reachable" | Make sure GraphDB is running: `curl http://localhost:7200/rest/repositories` |
| "Repository already exists" | Use `--drop` flag to recreate, or skip the setup step. |
| DBpedia queries time out | Reduce `--limit` or try again later (public endpoint may be slow). |
| Authentication errors | Check `GRAPHDB_USERNAME` / `GRAPHDB_PASSWORD` env vars. |
| Docker port conflict | Map a different host port: `-p 7201:7200` and set `GRAPHDB_PORT=7201`. |
