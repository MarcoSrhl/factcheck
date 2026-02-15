"""Generate large-scale training data for BERT and T5 from DBpedia.

Queries DBpedia SPARQL endpoint for real facts, then creates:
  - SUPPORTED examples from real triples
  - REFUTED examples by corrupting real triples (entity swapping)
  - NOT ENOUGH INFO examples from unrelated entity pairs

Uses ThreadPoolExecutor for parallel SPARQL queries.
"""

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.sparql_queries import (
    QUERY_REGISTRY,
    fetch_mixed_triplets,
    _uri_to_label,
    _run_query,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Additional SPARQL queries for more diversity
# ---------------------------------------------------------------------------

EXTRA_QUERIES: dict[str, dict] = {
    "nationalities": {
        "query": """
        SELECT DISTINCT ?person ?country WHERE {{
            ?person dbo:nationality ?country .
            ?person a dbo:Person .
            ?country a dbo:Country .
        }}
        LIMIT {limit}
        """,
        "vars": ["person", "country"],
        "predicate": "has nationality",
    },
    "genres": {
        "query": """
        SELECT DISTINCT ?work ?genre WHERE {{
            ?work dbo:genre ?genre .
            ?work a dbo:Work .
        }}
        LIMIT {limit}
        """,
        "vars": ["work", "genre"],
        "predicate": "belongs to genre",
    },
    "companies_founders": {
        "query": """
        SELECT DISTINCT ?company ?founder WHERE {{
            ?company dbo:founder ?founder .
            ?founder a dbo:Person .
        }}
        LIMIT {limit}
        """,
        "vars": ["company", "founder"],
        "predicate": "was founded by",
    },
    "spouse": {
        "query": """
        SELECT DISTINCT ?person ?spouse WHERE {{
            ?person dbo:spouse ?spouse .
            ?person a dbo:Person .
            ?spouse a dbo:Person .
        }}
        LIMIT {limit}
        """,
        "vars": ["person", "spouse"],
        "predicate": "is married to",
    },
    "country_of": {
        "query": """
        SELECT DISTINCT ?place ?country WHERE {{
            ?place dbo:country ?country .
            ?place a dbo:Place .
            ?country a dbo:Country .
        }}
        LIMIT {limit}
        """,
        "vars": ["place", "country"],
        "predicate": "is in",
    },
    "languages": {
        "query": """
        SELECT DISTINCT ?country ?lang WHERE {{
            ?country dbo:language ?lang .
            ?country a dbo:Country .
        }}
        LIMIT {limit}
        """,
        "vars": ["country", "lang"],
        "predicate": "has official language",
    },
    "awards": {
        "query": """
        SELECT DISTINCT ?person ?award WHERE {{
            ?person dbo:award ?award .
            ?person a dbo:Person .
        }}
        LIMIT {limit}
        """,
        "vars": ["person", "award"],
        "predicate": "received",
    },
    "known_for": {
        "query": """
        SELECT DISTINCT ?person ?thing WHERE {{
            ?person dbo:knownFor ?thing .
            ?person a dbo:Person .
        }}
        LIMIT {limit}
        """,
        "vars": ["person", "thing"],
        "predicate": "is known for",
    },
    "death_places": {
        "query": """
        SELECT DISTINCT ?person ?place WHERE {{
            ?person dbo:deathPlace ?place .
            ?person a dbo:Person .
            ?place a dbo:Place .
        }}
        LIMIT {limit}
        """,
        "vars": ["person", "place"],
        "predicate": "died in",
    },
    "headquarters": {
        "query": """
        SELECT DISTINCT ?org ?place WHERE {{
            ?org dbo:headquarter ?place .
            ?place a dbo:Place .
        }}
        LIMIT {limit}
        """,
        "vars": ["org", "place"],
        "predicate": "has headquarters in",
    },
    "developers": {
        "query": """
        SELECT DISTINCT ?product ?dev WHERE {{
            ?product dbo:developer ?dev .
        }}
        LIMIT {limit}
        """,
        "vars": ["product", "dev"],
        "predicate": "was developed by",
    },
    "rivers_countries": {
        "query": """
        SELECT DISTINCT ?river ?country WHERE {{
            ?river dbo:country ?country .
            ?river a dbo:River .
            ?country a dbo:Country .
        }}
        LIMIT {limit}
        """,
        "vars": ["river", "country"],
        "predicate": "flows through",
    },
}


def _fetch_extra_category(
    name: str, info: dict, limit: int
) -> list[tuple[str, str, str]]:
    """Fetch triplets for one extra category."""
    query = info["query"].format(limit=limit)
    rows = _run_query(query, info["vars"])
    predicate = info["predicate"]
    return [
        (_uri_to_label(row[0]), predicate, _uri_to_label(row[1]))
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Claim templates (predicate -> list of sentence templates)
# ---------------------------------------------------------------------------

CLAIM_TEMPLATES: dict[str, list[str]] = {
    "is capital of": [
        "{subject} is the capital of {object}",
        "The capital of {object} is {subject}",
        "{subject} serves as the capital of {object}",
    ],
    "was born in": [
        "{subject} was born in {object}",
        "{subject} is from {object}",
        "{object} is the birthplace of {subject}",
    ],
    "has occupation": [
        "{subject} works as a {object}",
        "{subject} is a {object}",
    ],
    "is located in": [
        "{subject} is located in {object}",
        "{subject} can be found in {object}",
        "{subject} is in {object}",
    ],
    "was founded in": [
        "{subject} was founded in {object}",
        "{subject} was established in {object}",
    ],
    "wrote": [
        "{subject} wrote {object}",
        "{object} was written by {subject}",
        "{subject} is the author of {object}",
    ],
    "is leader of": [
        "{subject} is the leader of {object}",
        "{subject} leads {object}",
    ],
    "studied at": [
        "{subject} studied at {object}",
        "{subject} attended {object}",
        "{subject} is an alumnus of {object}",
    ],
    "has nationality": [
        "{subject} is from {object}",
        "{subject} has {object} nationality",
    ],
    "belongs to genre": [
        "{subject} belongs to the {object} genre",
        "{subject} is a {object} work",
    ],
    "was founded by": [
        "{subject} was founded by {object}",
        "{object} founded {subject}",
    ],
    "is married to": [
        "{subject} is married to {object}",
        "{subject} and {object} are married",
    ],
    "is in": [
        "{subject} is in {object}",
        "{subject} is located in {object}",
    ],
    "has official language": [
        "The official language of {subject} is {object}",
        "{subject} speaks {object}",
    ],
    "received": [
        "{subject} received the {object}",
        "{subject} won the {object}",
    ],
    "is known for": [
        "{subject} is known for {object}",
        "{subject} is famous for {object}",
    ],
    "died in": [
        "{subject} died in {object}",
        "{object} is where {subject} died",
    ],
    "has headquarters in": [
        "{subject} has its headquarters in {object}",
        "{subject} is headquartered in {object}",
    ],
    "was developed by": [
        "{subject} was developed by {object}",
        "{object} developed {subject}",
    ],
    "flows through": [
        "{subject} flows through {object}",
        "The {subject} river passes through {object}",
    ],
}

# Evidence templates per predicate
EVIDENCE_TEMPLATES: dict[str, list[str]] = {
    "is capital of": [
        "DBpedia confirms {subject} is related to {object} via dbo:capital",
        "The knowledge base shows {subject} as the capital of {object} through dbo:capital, dbo:country",
    ],
    "was born in": [
        "DBpedia confirms {subject} is related to {object} via dbo:birthPlace",
        "The knowledge base records {object} as the birthplace of {subject}",
    ],
    "has occupation": [
        "DBpedia confirms {subject} has occupation {object} via dbo:occupation",
    ],
    "is located in": [
        "DBpedia confirms {subject} is related to {object} via dbo:location",
        "The knowledge base shows {subject} is located in {object} via dbo:location, wikiPageWikiLink",
    ],
    "was founded in": [
        "DBpedia confirms {subject} has foundingDate {object}",
    ],
    "wrote": [
        "DBpedia confirms {object} is related to {subject} via dbo:author",
    ],
    "is leader of": [
        "DBpedia confirms {subject} is related to {object} via dbo:leader",
    ],
    "studied at": [
        "DBpedia confirms {subject} is related to {object} via dbo:almaMater",
    ],
    "has nationality": [
        "DBpedia confirms {subject} is related to {object} via dbo:nationality",
    ],
    "belongs to genre": [
        "DBpedia confirms {subject} is related to {object} via dbo:genre",
    ],
    "was founded by": [
        "DBpedia confirms {subject} is related to {object} via dbo:founder",
    ],
    "is married to": [
        "DBpedia confirms {subject} is related to {object} via dbo:spouse",
    ],
    "is in": [
        "DBpedia confirms {subject} is related to {object} via dbo:country",
    ],
    "has official language": [
        "DBpedia confirms {subject} is related to {object} via dbo:language",
    ],
    "received": [
        "DBpedia confirms {subject} is related to {object} via dbo:award",
    ],
    "is known for": [
        "DBpedia confirms {subject} is related to {object} via dbo:knownFor",
    ],
    "died in": [
        "DBpedia confirms {subject} is related to {object} via dbo:deathPlace",
    ],
    "has headquarters in": [
        "DBpedia confirms {subject} is related to {object} via dbo:headquarter",
    ],
    "was developed by": [
        "DBpedia confirms {subject} is related to {object} via dbo:developer",
    ],
    "flows through": [
        "DBpedia confirms {subject} is related to {object} via dbo:country",
    ],
}

# T5 explanation templates per verdict
T5_EXPLANATION_TEMPLATES = {
    "SUPPORTED": [
        "This claim is supported by evidence. The knowledge base confirms that {subject} {predicate} {object} through a direct relationship in DBpedia.",
        "The evidence confirms this claim. DBpedia's knowledge graph shows that {subject} {predicate} {object}, as verified by the {pred_short} predicate.",
        "This claim is verified. According to DBpedia, {subject} {predicate} {object}. The knowledge base contains a direct relation confirming this fact.",
        "The claim is supported by the knowledge base. {subject} is confirmed to have a relationship with {object} via the {pred_short} property in DBpedia.",
        "This is a verified claim. The knowledge base records that {subject} {predicate} {object}, supporting the claim through structured data in DBpedia.",
    ],
    "REFUTED": [
        "This claim is refuted by the evidence. The knowledge base shows that {real_subject} {predicate} {real_object}, not {wrong_entity}. The claim contradicts established facts in DBpedia.",
        "The claim is incorrect. According to DBpedia, {real_subject} {predicate} {real_object}. The claim incorrectly states {wrong_entity} instead.",
        "This claim is refuted. DBpedia confirms that {real_subject} {predicate} {real_object}, which contradicts the claim involving {wrong_entity}.",
        "The evidence refutes this claim. The knowledge base records {real_subject} as being associated with {real_object}, not {wrong_entity} as the claim suggests.",
        "This is an incorrect claim. DBpedia data shows that {real_subject} {predicate} {real_object}. There is no such relationship with {wrong_entity}.",
    ],
    "NOT ENOUGH INFO": [
        "There is not enough information to verify or refute this claim. The knowledge base does not contain a direct relationship between {subject} and {object}.",
        "The available evidence is insufficient to confirm or deny this claim. No direct relation between {subject} and {object} was found in DBpedia.",
        "The evidence is insufficient for a definitive verdict. While both {subject} and {object} exist in the knowledge base, no direct connection between them is recorded.",
        "There is not enough data to verify this claim. DBpedia does not establish a clear relationship between {subject} and {object}, leaving the claim unverifiable.",
        "The knowledge base lacks sufficient evidence to verify this claim. No established connection between {subject} and {object} was found in the available data.",
    ],
}


# Predicate to short dbo name
PRED_TO_SHORT: dict[str, str] = {
    "is capital of": "dbo:capital",
    "was born in": "dbo:birthPlace",
    "has occupation": "dbo:occupation",
    "is located in": "dbo:location",
    "was founded in": "dbo:foundingDate",
    "wrote": "dbo:author",
    "is leader of": "dbo:leader",
    "studied at": "dbo:almaMater",
    "has nationality": "dbo:nationality",
    "belongs to genre": "dbo:genre",
    "was founded by": "dbo:founder",
    "is married to": "dbo:spouse",
    "is in": "dbo:country",
    "has official language": "dbo:language",
    "received": "dbo:award",
    "is known for": "dbo:knownFor",
    "died in": "dbo:deathPlace",
    "has headquarters in": "dbo:headquarter",
    "was developed by": "dbo:developer",
    "flows through": "dbo:country",
}


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------


def fetch_all_triplets(
    per_category: int = 1000,
    max_workers: int = 8,
) -> dict[str, list[tuple[str, str, str]]]:
    """Fetch triplets from all categories in parallel.

    Returns a dict mapping category name -> list of triplets.
    """
    all_triplets: dict[str, list[tuple[str, str, str]]] = {}

    def _fetch_standard(cat_name: str, fetch_fn, limit: int):
        return cat_name, fetch_fn(limit=limit)

    def _fetch_extra(cat_name: str, info: dict, limit: int):
        return cat_name, _fetch_extra_category(cat_name, info, limit)

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Standard categories from QUERY_REGISTRY
        for name, fn in QUERY_REGISTRY.items():
            futures.append(
                executor.submit(_fetch_standard, name, fn, per_category)
            )
        # Extra categories
        for name, info in EXTRA_QUERIES.items():
            futures.append(
                executor.submit(_fetch_extra, name, info, per_category)
            )

        for future in as_completed(futures):
            try:
                cat_name, triplets = future.result()
                all_triplets[cat_name] = triplets
                logger.info(
                    "Fetched %d triplets for '%s'", len(triplets), cat_name
                )
            except Exception as exc:
                logger.error("Failed to fetch category: %s", exc)

    total = sum(len(v) for v in all_triplets.values())
    logger.info("Total triplets fetched: %d across %d categories", total, len(all_triplets))
    return all_triplets


def generate_bert_data(
    triplets_by_cat: dict[str, list[tuple[str, str, str]]],
) -> list[dict]:
    """Generate BERT training data from fetched triplets.

    Creates SUPPORTED, REFUTED, and NOT ENOUGH INFO examples.
    """
    data: list[dict] = []
    all_triplets: list[tuple[str, str, str]] = []
    for cat_triplets in triplets_by_cat.values():
        all_triplets.extend(cat_triplets)

    if not all_triplets:
        logger.warning("No triplets to generate data from!")
        return data

    # Collect all subjects and objects for entity swapping
    all_subjects = list({t[0] for t in all_triplets})
    all_objects = list({t[2] for t in all_triplets})

    for subj, pred, obj in all_triplets:
        templates = CLAIM_TEMPLATES.get(pred, ["{subject} {predicate} {object}"])
        evidence_tmpls = EVIDENCE_TEMPLATES.get(
            pred, ["DBpedia confirms {subject} is related to {object}"]
        )

        # --- SUPPORTED ---
        claim = random.choice(templates).format(subject=subj, object=obj)
        evidence = random.choice(evidence_tmpls).format(subject=subj, object=obj)
        data.append({
            "claim": claim,
            "evidence": evidence,
            "label": "SUPPORTED",
        })

        # --- REFUTED (swap object with a random different one) ---
        wrong_obj = random.choice(all_objects)
        attempts = 0
        while wrong_obj == obj and attempts < 10:
            wrong_obj = random.choice(all_objects)
            attempts += 1
        if wrong_obj != obj:
            claim_ref = random.choice(templates).format(subject=subj, object=wrong_obj)
            evidence_ref = (
                f"DBpedia confirms {subj} is related to {obj} via "
                f"{PRED_TO_SHORT.get(pred, pred)}, not {wrong_obj}"
            )
            data.append({
                "claim": claim_ref,
                "evidence": evidence_ref,
                "label": "REFUTED",
            })

        # --- NOT ENOUGH INFO (50% chance to avoid imbalance) ---
        if random.random() < 0.5:
            rand_subj = random.choice(all_subjects)
            rand_obj = random.choice(all_objects)
            attempts = 0
            while (rand_subj, pred, rand_obj) in {(s, p, o) for s, p, o in all_triplets} and attempts < 10:
                rand_subj = random.choice(all_subjects)
                rand_obj = random.choice(all_objects)
                attempts += 1
            nei_templates = [
                f"{rand_subj} {pred} {rand_obj}",
                f"There is a connection between {rand_subj} and {rand_obj}",
            ]
            claim_nei = random.choice(nei_templates)
            evidence_nei = f"No direct relation found between {rand_subj} and {rand_obj} in DBpedia"
            data.append({
                "claim": claim_nei,
                "evidence": evidence_nei,
                "label": "NOT ENOUGH INFO",
            })

    random.shuffle(data)
    logger.info(
        "Generated %d BERT examples (S=%d, R=%d, N=%d)",
        len(data),
        sum(1 for d in data if d["label"] == "SUPPORTED"),
        sum(1 for d in data if d["label"] == "REFUTED"),
        sum(1 for d in data if d["label"] == "NOT ENOUGH INFO"),
    )
    return data


def generate_t5_data(
    triplets_by_cat: dict[str, list[tuple[str, str, str]]],
    max_examples: int = 5000,
) -> list[dict]:
    """Generate T5 explanation training data from fetched triplets."""
    data: list[dict] = []
    all_triplets: list[tuple[str, str, str]] = []
    for cat_triplets in triplets_by_cat.values():
        all_triplets.extend(cat_triplets)

    if not all_triplets:
        return data

    all_subjects = list({t[0] for t in all_triplets})
    all_objects = list({t[2] for t in all_triplets})

    random.shuffle(all_triplets)

    for subj, pred, obj in all_triplets:
        if len(data) >= max_examples:
            break

        pred_short = PRED_TO_SHORT.get(pred, pred)
        claim_templates = CLAIM_TEMPLATES.get(pred, ["{subject} {predicate} {object}"])
        evidence_tmpls = EVIDENCE_TEMPLATES.get(
            pred, ["DBpedia confirms {subject} is related to {object}"]
        )

        # --- SUPPORTED ---
        claim = random.choice(claim_templates).format(subject=subj, object=obj)
        evidence = random.choice(evidence_tmpls).format(subject=subj, object=obj)
        explanation = random.choice(T5_EXPLANATION_TEMPLATES["SUPPORTED"]).format(
            subject=subj, predicate=pred, object=obj, pred_short=pred_short,
        )
        data.append({
            "input": f"explain: claim: {claim} [SEP] verdict: SUPPORTED [SEP] evidence: {evidence}",
            "target": explanation,
        })

        # --- REFUTED ---
        wrong_obj = random.choice(all_objects)
        attempts = 0
        while wrong_obj == obj and attempts < 10:
            wrong_obj = random.choice(all_objects)
            attempts += 1
        if wrong_obj != obj:
            claim_ref = random.choice(claim_templates).format(subject=subj, object=wrong_obj)
            evidence_ref = (
                f"DBpedia confirms {subj} is related to {obj} via {pred_short}, not {wrong_obj}"
            )
            explanation_ref = random.choice(T5_EXPLANATION_TEMPLATES["REFUTED"]).format(
                real_subject=subj, predicate=pred, real_object=obj, wrong_entity=wrong_obj,
            )
            data.append({
                "input": f"explain: claim: {claim_ref} [SEP] verdict: REFUTED [SEP] evidence: {evidence_ref}",
                "target": explanation_ref,
            })

        # --- NOT ENOUGH INFO (50%) ---
        if random.random() < 0.5:
            rand_subj = random.choice(all_subjects)
            rand_obj = random.choice(all_objects)
            claim_nei = f"{rand_subj} {pred} {rand_obj}"
            evidence_nei = f"No direct relation found between {rand_subj} and {rand_obj} in DBpedia"
            explanation_nei = random.choice(T5_EXPLANATION_TEMPLATES["NOT ENOUGH INFO"]).format(
                subject=rand_subj, object=rand_obj,
            )
            data.append({
                "input": f"explain: claim: {claim_nei} [SEP] verdict: NOT ENOUGH INFO [SEP] evidence: {evidence_nei}",
                "target": explanation_nei,
            })

    random.shuffle(data)
    logger.info("Generated %d T5 examples", len(data))
    return data


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_all(
    per_category: int = 1000,
    max_workers: int = 8,
    t5_max: int = 5000,
    output_dir: str = "data",
) -> tuple[str, str]:
    """Fetch from DBpedia and generate both BERT and T5 training data.

    Returns paths to the saved JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching triplets from DBpedia ({len(QUERY_REGISTRY) + len(EXTRA_QUERIES)} categories, "
          f"up to {per_category} each, {max_workers} workers)...")
    start = time.time()
    triplets_by_cat = fetch_all_triplets(
        per_category=per_category, max_workers=max_workers
    )
    elapsed = time.time() - start

    total = sum(len(v) for v in triplets_by_cat.values())
    print(f"Fetched {total} triplets in {elapsed:.1f}s")
    for cat, trips in sorted(triplets_by_cat.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:25s}: {len(trips):>5d}")

    # Generate BERT data
    print("\nGenerating BERT training data...")
    bert_data = generate_bert_data(triplets_by_cat)
    bert_path = os.path.join(output_dir, "bert_training_data.json")
    with open(bert_path, "w") as f:
        json.dump(bert_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(bert_data)} BERT examples to {bert_path}")

    # Generate T5 data
    print(f"\nGenerating T5 training data (max {t5_max})...")
    t5_data = generate_t5_data(triplets_by_cat, max_examples=t5_max)
    t5_path = os.path.join(output_dir, "t5_training_data.json")
    with open(t5_path, "w") as f:
        json.dump(t5_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(t5_data)} T5 examples to {t5_path}")

    # Stats
    s = sum(1 for d in bert_data if d["label"] == "SUPPORTED")
    r = sum(1 for d in bert_data if d["label"] == "REFUTED")
    n = sum(1 for d in bert_data if d["label"] == "NOT ENOUGH INFO")
    print(f"\nBERT data distribution: SUPPORTED={s}, REFUTED={r}, NOT ENOUGH INFO={n}")

    return bert_path, t5_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate training data from DBpedia"
    )
    parser.add_argument(
        "--per-category", type=int, default=1000,
        help="Max triplets per SPARQL category (default: 1000)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel SPARQL workers (default: 8)",
    )
    parser.add_argument(
        "--t5-max", type=int, default=5000,
        help="Max T5 training examples (default: 5000)",
    )
    parser.add_argument(
        "--output", type=str, default="data",
        help="Output directory (default: data/)",
    )
    args = parser.parse_args()

    generate_all(
        per_category=args.per_category,
        max_workers=args.workers,
        t5_max=args.t5_max,
        output_dir=args.output,
    )
