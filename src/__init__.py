"""Factcheck package for automated fact verification.

Main components:
- FactChecker: Full pipeline for fact-checking claims
- FactClassifier: BERT-based neural classifier
- TripletExtractor: Extract subject-relation-object triplets
- EntityLinker: Link entities to DBpedia
- KnowledgeQuery: Query DBpedia knowledge base
"""

from factcheck.fact_checker import FactChecker
from factcheck.model import FactClassifier, LABEL_MAP, LABEL_TO_ID, NUM_LABELS
from factcheck.triplet_extractor import TripletExtractor
from factcheck.entity_linker import EntityLinker
from factcheck.knowledge_query import KnowledgeQuery

__version__ = "0.1.0"

__all__ = [
    "FactChecker",
    "FactClassifier",
    "TripletExtractor",
    "EntityLinker",
    "KnowledgeQuery",
    "LABEL_MAP",
    "LABEL_TO_ID",
    "NUM_LABELS",
]
