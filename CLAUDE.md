# Automated Fact Checking System

## Objectif
Système de fact-checking automatisé qui, à partir d'une phrase (claim), 
détermine si elle est vraie ou fausse en utilisant DBpedia comme knowledge base.

## Architecture (Pipeline multi-étapes)
1. **Triplet Extraction** : Extraire (sujet, relation, objet) d'une phrase
2. **Entity Linking** : Mapper les entités extraites vers DBpedia
3. **Knowledge Base Query** : Interroger DBpedia via SPARQL/JSON
4. **Verdict** : Comparer et produire un verdict (TRUE/FALSE/UNVERIFIABLE)

## Stack technique
- Python 3.10+
- PyTorch + Transformers (HuggingFace)
- spaCy (NLP preprocessing)
- SPARQLWrapper (queries DBpedia)
- Jupyter Notebook pour la démo

## Structure des fichiers
- `src/triplet_extractor.py` — extraction de triplets
- `src/entity_linker.py` — liaison des entités vers DBpedia
- `src/knowledge_query.py` — requêtes SPARQL/JSON sur DBpedia  
- `src/fact_checker.py` — pipeline principal
- `src/model.py` — modèle neural (classifier)
- `notebooks/demo.ipynb` — notebook de démonstration
- `data/` — datasets
- `tests/` — tests unitaires
