# Automated Fact-Checking System

An automated fact-checking pipeline that determines whether English claims are **SUPPORTED**, **REFUTED**, or if there is **NOT ENOUGH INFO**, using DBpedia as a knowledge base and BERT as a neural classifier.

## Architecture

```
Input Claim
    │
    ▼
┌─────────────────────┐
│ 1. Triplet Extractor │  spaCy dependency parsing
│    (subject, pred,   │  extracts (S, P, O) triplets
│     object)          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 2. Entity Linker     │  DBpedia Lookup API
│    text → URI        │  maps entities to DBpedia
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 3. Knowledge Query   │  SPARQL + JSON endpoints
│    verify triplet    │  checks if relation exists
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 4. Neural Classifier │  BERT (bert-base-uncased)
│    claim + evidence  │  3-class classification
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 5. Verdict           │  Combines KB evidence
│    SUPPORTED /       │  with neural prediction
│    REFUTED /         │
│    NOT ENOUGH INFO   │
└─────────────────────┘
```

## Project Structure

```
fact-checker/
├── src/
│   ├── triplet_extractor.py   # spaCy-based triplet extraction
│   ├── entity_linker.py       # DBpedia Lookup entity linking
│   ├── knowledge_query.py     # SPARQL/JSON knowledge base queries
│   ├── model.py               # BERT classifier wrapper
│   ├── train.py               # Training script
│   └── fact_checker.py        # Main pipeline orchestrator
├── tests/
│   ├── test_triplet_extractor.py
│   ├── test_entity_linker.py
│   ├── test_knowledge_query.py
│   └── test_fact_checker.py
├── notebooks/
│   └── demo.ipynb             # Interactive demo notebook
├── models/                    # Saved trained models
├── data/                      # Datasets
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd fact-checker

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

### Quick Start

```python
from src.fact_checker import FactChecker, format_result

checker = FactChecker()
result = checker.check("Paris is the capital of France")
print(format_result(result))
```

### Train the Model

```bash
# Train with synthetic data (default)
python -m src.train --epochs 10

# Train with custom dataset
python -m src.train --data data/train.json --epochs 10 --batch-size 8
```

The training data JSON format:
```json
[
  {
    "claim": "Paris is the capital of France",
    "evidence": "Paris is the capital and most populous city of France.",
    "label": "SUPPORTED"
  }
]
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Demo Notebook

```bash
cd notebooks
jupyter notebook demo.ipynb
```

## Components

### Triplet Extractor (`src/triplet_extractor.py`)
Uses spaCy dependency parsing to extract (subject, predicate, object) triplets from English sentences. Handles copular constructions ("X is Y"), passive voice ("X was born in Y"), and active verbal sentences.

### Entity Linker (`src/entity_linker.py`)
Maps text entities to DBpedia URIs using the DBpedia Lookup API. Includes result caching and error handling.

### Knowledge Query (`src/knowledge_query.py`)
Two verification methods:
- **SPARQL**: Queries the DBpedia SPARQL endpoint to find relations between entities
- **JSON**: Fetches entity data from DBpedia's JSON endpoint as fallback

### Neural Classifier (`src/model.py`)
BERT-based (bert-base-uncased) 3-class classifier. Predicts SUPPORTED/REFUTED/NOT ENOUGH INFO given a claim and optional evidence text. Supports CPU and GPU.

### Pipeline (`src/fact_checker.py`)
Orchestrates all components and combines KB evidence with neural predictions for a final verdict with confidence score.

## Results

On the synthetic evaluation set (10 claims):

| Metric    | Score |
|-----------|-------|
| Accuracy  | ~70%  |
| Precision | ~0.70 |
| Recall    | ~0.70 |
| F1-Score  | ~0.70 |

Note: Performance is limited by the synthetic training data. Fine-tuning on larger datasets (FEVER, LIAR) would significantly improve results.

## Tech Stack

- **Python 3.10+**
- **spaCy** - NLP preprocessing and dependency parsing
- **Transformers** (HuggingFace) - BERT model
- **PyTorch** - Deep learning backend
- **SPARQLWrapper** - DBpedia SPARQL queries
- **scikit-learn** - Metrics and evaluation
