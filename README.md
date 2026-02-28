# Automated Fact-Checking System

An automated fact-checking pipeline that determines whether English claims are **SUPPORTED**, **REFUTED**, or if there is **NOT ENOUGH INFO**, using DBpedia as a knowledge base, BERT as the primary neural classifier, and T5 for natural language explanations. An optional BERT-based GAN provides triplet plausibility scoring.

## Architecture

```
Input Claim
    │
    ▼
┌──────────────────────────┐
│ 1. Triplet Extraction     │  spaCy dependency parsing
│    (subject, predicate,   │  Copular / verbal / fallback strategies
│     object)               │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ 2. Entity Linking         │  DBpedia Lookup API
│    text → DBpedia URI     │  Disambiguation scoring (label similarity,
│                           │  exact match, URI quality, popularity)
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ 3. Knowledge Base Query   │  SPARQL + JSON endpoints
│    Verify triplet against │  DBpedia SPARQL and JSON API
│    DBpedia                │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ 4. Evidence Building      │  Formats KB relations as natural
│    KB facts → text        │  language matching training templates
│                           │  (e.g. "DBpedia confirms X via Y")
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ 5. Neural Classifier      │  BERT (bert-base-uncased)
│    claim + evidence →     │  3-class sequence classification
│    label + confidence     │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ 6. GAN Discriminator      │  BERT-based GAN
│    (optional)             │  Triplet plausibility scoring
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ 7. Verdict Combination    │  Neural classifier is primary signal;
│    + Explainability       │  KB evidence adjusts confidence
│                           │  (agreement ↑ / disagreement ↓)
│    SUPPORTED / REFUTED /  │
│    NOT ENOUGH INFO        │  Optional: T5 explanation, KB
│                           │  reasoning chain, attention analysis
└──────────────────────────┘
```

## Project Structure

```
fact-checker/
├── src/
│   ├── triplet_extractor.py      # spaCy-based triplet extraction
│   ├── entity_linker.py          # DBpedia Lookup entity linking + disambiguation
│   ├── knowledge_query.py        # SPARQL/JSON knowledge base queries
│   ├── sparql_queries.py         # SPARQL query templates for 8 DBpedia categories
│   ├── model.py                  # BERT 3-class classifier
│   ├── gan_model.py              # BERT-based GAN (generator + discriminator)
│   ├── gan_trainer.py            # GAN training loop
│   ├── explainer.py              # Explainability module (T5, KB reasoning, attention)
│   ├── fact_checker.py           # Main pipeline orchestrator
│   ├── train.py                  # BERT classifier training
│   ├── train_explainer.py        # T5 explainer training
│   ├── generate_training_data.py # DBpedia-based training data generation
│   ├── split_data.py             # Stratified train/validation split
│   └── validate_pipeline.py      # End-to-end pipeline validation
├── tests/
│   ├── test_triplet_extractor.py
│   ├── test_entity_linker.py
│   ├── test_knowledge_query.py
│   ├── test_fact_checker.py
│   ├── test_explainer.py
│   └── test_gan_model.py
├── notebooks/
│   └── demo.ipynb                # Interactive demo notebook
├── models/                       # Saved trained models
│   ├── fact_checker/             #   BERT classifier
│   ├── explainer/                #   T5 explanation generator
│   └── gan/                      #   BERT-based GAN
├── data/                         # Datasets (55K+ examples)
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

### With Explainability

```python
checker = FactChecker(use_gan=True, use_explainer=True)
result = checker.check("Barack Obama was born in Hawaii")
print(format_result(result))
# Includes: verdict, confidence, KB reasoning chain, T5 explanation,
#           attention highlights, and confidence decomposition
```

### Train the Models

```bash
# Generate training data from DBpedia (55K+ examples)
python -m src.generate_training_data --per-category 1000 --workers 8

# Split into train/validation
python -m src.split_data

# Train BERT classifier
python -m src.train --data data/train.json --epochs 10 --batch-size 8

# Train T5 explainer
python -m src.train_explainer --epochs 8

# Train BERT-based GAN
python -m src.gan_trainer --epochs 50 --batch-size 16 --output models/gan
```

### Validate the Pipeline

```bash
# End-to-end validation on held-out data
python -m src.validate_pipeline

# Quick regression benchmark (20 curated claims)
python -m src.validate_pipeline --benchmark
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

## Pipeline Components

### 1. Triplet Extraction (`src/triplet_extractor.py`)

Uses spaCy dependency parsing to extract (subject, predicate, object) triplets via three strategies:
- **Copular**: "X is Y" constructions (AUX root → nsubj + attr/acomp)
- **Verbal**: Active and passive voice ("X was born in Y", "X wrote Y")
- **Fallback**: Scans for any nsubj/nsubjpass and related objects

Noun chunks are expanded to full spans for richer entity mentions.

### 2. Entity Linking (`src/entity_linker.py`)

Maps text entities to DBpedia URIs using the DBpedia Lookup API with multi-factor disambiguation scoring:

| Factor | Weight | Description |
|--------|--------|-------------|
| Label similarity | 0.40 | SequenceMatcher ratio between query and candidate label |
| Exact match | 0.20 | Bonus for exact string match |
| URI quality | 0.25 | Penalizes `List_of_`, `Category:`, `_(disambiguation)` URIs |
| Popularity | 0.15 | Log-normalized `refCount` from DBpedia |

Includes entity text cleaning (strips determiners, possessives, parentheticals), result caching, and a fallback that constructs and verifies direct DBpedia URIs.

### 3. Knowledge Base Query (`src/knowledge_query.py`)

Queries the public DBpedia SPARQL endpoint + JSON API for fact verification.

Methods:
- `sparql_check_relation`: Finds all predicates between two entities
- `sparql_get_property` / `json_get_property_values`: Gets values for a specific property
- `sparql_ask`: Boolean ASK query for a specific triple
- `get_entity_properties`: Fetches key properties (birthPlace, capital, country, etc.) for evidence building
- `verify_triplet`: SPARQL-first with JSON fallback

### 4. Neural Classifier (`src/model.py`)

BERT-based (`bert-base-uncased`) 3-class sequence classifier:
- Input: Proper BERT sentence-pair encoding `(claim, evidence)` with `token_type_ids` (max 256 tokens)
- Output: predicted label + softmax confidence
- Labels: SUPPORTED, REFUTED, NOT ENOUGH INFO
- Supports CPU, CUDA, and MPS (Apple Silicon)

### 5. BERT-based GAN (`src/gan_model.py`)

Adversarial architecture for triplet plausibility scoring:

- **Generator**: `BertForMaskedLM` that masks subject/object tokens, injects noise, and reconstructs via Gumbel-Softmax for differentiable sampling
- **Discriminator**: `BertModel` with spectral-normalized classification head, scores triplets in [0, 1]
- **Training stabilization**: Feature matching loss, instance noise, diversity loss, MLM reconstruction loss, label smoothing (0.9), 3:1 G:D update ratio, temperature annealing

### 6. Explainability Module (`src/explainer.py`)

Four complementary explanation strategies:

| Strategy | Method |
|----------|--------|
| **KB Reasoning** | Structured step-by-step chain: claim assertion → entity linking → KB verification → semantic match → summary |
| **T5 Explanation** | Free-form natural language explanation via fine-tuned T5-small (beam search, 150 tokens) |
| **Attention Analysis** | Highlights top-attended tokens from BERT's final layer (aggregated across heads) |
| **Confidence Decomposition** | Per-component breakdown of how KB, neural classifier, and GAN each contributed |

### 7. Verdict Combination (`src/fact_checker.py`)

The neural classifier is the primary signal; KB evidence adjusts confidence but never overrides the neural label:
- KB agrees with neural prediction → confidence boosted (×1.1–1.2)
- KB disagrees with neural prediction → confidence reduced (×0.7–0.8)
- No neural model available → falls back to KB-only heuristic
- All confidences clamped to [0.1, 0.99]

## Training Data

Training data is automatically generated from DBpedia using 20 SPARQL categories:

**Standard categories** (8): capitals, birth places, occupations, locations, founding dates, authored works, country leaders, alma maters

**Extended categories** (12): nationalities, genres, company founders, spouses, countries, languages, awards, known for, death places, headquarters, developers, rivers/countries

Generation process:
1. Parallel SPARQL queries fetch up to 1000 triplets per category
2. **SUPPORTED**: Real triplets formatted via natural language templates (2-3 per predicate); 30% use inference-style evidence (`"DBpedia confirms X via dbo:pred"`)
3. **REFUTED**: Same-type entity swapping (harder negatives from the same predicate pool); 50% use property-based evidence format
4. **NOT ENOUGH INFO**: Random unrelated entity pairs (80% probability) + cross-predicate NEI examples (valid entities, wrong relation)

| Dataset | Examples | Description |
|---------|----------|-------------|
| `bert_training_data.json` | 55,736 | Full BERT training data (SUPPORTED: 18,339 / REFUTED: 18,339 / NEI: 19,058) |
| `train.json` | 44,591 | 80% stratified train split |
| `validation.json` | 11,145 | 20% stratified validation split |
| `t5_training_data.json` | 5,000 | T5 explainer training data |

## ML Experiment Tracking

The system can be integrated with MLflow for experiment tracking and model versioning:

```bash
# Train models (MLflow integration can be added to training scripts)
python -m src.train --data data/train.json --epochs 10
python -m src.train_explainer --epochs 8
```

> **Note**: The previous GraphDB local database setup has been removed. The system now queries DBpedia directly via SPARQL/JSON APIs.

## Results

### Pipeline Validation (49 claims, end-to-end)

| Metric | Score |
|--------|-------|
| **Accuracy** | **53.06%** |

> **Note:** These results were obtained before the latest changes (proper sentence-pair encoding, simplified verdict logic, improved evidence alignment). Re-run `python -m src.validate_pipeline` and `python -m src.validate_pipeline --benchmark` to get updated numbers.

Confusion matrix:

| | Predicted SUPPORTED | Predicted REFUTED | Predicted NEI |
|---|---|---|---|
| **Actual SUPPORTED** | 8 | 5 | 7 |
| **Actual REFUTED** | 5 | 11 | 4 |
| **Actual NEI** | 0 | 2 | 7 |

Key observations:
- Best performance on **REFUTED** claims (55% recall)
- **NOT ENOUGH INFO** correctly identified when no KB evidence is found
- Main error source: SUPPORTED claims misclassified as NEI when DBpedia lacks the relevant relation

### Limitations

- Performance is bounded by DBpedia coverage — claims about entities or relations not in DBpedia default to neural-only classification
- The public DBpedia SPARQL endpoint has rate limits that can slow batch evaluation
- Training data is synthetically generated from DBpedia templates, which limits linguistic diversity

## Tech Stack

- **Python 3.10+**
- **spaCy** — NLP preprocessing and dependency parsing
- **Transformers** (HuggingFace) — BERT classifier, BERT-based GAN, T5 explainer
- **PyTorch** — Deep learning backend
- **SPARQLWrapper** — DBpedia SPARQL queries
- **scikit-learn** — Metrics and evaluation
- **sentence-transformers** — Semantic similarity
