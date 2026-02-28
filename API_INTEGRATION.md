# Factcheck Package - API Integration Guide

This guide shows how to integrate the `factcheck` package into your API for automated fact-checking.

## Installation

```bash
pip install git+https://github.com/MarcoSrhl/factcheck.git
python -m spacy download en_core_web_sm
```

Or add to `requirements.txt`:
```txt
factcheck @ git+https://github.com/MarcoSrhl/factcheck.git
spacy>=3.5.0
```

---

## Quick Start

### Option 1: Full Pipeline (Recommended)

Uses triplet extraction, entity linking, DBpedia queries, and neural classification:

```python
from factcheck import FactChecker

# Initialize once (e.g., at app startup)
checker = FactChecker(
    model_path="path/to/your/mlflow/model",  # or local model directory
    use_neural=True
)

# Check a claim
result = checker.check("Paris is the capital of France")

print(result)
# {
#     "claim": "Paris is the capital of France",
#     "verdict": "SUPPORTED",
#     "confidence": 0.98,
#     "evidence": [...],
#     "triplets": [("Paris", "be", "the capital of France")],
#     "entities": {"Paris": "http://dbpedia.org/resource/Paris", ...}
# }
```

### Option 2: Classifier Only (Fast)

If you already have evidence or want to skip entity linking:

```python
from factcheck import FactClassifier

# Initialize once
classifier = FactClassifier(model_path="path/to/your/mlflow/model")

# Predict with claim + evidence
result = classifier.predict(
    claim="Paris is the capital of France",
    evidence="Paris is the capital and most populous city of France."
)

print(result)
# {
#     "label": "SUPPORTED",
#     "confidence": 0.98,
#     "probabilities": {
#         "SUPPORTED": 0.98,
#         "REFUTED": 0.01,
#         "NOT ENOUGH INFO": 0.01
#     }
# }
```

---

## Loading Model from MLflow/DagsHub

### Method 1: Download model locally first

```bash
# Use your friend's push_to_mlflow script or download from DagsHub UI
# Model will be in a local directory like models/fact_checker
```

```python
classifier = FactClassifier(model_path="models/fact_checker")
```

### Method 2: Load directly from MLflow (if configured)

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("https://dagshub.com/MarcoSrhl/NLP-Fact-checking.mlflow")

# Load model by name and version
model_uri = "models:/fact-checker-bert/2"  # version 2
model = mlflow.pyfunc.load_model(model_uri)

# Or use the local model path after downloading
classifier = FactClassifier(model_path="path/to/downloaded/model")
```

---

## API Integration Example (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from factcheck import FactChecker

app = FastAPI()

# Initialize at startup
checker = FactChecker(model_path="models/fact_checker", use_neural=True)

class ClaimRequest(BaseModel):
    claim: str
    evidence: str | None = None

class FactCheckResponse(BaseModel):
    claim: str
    verdict: str
    confidence: float
    evidence: list[dict] | None = None

@app.post("/factcheck", response_model=FactCheckResponse)
async def fact_check(request: ClaimRequest):
    try:
        result = checker.check(request.claim)
        return FactCheckResponse(
            claim=result["claim"],
            verdict=result["verdict"],
            confidence=result["confidence"],
            evidence=result.get("evidence")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": checker.classifier is not None}
```

### Example Request

```bash
curl -X POST "http://localhost:8000/factcheck" \
  -H "Content-Type: application/json" \
  -d '{"claim": "Paris is the capital of France"}'
```

### Example Response

```json
{
  "claim": "Paris is the capital of France",
  "verdict": "SUPPORTED",
  "confidence": 0.98,
  "evidence": [
    {
      "source": "DBpedia",
      "text": "Paris is the capital and most populous city of France...",
      "uri": "http://dbpedia.org/resource/Paris"
    }
  ]
}
```

---

## Component-Level Usage

If you need more control, use individual components:

```python
from factcheck import TripletExtractor, EntityLinker, KnowledgeQuery, FactClassifier

# Extract triplets
extractor = TripletExtractor()
triplets = extractor.extract("Paris is the capital of France")
# [("Paris", "be", "the capital of France")]

# Link entities to DBpedia
linker = EntityLinker()
paris_uri = linker.link("Paris")
# "http://dbpedia.org/resource/Paris"

# Query knowledge base
kb = KnowledgeQuery()
evidence = kb.get_json_property(paris_uri, "dbo:abstract")
# "Paris is the capital and most populous city of France..."

# Classify
classifier = FactClassifier(model_path="models/fact_checker")
result = classifier.predict(
    claim="Paris is the capital of France",
    evidence=evidence
)
# {"label": "SUPPORTED", "confidence": 0.98, ...}
```

---

## Model Input/Output

### Input
- **claim** (required): String - the statement to fact-check
- **evidence** (optional): String - supporting context/text

### Output
```python
{
    "label": str,          # "SUPPORTED" | "REFUTED" | "NOT ENOUGH INFO"
    "confidence": float,   # 0.0 to 1.0
    "probabilities": {     # Softmax scores for all classes
        "SUPPORTED": float,
        "REFUTED": float,
        "NOT ENOUGH INFO": float
    }
}
```

---

## Error Handling

```python
from factcheck import FactChecker

checker = FactChecker(model_path="models/fact_checker")

try:
    result = checker.check("Some claim")
except FileNotFoundError:
    print("Model not found - check model_path")
except ConnectionError:
    print("DBpedia connection failed - KB queries unavailable")
except Exception as e:
    print(f"Fact-checking failed: {e}")
```

---

## Performance Tips

1. **Initialize once**: Create `FactChecker` or `FactClassifier` at app startup, not per request
2. **Batch predictions**: Use `classifier.predict_batch()` for multiple claims
3. **Skip KB queries**: If you have evidence, use `FactClassifier` directly instead of full pipeline
4. **Cache results**: Cache frequently checked claims
5. **Timeout DBpedia**: Set timeouts for API calls to avoid hanging

```python
# Batch example
claims = ["Claim 1", "Claim 2", "Claim 3"]
evidences = ["Evidence 1", "Evidence 2", "Evidence 3"]
results = classifier.predict_batch(claims, evidences)
```

---

## Environment Variables (Optional)

If using database tracking:

```bash
# .env file
NEON_DB_URL=postgresql://user:pass@host/db?sslmode=require
```

```python
from factcheck.database import NeonDB

db = NeonDB()  # Reads from NEON_DB_URL env var
```

---

## Troubleshooting

### Issue: "No module named 'en_core_web_sm'"
```bash
python -m spacy download en_core_web_sm
```

### Issue: "Model not found"
Ensure model is downloaded locally or path is correct:
```python
import os
print(os.path.exists("models/fact_checker"))  # Should be True
print(os.listdir("models/fact_checker"))      # Should show model files
```

### Issue: DBpedia timeout
DBpedia queries may be slow or timeout. Use shorter timeout or skip KB:
```python
checker = FactChecker(model_path="...", use_neural=True)
# Triplet extraction and entity linking will still work
```

---

## API Endpoints Reference

### Minimal API Implementation

```python
from fastapi import FastAPI
from factcheck import FactClassifier

app = FastAPI()
classifier = FactClassifier(model_path="models/fact_checker")

@app.post("/predict")
def predict(claim: str, evidence: str = ""):
    return classifier.predict(claim, evidence)
```

### Full Pipeline API

```python
from fastapi import FastAPI
from factcheck import FactChecker

app = FastAPI()
checker = FactChecker(model_path="models/fact_checker")

@app.post("/check")
def check(claim: str):
    return checker.check(claim)
```

---

## Package Components

Available imports:
```python
from factcheck import (
    FactChecker,        # Full pipeline
    FactClassifier,     # Neural classifier only
    TripletExtractor,   # Extract (subject, relation, object)
    EntityLinker,       # Link entities to DBpedia
    KnowledgeQuery,     # Query DBpedia
    LABEL_MAP,          # {0: "SUPPORTED", 1: "REFUTED", 2: "NOT ENOUGH INFO"}
    LABEL_TO_ID,        # {"SUPPORTED": 0, "REFUTED": 1, "NOT ENOUGH INFO": 2}
    NUM_LABELS,         # 3
)
```

---

## Updates

To update the package when code changes:
```bash
pip install --upgrade --force-reinstall git+https://github.com/MarcoSrhl/factcheck.git
```

Or install from specific commit:
```bash
pip install git+https://github.com/MarcoSrhl/factcheck.git@7d803f1
```

---

## Support

For issues or questions about the package:
- GitHub: https://github.com/MarcoSrhl/factcheck
- Check model on DagsHub: https://dagshub.com/MarcoSrhl/NLP-Fact-checking.mlflow
