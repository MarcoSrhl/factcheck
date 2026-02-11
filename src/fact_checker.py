"""Main fact-checking pipeline orchestrating all components."""

import logging
import os

from src.triplet_extractor import TripletExtractor
from src.entity_linker import EntityLinker
from src.knowledge_query import KnowledgeQuery
from src.model import FactClassifier, LABEL_MAP

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/fact_checker"


class FactChecker:
    """End-to-end fact-checking pipeline.

    Steps:
    1. Extract triplets from the claim
    2. Link entities to DBpedia
    3. Query DBpedia for evidence
    4. Use neural classifier for verdict
    5. Combine KB evidence + neural prediction for final verdict
    """

    def __init__(self, model_path: str | None = None, use_neural: bool = True):
        self.extractor = TripletExtractor()
        self.linker = EntityLinker()
        self.kb = KnowledgeQuery()

        self.use_neural = use_neural
        self.classifier = None
        if use_neural:
            path = model_path or DEFAULT_MODEL_PATH
            if os.path.exists(path):
                self.classifier = FactClassifier(model_path=path)
                logger.info(f"Loaded neural model from {path}")
            else:
                logger.warning(
                    f"No model found at {path}. Using base BERT (untrained)."
                )
                self.classifier = FactClassifier()

    def check(self, claim: str) -> dict:
        """Run the full fact-checking pipeline on a claim.

        Returns a dict with:
          - claim: original claim text
          - triplets: extracted (subject, predicate, object) tuples
          - entities: linked DBpedia URIs
          - kb_evidence: knowledge base verification results
          - neural_prediction: neural model prediction (if enabled)
          - verdict: final verdict (SUPPORTED / REFUTED / NOT ENOUGH INFO)
          - confidence: confidence score
        """
        result = {
            "claim": claim,
            "triplets": [],
            "entities": {},
            "kb_evidence": [],
            "neural_prediction": None,
            "verdict": "NOT ENOUGH INFO",
            "confidence": 0.0,
        }

        # Step 1: Extract triplets
        triplets = self.extractor.extract(claim)
        result["triplets"] = [(s, p, o) for s, p, o in triplets]
        logger.info(f"Extracted {len(triplets)} triplet(s) from: {claim}")

        # Step 2: Entity linking
        entity_uris = {}
        for subject, _, obj in triplets:
            if subject not in entity_uris:
                uri = self.linker.link(subject)
                if uri:
                    entity_uris[subject] = uri
            if obj not in entity_uris:
                uri = self.linker.link(obj)
                if uri:
                    entity_uris[obj] = uri
        result["entities"] = entity_uris

        # Step 3: Query knowledge base
        kb_results = []
        for subject, predicate, obj in triplets:
            subj_uri = entity_uris.get(subject)
            obj_uri = entity_uris.get(obj)
            verification = self.kb.verify_triplet(subj_uri, obj_uri)
            kb_results.append({
                "triplet": (subject, predicate, obj),
                "subject_uri": subj_uri,
                "object_uri": obj_uri,
                **verification,
            })
        result["kb_evidence"] = kb_results

        # Step 4: Neural classification
        evidence_text = self._build_evidence_text(kb_results, entity_uris)
        if self.classifier:
            neural_result = self.classifier.predict(claim, evidence_text)
            result["neural_prediction"] = neural_result

        # Step 5: Final verdict
        verdict, confidence = self._combine_verdicts(kb_results, result.get("neural_prediction"))
        result["verdict"] = verdict
        result["confidence"] = confidence

        return result

    def _build_evidence_text(self, kb_results: list[dict], entity_uris: dict) -> str:
        """Build evidence text from KB results for the neural model."""
        parts = []
        for kr in kb_results:
            if kr["found"]:
                subj = kr["triplet"][0]
                obj = kr["triplet"][2]
                preds = kr["predicates"][:3]
                pred_names = [p.split("/")[-1] for p in preds]
                parts.append(f"{subj} is related to {obj} via {', '.join(pred_names)}")
            else:
                parts.append(f"No relation found between {kr['triplet'][0]} and {kr['triplet'][2]}")
        return ". ".join(parts) if parts else ""

    def _combine_verdicts(
        self,
        kb_results: list[dict],
        neural_prediction: dict | None,
    ) -> tuple[str, float]:
        """Combine KB evidence and neural prediction into a final verdict."""
        kb_found = any(kr["found"] for kr in kb_results) if kb_results else False
        kb_all_found = all(kr["found"] for kr in kb_results) if kb_results else False

        if neural_prediction:
            neural_label = neural_prediction["label"]
            neural_conf = neural_prediction["confidence"]

            # If KB confirms relation exists and neural says SUPPORTED -> high confidence SUPPORTED
            if kb_all_found and neural_label == "SUPPORTED":
                return "SUPPORTED", min(0.95, (neural_conf + 1.0) / 2)

            # If KB confirms relation exists but neural says REFUTED -> trust neural with lower confidence
            if kb_found and neural_label == "REFUTED":
                return "REFUTED", neural_conf * 0.7

            # If KB finds nothing and neural says REFUTED -> REFUTED
            if not kb_found and neural_label == "REFUTED":
                return "REFUTED", neural_conf

            # If KB finds something and neural says NOT ENOUGH INFO -> lean SUPPORTED
            if kb_found and neural_label == "NOT ENOUGH INFO":
                return "SUPPORTED", 0.5

            # If KB finds nothing and neural says SUPPORTED -> trust neural with lower confidence
            if not kb_found and neural_label == "SUPPORTED":
                return "SUPPORTED", neural_conf * 0.6

            # Otherwise, trust neural prediction
            return neural_label, neural_conf

        # No neural model: rely on KB only
        if kb_all_found:
            return "SUPPORTED", 0.7
        elif kb_found:
            return "SUPPORTED", 0.5
        else:
            return "NOT ENOUGH INFO", 0.3

    def check_batch(self, claims: list[str]) -> list[dict]:
        """Check multiple claims."""
        return [self.check(claim) for claim in claims]


def format_result(result: dict) -> str:
    """Format a fact-check result for display."""
    lines = [
        f"Claim: {result['claim']}",
        f"Verdict: {result['verdict']} (confidence: {result['confidence']:.2f})",
    ]
    if result["triplets"]:
        lines.append("Triplets:")
        for s, p, o in result["triplets"]:
            lines.append(f"  ({s}, {p}, {o})")
    if result["entities"]:
        lines.append("Linked Entities:")
        for name, uri in result["entities"].items():
            lines.append(f"  {name} -> {uri}")
    if result["kb_evidence"]:
        lines.append("KB Evidence:")
        for ev in result["kb_evidence"]:
            status = "FOUND" if ev["found"] else "NOT FOUND"
            lines.append(f"  {ev['triplet'][0]} <-> {ev['triplet'][2]}: {status}")
            if ev["predicates"]:
                for p in ev["predicates"][:3]:
                    lines.append(f"    via {p}")
    if result["neural_prediction"]:
        np = result["neural_prediction"]
        lines.append(f"Neural: {np['label']} ({np['confidence']:.3f})")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    checker = FactChecker()

    claims = [
        "Paris is the capital of France",
        "Barack Obama was born in Hawaii",
        "The Earth is flat",
        "The Eiffel Tower is located in Paris",
        "Albert Einstein developed the theory of relativity",
    ]

    for claim in claims:
        result = checker.check(claim)
        print("\n" + "=" * 60)
        print(format_result(result))
