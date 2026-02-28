"""Main fact-checking pipeline orchestrating all components."""

from __future__ import annotations

import logging
import os
from typing import Optional

from factcheck.triplet_extractor import TripletExtractor
from factcheck.entity_linker import EntityLinker
from factcheck.knowledge_query import KnowledgeQuery
from factcheck.model import FactClassifier, LABEL_MAP

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/fact_checker"
DEFAULT_GAN_PATH = "models/gan"
DEFAULT_EXPLAINER_PATH = "models/explainer"


class FactChecker:
    """End-to-end fact-checking pipeline.

    Steps:
    1. Extract triplets from the claim
    2. Link entities to DBpedia
    3. Query DBpedia for evidence
    4. Use neural classifier for verdict
    5. (Optional) Run the GAN discriminator on triplet embeddings
    6. Combine KB evidence + neural prediction + GAN score for final verdict
    7. (Optional) Generate an explanation for the verdict
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_neural: bool = True,
        use_gan: bool = False,
        gan_path: Optional[str] = None,
        use_explainer: bool = False,
        explainer_model_path: Optional[str] = None,
    ) -> None:
        self.extractor = TripletExtractor()
        self.linker = EntityLinker()
        self.kb = KnowledgeQuery()

        # ----- Neural classifier ------------------------------------------
        self.use_neural = use_neural
        self.classifier: Optional[FactClassifier] = None
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

        # ----- GAN discriminator ------------------------------------------
        self.use_gan = use_gan
        self.gan = None
        if use_gan:
            self.gan = self._load_gan(gan_path)

        # ----- Explainer --------------------------------------------------
        self.use_explainer = use_explainer
        self.explainer = None
        if use_explainer:
            self.explainer = self._load_explainer(explainer_model_path)

    # ----- Explainer loading helper ---------------------------------------

    @staticmethod
    def _load_explainer(
        explainer_model_path: Optional[str] = None,
    ) -> Optional["FactExplainer"]:
        """Try to load the FactExplainer.  Returns ``None`` on failure."""
        from factcheck.explainer import FactExplainer

        path = explainer_model_path or DEFAULT_EXPLAINER_PATH
        try:
            explainer = FactExplainer(
                use_t5=True,
                t5_model_path=path,
                use_attention=True,
            )
            logger.info("Loaded FactExplainer with T5 from %s", path)
            return explainer
        except Exception as exc:
            logger.warning(
                "Could not load FactExplainer: %s. Explanations will be unavailable.",
                exc,
            )
            return None

    # ----- GAN loading helper --------------------------------------------

    @staticmethod
    def _load_gan(
        gan_path: Optional[str] = None,
    ) -> Optional["FactGAN"]:
        """Try to load a trained FactGAN.  Returns ``None`` on failure."""
        from factcheck.gan_model import FactGAN

        path = gan_path or DEFAULT_GAN_PATH
        try:
            gan = FactGAN()
        except Exception as exc:
            logger.warning(
                "Could not initialise FactGAN: %s. GAN disabled.", exc
            )
            return None

        if os.path.isdir(path):
            try:
                gan.load(path)
                logger.info("Loaded FactGAN from %s", path)
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Could not load GAN weights from %s: %s. "
                    "Old MLP checkpoint may be incompatible with BERT architecture. "
                    "Using untrained GAN.",
                    path,
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "Could not load GAN weights from %s: %s. "
                    "Using untrained GAN.",
                    path,
                    exc,
                )
        else:
            logger.warning(
                "GAN directory %s does not exist. Using untrained GAN.",
                path,
            )
        return gan

    def check(self, claim: str) -> dict:
        """Run the full fact-checking pipeline on a claim.

        Returns a dict with:
          - claim: original claim text
          - triplets: extracted (subject, predicate, object) tuples
          - entities: linked DBpedia URIs
          - kb_evidence: knowledge base verification results
          - neural_prediction: neural model prediction (if enabled)
          - gan_score: GAN discriminator score (if enabled, float 0-1)
          - verdict: final verdict (SUPPORTED / REFUTED / NOT ENOUGH INFO)
          - confidence: confidence score
          - explanation: multi-layered explanation dict (if explainer enabled)
        """
        result: dict = {
            "claim": claim,
            "triplets": [],
            "entities": {},
            "kb_evidence": [],
            "neural_prediction": None,
            "gan_score": None,
            "verdict": "NOT ENOUGH INFO",
            "confidence": 0.0,
            "explanation": None,
        }

        # Step 1: Extract triplets
        triplets = self.extractor.extract(claim)
        result["triplets"] = [(s, p, o) for s, p, o in triplets]
        logger.info(f"Extracted {len(triplets)} triplet(s) from: {claim}")

        # Step 2: Entity linking
        entity_uris: dict[str, str] = {}
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
        kb_results: list[dict] = []
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

        # Step 5: GAN discriminator scoring
        gan_score = self._run_gan_discriminator(triplets)
        result["gan_score"] = gan_score

        # Step 6: Final verdict
        verdict, confidence = self._combine_verdicts(
            kb_results,
            result.get("neural_prediction"),
            gan_score=gan_score,
        )
        result["verdict"] = verdict
        result["confidence"] = confidence

        # Step 7: Generate explanation (if enabled)
        if self.use_explainer and self.explainer is not None:
            try:
                explanation = self.explainer.explain(
                    fact_check_result=result,
                    classifier=self.classifier,
                )
                result["explanation"] = explanation
            except Exception as exc:
                logger.warning("Explainer failed: %s", exc)

        return result

    # ----- GAN helper ----------------------------------------------------

    def _run_gan_discriminator(
        self, triplets: list[tuple[str, str, str]]
    ) -> Optional[float]:
        """Score the claim's triplets through the GAN discriminator.

        Returns the *average* discriminator score across all triplets,
        or ``None`` if the GAN is disabled or no triplets were extracted.
        """
        if not self.use_gan or self.gan is None or not triplets:
            return None

        try:
            scores = self.gan.discriminate_triplets(triplets)  # (n, 1)
            avg_score: float = scores.mean().item()
            logger.info("GAN discriminator avg score: %.4f", avg_score)
            return avg_score
        except Exception as exc:
            logger.warning("GAN discriminator failed: %s", exc)
            return None

    def _build_evidence_text(self, kb_results: list[dict], entity_uris: dict) -> str:
        """Build evidence text from KB results for the neural model.

        Formats evidence to match training data patterns so the neural model
        can recognise them. Training uses formats like:
          - "DBpedia confirms X is related to Y via dbo:pred"
          - "No direct relation found between X and Y in DBpedia"
        """
        parts = []
        for kr in kb_results:
            subj = kr["triplet"][0]
            obj = kr["triplet"][2]

            if kr["found"]:
                preds = kr["predicates"][:3]
                pred_names = [p.split("/")[-1] for p in preds]
                dbo_preds = ", ".join(pred_names)
                parts.append(
                    f"DBpedia confirms {subj} is related to {obj} via {dbo_preds}"
                )
            else:
                subj_uri = kr.get("subject_uri")
                if subj_uri:
                    props = self.kb.get_entity_properties(subj_uri)
                    if props:
                        prop_parts = []
                        for prop_name, values in list(props.items())[:3]:
                            val_str = ", ".join(values[:2])
                            prop_parts.append(f"{subj} {prop_name} is {val_str}")
                        parts.append(
                            f"No direct relation found between {subj} and {obj} in DBpedia. "
                            + ". ".join(prop_parts)
                        )
                    else:
                        parts.append(
                            f"No direct relation found between {subj} and {obj} in DBpedia"
                        )
                else:
                    parts.append(
                        f"No direct relation found between {subj} and {obj} in DBpedia"
                    )
        return ". ".join(parts) if parts else ""

    # Mapping from claim predicates to DBpedia property names for contradiction detection
    _PREDICATE_TO_PROPERTY = {
        "born in": "birthPlace",
        "was born in": "birthPlace",
        "born": "birthPlace",
        "capital": "capital",
        "located in": "location",
        "is located in": "location",
        "country": "country",
        "be": "country",  # for "X is the capital of Y" patterns
    }

    def _check_property_contradiction(
        self, kb_results: list[dict]
    ) -> bool | None:
        """Check if subject properties contradict the claimed object.

        Returns True if a contradiction is found, False if properties support
        the claim, None if no relevant properties available.
        """
        for kr in kb_results:
            if kr["found"]:
                continue  # Direct relation found, no need to check
            subj_uri = kr.get("subject_uri")
            obj_uri = kr.get("object_uri")
            if not subj_uri:
                continue

            triplet = kr["triplet"]
            predicate = triplet[1]
            obj_text = triplet[2].lower()

            # Get the subject's properties
            props = self.kb.get_entity_properties(subj_uri)
            if not props:
                continue

            # Check if any property values match or contradict the claimed object
            for prop_name, values in props.items():
                for value in values:
                    value_lower = value.lower()
                    # If the property value contains the claimed object → no contradiction
                    if obj_text in value_lower or value_lower in obj_text:
                        return False
                    # If the object URI name is in the value → no contradiction
                    if obj_uri:
                        obj_name = obj_uri.split("/")[-1].replace("_", " ").lower()
                        if obj_name in value_lower or value_lower in obj_name:
                            return False

            # We have properties but none match the claimed object → contradiction
            return True

        return None

    def _combine_verdicts(
        self,
        kb_results: list[dict],
        neural_prediction: Optional[dict] = None,
        gan_score: Optional[float] = None,
    ) -> tuple[str, float]:
        """Combine KB evidence and neural prediction into a final verdict.

        Strategy: the neural classifier is the primary signal. KB evidence
        adjusts confidence up (agreement) or down (disagreement) but never
        overrides the neural label.
        """
        kb_found = any(kr["found"] for kr in kb_results) if kb_results else False

        # No neural model: fall back to KB-only
        if not neural_prediction:
            if kb_found:
                return "SUPPORTED", 0.6
            return "NOT ENOUGH INFO", 0.3

        neural_label = neural_prediction["label"]
        neural_conf = neural_prediction["confidence"]

        # KB agreement boost / disagreement penalty
        if kb_found and neural_label == "SUPPORTED":
            confidence = min(0.95, neural_conf * 1.2)
        elif kb_found and neural_label == "REFUTED":
            confidence = neural_conf * 0.8
        elif not kb_found and neural_label == "SUPPORTED":
            confidence = neural_conf * 0.7
        elif not kb_found and neural_label == "REFUTED":
            confidence = min(0.95, neural_conf * 1.1)
        else:
            confidence = neural_conf

        confidence = min(0.99, max(0.1, confidence))
        return neural_label, confidence

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
        np_pred = result["neural_prediction"]
        lines.append(f"Neural: {np_pred['label']} ({np_pred['confidence']:.3f})")
    if result.get("gan_score") is not None:
        lines.append(f"GAN Discriminator Score: {result['gan_score']:.4f}")
    if result.get("explanation"):
        explanation = result["explanation"]
        nl = explanation.get("natural_explanation")
        if nl:
            lines.append(f"Explanation: {nl}")
        breakdown = explanation.get("confidence_breakdown")
        if breakdown and breakdown.get("formatted"):
            lines.append(breakdown["formatted"])
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse as _argparse

    logging.basicConfig(level=logging.INFO)

    _parser = _argparse.ArgumentParser(description="Run the fact-checking pipeline.")
    _parser.add_argument(
        "--gan", action="store_true", help="Enable GAN discriminator scoring."
    )
    _parser.add_argument(
        "--gan-path", type=str, default=None, help="Path to trained GAN model."
    )
    _parser.add_argument(
        "--explain", action="store_true", help="Enable explainability module."
    )
    _parser.add_argument(
        "--explainer-path", type=str, default=None, help="Path to trained explainer model."
    )
    _args = _parser.parse_args()

    checker = FactChecker(
        use_gan=_args.gan,
        gan_path=_args.gan_path,
        use_explainer=_args.explain,
        explainer_model_path=_args.explainer_path,
    )

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
